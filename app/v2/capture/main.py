from __future__ import annotations

import importlib
import inspect
import os
from contextlib import asynccontextmanager
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from app.v2.capture.config import CaptureConfig
from app.v2.capture.journal import DisabledObserver, InMemoryCaptureJournal
from app.v2.capture.reference_api import (
    ReferenceQueryConfig,
    ReferenceQueryController,
    ReferenceQueryFailed,
    ReferenceQueryTimeout,
    ReferenceQueryUnavailable,
)
from app.v2.capture.service import RithmicCaptureService


_FORBIDDEN_FACTORY_FRAGMENTS = (
    ".api.",
    ".copier.",
    ".execution.",
    ".intelligence.",
    ".learning.",
    ".shadow",
    "openai",
    "scout",
)


def _load_factory(path: str) -> Callable[[CaptureConfig], Any]:
    module_name, separator, attribute = path.partition(":")
    normalized = f".{module_name.lower()}."
    if not separator or not module_name.startswith("app.v2.") or any(
        fragment in normalized for fragment in _FORBIDDEN_FACTORY_FRAGMENTS
    ):
        raise ValueError("capture factory path is not allowed")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("capture factory is not callable")
    return factory


async def _construct(factory_path: str, config: CaptureConfig):
    value = _load_factory(factory_path)(config)
    return await value if inspect.isawaitable(value) else value


async def build_service(config: CaptureConfig | None = None) -> RithmicCaptureService:
    settings = config or CaptureConfig.from_mapping()
    if settings.preflight_blockers:
        return RithmicCaptureService(
            settings,
            observer=DisabledObserver(),
            journal=InMemoryCaptureJournal(),
        )

    try:
        journal = await _construct(settings.journal_factory or "", settings)
        # Construct the non-networked durable sink first.  If persistence is
        # unavailable, no observer (and therefore no external transport
        # resource) is created.
        observer = await _construct(settings.observer_factory or "", settings)
        return RithmicCaptureService(settings, observer=observer, journal=journal)
    except Exception:
        # Factory errors are intentionally reduced to a non-sensitive blocker.
        # The HTTP process remains live so Render can surface the failed state.
        return RithmicCaptureService(
            settings,
            observer=DisabledObserver(),
            journal=InMemoryCaptureJournal(),
            startup_blockers=("capture_component_factory_failed",),
        )


@asynccontextmanager
async def lifespan(application: FastAPI):
    service = await build_service()
    application.state.capture_service = service
    application.state.reference_queries = ReferenceQueryController(
        service,
        ReferenceQueryConfig.from_mapping(os.environ),
    )
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(
    title="MooneyCapitol Rithmic Read-Only Capture",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


def _service() -> RithmicCaptureService:
    return app.state.capture_service


def _reference_queries() -> ReferenceQueryController:
    return app.state.reference_queries


def _authorize_reference_query(token: str | None) -> ReferenceQueryController:
    controller = _reference_queries()
    authorized = controller.authorize(token)
    if not controller.config.available:
        raise HTTPException(status_code=503, detail="reference service unavailable")
    if not authorized:
        raise HTTPException(
            status_code=401,
            detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return controller


def _reference_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ReferenceQueryUnavailable):
        return HTTPException(status_code=503, detail="reference service unavailable")
    if isinstance(exc, ReferenceQueryTimeout):
        return HTTPException(status_code=504, detail="reference query timed out")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail="reference query rejected")
    if isinstance(exc, ReferenceQueryFailed):
        return HTTPException(status_code=502, detail="reference query failed")
    return HTTPException(status_code=502, detail="reference query failed")


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    # Do not echo query text, tokens, or framework validation internals.
    return JSONResponse({"detail": "request validation failed"}, status_code=422)


@app.get("/health")
async def health() -> JSONResponse:
    snapshot = _service().health()
    return JSONResponse(snapshot.as_dict(), status_code=200 if snapshot.live else 503)


@app.get("/ready")
async def ready() -> JSONResponse:
    snapshot = _service().health()
    return JSONResponse(snapshot.as_dict(), status_code=200 if snapshot.ready else 503)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> Response:
    snapshot = _service().health()
    reference = _reference_queries()
    body = "\n".join(
        (
            "# TYPE mooney_rithmic_capture_live gauge",
            f"mooney_rithmic_capture_live {1 if snapshot.live else 0}",
            "# TYPE mooney_rithmic_capture_ready gauge",
            f"mooney_rithmic_capture_ready {1 if snapshot.ready else 0}",
            "# TYPE mooney_rithmic_capture_buffered_events gauge",
            f"mooney_rithmic_capture_buffered_events {snapshot.buffered_event_count}",
            "# TYPE mooney_rithmic_capture_journal_depth gauge",
            f"mooney_rithmic_capture_journal_depth {snapshot.journal_depth}",
            "# TYPE mooney_rithmic_capture_submission_enabled gauge",
            "mooney_rithmic_capture_submission_enabled 0",
            "# TYPE mooney_rithmic_reference_query_available gauge",
            "mooney_rithmic_reference_query_available "
            f"{1 if reference.config.available and _service().ticker_reference_available else 0}",
            "",
        )
    )
    return PlainTextResponse(body)


@app.get("/reference/symbols")
async def reference_symbols(
    q: str = Query(min_length=2, max_length=32),
    exchange: str | None = Query(default=None, max_length=32),
    product_code: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=25, ge=1, le=100),
    token: str | None = Header(
        default=None, alias="X-Rithmic-Reference-Token", max_length=512
    ),
) -> JSONResponse:
    controller = _authorize_reference_query(token)
    try:
        result = await controller.search(
            q,
            exchange=exchange,
            product_code=product_code,
            limit=limit,
        )
    except Exception as exc:
        raise _reference_error(exc) from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.get("/reference/contracts/{symbol}")
async def contract_reference(
    symbol: str = Path(min_length=1, max_length=32),
    exchange: str = Query(min_length=1, max_length=32),
    token: str | None = Header(
        default=None, alias="X-Rithmic-Reference-Token", max_length=512
    ),
) -> JSONResponse:
    controller = _authorize_reference_query(token)
    try:
        result = await controller.reference(symbol, exchange)
    except Exception as exc:
        raise _reference_error(exc) from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.get("/reference/tick-sizes/{tick_size_type}")
async def reference_tick_sizes(
    tick_size_type: str = Path(min_length=1, max_length=64),
    token: str | None = Header(
        default=None, alias="X-Rithmic-Reference-Token", max_length=512
    ),
) -> JSONResponse:
    controller = _authorize_reference_query(token)
    try:
        result = await controller.tick_sizes(tick_size_type)
    except Exception as exc:
        raise _reference_error(exc) from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


def main() -> None:
    port = int(os.getenv("PORT", "10000"))
    # Query text and operational headers never belong in access logs. Aggregate
    # health/metrics remain available for operating the service.
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
