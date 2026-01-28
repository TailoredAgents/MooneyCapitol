from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.core.config_store import CONFIG, ensure_config_initialized, refresh_config
from app.observability.logging import get_logger
from app.observability.sentry import init_sentry
from app.services.kv_store import assert_store_available
from app.services.live_state import ensure_lanes_initialized
from app.services.runtime import ensure_worker_tick_initialized
from app.api.routes.health import router as health_router
from app.api.routes.config import router as config_router
from app.api.routes.watchlist import router as watchlist_router
from app.api.routes.setups import router as setups_router
from app.api.routes.reports import router as reports_router
from app.api.routes.learning import router as learning_router
from app.api.routes.slash import router as slash_router
from app.api.routes.slack import router as slack_router
from app.api.ws import router as ws_router
from app.api.pages import router as pages_router


logger = get_logger("api")
settings = Settings()  # env-based


@asynccontextmanager
async def lifespan(app: FastAPI):
    dsn = init_sentry()
    if dsn:
        logger.info("sentry.init", dsn=True)
    else:
        logger.info("sentry.init", dsn=False)
    assert_store_available()
    ensure_config_initialized()
    ensure_lanes_initialized()
    ensure_worker_tick_initialized()
    refresh_config()
    yield


app = FastAPI(title="Trading Agent API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(config_router)
app.include_router(watchlist_router)
app.include_router(setups_router)
app.include_router(reports_router)
app.include_router(learning_router)
app.include_router(slash_router)
app.include_router(slack_router)
app.include_router(ws_router)
app.include_router(pages_router)
