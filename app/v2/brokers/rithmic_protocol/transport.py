from __future__ import annotations

import inspect
import ssl
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from .constants import BROKER_MUTATION_TEMPLATE_IDS, OUTBOUND_READ_ONLY_TEMPLATE_IDS, template_name
from .errors import InsecureEndpoint, InvalidFrame, MutationTemplateRejected, OutboundTemplateRejected
from .framing import extract_template_id


class WebSocketLike(Protocol):
    async def send(self, message: bytes) -> Any: ...

    async def recv(self) -> bytes | str: ...

    async def close(self, code: int = 1000, reason: str = "") -> Any: ...


@dataclass(frozen=True)
class Endpoint:
    uri: str
    hostname: str
    port: int | None


def validate_wss_endpoint(uri: str) -> Endpoint:
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError as exc:
        raise InsecureEndpoint("invalid Rithmic endpoint") from exc
    if parsed.scheme.lower() != "wss":
        raise InsecureEndpoint("Rithmic transport requires a wss:// endpoint")
    if not parsed.hostname:
        raise InsecureEndpoint("Rithmic WSS endpoint requires a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise InsecureEndpoint("credentials may not appear in the endpoint URI")
    if parsed.query or parsed.fragment:
        raise InsecureEndpoint("Rithmic endpoint may not contain a query or fragment")
    return Endpoint(uri=uri, hostname=parsed.hostname, port=port)


def create_client_ssl_context(*, cafile: str | None = None) -> ssl.SSLContext:
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cafile)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def assert_secure_ssl_context(context: ssl.SSLContext) -> None:
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise InsecureEndpoint("TLS certificate and hostname verification are mandatory")
    if context.minimum_version < ssl.TLSVersion.TLSv1_2:
        raise InsecureEndpoint("TLS 1.2 or newer is mandatory")


class ReadOnlyOutboundPolicy:
    """Last-line, default-deny template guard immediately before transport."""

    def __init__(self, allowed: frozenset[int] = OUTBOUND_READ_ONLY_TEMPLATE_IDS) -> None:
        normalized = frozenset(int(item) for item in allowed)
        overlap = normalized.intersection(int(item) for item in BROKER_MUTATION_TEMPLATE_IDS)
        if overlap:
            raise ValueError("read-only allowlist contains broker mutation templates")
        self._allowed = normalized

    @property
    def allowed(self) -> frozenset[int]:
        return self._allowed

    def assert_allowed(self, template_id: int) -> None:
        value = int(template_id)
        if value in {int(item) for item in BROKER_MUTATION_TEMPLATE_IDS}:
            raise MutationTemplateRejected(
                f"broker mutation template {value} ({template_name(value)}) is hard-disabled"
            )
        if value not in self._allowed:
            raise OutboundTemplateRejected(
                f"outbound template {value} ({template_name(value)}) is not read-only allowlisted"
            )


ConnectCallable = Callable[..., Awaitable[WebSocketLike] | WebSocketLike]


class WssTransport:
    """Secure one-protobuf-per-binary-WebSocket-message transport."""

    def __init__(
        self,
        endpoint: str,
        *,
        policy: ReadOnlyOutboundPolicy | None = None,
        ssl_context: ssl.SSLContext | None = None,
        connect_impl: ConnectCallable | None = None,
        open_timeout_seconds: float = 15.0,
        close_timeout_seconds: float = 10.0,
    ) -> None:
        self.endpoint = validate_wss_endpoint(endpoint)
        self.policy = policy or ReadOnlyOutboundPolicy()
        self.ssl_context = ssl_context or create_client_ssl_context()
        assert_secure_ssl_context(self.ssl_context)
        self._connect_impl = connect_impl
        self._socket: WebSocketLike | None = None
        self._open_timeout_seconds = open_timeout_seconds
        self._close_timeout_seconds = close_timeout_seconds

    @property
    def connected(self) -> bool:
        return self._socket is not None

    async def connect(self) -> None:
        if self._socket is not None:
            return
        connector = self._connect_impl
        if connector is None:
            try:
                # websockets >= 13
                from websockets.asyncio.client import connect as connector
            except ImportError:  # pragma: no cover - exercised on the pinned v12 deployment
                from websockets import connect as connector
        result = connector(
            self.endpoint.uri,
            ssl=self.ssl_context,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=self._open_timeout_seconds,
            close_timeout=self._close_timeout_seconds,
            max_size=8 * 1024 * 1024,
            compression=None,
        )
        self._socket = await result if inspect.isawaitable(result) else result

    async def send_frame(self, frame: bytes) -> int:
        if self._socket is None:
            raise ConnectionError("Rithmic transport is not connected")
        template_id = extract_template_id(frame)
        self.policy.assert_allowed(template_id)
        await self._socket.send(bytes(frame))
        return template_id

    async def receive_frame(self) -> bytes:
        if self._socket is None:
            raise ConnectionError("Rithmic transport is not connected")
        message = await self._socket.recv()
        if not isinstance(message, bytes):
            raise InvalidFrame("Rithmic server sent a text WebSocket message")
        extract_template_id(message)
        return message

    async def close(self, *, reason: str = "client shutdown") -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.close(code=1000, reason=reason[:123])
