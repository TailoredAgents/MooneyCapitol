from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .bindings import ExternalBindingRegistry
from .constants import Plant, Template
from .dispatch import DispatchEvent, DispatchKind, MessageDispatcher
from .errors import ForcedLogoutReceived, ProtocolRejected, UnsupportedTemplate
from .framing import encode_message, extract_template_id
from .state import PlantSessionState, ReconnectPolicy, SessionState
from .transport import WssTransport


@dataclass(frozen=True)
class ReceivedMessage:
    plant: Plant
    generation_id: str
    template_id: int
    fingerprint: str
    dispatch: DispatchEvent
    raw_frame: bytes = field(repr=False)


EventCallback = Callable[[ReceivedMessage], Awaitable[None] | None]
DisconnectCallback = Callable[[Plant, str, str], Awaitable[None] | None]
MessageFactory = Callable[[], Any]


class PlantSession:
    """One plant connection with independent generation, health and reconnect state."""

    def __init__(
        self,
        plant: Plant,
        transport: WssTransport,
        registry: ExternalBindingRegistry,
        *,
        required: bool = True,
        dispatcher: MessageDispatcher | None = None,
        reconnect_policy: ReconnectPolicy | None = None,
        login_timeout_seconds: float = 30.0,
        heartbeat_timeout_multiplier: float = 2.0,
    ) -> None:
        self.plant = plant
        self.transport = transport
        self.registry = registry
        self.state = PlantSessionState(plant, required=required)
        self.dispatcher = dispatcher or MessageDispatcher()
        self.reconnect_policy = reconnect_policy or ReconnectPolicy()
        if login_timeout_seconds <= 0:
            raise ValueError("login_timeout_seconds must be positive")
        self.login_timeout_seconds = login_timeout_seconds
        if heartbeat_timeout_multiplier < 1:
            raise ValueError("heartbeat_timeout_multiplier must be at least one")
        self.heartbeat_timeout_multiplier = heartbeat_timeout_multiplier

    async def open(self) -> None:
        self.state.begin_connect()
        try:
            await self.transport.connect()
        except Exception:
            self.state.connection_lost("transport connection failed")
            raise
        self.state.transport_connected()

    async def send(self, message: Any, *, heartbeat: bool = False) -> int:
        encoded = encode_message(message)
        template_id = await self.transport.send_frame(encoded.payload)
        self.state.record_outbound(heartbeat=heartbeat)
        return template_id

    async def receive_once(self) -> ReceivedMessage:
        frame = await self.transport.receive_frame()
        template_id = extract_template_id(frame)
        try:
            message = self.registry.decode(frame)
        except UnsupportedTemplate:
            # Unknown server additions remain journalable without making up a
            # schema interpretation. The raw bytes are never rendered/logged.
            message = {"template_id": template_id}
        event = self.dispatcher.dispatch(template_id, message, self.state)
        return ReceivedMessage(
            plant=self.plant,
            generation_id=str(self.state.generation_id or ""),
            template_id=template_id,
            fingerprint=hashlib.sha256(frame).hexdigest(),
            dispatch=event,
            raw_frame=frame,
        )

    async def send_heartbeat_if_due(self, heartbeat_factory: MessageFactory) -> bool:
        if not self.state.heartbeat_due():
            return False
        await self.send(heartbeat_factory(), heartbeat=True)
        return True

    async def close(self) -> None:
        await self.transport.close()
        self.state.stop()

    async def run_forever(
        self,
        *,
        login_factory: MessageFactory,
        heartbeat_factory: MessageFactory,
        on_event: EventCallback,
        stop_event: asyncio.Event,
        on_disconnect: DisconnectCallback | None = None,
    ) -> None:
        """Reconnect transport failures; stop on login failure or forced logout."""

        while not stop_event.is_set():
            try:
                await self.open()
                await self.send(login_factory())
                login_deadline = (
                    asyncio.get_running_loop().time() + self.login_timeout_seconds
                )
                while not stop_event.is_set():
                    timeout = min(self.state.heartbeat_interval_seconds or 5.0, 5.0)
                    if self.state.state is SessionState.AUTHENTICATING:
                        login_remaining = (
                            login_deadline - asyncio.get_running_loop().time()
                        )
                        if login_remaining <= 0:
                            raise TimeoutError("Rithmic login response timed out")
                        timeout = min(timeout, login_remaining)
                    try:
                        received = await asyncio.wait_for(self.receive_once(), timeout=timeout)
                    except TimeoutError:
                        if (
                            self.state.state is SessionState.AUTHENTICATING
                            and asyncio.get_running_loop().time() >= login_deadline
                        ):
                            raise TimeoutError("Rithmic login response timed out")
                        if self.state.heartbeat_response_overdue(
                            timeout_multiplier=self.heartbeat_timeout_multiplier
                        ):
                            raise TimeoutError("Rithmic application heartbeat response timed out")
                        await self.send_heartbeat_if_due(heartbeat_factory)
                        continue
                    result = on_event(received)
                    if inspect.isawaitable(result):
                        await result
                    if received.dispatch.kind is DispatchKind.LOGIN_FAILED:
                        raise ProtocolRejected("Rithmic login failed")
                    if received.dispatch.kind is DispatchKind.REJECT:
                        raise ProtocolRejected("Rithmic rejected a protocol request")
                    if received.dispatch.kind is DispatchKind.FORCED_LOGOUT:
                        raise ForcedLogoutReceived("Rithmic forced logout")
            except asyncio.CancelledError:
                raise
            except (ProtocolRejected, ForcedLogoutReceived):
                await self.transport.close(reason="terminal protocol response")
                await self._notify_disconnect(
                    on_disconnect, "terminal_protocol_response"
                )
                return
            except Exception:
                if self.state.state in {
                    SessionState.DEGRADED,
                    SessionState.FORCED_LOGOUT,
                }:
                    await self.transport.close(reason="terminal protocol state")
                    await self._notify_disconnect(
                        on_disconnect, "terminal_protocol_state"
                    )
                    return
                await self.transport.close(reason="connection recovery")
                # ``open`` has already recorded a failed transport connect.
                # Avoid incrementing the reconnect attempt twice for the same
                # failure; established connections still transition here.
                if self.state.state is not SessionState.DISCONNECTED:
                    self.state.connection_lost("connection interrupted")
                await self._notify_disconnect(on_disconnect, "connection_interrupted")
                delay = self.reconnect_policy.delay(max(1, self.state.reconnect_attempt))
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                except TimeoutError:
                    continue
        await self.transport.close(reason="service shutdown")
        await self._notify_disconnect(on_disconnect, "service_shutdown")
        if self.state.state is not SessionState.STOPPED:
            self.state.stop()

    async def _notify_disconnect(
        self,
        callback: DisconnectCallback | None,
        reason: str,
    ) -> None:
        generation_id = str(self.state.generation_id or "")
        if callback is None or not generation_id:
            return
        result = callback(self.plant, generation_id, reason)
        if inspect.isawaitable(result):
            await result
