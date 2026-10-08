from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .constants import PROTOCOL_TEMPLATE_VERSION, Template
from .fields import FieldView
from .multipart import MultipartResponseTracker, ResponseProgress
from .normalization import (
    GatewayInfoObservation,
    LoginObservation,
    SystemInfoObservation,
    normalize_gateway_info,
    normalize_login,
    normalize_system_info,
)
from .state import PlantSessionState


class DispatchKind(str, Enum):
    MESSAGE = "MESSAGE"
    LOGIN_SUCCEEDED = "LOGIN_SUCCEEDED"
    LOGIN_FAILED = "LOGIN_FAILED"
    HEARTBEAT = "HEARTBEAT"
    REJECT = "REJECT"
    FORCED_LOGOUT = "FORCED_LOGOUT"
    SYSTEM_INFO = "SYSTEM_INFO"
    GATEWAY_INFO = "GATEWAY_INFO"


@dataclass(frozen=True)
class DispatchEvent:
    kind: DispatchKind
    template_id: int
    message: Any = None
    control: LoginObservation | SystemInfoObservation | GatewayInfoObservation | None = None
    response_codes: tuple[str, ...] = ()
    multipart: ResponseProgress | None = None


class MessageDispatcher:
    """Handles connection-control templates and leaves business events untouched."""

    def __init__(self, multipart: MultipartResponseTracker | None = None) -> None:
        self.multipart = multipart or MultipartResponseTracker()

    def dispatch(
        self,
        template_id: int,
        message: Mapping[str, Any] | Any,
        state: PlantSessionState | None = None,
    ) -> DispatchEvent:
        template_id = int(template_id)
        view = FieldView(message)
        if state is not None:
            state.record_message()
        if template_id == Template.LOGIN_RESPONSE:
            login = normalize_login(message)
            if (
                login.success
                and login.heartbeat_interval_seconds is not None
                and login.template_version == PROTOCOL_TEMPLATE_VERSION
            ):
                if state is not None:
                    state.login_succeeded(float(login.heartbeat_interval_seconds))
                kind = DispatchKind.LOGIN_SUCCEEDED
            else:
                if state is not None:
                    state.rejected(
                        "login failed, omitted heartbeat interval, or returned an incompatible template version"
                    )
                kind = DispatchKind.LOGIN_FAILED
            return DispatchEvent(kind, template_id, message, login, login.response_codes)
        if template_id == Template.HEARTBEAT_RESPONSE:
            if state is not None:
                state.record_heartbeat()
            return DispatchEvent(
                DispatchKind.HEARTBEAT,
                template_id,
                message,
                response_codes=view.strings("rp_code"),
            )
        if template_id == Template.REJECT:
            codes = view.strings("rp_code")
            if state is not None:
                state.rejected("protocol reject received")
            return DispatchEvent(DispatchKind.REJECT, template_id, message, response_codes=codes)
        if template_id == Template.FORCED_LOGOUT:
            codes = view.strings("rp_code")
            if state is not None:
                state.forced_logout()
            return DispatchEvent(
                DispatchKind.FORCED_LOGOUT,
                template_id,
                message,
                response_codes=codes,
            )
        if template_id == Template.SYSTEM_INFO_RESPONSE:
            control = normalize_system_info(message)
            return DispatchEvent(
                DispatchKind.SYSTEM_INFO,
                template_id,
                message,
                control,
                control.response_codes,
            )
        if template_id == Template.GATEWAY_INFO_RESPONSE:
            control = normalize_gateway_info(message)
            return DispatchEvent(
                DispatchKind.GATEWAY_INFO,
                template_id,
                message,
                control,
                control.response_codes,
            )
        multipart = self.multipart.consume(template_id, message)
        return DispatchEvent(
            DispatchKind.MESSAGE,
            template_id,
            message,
            response_codes=view.strings("rp_code"),
            multipart=multipart,
        )
