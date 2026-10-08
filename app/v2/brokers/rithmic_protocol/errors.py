from __future__ import annotations


class RithmicProtocolError(RuntimeError):
    """Base error for the read-only R|Protocol runtime."""


class BindingsUnavailable(RithmicProtocolError):
    """External generated protobuf bindings are missing or invalid."""


class InvalidFrame(RithmicProtocolError):
    """A WebSocket message is not one complete binary protobuf frame."""


class UnsupportedTemplate(RithmicProtocolError):
    """No external protobuf binding is registered for a template."""


class OutboundTemplateRejected(RithmicProtocolError):
    """An outbound template is not on the explicit read-only allowlist."""


class MutationTemplateRejected(OutboundTemplateRejected):
    """A broker mutation was rejected before reaching the socket."""


class InsecureEndpoint(RithmicProtocolError):
    """An endpoint or TLS context does not meet the WSS-only policy."""


class UnauthorizedAccount(RithmicProtocolError):
    """An account-scoped request targeted a non-allowlisted account."""


class ProtocolRejected(RithmicProtocolError):
    """Rithmic returned a reject or unsuccessful response code."""


class ForcedLogoutReceived(RithmicProtocolError):
    """Rithmic forced the current login session to close."""


class InvalidStateTransition(RithmicProtocolError):
    """A session or recovery operation occurred out of order."""


class NormalizationError(RithmicProtocolError):
    """An official field was present but could not be losslessly normalized."""
