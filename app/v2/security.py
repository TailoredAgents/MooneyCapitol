"""V2-facing export of the repository-wide redaction boundary."""

from app.security import redact_sensitive

__all__ = ["redact_sensitive"]
