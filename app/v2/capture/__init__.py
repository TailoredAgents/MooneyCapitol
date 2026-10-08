"""Dedicated, broker-read-only Rithmic capture service.

This package intentionally does not depend on the execution, copier, Scout,
learning, or AI packages.  The only adapter contract exported here is
observational; it has no broker-mutation methods.
"""

from app.v2.capture.config import CaptureConfig
from app.v2.capture.service import CaptureHealth, RithmicCaptureService

__all__ = ["CaptureConfig", "CaptureHealth", "RithmicCaptureService"]
