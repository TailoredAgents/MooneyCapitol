import os
from typing import Optional

import sentry_sdk


def init_sentry() -> Optional[str]:
    dsn = os.getenv("SENTRY_DSN")
    if dsn:
        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.1)
        return dsn
    return None

