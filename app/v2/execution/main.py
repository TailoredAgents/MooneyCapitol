from __future__ import annotations

import asyncio
import signal

from app.v2.execution.config import ExecutionConfigSnapshot
from app.v2.execution.service import FuturesExecutionService


async def serve() -> None:
    service = FuturesExecutionService(ExecutionConfigSnapshot(config_version="v2-foundation-1"))
    await service.start()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:  # Windows event loops
            pass
    await stopped.wait()
    await service.stop()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
