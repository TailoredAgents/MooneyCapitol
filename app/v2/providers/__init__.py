"""V2 market-data provider adapters.

Provider-specific code is intentionally isolated from the research contracts in
``app.v2.market_data``.
"""

from app.v2.providers.massive_futures import AsyncJsonTransport, MassiveFuturesProvider

__all__ = ["AsyncJsonTransport", "MassiveFuturesProvider"]
