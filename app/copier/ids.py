from __future__ import annotations

import hashlib


def copy_client_order_id(master_execution_id: str, target_account_key: str, sequence: int = 0) -> str:
    """Return a deterministic Webull-compatible client order id.

    Webull documents `client_order_id` as client-defined and unique per account
    with a max length of 32 characters. Keep this stable so retries after a
    worker restart do not create duplicate copied orders.
    """
    raw = f"{master_execution_id}|{target_account_key}|{sequence}".encode("utf-8")
    digest = hashlib.blake2s(raw, digest_size=15).hexdigest()
    return f"mc{digest}"

