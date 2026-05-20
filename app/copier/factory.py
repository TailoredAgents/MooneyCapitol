from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy import desc, select

from app.db.models import AccountSnapshot
from app.db.session import get_session
from app.copier.engine import CopyTarget, TradingClient
from app.copier.risk import RiskPolicy
from app.copier.sizing import SizingPolicy
from app.copier.webull_client import WebullTradingClient
from app.core.config import CopierConfig, CopyTargetAccountConfig
from app.core.config_store import CONFIG


ClientFactory = Callable[[CopyTargetAccountConfig], TradingClient]
EquityProvider = Callable[[str], float | None]
SessionScope = Callable[[], AbstractContextManager]


def build_copy_targets(
    config: CopierConfig | None = None,
    client_factory: ClientFactory | None = None,
    allow_missing_accounts: bool = False,
    equity_provider: EquityProvider | None = None,
    session_scope: SessionScope = get_session,
    use_latest_snapshot_equity: bool = False,
) -> list[CopyTarget]:
    config = config or CONFIG.copier
    client_factory = client_factory or _webull_client_from_target_config
    return [
        _build_target(
            config,
            target_cfg,
            client_factory,
            allow_missing_accounts=allow_missing_accounts,
            equity_provider=equity_provider,
            session_scope=session_scope,
            use_latest_snapshot_equity=use_latest_snapshot_equity,
        )
        for target_cfg in config.targets
    ]


def _build_target(
    config: CopierConfig,
    target_cfg: CopyTargetAccountConfig,
    client_factory: ClientFactory,
    allow_missing_accounts: bool = False,
    equity_provider: EquityProvider | None = None,
    session_scope: SessionScope = get_session,
    use_latest_snapshot_equity: bool = False,
) -> CopyTarget:
    account_id = target_cfg.account_ref or os.getenv(target_cfg.account_id_env)
    if not account_id:
        if not allow_missing_accounts:
            raise ValueError(f"Missing account id for copier target {target_cfg.name}: {target_cfg.account_id_env}")
        account_id = f"dry-run:{target_cfg.name}"
    master_account_id = config.master_account or os.getenv(config.master_account_env)
    master_equity = _equity_for_account(
        master_account_id,
        config.master_equity,
        config.master_equity_env,
        equity_provider=equity_provider,
        session_scope=session_scope,
        use_latest_snapshot_equity=use_latest_snapshot_equity,
    )
    target_equity = _equity_for_account(
        account_id,
        target_cfg.equity,
        target_cfg.equity_env,
        equity_provider=equity_provider,
        session_scope=session_scope,
        use_latest_snapshot_equity=use_latest_snapshot_equity,
    )

    return CopyTarget(
        name=target_cfg.name,
        account_id=account_id,
        client=client_factory(target_cfg),
        sizing=SizingPolicy(
            mode=target_cfg.sizing_mode,  # type: ignore[arg-type]
            value=target_cfg.sizing_value,
            min_notional=target_cfg.min_notional,
        ),
        risk=RiskPolicy(
            enabled=target_cfg.enabled and config.enabled,
            global_kill_switch=config.global_kill_switch,
            max_notional_per_trade=target_cfg.max_notional_per_trade,
            max_position_pct=target_cfg.max_position_pct,
            max_daily_notional=target_cfg.max_daily_notional,
            max_daily_trades=target_cfg.max_daily_trades,
            max_orders_per_minute=config.max_orders_per_minute,
            shorting_enabled=target_cfg.shorting_enabled and config.copy_shorts,
            allowlist={item.upper() for item in target_cfg.allowlist},
            blocklist={item.upper() for item in target_cfg.blocklist},
        ),
        master_equity=master_equity,
        target_equity=target_equity,
    )


def _webull_client_from_target_config(target_cfg: CopyTargetAccountConfig) -> WebullTradingClient:
    return WebullTradingClient.from_env(
        app_key_env=target_cfg.api_key_env,
        app_secret_env=target_cfg.api_secret_env,
        endpoint_env=target_cfg.endpoint_env,
        account_id_env=target_cfg.account_id_env,
        environment=target_cfg.environment,
    )


def _float_from_config_or_env(config_value: float | None, env_name: str | None) -> float | None:
    if config_value is not None and config_value > 0:
        return float(config_value)
    if not env_name:
        return None
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _equity_for_account(
    account_ref: str | None,
    config_value: float | None,
    env_name: str | None,
    *,
    equity_provider: EquityProvider | None,
    session_scope: SessionScope,
    use_latest_snapshot_equity: bool,
) -> float | None:
    if account_ref:
        if equity_provider is not None:
            value = equity_provider(account_ref)
            if value and value > 0:
                return float(value)
        if use_latest_snapshot_equity:
            value = _latest_snapshot_equity(account_ref, session_scope)
            if value and value > 0:
                return float(value)
    return _float_from_config_or_env(config_value, env_name)


def _latest_snapshot_equity(account_ref: str, session_scope: SessionScope) -> float | None:
    try:
        with session_scope() as session:
            row = (
                session.execute(
                    select(AccountSnapshot)
                    .where(AccountSnapshot.account_ref == account_ref)
                    .order_by(desc(AccountSnapshot.snapshot_time))
                    .limit(1)
                )
                .scalars()
                .first()
            )
    except Exception:
        return None
    if row is None:
        return None
    value = row.equity_value if row.equity_value is not None else row.total_value
    return float(value) if value is not None and value > 0 else None
