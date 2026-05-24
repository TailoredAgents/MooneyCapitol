from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Date,
    BigInteger,
    ForeignKey,
    Boolean,
    UniqueConstraint,
    Index,
    Text,
)
from sqlalchemy import LargeBinary
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class Symbol(Base):
    __tablename__ = "symbols"
    id = Column(Integer, primary_key=True)
    ticker = Column(String(16), unique=True, nullable=False, index=True)
    exchange = Column(String(16))


class WatchlistEntry(Base):
    __tablename__ = "watchlist_entries"
    id = Column(BigInteger, primary_key=True)
    trade_date = Column(Date, index=True, nullable=False)
    ticker = Column(String(16), index=True, nullable=False)
    rank = Column(Integer, nullable=False)
    gap_pct = Column(Float, nullable=False)
    direction = Column(String(4), nullable=False)
    premkt_volume = Column(BigInteger, nullable=True)
    price = Column(Float, nullable=True)
    source = Column(String(16), nullable=False, default="top100")  # top100 | longlist
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("trade_date", "ticker", "source", name="uq_watchlist_trade_ticker_source"),)


class Candle(Base):
    __tablename__ = "candles"
    id = Column(BigInteger, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), index=True, nullable=False)
    ts = Column(DateTime(timezone=True), index=True, nullable=False)
    tf = Column(String(8), index=True, nullable=False)
    o = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    l = Column(Float, nullable=False)
    c = Column(Float, nullable=False)
    v = Column(BigInteger, nullable=False)
    __table_args__ = (UniqueConstraint("symbol_id", "ts", "tf", name="uq_candles_symbol_ts_tf"),)


class L2Snapshot(Base):
    __tablename__ = "l2_snapshots"
    id = Column(BigInteger, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), index=True, nullable=False)
    ts = Column(DateTime(timezone=True), index=True, nullable=False)
    bid_total = Column(Float, nullable=False)
    ask_total = Column(Float, nullable=False)
    levels = Column(JSONB, nullable=True)  # compact top-5
    imbalance = Column(Float, nullable=False)
    __table_args__ = (UniqueConstraint("symbol_id", "ts", name="uq_l2_symbol_ts"),)


class Box(Base):
    __tablename__ = "boxes"
    id = Column(BigInteger, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), index=True, nullable=False)
    tf = Column(String(8), nullable=False)
    start_ts = Column(DateTime(timezone=True), index=True, nullable=False)
    end_ts = Column(DateTime(timezone=True), index=True, nullable=False)
    hi = Column(Float, nullable=False)
    lo = Column(Float, nullable=False)
    bars = Column(Integer, nullable=False)
    height = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=False)
    rvol = Column(Float, nullable=True)
    spread_cents = Column(Float, nullable=True)


class Setup(Base):
    __tablename__ = "setups"
    id = Column(BigInteger, primary_key=True)
    box_id = Column(BigInteger, ForeignKey("boxes.id"), index=True, nullable=False)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), index=True, nullable=False)
    tf = Column(String(8), nullable=False)
    direction = Column(String(8), nullable=False)  # long/short
    detected_ts = Column(DateTime(timezone=True), index=True, nullable=False)
    entry_price = Column(Float, nullable=True)
    invalidation = Column(Float, nullable=True)
    targets = Column(ARRAY(Float), nullable=True)
    rr_min = Column(Float, nullable=True)
    score = Column(Integer, nullable=True)
    l2_confirm = Column(Boolean, nullable=False, default=False)
    state = Column(String(12), index=True, nullable=False, default="armed")
    payload_json = Column(JSONB, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(BigInteger, primary_key=True)
    setup_id = Column(BigInteger, ForeignKey("setups.id"), index=True, nullable=True)
    sent_ts = Column(DateTime(timezone=True), index=True, nullable=False)
    channel = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False)
    ack_by = Column(String(64), nullable=True)
    ack_ts = Column(DateTime(timezone=True), nullable=True)
    type = Column(String(16), nullable=False, default="trigger")
    symbol = Column(String(16), index=True, nullable=True)
    direction = Column(String(8), nullable=True)
    slack_thread_ts = Column(String(32), index=True, nullable=True)
    slack_message_ts = Column(String(32), index=True, nullable=True)
    payload_json = Column(JSONB, nullable=True)


class Fill(Base):
    __tablename__ = "fills"
    id = Column(BigInteger, primary_key=True)
    ext_trade_id = Column(String(128), index=True)
    account = Column(String(64), index=True)
    symbol = Column(String(16), index=True)
    ts = Column(DateTime(timezone=True), index=True, nullable=False)
    side = Column(String(8), nullable=False)
    qty = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, nullable=True)
    setup_id = Column(BigInteger, ForeignKey("setups.id"), index=True, nullable=True)
    setup_match_score = Column(Float, nullable=True)
    setup_match_confidence = Column(String(16), nullable=True, index=True)
    setup_match_reason = Column(JSONB, nullable=True)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(BigInteger, primary_key=True)
    setup_id = Column(BigInteger, ForeignKey("setups.id"), index=True, nullable=False)
    account = Column(String(64), index=True)
    symbol = Column(String(16), index=True)
    open_ts = Column(DateTime(timezone=True), nullable=False)
    close_ts = Column(DateTime(timezone=True), nullable=True)
    qty = Column(Integer, nullable=False)
    basis = Column(Float, nullable=True)
    p_and_l = Column(Float, nullable=True)
    realized_r = Column(Float, nullable=True)
    exit_reason = Column(String(32), nullable=True)


class CopyTargetAccount(Base):
    __tablename__ = "copy_target_accounts"
    id = Column(BigInteger, primary_key=True)
    name = Column(String(64), unique=True, nullable=False, index=True)
    broker = Column(String(32), nullable=False, default="webull")
    environment = Column(String(16), nullable=False, default="paper")
    enabled = Column(Boolean, nullable=False, default=False)
    account_ref = Column(String(128), nullable=True)
    equity = Column(Float, nullable=True)
    sizing_mode = Column(String(32), nullable=False, default="disabled")
    sizing_value = Column(Float, nullable=False, default=0.0)
    min_notional = Column(Float, nullable=False, default=0.0)
    max_notional_per_trade = Column(Float, nullable=False, default=0.0)
    max_position_pct = Column(Float, nullable=False, default=0.0)
    max_daily_notional = Column(Float, nullable=False, default=0.0)
    max_daily_trades = Column(Integer, nullable=False, default=0)
    regular_hours_only = Column(Boolean, nullable=False, default=True)
    shorting_enabled = Column(Boolean, nullable=False, default=False)
    allowlist = Column(ARRAY(String(16)), nullable=True)
    blocklist = Column(ARRAY(String(16)), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class MasterExecution(Base):
    __tablename__ = "master_executions"
    id = Column(BigInteger, primary_key=True)
    broker = Column(String(32), nullable=False, default="webull")
    account_ref = Column(String(128), nullable=True, index=True)
    broker_execution_id = Column(String(128), nullable=False)
    broker_order_id = Column(String(128), nullable=True)
    symbol = Column(String(16), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    asset_class = Column(String(32), nullable=False, default="equity")
    executed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    received_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    raw_payload = Column(JSONB, nullable=True)
    __table_args__ = (
        UniqueConstraint("broker", "account_ref", "broker_execution_id", name="uq_master_exec_broker_account_exec"),
    )


class CopyOrder(Base):
    __tablename__ = "copy_orders"
    id = Column(BigInteger, primary_key=True)
    master_execution_id = Column(BigInteger, ForeignKey("master_executions.id"), index=True, nullable=False)
    target_account_id = Column(BigInteger, ForeignKey("copy_target_accounts.id"), index=True, nullable=False)
    broker = Column(String(32), nullable=False, default="webull")
    client_order_id = Column(String(128), nullable=False, unique=True, index=True)
    broker_order_id = Column(String(128), nullable=True, index=True)
    symbol = Column(String(16), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    order_type = Column(String(16), nullable=False, default="market")
    time_in_force = Column(String(16), nullable=False, default="day")
    status = Column(String(32), nullable=False, default="created", index=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    filled_qty = Column(Float, nullable=True)
    avg_fill_price = Column(Float, nullable=True)
    reject_reason = Column(String(512), nullable=True)
    latency_ms = Column(Float, nullable=True)
    raw_submit_payload = Column(JSONB, nullable=True)
    raw_response_payload = Column(JSONB, nullable=True)


class CopyOrderEvent(Base):
    __tablename__ = "copy_order_events"
    id = Column(BigInteger, primary_key=True)
    copy_order_id = Column(BigInteger, ForeignKey("copy_orders.id"), index=True, nullable=False)
    event_type = Column(String(64), nullable=False)
    status = Column(String(32), nullable=True)
    event_at = Column(DateTime(timezone=True), nullable=True)
    received_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    raw_payload = Column(JSONB, nullable=True)


class CopyReconciliation(Base):
    __tablename__ = "copy_reconciliations"
    id = Column(BigInteger, primary_key=True)
    target_account_id = Column(BigInteger, ForeignKey("copy_target_accounts.id"), index=True, nullable=True)
    symbol = Column(String(16), nullable=True, index=True)
    severity = Column(String(16), nullable=False, default="warning")
    status = Column(String(32), nullable=False, default="open", index=True)
    message = Column(String(1024), nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    raw_context = Column(JSONB, nullable=True)


class CopierAuditEvent(Base):
    __tablename__ = "copier_audit_events"
    id = Column(BigInteger, primary_key=True)
    event_type = Column(String(64), nullable=False, index=True)
    actor = Column(String(128), nullable=True)
    target_account_id = Column(BigInteger, ForeignKey("copy_target_accounts.id"), index=True, nullable=True)
    message = Column(String(1024), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    payload = Column(JSONB, nullable=True)


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"
    id = Column(BigInteger, primary_key=True)
    account_ref = Column(String(128), nullable=False, index=True)
    account_name = Column(String(64), nullable=False, index=True)
    account_type = Column(String(20), nullable=False)
    broker = Column(String(32), nullable=False, default="webull")
    snapshot_time = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    market_session = Column(String(20), nullable=False, default="unknown")
    cash_balance = Column(Float, nullable=True)
    equity_value = Column(Float, nullable=True)
    total_value = Column(Float, nullable=False)
    buying_power = Column(Float, nullable=True)
    day_trades_used = Column(Integer, nullable=True)
    day_trades_remaining = Column(Integer, nullable=True)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    realized_pnl_today = Column(Float, nullable=False, default=0.0)
    total_pnl_today = Column(Float, nullable=False, default=0.0)
    max_drawdown_today = Column(Float, nullable=False, default=0.0)
    max_profit_today = Column(Float, nullable=False, default=0.0)
    position_count = Column(Integer, nullable=False, default=0)
    total_exposure = Column(Float, nullable=False, default=0.0)
    risk_level = Column(String(20), nullable=False, default="normal")
    data_source = Column(String(50), nullable=False, default="webull_api")
    raw_payload = Column(JSONB, nullable=True)
    __table_args__ = (
        Index("idx_account_snapshot_ref_time", "account_ref", "snapshot_time"),
        Index("idx_account_snapshot_name_time", "account_name", "snapshot_time"),
    )


class PositionSnapshot(Base):
    __tablename__ = "position_snapshots"
    id = Column(BigInteger, primary_key=True)
    account_ref = Column(String(128), nullable=False, index=True)
    account_name = Column(String(64), nullable=False, index=True)
    symbol = Column(String(16), nullable=False, index=True)
    snapshot_time = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    qty = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    market_value = Column(Float, nullable=False, default=0.0)
    cost_basis = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl_pct = Column(Float, nullable=True)
    side = Column(String(10), nullable=False, default="long")
    raw_payload = Column(JSONB, nullable=True)
    __table_args__ = (
        Index("idx_position_snapshot_account_symbol_time", "account_ref", "symbol", "snapshot_time"),
    )


class PnlAlert(Base):
    __tablename__ = "pnl_alerts"
    id = Column(BigInteger, primary_key=True)
    account_ref = Column(String(128), nullable=False, index=True)
    account_name = Column(String(64), nullable=False, index=True)
    account_type = Column(String(20), nullable=False)
    alert_time = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    threshold_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    account_value = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    message = Column(Text, nullable=False)
    triggered_by = Column(String(50), nullable=False, default="pnl_monitor")
    action_taken = Column(String(100), nullable=True)
    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_by = Column(String(128), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    raw_context = Column(JSONB, nullable=True)
    __table_args__ = (
        Index("idx_pnl_alert_account_time", "account_ref", "alert_time"),
    )


class TradingSession(Base):
    __tablename__ = "trading_sessions"
    id = Column(BigInteger, primary_key=True)
    account_ref = Column(String(128), nullable=False, index=True)
    account_name = Column(String(64), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    session_start = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    session_end = Column(DateTime(timezone=True), nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    starting_value = Column(Float, nullable=False)
    ending_value = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    max_profit = Column(Float, nullable=False, default=0.0)
    trades_count = Column(Integer, nullable=False, default=0)
    notes = Column(Text, nullable=True)
    __table_args__ = (
        UniqueConstraint("account_ref", "trade_date", name="uq_trading_session_account_date"),
    )


class AIArtifact(Base):
    __tablename__ = "ai_artifacts"
    id = Column(BigInteger, primary_key=True)
    artifact_type = Column(String(64), nullable=False, index=True)
    source_type = Column(String(64), nullable=False, index=True)
    source_id = Column(String(128), nullable=True, index=True)
    symbol = Column(String(16), nullable=True, index=True)
    model = Column(String(64), nullable=False)
    prompt_version = Column(String(64), nullable=False)
    input_json = Column(JSONB, nullable=True)
    output_json = Column(JSONB, nullable=True)
    output_text = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="created", index=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    __table_args__ = (
        Index("idx_ai_artifact_source", "artifact_type", "source_type", "source_id"),
        Index("idx_ai_artifact_symbol_created", "symbol", "created_at"),
    )


class KVStore(Base):
    """Small key/value store for cross-service state.

    Render runs API and worker in separate services; their local files and
    in-memory globals are not shared. This table is used to persist:
    - app config
    - worker tick
    - dashboard lanes
    - learning artifacts (thresholds/report/model bytes)
    """

    __tablename__ = "kv_store"
    key = Column(String(128), primary_key=True)
    value_json = Column(JSONB, nullable=True)
    value_bytes = Column(LargeBinary, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
