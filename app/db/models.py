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
