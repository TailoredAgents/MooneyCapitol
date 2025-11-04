from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Literal, Optional

import numpy as np

from sqlalchemy import select

from app.adapters.depth_provider import DepthSnapshot, get_depth_provider
from app.adapters.ibkr_depth import DemoDepthProvider
from app.adapters.polygon_client import PolygonClient
from app.adapters.slack import SlackAdapter
from app.core.config_store import CONFIG
from app.core.detectors.consolidation import Box, Candle, find_consolidation_box
from app.core.detectors.trigger import check_break_and_retest
from app.core.rr import project_rr
from app.db.models import Alert, Box as BoxModel
from app.db.models import Setup
from app.db.session import get_session
from app.observability.logging import get_logger
from app.services.depth import mark_snapshot, mark_stale
from app.services.learning import bucket_price, bucket_time, get_learning_service
from app.services.ledger import get_ledger_service
from app.services.levels import find_gap_edges, find_htf_levels, nearest_target
from app.services.live_state import set_lanes
from app.services.runtime import update_worker_tick
from app.services.watchlist import (
    WatchlistItem,
    get_or_create_symbol,
    upsert_watchlist_entries,
)
from app.utils.time import is_rth, now_et, seconds_until
from app.workers.budget import PingBudget


logger = get_logger("worker")


@dataclass
class DirectionState:
    direction: Literal["long", "short"]
    target: float | None = None
    rr: float | None = None
    gating_note: str | None = None
    primed: bool = False
    primed_started: datetime | None = None
    primed_expires: datetime | None = None
    primed_ttl_bars: int = 3
    l2_buffer: Deque[tuple[datetime, float, float]] = field(default_factory=lambda: deque(maxlen=30))
    primed_pinged: bool = False
    active: bool = False
    active_started: datetime | None = None
    active_expires: datetime | None = None
    trigger_payload: dict | None = None
    setup_id: int | None = None
    last_trigger_ts: datetime | None = None
    no_l2: bool = False
    score: int = 0
    last_l2_display: str | None = None
    last_alert_ts: str | None = None
    alert_type: str | None = None
    last_alert_id: int | None = None
    action_status: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    p2r: float | None = None


@dataclass
class BoxState:
    symbol: str
    box: Box
    thread_key: str
    slack_thread_ts: str | None = None
    levels: dict = field(default_factory=dict)
    gap_edges: dict = field(default_factory=dict)
    contexts: Dict[str, DirectionState] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_et)
    last_heads_up_ts: datetime | None = None
    rr_under: bool = False
    box_db_id: int | None = None


@dataclass
class WatchlistState:
    trade_date: date
    longlist: list[WatchlistItem] = field(default_factory=list)
    longlist_frozen: bool = False
    finalized: list[WatchlistItem] = field(default_factory=list)
    finalized_at: datetime | None = None

    @classmethod
    def new(cls) -> "WatchlistState":
        return cls(trade_date=now_et().date())


class WorkerContext:
    def __init__(self):
        self.polygon = PolygonClient()
        self.slack = SlackAdapter()
        self.depth = get_depth_provider()
        self.depth_mode = os.getenv("DEPTH_MODE", "demo").lower()
        self.depth_is_demo = isinstance(self.depth, DemoDepthProvider)
        self.ledger = get_ledger_service()
        self.learning = get_learning_service()
        self.model_loaded = False
        self.rank_model = None
        self.rank_scaler = None
        self.thresholds = self.learning.load_thresholds()
        self.threshold_canary = self.learning.load_canary()
        self.watchlist = WatchlistState.new()
        self.box_states: Dict[str, BoxState] = {}
        self.symbol_cursor = 0
        self.batch_size = 12
        self.depth_symbols: set[str] = set()
        self.ping_budget = PingBudget(limit=CONFIG.alerts.per_minute_budget_open)
        self.last_learning_report: dict | None = None

    def load_ranker(self):
        if self.model_loaded:
            return
        try:
            self.rank_model, self.rank_scaler = self.learning.load_model()
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False

    def score_features(self, symbol: str, features: dict[str, float]) -> float | None:
        self.load_ranker()
        if not self.model_loaded or not self.rank_model or not self.rank_scaler:
            return None
        vector = np.array([[features.get(k, 0.0) for k in [
            "box_height",
            "box_bars",
            "rvol_break",
            "l2_mean",
            "l2_persist",
            "dist_htf",
            "dist_gap",
            "spread_cents",
        ]]])
        transformed = self.rank_scaler.transform(vector)
        return float(self.rank_model.predict_proba(transformed)[0, 1])
    
    def threshold_for(self, symbol: str, features: dict[str, float]) -> dict[str, float] | None:
        key = f"{features.get('price_bucket')}|{features.get('time_bucket')}"
        if symbol and symbol in self.threshold_canary.get("symbols", []):
            return self.threshold_canary.get("buckets", {}).get(key)
        return self.thresholds.get("buckets", {}).get(key)

    def reset_if_new_day(self):
        today = now_et().date()
        if self.watchlist.trade_date != today:
            logger.info("watchlist.reset", previous=str(self.watchlist.trade_date), current=str(today))
            self.watchlist = WatchlistState.new()
            self.box_states.clear()
            self.symbol_cursor = 0

    def finalized_symbols(self) -> List[str]:
        if not self.watchlist.finalized:
            return []
        return [item.ticker for item in self.watchlist.finalized]


CTX = WorkerContext()


async def every(interval_s: float, func, *args, **kwargs):
    while True:
        try:
            await func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - guardrails
            logger.error("job.error", err=str(exc), job=getattr(func, "__name__", "unknown"))
        await asyncio.sleep(interval_s)


def _to_candles(raw: list[dict]) -> list[Candle]:
    return [
        Candle(
            ts=int(item.get("t")),
            o=float(item.get("o")),
            h=float(item.get("h")),
            l=float(item.get("l")),
            c=float(item.get("c")),
            v=int(item.get("v")),
        )
        for item in raw
    ]


def _to_float(value) -> float | None:
    try:
        if value in (None, "n/a", ""):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _nearest_htf_distance(levels: dict, direction: str, entry: float | None) -> float | None:
    if entry is None:
        return None
    best = None
    if direction == "long":
        for lvl in levels.get("levels_up", []):
            if lvl and lvl > entry:
                diff = lvl - entry
                if best is None or diff < best:
                    best = diff
    else:
        for lvl in levels.get("levels_down", []):
            if lvl and lvl < entry:
                diff = entry - lvl
                if best is None or diff < best:
                    best = diff
    return best


def _gap_distance(levels: dict, direction: str, entry: float | None) -> float | None:
    if entry is None:
        return None
    gap_key = "gap_high" if direction == "long" else "gap_low"
    gap_level = levels.get(gap_key)
    if not gap_level:
        return None
    return abs(gap_level - entry)


def _persist_box(symbol: str, box: Box) -> int | None:
    start_dt = datetime.fromtimestamp(box.start_ts / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(box.end_ts / 1000, tz=timezone.utc)
    with get_session() as session:
        symbol_row = get_or_create_symbol(session, symbol)
        stmt = (
            select(BoxModel)
            .where(
                BoxModel.symbol_id == symbol_row.id,
                BoxModel.tf == box.tf,
                BoxModel.start_ts == start_dt,
            )
            .limit(1)
        )
        existing = session.execute(stmt).scalar_one_or_none()
        if existing:
            existing.end_ts = end_dt
            existing.hi = box.hi
            existing.lo = box.lo
            existing.bars = box.bars
            existing.height = box.height
            existing.quality_score = box.quality_score
            existing.rvol = box.rvol
            existing.spread_cents = box.spread_cents
            session.flush()
            return existing.id
        new_box = BoxModel(
            symbol_id=symbol_row.id,
            tf=box.tf,
            start_ts=start_dt,
            end_ts=end_dt,
            hi=box.hi,
            lo=box.lo,
            bars=box.bars,
            height=box.height,
            quality_score=box.quality_score,
            rvol=box.rvol,
            spread_cents=box.spread_cents,
        )
        session.add(new_box)
        session.flush()
        return new_box.id


def _persist_setup(symbol: str, direction: str, payload: dict, box: Box, box_id: int | None) -> int | None:
    with get_session() as session:
        symbol_row = get_or_create_symbol(session, symbol)
        entry_price = _to_float(payload.get("entry"))
        stop_price = _to_float(payload.get("stop"))
        rr_value = _to_float(payload.get("rr"))
        target_value = _to_float(payload.get("primary_target"))
        payload.setdefault("features", {})
        payload.setdefault("symbol", symbol)
        payload["entry_price"] = entry_price
        payload["direction"] = direction
        payload["detected_ts"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        if rr_value is not None:
            payload["rr_value"] = rr_value
        setup = Setup(
            box_id=box_id,
            symbol_id=symbol_row.id,
            tf=box.tf,
            direction=direction,
            detected_ts=datetime.utcnow().replace(tzinfo=timezone.utc),
            entry_price=entry_price,
            invalidation=stop_price,
            targets=[target_value] if target_value is not None else None,
            rr_min=rr_value,
            score=int(payload.get("score", 0)),
            l2_confirm=payload.get("l2_confirm", False),
            state="active",
            payload_json=payload,
        )
        session.add(setup)
        session.flush()
        return setup.id


def record_alert(
    setup_id: int | None,
    alert_type: str,
    channel: str,
    thread_ts: str | None,
    message_ts: str | None,
    symbol: str,
    direction: str | None,
    payload: dict,
) -> int:
    with get_session() as session:
        alert = Alert(
            setup_id=setup_id,
            sent_ts=datetime.utcnow().replace(tzinfo=timezone.utc),
            channel=channel,
            status="open",
            type=alert_type,
            symbol=symbol,
            direction=direction,
            slack_thread_ts=thread_ts,
            slack_message_ts=message_ts,
            payload_json=payload,
        )
        session.add(alert)
        session.flush()
        return alert.id


def fetch_alert_status(alert_id: int) -> dict | None:
    with get_session() as session:
        alert = session.get(Alert, alert_id)
        if not alert:
            return None
        return {
            "status": alert.status,
            "ack_by": alert.ack_by,
            "ack_ts": alert.ack_ts.isoformat() if alert.ack_ts else None,
            "type": alert.type,
            "payload": alert.payload_json,
            "slack_message_ts": alert.slack_message_ts,
        }


async def build_longlist(ctx: WorkerContext):
    ctx.reset_if_new_day()
    now = now_et()
    if not (4 <= now.hour < 9 or (now.hour == 9 and now.minute <= 10)):
        return
    if ctx.watchlist.longlist_frozen:
        return
    cfg = CONFIG.universe
    discovery = cfg.discovery
    try:
        raw_items = await ctx.polygon.list_top_gappers(
            limit=discovery.top_n * 2,
            gap_threshold_abs=discovery.gap_threshold_abs,
            price_min=cfg.price_min,
            price_max=cfg.price_max,
            min_volume_prev_day=cfg.min_volume_prev_day,
        )
    except Exception as exc:  # pragma: no cover - network failure guard
        logger.error("watchlist.longlist.error", err=str(exc))
        return
    items = [
        WatchlistItem(
            ticker=entry["ticker"],
            price=entry["price"],
            gap_pct=entry["gap_pct"],
            direction=entry["direction"],
            premkt_volume=entry["premkt_volume"],
            rank=idx + 1,
        )
        for idx, entry in enumerate(raw_items)
    ]
    ctx.watchlist.longlist = items
    with get_session() as session:
        upsert_watchlist_entries(session, ctx.watchlist.trade_date, items, source="longlist")
    logger.info("watchlist.longlist", count=len(items), date=str(ctx.watchlist.trade_date))


async def freeze_longlist(ctx: WorkerContext):
    ctx.reset_if_new_day()
    if ctx.watchlist.longlist_frozen:
        return
    ctx.watchlist.longlist_frozen = True
    logger.info("watchlist.freeze", date=str(ctx.watchlist.trade_date))


async def finalize_watchlist(ctx: WorkerContext):
    ctx.reset_if_new_day()
    cfg = CONFIG.universe
    discovery = cfg.discovery

    if not ctx.watchlist.longlist:
        await build_longlist(ctx)

    candidates = list(ctx.watchlist.longlist)
    blockset = {t.upper() for t in cfg.blocklist}
    candidates = [item for item in candidates if item.ticker.upper() not in blockset]

    pinned = []
    for symbol in cfg.pinned:
        if any(item.ticker.upper() == symbol.upper() for item in candidates):
            continue
        pinned.append(
            WatchlistItem(
                ticker=symbol.upper(),
                price=0.0,
                gap_pct=0.0,
                direction="up",
                premkt_volume=0,
            )
        )

    combined = pinned + candidates
    combined = combined[: discovery.top_n]
    for idx, item in enumerate(combined):
        item.rank = idx + 1

    ctx.watchlist.finalized = combined
    ctx.watchlist.finalized_at = now_et()

    with get_session() as session:
        upsert_watchlist_entries(session, ctx.watchlist.trade_date, combined, source="top100")

    if combined:
        abs_gaps = [abs(item.gap_pct) for item in combined if item.gap_pct is not None]
        med_gap = (sum(abs_gaps) / len(abs_gaps)) if abs_gaps else 0.0
        up_count = sum(1 for item in combined if item.direction == "up")
        down_count = sum(1 for item in combined if item.direction == "down")
        summary = (
            f"Watchlist {discovery.top_n} ready · {ctx.watchlist.trade_date.isoformat()}\n"
            f"Up: {up_count} · Down: {down_count} · Median |gap|: {med_gap * 100:.1f}%\n"
            "CSV: (link forthcoming)"
        )
        ctx.slack.post(summary)

    logger.info(
        "watchlist.finalize",
        count=len(combined),
        date=str(ctx.watchlist.trade_date),
        pinned=len(pinned),
    )


async def _fetch_levels(symbol: str, candles_1m: list[Candle], ctx: WorkerContext) -> dict:
    try:
        candles_5m_raw = await ctx.polygon.get_aggregates(symbol, multiplier=5, timespan="minute", limit=60)
        candles_15m_raw = await ctx.polygon.get_aggregates(symbol, multiplier=15, timespan="minute", limit=60)
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("levels.fetch.error", symbol=symbol, err=str(exc))
        return {}
    htf_levels = find_htf_levels(candles_5m_raw, candles_15m_raw)
    prev_close = await ctx.polygon.get_prev_close(symbol)
    gap_edges = find_gap_edges(prev_close, [{"h": c.h, "l": c.l} for c in candles_1m[:10]])
    return {**htf_levels, **gap_edges}


def _format_price(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _in_burst_window(now: datetime) -> bool:
    if now.tzinfo is None:
        now = now_et()
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=10, minute=0, second=0, microsecond=0)
    return start <= now <= end


async def _update_depth_subscriptions(ctx: WorkerContext):
    desired: set[str] = set()
    for symbol, state in ctx.box_states.items():
        # prioritize active, then primed, then armed with good RR
        if any(d.active for d in state.contexts.values()):
            desired.add(symbol)
            continue
        if any(d.primed for d in state.contexts.values()):
            desired.add(symbol)
            continue
    if len(desired) < CONFIG.scan.l2_active_max:
        remaining = CONFIG.scan.l2_active_max - len(desired)
        armed_candidates = sorted(
            ctx.box_states.values(),
            key=lambda bs: max((d.rr or 0) for d in bs.contexts.values()),
            reverse=True,
        )
        for bs in armed_candidates:
            if bs.symbol in desired:
                continue
            desired.add(bs.symbol)
            if len(desired) >= CONFIG.scan.l2_active_max:
                break

    subscribe = desired - ctx.depth_symbols
    unsubscribe = ctx.depth_symbols - desired
    if subscribe:
        await ctx.depth.subscribe(subscribe)
    if unsubscribe:
        await ctx.depth.unsubscribe(unsubscribe)
    ctx.depth_symbols = desired


def _update_dashboard(ctx: WorkerContext) -> None:
    armed_cards = []
    primed_cards = []
    active_cards = []

    for state in ctx.box_states.values():
        for direction, dstate in state.contexts.items():
            if dstate.last_alert_id:
                info = fetch_alert_status(dstate.last_alert_id)
                if info:
                    dstate.action_status = info.get("status")
                    if info.get("payload"):
                        dstate.last_l2_display = info["payload"].get("l2", dstate.last_l2_display)
            pill = []
            if dstate.rr is None or dstate.rr < CONFIG.gating.min_rr:
                pill.append("RR<2.0")
            if ctx.depth_is_demo and ctx.depth_mode == "demo":
                pill.append("L2 demo")
            elif ctx.depth_mode == "ibkr" and ctx.depth_is_demo:
                pill.append("No L2 (fallback demo)")
            elif dstate.no_l2:
                pill.append("No L2")
            if state.box.spread_cents > 1.0:
                pill.append("Spread>1¢")
            if dstate.action_status:
                if dstate.action_status != "open":
                    status_label = {
                        "acknowledged": "ACK",
                        "passed": "PASS",
                        "discuss": "DISC",
                    }.get(dstate.action_status, dstate.action_status.upper())
                    pill.append(status_label)
            if dstate.p2r is not None:
                pill.append(f"p2R {dstate.p2r:.2f}")

            card = {
                "symbol": state.symbol,
                "direction": direction,
                "entry": state.box.hi if direction == "long" else state.box.lo,
                "stop": state.box.lo if direction == "long" else state.box.hi,
                "target": dstate.target,
                "rr": dstate.rr,
                "boxes": state.box.bars,
                "height": state.box.height,
                "spread": state.box.spread_cents,
                "rvol": state.box.rvol,
                "ttl": dstate.primed_expires.isoformat() if dstate.primed_expires else None,
                "l2": dstate.last_l2_display,
                "status": dstate.action_status,
                "pills": pill,
            }

            if dstate.active:
                active_cards.append(card | {"expires": dstate.active_expires.isoformat() if dstate.active_expires else None})
            elif dstate.primed:
                primed_cards.append(card)
            else:
                armed_cards.append(card)

    set_lanes(armed_cards, primed_cards, active_cards, meta={"symbols": list(ctx.box_states.keys())})
    update_worker_tick({
        "armed": len(armed_cards),
        "primed": len(primed_cards),
        "active": len(active_cards),
        "symbols": list(ctx.box_states.keys()),
    })


async def scan_consolidations(ctx: WorkerContext):
    ctx.reset_if_new_day()
    if not ctx.watchlist.finalized:
        return

    ctx.ping_budget.limit = CONFIG.alerts.per_minute_budget_open

    symbols = ctx.finalized_symbols()
    if not symbols:
        return
    batch = symbols[ctx.symbol_cursor : ctx.symbol_cursor + ctx.batch_size]
    ctx.symbol_cursor += len(batch)
    if ctx.symbol_cursor >= len(symbols):
        ctx.symbol_cursor = 0
    if not batch:
        return

    cfg = CONFIG.detectors.consolidation
    gating_cfg = CONFIG.gating
    mention = CONFIG.alerts.mention

    for symbol in batch:
        try:
            candles_1m_raw = await ctx.polygon.get_aggregates(symbol, multiplier=1, timespan="minute", limit=120)
            candles_2m_raw = await ctx.polygon.get_aggregates(symbol, multiplier=2, timespan="minute", limit=120)
        except Exception as exc:  # pragma: no cover - network guard
            logger.error("scan.candles.error", symbol=symbol, err=str(exc))
            continue

        candles_1m = _to_candles(candles_1m_raw)
        candles_2m = _to_candles(candles_2m_raw)
        if not candles_1m:
            continue

        box = find_consolidation_box(candles_1m, candles_2m, cfg)
        if not box:
            ctx.box_states.pop(symbol, None)
            continue

        state = ctx.box_states.get(symbol)
        box_changed = True
        if state and state.box.start_ts == box.start_ts and abs(state.box.hi - box.hi) < 0.01 and abs(state.box.lo - box.lo) < 0.01:
            box_changed = False
        else:
            thread_key = f"{symbol}-{box.tf}-{box.start_ts}-{box.end_ts}"
            ctx.box_states[symbol] = BoxState(symbol=symbol, box=box, thread_key=thread_key)
            state = ctx.box_states[symbol]

        if box_changed:
            state.box_db_id = _persist_box(symbol, box)
            # levels / targets
            levels = await _fetch_levels(symbol, candles_1m, ctx)
            state.levels = levels
            state.gap_edges = {k: v for k, v in levels.items() if k.startswith("gap")}
            state.contexts = {
                "long": DirectionState(direction="long"),
                "short": DirectionState(direction="short"),
            }

            for direction, dstate in state.contexts.items():
                entry = box.hi if direction == "long" else box.lo
                stop = box.lo if direction == "long" else box.hi
                target = nearest_target(direction, entry, levels)
                rr = project_rr(direction, entry, stop, target)
                dstate.target = target
                dstate.rr = rr
                if rr is None or rr < gating_cfg.min_rr:
                    dstate.gating_note = "RR<min"
                dstate.score = int((rr or 0) * 20 + box.quality_score * 10 + (box.rvol or 0) * 5)
                detected_ts = now_et()
                dist_htf = _nearest_htf_distance(levels, direction, entry)
                dist_gap = _gap_distance(levels, direction, entry)
                features = {
                    "box_height": box.height,
                    "box_bars": box.bars,
                    "rvol_break": box.rvol or 0.0,
                    "l2_mean": 0.0,
                    "l2_persist": 0.0,
                    "dist_htf": dist_htf or 0.0,
                    "dist_gap": dist_gap or 0.0,
                    "spread_cents": box.spread_cents,
                    "price": entry,
                    "price_bucket": bucket_price(entry),
                    "time_bucket": bucket_time(detected_ts),
                    "direction_long": 1.0 if direction == "long" else 0.0,
                }
                dstate.features = features

            rr_under = all((d.rr or 0) < gating_cfg.min_rr for d in state.contexts.values())
            state.rr_under = rr_under

            thread_ts = ctx.slack.post_heads_up(
                symbol,
                box,
                {
                    "rvol": box.rvol,
                    "nearest_level": state.contexts["long"].target or state.contexts["short"].target,
                    "ttl": "~3 bars",
                    "notes": "RR<2.0" if rr_under else "",
                },
                thread_key=state.thread_key,
                mention=mention if CONFIG.alerts.heads_up_ping else None,
            )
            state.slack_thread_ts = thread_ts
            state.last_heads_up_ts = now_et()

        # Evaluate depth/PRIMED
        snapshot: Optional[DepthSnapshot] = None
        try:
            snapshot = await ctx.depth.snapshot(symbol)
        except Exception as exc:  # pragma: no cover
            logger.error("depth.snapshot.error", symbol=symbol, err=str(exc))
        for direction, dstate in state.contexts.items():
            dstate.no_l2 = snapshot is None
            now = now_et()
            if snapshot:
                if not (ctx.depth_mode == "ibkr" and ctx.depth_is_demo):
                    mark_snapshot(
                        "demo" if ctx.depth_mode == "demo" else "ibkr",
                        "demo" if ctx.depth_is_demo else "ibkr",
                    )
                bid_ratio = snapshot.bid_ratio
                ask_ratio = snapshot.ask_ratio
                l2_display = f"{ask_ratio*100:.0f}% ask" if direction == "long" else f"{bid_ratio*100:.0f}% bid"
                if ctx.depth_is_demo:
                    l2_display = f"{l2_display} (demo)"
                dstate.last_l2_display = l2_display
                dstate.l2_buffer.append((now, bid_ratio, ask_ratio))
                threshold = CONFIG.detectors.l2_confirm.imbalance_threshold
                persistence = CONFIG.detectors.l2_confirm.persistence_seconds
                recent = [sample for sample in dstate.l2_buffer if (now - sample[0]).total_seconds() <= persistence]
                if not recent:
                    lean_ok = False
                    dstate.features["l2_mean"] = 0.0
                    dstate.features["l2_persist"] = 0.0
                else:
                    avg_bid = sum(s[1] for s in recent) / len(recent)
                    avg_ask = sum(s[2] for s in recent) / len(recent)
                    if direction == "long":
                        lean_ok = avg_ask >= threshold
                        l2_display = f"{avg_ask*100:.0f}% ask"
                        dstate.features["l2_mean"] = avg_ask
                    else:
                        lean_ok = avg_bid >= threshold
                        l2_display = f"{avg_bid*100:.0f}% bid"
                        dstate.features["l2_mean"] = avg_bid
                    if ctx.depth_is_demo:
                        l2_display = f"{l2_display} (demo)"
                    dstate.last_l2_display = l2_display
                    if len(recent) >= 2:
                        dstate.features["l2_persist"] = (recent[-1][0] - recent[0][0]).total_seconds()
                    else:
                        dstate.features["l2_persist"] = len(recent)
                if lean_ok and dstate.rr and dstate.rr >= gating_cfg.min_rr and box.spread_cents <= 1.0:
                    if not dstate.primed:
                        dstate.primed = True
                        dstate.primed_started = now
                        dstate.primed_expires = now + timedelta(minutes=dstate.primed_ttl_bars)
                        logger.info(
                            "alert.transition",
                            symbol=symbol,
                            state_from="armed",
                            state_to="primed",
                            direction=direction,
                            setup_id=direction_state.setup_id,
                        )
                        threshold = ctx.threshold_for(symbol, dstate.features)
                        meets_threshold = True
                        if threshold:
                            if dstate.features.get("l2_mean", 0.0) < threshold.get("l2", 0.0):
                                meets_threshold = False
                            if dstate.features.get("rvol_break", 0.0) < threshold.get("rvol", 0.0):
                                meets_threshold = False
                        if not meets_threshold:
                            dstate.gating_note = "below learned threshold"
                        if dstate.p2r is None:
                            dstate.p2r = ctx.score_features(symbol, dstate.features)
                        can_ping = CONFIG.alerts.primed_ping
                        if CONFIG.alerts.primed_ping:
                            if _in_burst_window(now):
                                if meets_threshold:
                                    can_ping = ctx.ping_budget.allow(now)
                                else:
                                    can_ping = False
                            else:
                                can_ping = ctx.ping_budget.allow(now)
                        primed_payload = {
                            "entry": _format_price(box.hi if direction == "long" else box.lo),
                            "stop": _format_price(box.lo if direction == "long" else box.hi),
                            "target": _format_price(dstate.target),
                            "rr": f"{dstate.rr:.2f}" if dstate.rr else "n/a",
                            "l2": l2_display if snapshot else "No L2",
                            "spread": f"{box.spread_cents:.1f}¢",
                            "ttl": "~3 bars",
                            "note": dstate.gating_note or "",
                            "p2r": f"{dstate.p2r:.2f}" if dstate.p2r is not None else "n/a",
                        }
                        primed_ts = ctx.slack.post_primed(
                            state.thread_key,
                            symbol,
                            direction,
                            primed_payload,
                            mention=mention,
                            ping=CONFIG.alerts.primed_ping and can_ping,
                        )
                        dstate.primed_pinged = True
                        dstate.last_alert_ts = primed_ts
                        dstate.alert_type = "primed"
                        dstate.last_alert_id = record_alert(
                            setup_id=None,
                            alert_type="primed",
                            channel=CONFIG.alerts.slack_channel,
                            thread_ts=state.slack_thread_ts,
                            message_ts=primed_ts,
                            symbol=symbol,
                            direction=direction,
                            payload={
                                "entry": primed_payload.get("entry"),
                                "stop": primed_payload.get("stop"),
                                "target": primed_payload.get("target"),
                                "rr": primed_payload.get("rr"),
                                "l2": primed_payload.get("l2"),
                                "spread": primed_payload.get("spread"),
                                "p2r": primed_payload.get("p2r"),
                                "features": dstate.features,
                            },
                        )
                        dstate.action_status = "open"
                else:
                    if dstate.primed and (not lean_ok or box.spread_cents > 1.0):
                        dstate.primed = False
                        dstate.primed_started = None
                        dstate.primed_expires = None
            else:
                if dstate.primed:
                    dstate.primed = False
                    dstate.primed_started = None
                    dstate.primed_expires = None
                dstate.last_l2_display = None
                if ctx.depth_mode == "ibkr" and not ctx.depth_is_demo:
                    mark_stale("ibkr", "no snapshot")

            if dstate.primed and dstate.primed_expires and now_et() > dstate.primed_expires:
                dstate.primed = False
                dstate.primed_started = None
                dstate.primed_expires = None
                logger.info(
                    "alert.transition",
                    symbol=symbol,
                    state_from="primed",
                    state_to="armed",
                    direction=direction,
                    reason="ttl_expired",
                    setup_id=dstate.setup_id,
                )

            if dstate.features:
                dstate.p2r = ctx.score_features(symbol, dstate.features)

        # Trigger detection
        breakout_cfg = CONFIG.detectors.breakout
        trigger = check_break_and_retest(
            candles_1m,
            box_hi=box.hi,
            box_lo=box.lo,
            close_outside_ticks=breakout_cfg.close_outside_ticks,
            burst_rvol=breakout_cfg.volume_burst_rvol,
            retest_max_bars=breakout_cfg.retest_max_bars,
        )
        if trigger:
            direction_state = state.contexts.get(trigger.direction)
            if direction_state and direction_state.last_trigger_ts != trigger.break_ts:
                if direction_state.rr and direction_state.rr >= gating_cfg.min_rr:
                    if trigger.volume_burst_rvol:
                        direction_state.features["rvol_break"] = trigger.volume_burst_rvol
                    direction_state.active = True
                    direction_state.active_started = now_et()
                    direction_state.active_expires = now_et() + timedelta(minutes=breakout_cfg.retest_max_bars)
                    direction_state.last_trigger_ts = trigger.break_ts
                    logger.info(
                        "alert.transition",
                        symbol=symbol,
                        state_from="primed" if direction_state.primed else "armed",
                        state_to="active",
                        direction=trigger.direction,
                        setup_id=setup_id,
                    )
                    if direction_state.p2r is None:
                        direction_state.p2r = ctx.score_features(symbol, direction_state.features)
                    payload = {
                        "entry": f"{trigger.entry_price:.2f}",
                        "stop": f"{trigger.invalidation:.2f}",
                        "targets": _format_price(direction_state.target),
                        "rr": f"{direction_state.rr:.2f}" if direction_state.rr else "n/a",
                        "p2r": f"{direction_state.p2r:.2f}" if direction_state.p2r is not None else "n/a",
                        "spread": f"{trigger.spread_cents:.1f}¢",
                        "rvol": f"{trigger.volume_burst_rvol:.1f}x",
                        "l2": direction_state.last_l2_display or ("No L2" if direction_state.no_l2 else "Confirmed"),
                        "box_summary": f"{box.bars} bars · {box.height:.3f}h",
                        "primary_target": direction_state.target,
                        "score": direction_state.score,
                        "l2_confirm": not direction_state.no_l2,
                        "features": direction_state.features,
                        "price_bucket": direction_state.features.get("price_bucket"),
                        "time_bucket": direction_state.features.get("time_bucket"),
                        "symbol": symbol,
                        "note": direction_state.gating_note,
                    }
                    direction_state.trigger_payload = payload
                    setup_id = _persist_setup(symbol, trigger.direction, payload, box, state.box_db_id)
                    direction_state.setup_id = setup_id
                    threshold = ctx.threshold_for(symbol, direction_state.features)
                    meets_threshold = True
                    if threshold:
                        if direction_state.features.get("l2_mean", 0.0) < threshold.get("l2", 0.0):
                            meets_threshold = False
                        if direction_state.features.get("rvol_break", 0.0) < threshold.get("rvol", 0.0):
                            meets_threshold = False
                    if not meets_threshold:
                        direction_state.gating_note = "below learned threshold"
                    can_ping = True
                    now_ts = now_et()
                    if _in_burst_window(now_ts):
                        if meets_threshold:
                            can_ping = ctx.ping_budget.allow(now_ts)
                        else:
                            can_ping = False
                    else:
                        can_ping = ctx.ping_budget.allow(now_ts)
                    trigger_ts = ctx.slack.post_trigger(
                        state.thread_key,
                        symbol,
                        payload | {"targets": payload["targets"]},
                        mention=mention,
                        ping=can_ping,
                    )
                    direction_state.last_alert_ts = trigger_ts
                    direction_state.alert_type = "trigger"
                    direction_state.last_alert_id = record_alert(
                        setup_id=setup_id,
                        alert_type="trigger",
                        channel=CONFIG.alerts.slack_channel,
                        thread_ts=state.slack_thread_ts,
                        message_ts=trigger_ts,
                        symbol=symbol,
                        direction=trigger.direction,
                        payload=payload,
                    )
                    direction_state.action_status = "open"

        for dstate in state.contexts.values():
            if dstate.active and dstate.active_expires and now_et() > dstate.active_expires:
                dstate.active = False
                dstate.trigger_payload = None
                logger.info(
                    "alert.transition",
                    symbol=state.symbol,
                    state_from="active",
                    state_to="armed",
                    reason="retest_window_closed",
                    setup_id=dstate.setup_id,
                )

    await _update_depth_subscriptions(ctx)
    _update_dashboard(ctx)


async def sync_trades(ctx: WorkerContext):
    trade_date = now_et().date()
    inserted = await asyncio.to_thread(ctx.ledger.sync, trade_date)
    if inserted:
        logger.info("ledger.sync", count=inserted, date=str(trade_date))


async def eod_summary(ctx: WorkerContext):
    trade_date = now_et().date()
    await sync_trades(ctx)
    summary = await asyncio.to_thread(ctx.ledger.daily_summary, trade_date)
    wins = summary.get("wins", 0)
    total = summary.get("total_trades", 0)
    win_rate = summary.get("win_rate", 0)
    avg_r = summary.get("avg_r", 0)
    expectancy = summary.get("expectancy", 0)
    net_pnl = summary.get("net_pnl", 0)
    text = (
        f"EOD Summary · {trade_date}\n"
        f"Wins {wins}/{total} ({win_rate}%) · Avg R {avg_r} · Expectancy {expectancy}\n"
        f"Net PnL ${net_pnl}"
    )
    top_setups = summary.get("by_setup", [])[:3]
    if top_setups:
        lines = [f"{row.get('symbol')} #{row.get('setup_id') or '-'} · PnL ${row.get('pnl')} · R {row.get('realized_r')}" for row in top_setups]
        text += "\nTop setups:\n" + "\n".join(lines)
    ctx.slack.post(text)
    logger.info("eod.summary", summary=summary)


async def nightly_learning_job(ctx: WorkerContext):
    trade_date = now_et().date()
    try:
        result = await asyncio.to_thread(ctx.learning.train, trade_date)
    except Exception as exc:  # pragma: no cover - safety net
        logger.error("learning.nightly.error", err=str(exc))
        ctx.slack.post(f"Nightly Learning failed: {exc}")
        return
    if result.get("status") == "trained":
        ctx.thresholds = ctx.learning.load_thresholds()
        ctx.threshold_canary = ctx.learning.load_canary()
        ctx.model_loaded = False
        ctx.last_learning_report = result
        message = (
            f"Nightly Learning Summary · {trade_date}\n"
            f"Rows {result.get('rows')} · Positives {result.get('positives')}\n"
            f"Thresholds buckets {len(result.get('thresholds', {}).get('buckets', {}))}"
        )
        ctx.slack.post(message)
    else:
        ctx.last_learning_report = result
        ctx.slack.post(f"Nightly Learning skipped: {result.get('reason', 'unknown')}")
    logger.info("learning.nightly", result=result)


async def scheduler():
    asyncio.create_task(every(120.0, build_longlist, CTX))
    asyncio.create_task(every(10.0, scan_consolidations, CTX))
    asyncio.create_task(every(300.0, sync_trades, CTX))

    async def freeze_job():
        while True:
            await asyncio.sleep(seconds_until(9, 10))
            await freeze_longlist(CTX)

    async def finalize_job():
        while True:
            await asyncio.sleep(seconds_until(9, 25))
            await finalize_watchlist(CTX)

    asyncio.create_task(freeze_job())
    asyncio.create_task(finalize_job())

    async def at_1605():
        while True:
            await asyncio.sleep(seconds_until(16, 5))
            await sync_trades(CTX)

    async def at_1610():
        while True:
            await asyncio.sleep(seconds_until(16, 10))
            await eod_summary(CTX)

    async def at_0200():
        while True:
            await asyncio.sleep(seconds_until(2, 0))
            await nightly_learning_job(CTX)

    asyncio.create_task(at_1605())
    asyncio.create_task(at_1610())
    asyncio.create_task(at_0200())

    while True:
        await asyncio.sleep(3600)


def main():
    logger.info("worker.start", pid=os.getpid(), t=datetime.utcnow().isoformat() + "Z")
    asyncio.run(scheduler())


if __name__ == "__main__":
    main()
