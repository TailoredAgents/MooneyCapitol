from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.adapters.slack import SlackAdapter
from app.db.session import get_session
from app.services.ai_artifacts import latest_ai_artifact
from app.services.ledger import get_ledger_service, ingest_fills
from app.utils.time import now_et


router = APIRouter(prefix="", tags=["reports"])


@router.get("/reports/eod")
def get_eod_report(date: date | None = Query(default=None)):
    trade_date = date or now_et().date()
    ledger = get_ledger_service()
    summary = dict(ledger.daily_summary(trade_date))
    summary["ai_recap"] = _latest_daily_recap(trade_date)
    return summary


@router.get("/reports/eod.csv")
def get_eod_report_csv(date: date | None = Query(default=None)):
    trade_date = date or now_et().date()
    ledger = get_ledger_service()
    csv_data = ledger.summary_csv(trade_date)
    filename = f"eod_{trade_date}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/reports/eod/send")
def send_eod_report(date: date | None = Query(default=None)):
    trade_date = date or now_et().date()
    ledger = get_ledger_service()
    summary = ledger.daily_summary(trade_date)
    text = (
        f"EOD Summary · {trade_date}\n"
        f"Wins {summary['wins']}/{summary['total_trades']} ({summary['win_rate']}%) · Avg R {summary['avg_r']} · Expectancy {summary['expectancy']}\n"
        f"Net PnL ${summary['net_pnl']}"
    )
    top = summary.get("by_setup", [])[:3]
    if top:
        text += "\nTop setups:\n" + "\n".join(
            f"{row.get('symbol')} #{row.get('setup_id') or '-'} · PnL ${row.get('pnl')} · R {row.get('realized_r')}" for row in top
        )
    SlackAdapter().post(text)
    return {"ok": True}


@router.get("/reports/eod/ai")
def get_eod_ai_recap(date: date | None = Query(default=None)):
    trade_date = date or now_et().date()
    recap = _latest_daily_recap(trade_date)
    if recap is None:
        return {"date": trade_date.isoformat(), "ai_recap": None}
    return {"date": trade_date.isoformat(), "ai_recap": recap}


@router.post("/ingest/fills")
def ingest_manual_fills(fills: list[dict]):
    ledger = get_ledger_service()
    inserted = ingest_fills(fills)
    if inserted:
        ledger.invalidate()
    return {"inserted": inserted}


def _latest_daily_recap(trade_date: date) -> dict | None:
    with get_session() as session:
        artifact = latest_ai_artifact(
            session,
            artifact_type="daily_recap",
            source_type="eod_report",
            source_id=trade_date.isoformat(),
        )
        if not artifact or artifact.status != "completed" or not artifact.output_text:
            return None
        return {
            "text": artifact.output_text,
            "model": artifact.model,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
        }
