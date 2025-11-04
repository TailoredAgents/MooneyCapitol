from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.adapters.slack import SlackAdapter
from app.services.ledger import get_ledger_service, ingest_fills
from app.utils.time import now_et


router = APIRouter(prefix="", tags=["reports"])


@router.get("/reports/eod")
def get_eod_report(date: date | None = Query(default=None)):
    trade_date = date or now_et().date()
    ledger = get_ledger_service()
    return ledger.daily_summary(trade_date)


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


@router.post("/ingest/fills")
def ingest_manual_fills(fills: list[dict]):
    ledger = get_ledger_service()
    inserted = ingest_fills(fills)
    if inserted:
        ledger.invalidate()
    return {"inserted": inserted}
