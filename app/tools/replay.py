from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.detectors.consolidation import Candle, find_consolidation_box
from app.core.detectors.trigger import check_break_and_retest


def load_tape(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("candles", [])


def replay(path: Path) -> None:
    candles_raw = load_tape(path)
    candles = [
        Candle(ts=c["t"], o=c["o"], h=c["h"], l=c["l"], c=c["c"], v=c["v"])
        for c in candles_raw
    ]
    box = find_consolidation_box(candles, candles, cfg=_dummy_cfg())
    if not box:
        print("No consolidation detected")
        return
    print(f"Box detected: {box.bars} bars {box.lo:.2f}-{box.hi:.2f}")
    trigger = check_break_and_retest(candles, box.hi, box.lo, 1, 1.5, 4)
    if trigger:
        print(
            f"Trigger: {trigger.direction} entry {trigger.entry_price:.2f} retest at {trigger.retest_ts} "
            f"burst {trigger.volume_burst_rvol:.2f}x"
        )
    else:
        print("No trigger detected")


def _dummy_cfg():
    class Cfg:
        min_bars = 8
        max_bars = 25
        box_height_caps = ["$0.10", "0.35pct_mid", "0.6*ATR14_1m"]
        compression_check = True
        rvol_min_inside = 1.0

    return Cfg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay 1m tape for debugging")
    parser.add_argument("path", type=Path, help="Path to tape JSON")
    args = parser.parse_args()
    replay(args.path)
