"""Automatic daily calibration.

Runs the backtest over the last week of 1-minute candles, then fits a volatility
multiplier that makes the probabilities honest. The live model applies the multiplier.
If the model is still badly calibrated after tuning, the dashboard blocks BET signals.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .backtest import evaluate, fit_vol_multiplier, max_calibration_gap, predict_windows, rescale
from .data.prices import fetch_candles_1m

log = logging.getLogger(__name__)

MIN_WINDOWS = 200
MAX_GAP = 0.06  # worst calibration bin may be off by at most 6 points
MAX_AGE_S = 24 * 3600


@dataclass(frozen=True)
class CalibrationResult:
    ts: float
    days: float
    source: str
    windows: int
    vol_multiplier: float
    brier_raw: float
    brier_tuned: float
    max_gap: float

    @property
    def ok(self) -> bool:
        return self.windows >= MIN_WINDOWS and self.max_gap <= MAX_GAP

    def summary(self) -> str:
        return (
            f"{self.days:g}d, {self.windows} windows, vol ×{self.vol_multiplier:.2f}, "
            f"Brier {self.brier_tuned:.3f}, worst bin off {self.max_gap:.1%}"
        )


def load_cached(path: str | Path, source: str) -> CalibrationResult | None:
    try:
        data = json.loads(Path(path).read_text())
        result = CalibrationResult(**data)
    except (OSError, ValueError, TypeError):
        return None
    if result.source != source or time.time() - result.ts > MAX_AGE_S:
        return None
    return result


def save(path: str | Path, result: CalibrationResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(result), indent=2))


async def run_calibration(client: httpx.AsyncClient, source: str, days: float = 7) -> CalibrationResult:
    end = int(time.time()) // 60 * 60
    start = end - int(days * 86400) - 3600
    candles = await fetch_candles_1m(client, source, start, end)
    return calibrate_candles(candles, source, days)


def calibrate_candles(candles, source: str, days: float) -> CalibrationResult:
    preds = predict_windows(candles)
    raw = evaluate(preds)
    k = fit_vol_multiplier(preds)
    tuned = evaluate(rescale(preds, k))
    return CalibrationResult(
        ts=time.time(),
        days=days,
        source=source,
        windows=tuned.windows,
        vol_multiplier=k,
        brier_raw=raw.brier,
        brier_tuned=tuned.brier,
        max_gap=max_calibration_gap(tuned),
    )
