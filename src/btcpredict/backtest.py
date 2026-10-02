"""Calibration backtest on historical 1-minute candles.

For every complete 15-minute window, at each whole minute m = 1..14 we compute the model's
P(UP) from the price at that minute, the window open, and volatility measured over the
preceding hour, and compare with what actually happened. A well-calibrated model's 70%
calls should come true about 70% of the time.

This checks the probability model only. Whether you can profit depends on Polymarket's
prices at those moments, which the live recorder collects going forward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .data.prices import Candle
from .model.probability import norm_cdf, prob_up
from .model.volatility import clamp_sigma, per_second_variance_from_closes
from .model.windows import WINDOW_SECONDS

VOL_LOOKBACK_MIN = 60


@dataclass(frozen=True)
class Prediction:
    window_start: int
    minute: int  # minutes elapsed when the prediction was made
    p_up: float
    outcome: int  # 1 if UP
    log_ratio: float = 0.0  # ln(price / open)
    scale: float = 0.0  # sigma * sqrt(seconds left)


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    hit_rate: float


@dataclass(frozen=True)
class Report:
    windows: int
    predictions: int
    up_rate: float
    brier: float
    brier_coinflip: float
    log_loss: float
    by_minute: list[tuple[int, int, float]]  # (minute, n, brier)
    bins: list[CalibrationBin]


def predict_windows(candles: Sequence[Candle]) -> list[Prediction]:
    by_ts = {c.ts: c for c in candles}
    ordered = sorted(by_ts)
    if not ordered:
        return []
    first = ((ordered[0] + WINDOW_SECONDS - 1) // WINDOW_SECONDS) * WINDOW_SECONDS
    preds: list[Prediction] = []
    for start in range(first, ordered[-1] + 1, WINDOW_SECONDS):
        bars = [by_ts.get(start + 60 * i) for i in range(15)]
        history = [by_ts.get(start - 60 * i) for i in range(VOL_LOOKBACK_MIN, 0, -1)]
        if any(b is None for b in bars) or sum(h is not None for h in history) < VOL_LOOKBACK_MIN * 0.9:
            continue
        open_px = bars[0].open
        outcome = int(bars[14].close >= open_px)
        closes = [h.close for h in history if h is not None]
        for m in range(1, 15):
            closes_now = closes + [b.close for b in bars[:m]]
            var = per_second_variance_from_closes(closes_now[-VOL_LOOKBACK_MIN:])
            if not var:
                continue
            sigma = clamp_sigma(math.sqrt(var))
            price, tau = bars[m - 1].close, (15 - m) * 60
            p = prob_up(price, open_px, sigma, tau)
            preds.append(Prediction(start, m, p, outcome, math.log(price / open_px), sigma * math.sqrt(tau)))
    return preds


def evaluate(preds: Sequence[Prediction], n_bins: int = 10) -> Report:
    if not preds:
        raise ValueError("no complete windows to evaluate")
    eps = 1e-6
    brier = sum((p.p_up - p.outcome) ** 2 for p in preds) / len(preds)
    log_loss = -sum(
        math.log(max(eps, p.p_up if p.outcome else 1 - p.p_up)) for p in preds
    ) / len(preds)
    windows = {p.window_start: p.outcome for p in preds}
    up_rate = sum(windows.values()) / len(windows)

    by_minute = []
    for m in range(1, 15):
        sub = [p for p in preds if p.minute == m]
        if sub:
            by_minute.append((m, len(sub), sum((p.p_up - p.outcome) ** 2 for p in sub) / len(sub)))

    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sub = [p for p in preds if lo <= p.p_up < hi or (i == n_bins - 1 and p.p_up == 1.0)]
        if sub:
            bins.append(
                CalibrationBin(
                    lo, hi, len(sub),
                    sum(p.p_up for p in sub) / len(sub),
                    sum(p.outcome for p in sub) / len(sub),
                )
            )
    return Report(len(windows), len(preds), up_rate, brier, 0.25, log_loss, by_minute, bins)


def rescale(preds: Sequence[Prediction], vol_multiplier: float) -> list[Prediction]:
    """Recompute P(UP) as if volatility were scaled by vol_multiplier."""
    out = []
    for p in preds:
        s = p.scale * vol_multiplier
        p_up = norm_cdf(p.log_ratio / s) if s > 0 else float(p.log_ratio >= 0)
        out.append(Prediction(p.window_start, p.minute, p_up, p.outcome, p.log_ratio, p.scale))
    return out


def fit_vol_multiplier(preds: Sequence[Prediction], lo: float = 0.6, hi: float = 2.0, step: float = 0.05) -> float:
    """Volatility scale that minimizes log loss on history. >1 means the raw model is overconfident
    (e.g. fat tails), <1 means it is underconfident."""
    best_k, best_loss = 1.0, float("inf")
    k = lo
    while k <= hi + 1e-9:
        loss = evaluate(rescale(preds, k)).log_loss
        if loss < best_loss:
            best_k, best_loss = round(k, 2), loss
        k += step
    return best_k


def max_calibration_gap(report: Report, min_n: int = 200) -> float:
    """Largest |actual - predicted| across calibration bins with enough samples."""
    gaps = [abs(b.hit_rate - b.mean_pred) for b in report.bins if b.n >= min_n]
    return max(gaps, default=0.0)
