"""Short-horizon volatility estimation.

We track variance of log returns *per second*. It is seeded from 1-minute candles
(per-minute variance / 60) and then updated with an EWMA over ~1s price samples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

SECONDS_PER_YEAR = 365 * 24 * 3600

# Guard rails: BTC per-second vol outside these bounds means bad data, not a real market.
MIN_SIGMA_PER_SEC = 0.15 / math.sqrt(SECONDS_PER_YEAR)  # 15% annualized
MAX_SIGMA_PER_SEC = 3.00 / math.sqrt(SECONDS_PER_YEAR)  # 300% annualized


def per_second_variance_from_closes(closes: Sequence[float], bar_seconds: int = 60) -> float | None:
    """Sample variance of log returns between consecutive closes, scaled to per-second."""
    if len(closes) < 3:
        return None
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return var / bar_seconds


class EwmaVolatility:
    """EWMA estimate of per-second log-return variance from irregular price samples."""

    def __init__(self, half_life_s: float = 900.0, seed_var_per_s: float | None = None):
        self.decay_per_s = math.log(2) / half_life_s
        self.var_per_s = seed_var_per_s
        self._last_ts: float | None = None
        self._last_px: float | None = None
        self.samples = 0

    def seed(self, var_per_s: float) -> None:
        self.var_per_s = var_per_s

    def update(self, ts: float, price: float) -> None:
        if price <= 0:
            return
        if self._last_ts is None or self._last_px is None:
            self._last_ts, self._last_px = ts, price
            return
        dt = ts - self._last_ts
        if dt < 1.0:  # sample at most ~1/s to limit microstructure noise
            return
        r = math.log(price / self._last_px)
        # A gap of dt seconds contributes one observation of variance r^2/dt, weighted by dt.
        obs = (r * r) / dt
        alpha = 1.0 - math.exp(-self.decay_per_s * dt)
        self.var_per_s = obs if self.var_per_s is None else (1 - alpha) * self.var_per_s + alpha * obs
        self._last_ts, self._last_px = ts, price
        self.samples += 1

    @property
    def sigma_per_s(self) -> float | None:
        if self.var_per_s is None:
            return None
        return clamp_sigma(math.sqrt(self.var_per_s))


def clamp_sigma(sigma_per_s: float) -> float:
    return min(MAX_SIGMA_PER_SEC, max(MIN_SIGMA_PER_SEC, sigma_per_s))


def annualized(sigma_per_s: float) -> float:
    return sigma_per_s * math.sqrt(SECONDS_PER_YEAR)
