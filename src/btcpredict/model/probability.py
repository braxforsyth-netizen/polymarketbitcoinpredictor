"""Probability that BTC finishes the window at or above its opening price.

Driftless random walk in log-price:  P(UP) = Phi( ln(S/K) / (sigma * sqrt(tau)) )
  S = current price, K = window open ("price to beat"), tau = seconds remaining,
  sigma = per-second volatility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_up(price: float, open_price: float, sigma_per_s: float, seconds_left: float) -> float:
    if seconds_left <= 0 or sigma_per_s <= 0:
        return 1.0 if price >= open_price else 0.0
    z = math.log(price / open_price) / (sigma_per_s * math.sqrt(seconds_left))
    return norm_cdf(z)


@dataclass(frozen=True)
class Projection:
    p_up: float
    p_low: float  # P(UP) if vol is higher/lower than estimated -> uncertainty band
    p_high: float
    sigma_per_s: float
    seconds_left: float
    expected_move_usd: float  # 1-sigma move over the remaining time
    vol_multiplier: float

    @property
    def p_down(self) -> float:
        return 1.0 - self.p_up


def project(
    price: float,
    open_price: float,
    sigma_per_s: float,
    seconds_left: float,
    vol_multiplier: float = 1.0,
    vol_uncertainty: float = 0.25,
) -> Projection:
    """Point estimate plus a band from re-running with sigma scaled by (1 ± vol_uncertainty).

    vol_multiplier > 1 widens the distribution, e.g. when high-impact news just hit.
    """
    sigma = sigma_per_s * vol_multiplier
    p = prob_up(price, open_price, sigma, seconds_left)
    p_a = prob_up(price, open_price, sigma * (1 + vol_uncertainty), seconds_left)
    p_b = prob_up(price, open_price, sigma * (1 - vol_uncertainty), seconds_left)
    return Projection(
        p_up=p,
        p_low=min(p_a, p_b),
        p_high=max(p_a, p_b),
        sigma_per_s=sigma,
        seconds_left=seconds_left,
        expected_move_usd=price * sigma * math.sqrt(max(seconds_left, 0.0)),
        vol_multiplier=vol_multiplier,
    )
