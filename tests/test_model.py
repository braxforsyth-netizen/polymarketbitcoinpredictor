import math
import random

import pytest

from btcpredict.model.edge import fee_per_share, recommend
from btcpredict.model.probability import prob_up, project
from btcpredict.model.volatility import EwmaVolatility, per_second_variance_from_closes
from btcpredict.model.windows import window_at

KW = dict(min_edge=0.04, fee_rate=0.0, kelly_fraction=0.25, bankroll=100, max_stake_fraction=0.05, min_seconds_left=45)
SIGMA = 0.5 / math.sqrt(365 * 24 * 3600)  # 50% annualized, per second


def test_window_alignment_and_slug():
    w = window_at(1_759_420_800 + 437)
    assert w.start == 1_759_420_800 and w.end - w.start == 900
    assert w.slug == "btc-updown-15m-1759420800"
    assert w.seconds_left(w.start + 600) == 300


def test_prob_up_basics():
    assert prob_up(100, 100, SIGMA, 600) == pytest.approx(0.5)
    assert prob_up(100.2, 100, SIGMA, 600) > 0.5 > prob_up(99.8, 100, SIGMA, 600)
    # same lead is worth more with less time left
    assert prob_up(100.1, 100, SIGMA, 60) > prob_up(100.1, 100, SIGMA, 600)
    assert prob_up(100, 100, SIGMA, 0) == 1.0  # tie resolves UP
    assert prob_up(99.9, 100, SIGMA, 0) == 0.0


def test_projection_band_and_multiplier():
    p = project(100.1, 100, SIGMA, 300)
    assert p.p_low <= p.p_up <= p.p_high
    wide = project(100.1, 100, SIGMA, 300, vol_multiplier=1.5)
    assert 0.5 < wide.p_up < p.p_up


def test_volatility_estimators_recover_sigma():
    rng = random.Random(1)
    px, closes = 60_000.0, []
    ewma = EwmaVolatility(half_life_s=3600)
    for t in range(6 * 3600):
        px *= math.exp(rng.gauss(0, SIGMA))
        ewma.update(float(t), px)
        if t % 60 == 0:
            closes.append(px)
    assert math.sqrt(per_second_variance_from_closes(closes)) == pytest.approx(SIGMA, rel=0.15)
    assert ewma.sigma_per_s == pytest.approx(SIGMA, rel=0.15)


def test_fee_model():
    assert fee_per_share(0.5, 0.03) == pytest.approx(0.015)
    assert fee_per_share(0.9, 0.03) == pytest.approx(0.003)


def test_recommend_flags_implausible_disagreement():
    rec = recommend(0.99, (0.98, 1.0), 0.55, 0.47, 300, **KW)
    assert rec.action == "NO BET" and "disagrees with the market" in rec.reasons[-1]


def test_recommend_bet_when_edge_clear():
    rec = recommend(0.75, (0.72, 0.78), 0.60, 0.42, 300, **KW)
    assert rec.action == "BET UP"
    assert rec.best.ev_per_dollar == pytest.approx(0.25)
    assert rec.best.kelly == pytest.approx((0.75 - 0.60) / 0.40)
    assert rec.stake == pytest.approx(5.0)  # capped at 5% of bankroll


def test_recommend_no_bet_cases():
    assert recommend(0.52, (0.5, 0.54), 0.51, 0.51, 300, **KW).action == "NO BET"  # thin edge
    assert recommend(0.75, (0.72, 0.78), 0.60, 0.42, 20, **KW).action == "NO BET"  # too late
    assert recommend(0.75, (0.72, 0.78), 0.60, 0.42, 300, news_shock=True, **KW).action == "NO BET"
    assert recommend(0.75, (0.55, 0.85), 0.60, 0.42, 300, **KW).action == "NO BET"  # band too wide
    assert recommend(0.75, (0.72, 0.78), None, None, 300, **KW).action == "NO BET"


def test_recommend_down_side():
    rec = recommend(0.20, (0.18, 0.22), 0.25, 0.70, 300, **KW)
    assert rec.action == "BET DOWN" and rec.best.side == "DOWN"
