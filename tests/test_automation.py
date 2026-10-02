"""Calibration, readiness, data check, startup checks and the panels that show them."""

import math
import random
import sqlite3
import time

import httpx
import pytest
from rich.console import Console

from btcpredict import calibration, review
from btcpredict.config import Settings
from btcpredict.dashboard import health_panel, paper_panel, render
from btcpredict.data.prices import Candle
from btcpredict.engine import Engine
from btcpredict.health import Check, check_exchange_rest, check_polymarket
from btcpredict.recorder import SCHEMA
from btcpredict.supervisor import Supervisor
from test_backtest import synthetic_candles


def jumpy_candles(days: float, seed: int = 11) -> list[Candle]:
    """Random walk whose 1m vol is mostly quiet with occasional bursts (fat tails)."""
    rng = random.Random(seed)
    start = 1_759_000_500 // 900 * 900
    px, out, base = 60_000.0, [], 0.4 / math.sqrt(365 * 24 * 60)
    for i in range(int(days * 1440)):
        o = px
        regime = 4.0 if (i // 15) % 10 == 0 else 0.6  # calm most windows, wild 1 in 10
        px *= math.exp(rng.gauss(0, base * regime))
        out.append(Candle(start + 60 * i, o, max(o, px), min(o, px), px))
    return out


def test_calibration_on_clean_random_walk_is_ok():
    r = calibration.calibrate_candles(synthetic_candles(days=10), "coinbase", 10)
    assert r.windows > 900 and 0.8 <= r.vol_multiplier <= 1.25 and r.ok


def test_calibration_tunes_vol_and_flags_unfixable_data():
    from btcpredict.backtest import evaluate, predict_windows, rescale

    candles = jumpy_candles(days=10)
    r = calibration.calibrate_candles(candles, "coinbase", 10)
    preds = predict_windows(candles)
    assert r.vol_multiplier != 1.0
    assert evaluate(rescale(preds, r.vol_multiplier)).log_loss < evaluate(preds).log_loss
    assert not r.ok  # still badly calibrated -> the dashboard will block bets


def test_calibration_cache_roundtrip(tmp_path):
    r = calibration.CalibrationResult(time.time(), 7, "coinbase", 600, 1.1, 0.12, 0.11, 0.02)
    calibration.save(tmp_path / "c.json", r)
    assert calibration.load_cached(tmp_path / "c.json", "coinbase") == r
    assert calibration.load_cached(tmp_path / "c.json", "binance") is None  # different source


def test_readiness_verdicts():
    paper_good = review.TradeSummary(40, 25, 40.0, 5.0, 2.0)
    r = review.Readiness(120, 0.10, 0.11, paper_good)
    assert not r.validated and "120/300" in r.verdict()
    r = review.Readiness(320, 0.12, 0.11, paper_good)
    assert not r.validated and "forecast better than the model" in r.verdict()
    r = review.Readiness(320, 0.10, 0.11, review.TradeSummary(40, 15, 40.0, -3.0, 6.0))
    assert not r.validated and "lost money" in r.verdict()
    r = review.Readiness(320, 0.10, 0.11, paper_good)
    assert r.validated and r.verdict().startswith("Validated")


W = 1_759_420_800


def resolved_market_handler(up_wins: dict[int, bool]):
    def handler(req: httpx.Request):
        slug = req.url.params.get("slug", "")
        if "gamma-api" in str(req.url) and "/markets" in str(req.url):
            w = int(slug.rsplit("-", 1)[-1])
            if w in up_wins:
                prices = '["1","0"]' if up_wins[w] else '["0","1"]'
                return httpx.Response(200, json=[{"slug": slug, "outcomes": '["Up","Down"]',
                                                  "clobTokenIds": '["a","b"]', "outcomePrices": prices, "closed": True}])
            return httpx.Response(200, json=[])
        return httpx.Response(404)
    return handler


def make_supervisor(tmp_path, handler) -> tuple[Supervisor, Engine]:
    settings = Settings(db_path=str(tmp_path / "s.sqlite"))
    engine = Engine(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    db = sqlite3.connect(settings.db_path)
    db.executescript(SCHEMA)
    return Supervisor(settings, engine, db), engine


async def test_data_check_matches_and_blocks_on_serious_mismatch(tmp_path):
    windows = [W + 900 * i for i in range(4)]
    # Polymarket says UP for all four windows
    sup, engine = make_supervisor(tmp_path, resolved_market_handler({w: True for w in windows}))
    engine.finals = {
        windows[0]: (100.0, 150.0, 1.0),  # ours UP: match
        windows[1]: (100.0, 95.0, 1.0),   # ours DOWN by $5: near-line miss
        windows[2]: (100.0, 40.0, 1.0),   # ours DOWN by $60: serious
        windows[3]: (100.0, 30.0, 1.0),   # serious
    }
    await sup.check_finished_windows()
    dc = sup.data_check
    assert (dc.checked, dc.matched, dc.near_line_misses, dc.serious_misses) == (4, 1, 1, 2)
    assert "data" in engine.blockers
    # outcomes are cached for the review
    assert dict(sup.db.execute("SELECT window_start, outcome FROM outcomes")) == {w: 1 for w in windows}
    await engine.client.aclose()


async def test_data_check_waits_for_final_seconds_and_resolution(tmp_path):
    sup, engine = make_supervisor(tmp_path, resolved_market_handler({}))
    engine.finals = {W: (100.0, 101.0, 1.0), W + 900: (100.0, 101.0, 120.0)}
    await sup.check_finished_windows()
    assert sup.data_check.checked == 0 and "data" not in engine.blockers
    await engine.client.aclose()


async def test_update_readiness_sets_validated_flag(tmp_path):
    sup, engine = make_supervisor(tmp_path, resolved_market_handler({}))
    sup.db.execute("INSERT INTO outcomes VALUES (?,?,?)", (W, 1, "polymarket"))
    sup.db.execute(
        "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (W + 300, W, 1, 1, 600, 0.5, 0.7, 0.6, 0.42, 0.58, 0.40, "BET UP", 2.0, 0, "chainlink", "chainlink"),
    )
    await sup.update_readiness()
    assert sup.readiness.windows == 1 and not engine.validated
    assert sup.readiness.paper.n == 1
    await engine.client.aclose()


async def test_engine_blockers_force_no_bet():
    engine = Engine(Settings(taker_fee_rate=0.0), httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    engine.vol.seed((0.5 / 5615.69) ** 2)
    engine.open_prices[W] = 60_000.0
    engine.on_price(W + 300, 60_060.0)
    engine.blockers["calibration"] = "Model miscalibrated"
    s = engine.snapshot(now=W + 300)
    assert s.recommendation.action == "NO BET" and "Model miscalibrated" in s.recommendation.reasons
    await engine.client.aclose()


async def test_health_checks_report_fixes():
    def handler(req):
        if "ticker" in str(req.url):
            return httpx.Response(200, json={"price": "61000"})
        if "gamma-api" in str(req.url):
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ok = await check_exchange_rest(Settings(), client)
        pm = await check_polymarket(client)
    assert ok.ok and "$61,000.00" in ok.detail
    assert not pm[0].ok and "renamed" in pm[0].fix


async def test_panels_render_with_supervisor_state(tmp_path):
    sup, engine = make_supervisor(tmp_path, resolved_market_handler({}))
    sup.checks = [Check("Coinbase price", True, "$1"), Check("Polymarket", False, "down", "try later")]
    sup.calibration = calibration.CalibrationResult(time.time(), 7, "coinbase", 600, 1.1, 0.12, 0.11, 0.02)
    sup.calibration_note = sup.calibration.summary()
    sup.readiness = review.Readiness(42, 0.10, 0.11, review.TradeSummary(3, 2, 6.0, 1.5, 1.0))
    console = Console(width=150, height=50, record=True, file=open("/dev/null", "w"))
    console.print(render(engine.snapshot(), engine.s, "", None, sup))
    text = console.export_text()
    for expected in ("Health", "Polymarket: down", "Calibration: 7d", "Paper trading", "42/300 windows"):
        assert expected in text, expected
    await engine.client.aclose()
