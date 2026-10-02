import sqlite3

import httpx

from btcpredict import review
from btcpredict.config import Settings
from btcpredict.engine import Engine

W = 1_759_420_800  # a window boundary


def engine():
    e = Engine(Settings(), httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    e.vol.seed((0.5 / 5615.69) ** 2)  # ~50% annualized per-second variance
    return e


def test_chainlink_open_captured_at_boundary_and_preferred():
    e = engine()
    e.open_prices[W] = 60_000.0  # exchange candle open
    e.on_price(W - 1, 59_990.0)
    e.on_chainlink(W - 1, 60_010.0)  # last tick of previous window
    e.on_chainlink(W + 1, 60_020.0)  # first tick of new window -> price to beat
    e.on_price(W + 2, 60_015.0)
    e.on_chainlink(W + 2, 60_030.0)
    s = e.snapshot(now=W + 3)
    assert s.price_source == "chainlink" and s.price == 60_030.0
    assert s.open_source == "chainlink" and s.open_price == 60_020.0


def test_basis_adjusted_open_when_started_mid_window():
    e = engine()
    e.open_prices[W] = 60_000.0
    for i in range(10):
        e.on_price(W + 100 + i, 60_100.0)
        e.on_chainlink(W + 100 + i, 60_108.0)  # Chainlink runs $8 above the exchange
    s = e.snapshot(now=W + 111)
    assert s.open_source == "exchange+basis" and abs(s.open_price - 60_008.0) < 1e-6
    assert s.price == 60_108.0


def test_falls_back_to_exchange_when_chainlink_stale():
    e = engine()
    e.cl_streaming = True
    e.open_prices[W] = 60_000.0
    e.on_price(W + 100, 60_050.0)
    e.on_chainlink(W + 100, 60_055.0)
    e.on_price(W + 200, 60_060.0)
    s = e.snapshot(now=W + 200)
    assert s.price_source == "exchange" and s.price == 60_060.0 and s.open_price == 60_000.0
    assert any("Chainlink feed unavailable" in m for m in s.status)


async def test_resolve_outcomes_prefers_polymarket_then_candles(tmp_path):
    def handler(req: httpx.Request):
        slug = req.url.params.get("slug", "")
        if "gamma-api" in str(req.url) and slug.endswith(str(W)):
            return httpx.Response(200, json=[{"slug": slug, "outcomes": '["Up","Down"]', "clobTokenIds": '["a","b"]',
                                              "outcomePrices": '["0","1"]', "closed": True}])
        if "gamma-api" in str(req.url):
            return httpx.Response(200, json=[])
        if "candles" in str(req.url):  # window W+900: close above open -> UP
            rows = [[W + 900 + 60 * i, 1, 2, 100 + i, 101 + i, 1] for i in range(15)]
            return httpx.Response(200, json=rows[::-1])
        return httpx.Response(404)

    db = sqlite3.connect(tmp_path / "r.sqlite")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await review.resolve_outcomes(db, client, [W, W + 900], "coinbase")
    assert out == {W: 0, W + 900: 1}
    sources = dict(db.execute("SELECT window_start, source FROM outcomes"))
    assert sources == {W: "polymarket", W + 900: "coinbase"}
