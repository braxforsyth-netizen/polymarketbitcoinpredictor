"""Engine + agent + dashboard against mocked HTTP APIs (no network)."""

import json
import time
from types import SimpleNamespace

import httpx
from rich.console import Console

from btcpredict.agent import Analyst, run_tool
from btcpredict.config import Settings
from btcpredict.dashboard import render
from btcpredict.engine import Engine
from btcpredict.model.windows import window_at

NOW = time.time()
W = window_at(NOW)
OPEN = 60_000.0
SPOT = 60_090.0

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Bitcoin ETF inflows surge as BTC nears record</title><link>https://x/1</link>
<pubDate>{}</pubDate></item>
<item><title>Celebrity buys a yacht</title><link>https://x/2</link></item>
</channel></rss>""".format(time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(NOW - 1800)))


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "candles" in url:
        start = int(time.mktime(time.strptime(request.url.params["start"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone)
        end = int(time.mktime(time.strptime(request.url.params["end"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone)
        rows = []
        for ts in range(start // 60 * 60, end, 60):
            px = OPEN + (5 if (ts // 60) % 2 else -5)
            o = OPEN if ts == W.start else px
            rows.append([ts, min(o, px), max(o, px), o, px, 1.0])
        return httpx.Response(200, json=rows[::-1])
    if "ticker" in url:
        return httpx.Response(200, json={"price": str(SPOT)})
    if "gamma-api" in url and "/markets" in url:
        assert request.url.params["slug"] == W.slug
        return httpx.Response(200, json=[{
            "slug": W.slug, "question": "Bitcoin Up or Down?",
            "outcomes": '["Up", "Down"]', "clobTokenIds": '["UP1", "DOWN1"]', "outcomePrices": '["0.6", "0.4"]',
        }])
    if "clob.polymarket.com/book" in url:
        tok = request.url.params["token_id"]
        ask = "0.55" if tok == "UP1" else "0.47"
        return httpx.Response(200, json={"bids": [{"price": "0.40", "size": "50"}], "asks": [{"price": ask, "size": "100"}]})
    if "cointelegraph" in url:
        return httpx.Response(200, text=RSS)
    return httpx.Response(404)


async def make_engine(**overrides):
    settings = Settings(**({"taker_fee_rate": 0.0, "anthropic_api_key": "test"} | overrides))
    engine = Engine(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await engine.refresh_all()
    return engine


async def test_engine_snapshot_end_to_end():
    engine = await make_engine()
    snap = engine.snapshot()
    assert snap.open_price == OPEN
    assert snap.price == SPOT
    assert snap.quote.ask_up == 0.55 and snap.quote.ask_down == 0.47
    assert snap.projection.p_up > 0.5
    assert snap.recommendation.action in {"BET UP", "BET DOWN", "NO BET"}
    assert [n.title for n in snap.news] == ["Bitcoin ETF inflows surge as BTC nears record"]
    d = snap.to_dict()
    json.dumps(d)  # serializable for the agent tools
    assert d["polymarket_up_ask"] == 0.55

    console = Console(width=140, height=45, record=True, file=open("/dev/null", "w"))
    console.print(render(snap, engine.s, "Call: NO BET", time.time()))
    text = console.export_text()
    assert "BTC 15m Up/Down Advisor" in text and "ETF inflows" in text
    await engine.client.aclose()


async def test_tools_return_engine_numbers():
    engine = await make_engine()
    snap = engine.snapshot()
    proj = json.loads(run_tool("get_projection", {}, snap))
    assert proj["model_p_up"] == round(snap.projection.p_up, 4)
    news = json.loads(run_tool("get_news", {"min_impact": "high"}, snap))
    assert news[0]["impact"] == "high"
    await engine.client.aclose()


class FakeMessages:
    """Mimics client.beta.messages: first asks for a tool, then answers."""

    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[SimpleNamespace(type="tool_use", id="t1", name="get_projection", input={})],
            )
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Call: NO BET")])


async def test_analyst_tool_loop():
    engine = await make_engine()
    analyst = Analyst(engine.s, engine.snapshot)
    fake = FakeMessages()
    analyst.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    assert await analyst.briefing() == "Call: NO BET"
    first, second = fake.calls
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and "model_p_up" in tool_result["content"]
    await engine.client.aclose()
