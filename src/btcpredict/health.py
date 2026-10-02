"""Startup checks: can we reach every data source? Each failed check comes with a fix."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

import httpx

from .config import Settings
from .data import chainlink, polymarket, prices
from .data.news import RSS_FEEDS, _fetch_rss
from .model.windows import window_at


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    required: bool = True  # False: the dashboard still works without it


async def _first_tick(stream_coro_factory, timeout: float) -> float | None:
    got: asyncio.Future = asyncio.get_running_loop().create_future()

    def cb(ts: float, px: float) -> None:
        if not got.done():
            got.set_result(px)

    task = asyncio.create_task(stream_coro_factory(cb))
    try:
        return await asyncio.wait_for(asyncio.shield(got), timeout)
    except asyncio.TimeoutError:
        return None
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def check_exchange_rest(s: Settings, client: httpx.AsyncClient) -> Check:
    name = f"{s.price_source.title()} price"
    try:
        px = await prices.fetch_spot(client, s.price_source)
        return Check(name, True, f"${px:,.2f}")
    except Exception as exc:
        fix = (
            "Binance blocks US IPs: set PRICE_SOURCE=coinbase in .env"
            if s.price_source == "binance"
            else "Check your internet connection or try PRICE_SOURCE=binance outside the US"
        )
        return Check(name, False, str(exc)[:80], fix)


async def check_exchange_ws(s: Settings, timeout: float = 10) -> Check:
    name = f"{s.price_source.title()} live"
    px = await _first_tick(lambda cb: prices.stream_trades(s.price_source, cb), timeout)
    if px:
        return Check(name, True, "streaming")
    return Check(name, False, f"no trades within {timeout:.0f}s",
                 "A firewall/VPN may block WebSockets; the dashboard will use stale prices")


async def check_chainlink(s: Settings, timeout: float = 15) -> Check:
    if not s.settlement_feed:
        return Check("Chainlink", True, "disabled (SETTLEMENT_FEED=off)", required=False)
    px = await _first_tick(chainlink.stream_chainlink, timeout)
    if px:
        return Check("Chainlink", True, f"${px:,.2f}", required=False)
    return Check("Chainlink", False, f"no data within {timeout:.0f}s",
                 "Falls back to exchange price automatically; set SETTLEMENT_FEED=off to silence",
                 required=False)


async def check_polymarket(client: httpx.AsyncClient) -> list[Check]:
    w = window_at(time.time())
    try:
        market = await polymarket.find_market(client, w.slug)
    except Exception as exc:
        return [Check("Polymarket", False, str(exc)[:80], "Polymarket may be blocked in your region or down")]
    if market is None:
        return [Check("Polymarket", False, f"no market for {w.slug}",
                      "Polymarket may have renamed these markets; report the URL of a live one")]
    book = await polymarket.fetch_book(client, market.up_token)
    clob = (
        Check("Order book", True, f"UP ask {book.best_ask:.3f}")
        if book.best_ask is not None
        else Check("Order book", False, "no asks returned", "Odds fall back to Polymarket's last prices")
    )
    return [Check("Polymarket", True, "market found"), clob]


async def check_news(client: httpx.AsyncClient) -> Check:
    results = await asyncio.gather(*(_fetch_rss(client, n, u) for n, u in RSS_FEEDS.items()))
    working = sum(1 for r in results if r)
    return Check("News", working > 0, f"{working}/{len(RSS_FEEDS)} feeds",
                 "" if working else "All RSS feeds failed; check your connection", required=False)


async def check_claude(s: Settings) -> Check:
    if not s.ai_enabled:
        return Check("Claude", True, "off (free mode)", required=False)
    import anthropic

    try:
        await anthropic.AsyncAnthropic(api_key=s.anthropic_api_key).models.retrieve(s.claude_model)
        return Check("Claude", True, s.claude_model, required=False)
    except anthropic.AuthenticationError:
        return Check("Claude", False, "invalid API key", "Fix ANTHROPIC_API_KEY in .env", required=False)
    except anthropic.NotFoundError:
        return Check("Claude", False, f"unknown model {s.claude_model}", "Fix CLAUDE_MODEL in .env", required=False)
    except anthropic.APIError as exc:
        return Check("Claude", False, str(exc)[:80], "AI briefings will retry later", required=False)


async def run_checks(s: Settings, client: httpx.AsyncClient) -> list[Check]:
    groups = await asyncio.gather(
        check_exchange_rest(s, client),
        check_exchange_ws(s),
        check_chainlink(s),
        check_polymarket(client),
        check_news(client),
        check_claude(s),
    )
    out: list[Check] = []
    for g in groups:
        out.extend(g if isinstance(g, list) else [g])
    return out
