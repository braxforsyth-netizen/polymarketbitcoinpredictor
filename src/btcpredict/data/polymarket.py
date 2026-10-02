"""Polymarket read-only market data (no account or key needed).

  * Gamma API finds the market for a window by slug: btc-updown-15m-<window start unix ts>
  * CLOB API gives the live order book for the UP and DOWN outcome tokens
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


@dataclass(frozen=True)
class MarketInfo:
    slug: str
    question: str
    up_token: str
    down_token: str
    up_mid: float | None  # Gamma's last outcome prices, used if the CLOB is unreachable
    down_mid: float | None
    closed: bool


@dataclass(frozen=True)
class Book:
    best_bid: float | None
    best_ask: float | None
    ask_size: float  # shares available at best ask


@dataclass(frozen=True)
class MarketQuote:
    market: MarketInfo
    up: Book
    down: Book

    @property
    def ask_up(self) -> float | None:
        return self.up.best_ask if self.up.best_ask is not None else self.market.up_mid

    @property
    def ask_down(self) -> float | None:
        return self.down.best_ask if self.down.best_ask is not None else self.market.down_mid


def _as_list(value) -> list:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value or [])


def parse_market(m: dict) -> MarketInfo | None:
    outcomes = [str(o).lower() for o in _as_list(m.get("outcomes"))]
    tokens = [str(t) for t in _as_list(m.get("clobTokenIds"))]
    prices = _as_list(m.get("outcomePrices"))
    if len(outcomes) != 2 or len(tokens) != 2 or "up" not in outcomes or "down" not in outcomes:
        return None
    iu, idn = outcomes.index("up"), outcomes.index("down")
    px = [float(p) for p in prices] if len(prices) == 2 else [None, None]
    return MarketInfo(
        slug=m.get("slug", ""),
        question=m.get("question", ""),
        up_token=tokens[iu],
        down_token=tokens[idn],
        up_mid=px[iu],
        down_mid=px[idn],
        closed=bool(m.get("closed", False)),
    )


def parse_book(data: dict) -> Book:
    bids = [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
    asks = [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]
    best_bid = max((p for p, _ in bids), default=None)
    best_ask = min((p for p, _ in asks), default=None)
    ask_size = sum(s for p, s in asks if p == best_ask) if best_ask is not None else 0.0
    return Book(best_bid, best_ask, ask_size)


async def find_market(client: httpx.AsyncClient, slug: str) -> MarketInfo | None:
    r = await client.get(f"{GAMMA}/markets", params={"slug": slug})
    r.raise_for_status()
    for m in r.json() or []:
        info = parse_market(m)
        if info:
            return info
    # Fallback: look the slug up as an event and take its first Up/Down market.
    r = await client.get(f"{GAMMA}/events", params={"slug": slug})
    r.raise_for_status()
    for ev in r.json() or []:
        for m in ev.get("markets", []):
            info = parse_market(m)
            if info:
                return info
    return None


async def fetch_book(client: httpx.AsyncClient, token_id: str) -> Book:
    try:
        r = await client.get(f"{CLOB}/book", params={"token_id": token_id})
        r.raise_for_status()
        return parse_book(r.json())
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("CLOB book fetch failed: %s", exc)
        return Book(None, None, 0.0)


async def fetch_quote(client: httpx.AsyncClient, market: MarketInfo) -> MarketQuote:
    up, down = await asyncio.gather(
        fetch_book(client, market.up_token), fetch_book(client, market.down_token)
    )
    return MarketQuote(market, up, down)
