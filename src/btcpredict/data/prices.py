"""BTC spot price: historical 1-minute candles (REST) and live trades (WebSocket).

Two free, keyless sources:
  * coinbase - Coinbase Exchange BTC-USD (works from the US)
  * binance  - Binance BTCUSDT (blocked from US IP addresses)

Note: Polymarket settles on the Chainlink BTC/USD stream, which aggregates many venues.
Exchange prices usually sit within a few dollars of it, but can differ near the line.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import websockets

log = logging.getLogger(__name__)

PriceCallback = Callable[[float, float], Awaitable[None] | None]  # (unix_ts, price)


@dataclass(frozen=True)
class Candle:
    ts: int  # open time, unix seconds
    open: float
    high: float
    low: float
    close: float


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def fetch_candles_1m(
    client: httpx.AsyncClient, source: str, start: int, end: int
) -> list[Candle]:
    """1-minute candles with open time in [start, end), oldest first. Paginates as needed."""
    out: dict[int, Candle] = {}
    if source == "binance":
        cursor = start
        while cursor < end:
            r = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "startTime": cursor * 1000,
                    "endTime": end * 1000 - 1,
                    "limit": 1000,
                },
            )
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            for k in rows:
                ts = int(k[0]) // 1000
                out[ts] = Candle(ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]))
            cursor = int(rows[-1][0]) // 1000 + 60
    else:
        page = 300 * 60  # Coinbase returns at most 300 candles per call
        cursor = start
        while cursor < end:
            stop = min(end, cursor + page)
            r = await client.get(
                "https://api.exchange.coinbase.com/products/BTC-USD/candles",
                params={"granularity": 60, "start": _iso(cursor), "end": _iso(stop)},
                headers={"User-Agent": "btcpredict/0.1"},
            )
            r.raise_for_status()
            for row in r.json():  # [time, low, high, open, close, volume], newest first
                ts = int(row[0])
                if start <= ts < end:
                    out[ts] = Candle(ts, float(row[3]), float(row[2]), float(row[1]), float(row[4]))
            cursor = stop
            if cursor < end:
                await asyncio.sleep(0.35)  # stay under public rate limits
    return [out[k] for k in sorted(out)]


async def stream_trades(source: str, on_price: PriceCallback, stop: asyncio.Event | None = None) -> None:
    """Call on_price(ts, price) for every trade/ticker update. Reconnects forever."""
    backoff = 1.0
    while stop is None or not stop.is_set():
        try:
            if source == "binance":
                await _binance_ws(on_price)
            else:
                await _coinbase_ws(on_price)
            backoff = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # network drops are routine; keep going
            log.warning("price stream error (%s): %s; reconnecting in %.0fs", source, exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def _emit(cb: PriceCallback, ts: float, px: float) -> None:
    res = cb(ts, px)
    if asyncio.iscoroutine(res):
        await res


async def _coinbase_ws(on_price: PriceCallback) -> None:
    async with websockets.connect("wss://ws-feed.exchange.coinbase.com", ping_interval=20) as ws:
        await ws.send(
            json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]})
        )
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") != "ticker" or "price" not in msg:
                continue
            ts = (
                datetime.fromisoformat(msg["time"].replace("Z", "+00:00")).timestamp()
                if msg.get("time")
                else datetime.now(timezone.utc).timestamp()
            )
            await _emit(on_price, ts, float(msg["price"]))


async def _binance_ws(on_price: PriceCallback) -> None:
    url = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
    async with websockets.connect(url, ping_interval=20) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            if "p" in msg:
                await _emit(on_price, int(msg["T"]) / 1000.0, float(msg["p"]))


async def fetch_spot(client: httpx.AsyncClient, source: str) -> float:
    """Latest traded price via REST (used by one-shot commands)."""
    if source == "binance":
        r = await client.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
    else:
        r = await client.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
            headers={"User-Agent": "btcpredict/0.1"},
        )
    r.raise_for_status()
    return float(r.json()["price"])
