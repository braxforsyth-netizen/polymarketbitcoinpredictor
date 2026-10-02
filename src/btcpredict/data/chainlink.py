"""Chainlink BTC/USD, the price Polymarket's 15-minute markets settle on.

Streamed for free (no key) from Polymarket's real-time data service (RTDS),
topic `crypto_prices_chainlink`. Exchange prices from Coinbase or Binance are usually
within a few dollars of it, but near the line those few dollars decide the market.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Iterator

import websockets

from .prices import PriceCallback, _emit

log = logging.getLogger(__name__)

RTDS_URL = "wss://ws-live-data.polymarket.com"
SUBSCRIBE = {
    "action": "subscribe",
    "subscriptions": [
        {"topic": "crypto_prices_chainlink", "type": "*", "filters": json.dumps({"symbol": "btc/usd"})}
    ],
}


def parse_rtds(raw: str | bytes) -> Iterator[tuple[float, float]]:
    """Yield (unix_ts, price) from one RTDS message; ignores anything that isn't BTC/USD."""
    try:
        msg = json.loads(raw)
    except (TypeError, ValueError):
        return  # "PONG" and other keep-alive frames
    if not isinstance(msg, dict) or msg.get("topic") != "crypto_prices_chainlink":
        return
    payload = msg.get("payload") or {}
    if str(payload.get("symbol", "btc/usd")).lower() != "btc/usd":
        return
    # Updates carry {timestamp, value}; the initial snapshot may carry a list under "data".
    points = payload.get("data") if isinstance(payload.get("data"), list) else [payload]
    for p in points:
        try:
            ts, value = float(p["timestamp"]), float(p["value"])
        except (KeyError, TypeError, ValueError):
            continue
        yield (ts / 1000.0 if ts > 1e11 else ts), value


async def _pinger(ws, every_s: float = 5.0) -> None:
    while True:
        await asyncio.sleep(every_s)
        await ws.send("PING")


async def stream_chainlink(on_price: PriceCallback) -> None:
    """Call on_price(ts, price) for each Chainlink BTC/USD update. Reconnects forever."""
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(RTDS_URL, ping_interval=None) as ws:
                await ws.send(json.dumps(SUBSCRIBE))
                pinger = asyncio.create_task(_pinger(ws))
                try:
                    async for raw in ws:
                        for ts, px in parse_rtds(raw):
                            await _emit(on_price, ts, px)
                            backoff = 1.0
                finally:
                    pinger.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pinger
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("chainlink stream error: %s; reconnecting in %.0fs", exc, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60.0)
