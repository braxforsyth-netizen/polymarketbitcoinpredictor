"""Live state: price feed + current window + Polymarket quote + news -> Snapshot."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from .config import Settings
from .data import chainlink, polymarket, prices
from .data.news import NewsItem, fetch_news, news_shock
from .model.edge import Recommendation, recommend
from .model.probability import Projection, project
from .model.volatility import EwmaVolatility, annualized, per_second_variance_from_closes
from .model.windows import Window, window_at

log = logging.getLogger(__name__)

NEWS_SHOCK_VOL_MULTIPLIER = 1.5
CHAINLINK_FRESH_S = 10.0  # use the settlement feed only while it is this fresh


@dataclass
class Snapshot:
    ts: float
    window: Window
    price: float | None
    open_price: float | None
    seconds_left: float
    sigma_annual: float | None
    projection: Projection | None
    quote: polymarket.MarketQuote | None
    recommendation: Recommendation | None
    news: list[NewsItem]
    shock: bool
    status: list[str] = field(default_factory=list)
    price_source: str = "exchange"
    open_source: str = "exchange"
    validated: bool = False  # True once paper trading shows the model beats the market

    def to_dict(self) -> dict:
        """Plain-data view used by the AI agent tools and the recorder."""
        p, q, r = self.projection, self.quote, self.recommendation
        return {
            "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.ts)),
            "window": self.window.label(),
            "market_slug": self.window.slug,
            "btc_price": self.price,
            "btc_price_source": self.price_source,
            "window_open_price": self.open_price,
            "window_open_source": self.open_source,
            "change_usd": (self.price - self.open_price) if self.price and self.open_price else None,
            "seconds_left": round(self.seconds_left),
            "volatility_annualized": round(self.sigma_annual, 3) if self.sigma_annual else None,
            "model_p_up": round(p.p_up, 4) if p else None,
            "model_p_up_band": [round(p.p_low, 4), round(p.p_high, 4)] if p else None,
            "expected_1sigma_move_usd": round(p.expected_move_usd, 2) if p else None,
            "polymarket_up_ask": q.ask_up if q else None,
            "polymarket_up_bid": q.up.best_bid if q else None,
            "polymarket_down_ask": q.ask_down if q else None,
            "polymarket_down_bid": q.down.best_bid if q else None,
            "recommendation": r.action if r else "NO BET",
            "recommendation_reasons": r.reasons if r else ["Waiting for data"],
            "suggested_stake_usd": r.stake if r else 0.0,
            "edge_per_dollar": {
                s.side: (round(s.ev_per_dollar, 4) if s.ev_per_dollar is not None else None)
                for s in (r.quotes if r else ())
            },
            "high_impact_news_last_10m": self.shock,
            "signal_validated_by_paper_trading": self.validated,
            "status": self.status,
        }


class Engine:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.s = settings
        self.client = client or httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self.vol = EwmaVolatility(half_life_s=900.0)
        self.price: float | None = None
        self.price_ts: float = 0.0
        self.window = window_at(time.time())
        self.open_prices: dict[int, float] = {}
        self.confirmed_opens: set[int] = set()  # opens taken from the exchange's 1m candle
        self.market: polymarket.MarketInfo | None = None
        self.quote: polymarket.MarketQuote | None = None
        self.quote_ts: float = 0.0
        self.news: list[NewsItem] = []
        # Chainlink settlement feed
        self.cl_price: float | None = None
        self.cl_ts: float = 0.0
        self.cl_opens: dict[int, float] = {}
        self.basis: float | None = None  # EWMA of (Chainlink - exchange)
        self.cl_streaming = False
        # Set by the calibration and validation loops
        self.vol_multiplier = 1.0
        self.blockers: dict[str, str] = {}  # reasons that force NO BET
        self.validated = False
        self.finals: dict[int, tuple[float, float, float]] = {}  # window -> (open, last price, seconds left)
        self.status: dict[str, str] = {}

    # ---------- setup ----------
    async def seed_volatility(self) -> None:
        now = int(time.time())
        candles = await prices.fetch_candles_1m(self.client, self.s.price_source, now - 3 * 3600, now)
        var = per_second_variance_from_closes([c.close for c in candles])
        if var:
            self.vol.seed(var)
        if candles and self.price is None:
            self.price, self.price_ts = candles[-1].close, float(candles[-1].ts + 60)
        for c in candles:
            if c.ts % 900 == 0:
                self.open_prices[c.ts] = c.open
                self.confirmed_opens.add(c.ts)

    # ---------- inputs ----------
    def on_price(self, ts: float, price: float) -> None:
        w = window_at(ts)
        if w.start not in self.open_prices and self.price is not None and ts - w.start < 2.0:
            # We saw the boundary live; use the first trade as a provisional open.
            self.open_prices[w.start] = price
        self.price, self.price_ts = price, ts
        self.vol.update(ts, price)

    def on_chainlink(self, ts: float, price: float) -> None:
        w = window_at(ts)
        if self.cl_ts and window_at(self.cl_ts).start < w.start and ts - w.start <= 5.0:
            # First settlement tick of a new window: this is (within a tick) the price to beat.
            self.cl_opens[w.start] = price
        if self.price is not None and abs(ts - self.price_ts) < 3.0:
            diff = price - self.price
            self.basis = diff if self.basis is None else 0.9 * self.basis + 0.1 * diff
        self.cl_price, self.cl_ts = price, ts

    async def refresh_window(self) -> None:
        w = window_at(time.time())
        if w != self.window:
            self.window, self.market, self.quote = w, None, None
        if w.start not in self.confirmed_opens:
            try:
                candles = await prices.fetch_candles_1m(self.client, self.s.price_source, w.start, w.start + 60)
                if candles and candles[0].ts == w.start:
                    self.open_prices[w.start] = candles[0].open
                    self.confirmed_opens.add(w.start)
                self.status.pop("open", None)
            except httpx.HTTPError as exc:
                self.status["open"] = f"open price fetch failed: {exc}"

    async def refresh_market(self) -> None:
        try:
            if self.market is None or self.market.slug != self.window.slug:
                self.market = await polymarket.find_market(self.client, self.window.slug)
                if self.market is None:
                    self.status["polymarket"] = f"market {self.window.slug} not found"
                    return
            self.quote = await polymarket.fetch_quote(self.client, self.market)
            self.quote_ts = time.time()
            self.status.pop("polymarket", None)
        except httpx.HTTPError as exc:
            self.status["polymarket"] = f"Polymarket error: {exc}"

    async def refresh_news(self) -> None:
        try:
            self.news = await fetch_news(self.client, self.s.cryptopanic_token)
            self.status.pop("news", None)
        except Exception as exc:  # never let news take down the dashboard
            self.status["news"] = f"news error: {exc}"

    async def refresh_spot(self) -> None:
        price = await prices.fetch_spot(self.client, self.s.price_source)
        self.on_price(time.time(), price)

    # ---------- loops for the dashboard ----------
    async def run_price_stream(self) -> None:
        await prices.stream_trades(self.s.price_source, self.on_price)

    async def run_chainlink_stream(self) -> None:
        self.cl_streaming = True
        await chainlink.stream_chainlink(self.on_chainlink)

    async def run_market_loop(self, every_s: float = 2.0) -> None:
        while True:
            await self.refresh_window()
            await self.refresh_market()
            await asyncio.sleep(every_s)

    async def run_news_loop(self, every_s: float = 60.0) -> None:
        while True:
            await self.refresh_news()
            await asyncio.sleep(every_s)

    async def refresh_all(self) -> None:
        """One-shot refresh over REST, for commands that don't run the live stream."""
        await self.seed_volatility()
        await asyncio.gather(self.refresh_spot(), self.refresh_window(), self.refresh_news())
        await self.refresh_market()

    # ---------- output ----------
    def _price_and_open(self, now: float, w: Window) -> tuple[float | None, str, float | None, str]:
        """Pick a consistent (price, open) pair, preferring the Chainlink settlement feed."""
        exch_open = self.open_prices.get(w.start)
        if self.cl_price is not None and now - self.cl_ts < CHAINLINK_FRESH_S:
            if w.start in self.cl_opens:
                return self.cl_price, "chainlink", self.cl_opens[w.start], "chainlink"
            if exch_open is not None and self.basis is not None:
                return self.cl_price, "chainlink", exch_open + self.basis, "exchange+basis"
        return self.price, "exchange", exch_open, "exchange"

    def snapshot(self, now: float | None = None) -> Snapshot:
        now = now or time.time()
        w = window_at(now)
        price, price_src, open_px, open_src = self._price_and_open(now, w)
        left = w.seconds_left(now)
        shock = news_shock(self.news, now=now)
        sigma = self.vol.sigma_per_s
        status = list(self.status.values())
        if self.price is not None and now - self.price_ts > 15:
            status.append(f"price is {now - self.price_ts:.0f}s stale")
        if self.cl_streaming and price_src != "chainlink":
            if self.cl_price is None or now - self.cl_ts >= CHAINLINK_FRESH_S:
                status.append("Chainlink feed unavailable: using exchange price")
            else:
                status.append("Chainlink connected, calibrating vs exchange: using exchange price")

        proj = rec = None
        if price and open_px and sigma:
            proj = project(
                price, open_px, sigma, left,
                vol_multiplier=self.vol_multiplier * (NEWS_SHOCK_VOL_MULTIPLIER if shock else 1.0),
            )
            self.finals[w.start] = (open_px, price, left)
            if len(self.finals) > 16:
                del self.finals[min(self.finals)]
            quote = self.quote if self.quote and self.quote.market.slug == w.slug else None
            stale_quote = quote is not None and now - self.quote_ts > 10
            rec = recommend(
                proj.p_up, (proj.p_low, proj.p_high),
                None if quote is None or stale_quote else quote.ask_up,
                None if quote is None or stale_quote else quote.ask_down,
                left,
                min_edge=self.s.min_edge,
                fee_rate=self.s.taker_fee_rate,
                kelly_fraction=self.s.kelly_fraction,
                bankroll=self.s.bankroll,
                max_stake_fraction=self.s.max_stake_fraction,
                min_seconds_left=self.s.min_seconds_left,
                news_shock=shock,
                blockers=list(self.blockers.values()),
            )
        elif open_px is None:
            status.append("waiting for window open price")
        if open_src == "exchange" and open_px is not None and w.start not in self.confirmed_opens:
            status.append("open price is provisional (first live trade)")

        return Snapshot(
            ts=now,
            window=w,
            price=price,
            open_price=open_px,
            seconds_left=left,
            sigma_annual=annualized(sigma) if sigma else None,
            projection=proj,
            quote=self.quote if self.quote and self.quote.market.slug == w.slug else None,
            recommendation=rec,
            news=self.news,
            shock=shock,
            status=status,
            price_source=price_src,
            open_source=open_src,
            validated=self.validated,
        )
