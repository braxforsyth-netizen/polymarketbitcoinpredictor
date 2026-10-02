"""Paper-trading review of what the dashboard recorded in data/snapshots.sqlite.

Answers the two questions that matter before risking money:
  1. Would the signals the dashboard showed have made money?
  2. Are the model's probabilities more accurate than Polymarket's own prices?
     If not, any "edge" is an illusion.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import httpx

from .data import polymarket, prices
from .model.edge import fee_per_share
from .model.windows import WINDOW_SECONDS, Window

TIME_BUCKETS = ((600, 901, "10–15m left"), (300, 600, "5–10m left"), (120, 300, "2–5m left"), (0, 120, "<2m left"))


@dataclass(frozen=True)
class Trade:
    window_start: int
    side: str
    cost: float  # ask + fee per share
    stake: float
    won: bool

    @property
    def pnl(self) -> float:
        return self.stake * (1.0 / self.cost - 1.0) if self.won else -self.stake


@dataclass(frozen=True)
class TradeSummary:
    n: int
    wins: int
    staked: float
    pnl: float
    max_drawdown: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def roi(self) -> float:
        return self.pnl / self.staked if self.staked else 0.0


def summarize(trades: Sequence[Trade]) -> TradeSummary:
    equity = peak = dd = 0.0
    for t in sorted(trades, key=lambda t: t.window_start):
        equity += t.pnl
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return TradeSummary(
        len(trades), sum(t.won for t in trades), sum(t.stake for t in trades), sum(t.pnl for t in trades), dd
    )


def _first_per_window(rows: Iterable[dict], pick) -> dict[int, tuple[dict, str]]:
    chosen: dict[int, tuple[dict, str]] = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        w = r["window_start"]
        if w in chosen:
            continue
        side = pick(r)
        if side:
            chosen[w] = (r, side)
    return chosen


def _cost(r: dict, side: str, fee_rate: float) -> float | None:
    ask = r["ask_up"] if side == "UP" else r["ask_down"]
    return None if ask is None else ask + fee_per_share(ask, fee_rate)


def recorded_trades(rows: Sequence[dict], outcomes: dict[int, int], fee_rate: float) -> list[Trade]:
    """The first BET signal the dashboard showed in each window, at the stake it suggested."""
    def pick(r):
        a = r["action"] or ""
        return a.removeprefix("BET ") if a.startswith("BET ") else None

    trades = []
    for w, (r, side) in _first_per_window(rows, pick).items():
        cost = _cost(r, side, fee_rate)
        if w in outcomes and cost and 0 < cost < 1 and r["stake"]:
            trades.append(Trade(w, side, cost, r["stake"], outcomes[w] == (side == "UP")))
    return trades


def threshold_sweep(
    rows: Sequence[dict],
    outcomes: dict[int, int],
    fee_rate: float,
    thresholds: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15),
    min_seconds_left: float = 45.0,
) -> list[tuple[float, TradeSummary]]:
    """$1 flat bets on the first moment each window's edge reached a threshold. Use it to tune MIN_EDGE."""
    out = []
    for th in thresholds:
        def pick(r, th=th):
            if r["p_up"] is None or r["seconds_left"] < min_seconds_left:
                return None
            best = None
            for side, p in (("UP", r["p_up"]), ("DOWN", 1 - r["p_up"])):
                c = _cost(r, side, fee_rate)
                if c and 0 < c < 1 and (p - c) / c >= th and (best is None or (p - c) / c > best[1]):
                    best = (side, (p - c) / c)
            return best[0] if best else None

        trades = [
            Trade(w, side, _cost(r, side, fee_rate), 1.0, outcomes[w] == (side == "UP"))
            for w, (r, side) in _first_per_window(rows, pick).items()
            if w in outcomes
        ]
        out.append((th, summarize(trades)))
    return out


@dataclass(frozen=True)
class BrierRow:
    label: str
    n: int
    model: float
    market: float


def brier_vs_market(rows: Sequence[dict], outcomes: dict[int, int]) -> list[BrierRow]:
    """Model P(UP) vs Polymarket's UP mid-price as forecasts of the outcome (lower is better)."""
    usable = [
        r for r in rows
        if r["window_start"] in outcomes and r["p_up"] is not None
        and r["bid_up"] is not None and r["ask_up"] is not None
    ]
    result = []
    for lo, hi, label in ((0, 901, "all"),) + TIME_BUCKETS:
        sub = [r for r in usable if lo <= r["seconds_left"] < hi]
        if not sub:
            continue
        y = [outcomes[r["window_start"]] for r in sub]
        model = sum((r["p_up"] - o) ** 2 for r, o in zip(sub, y)) / len(sub)
        market = sum(((r["bid_up"] + r["ask_up"]) / 2 - o) ** 2 for r, o in zip(sub, y)) / len(sub)
        result.append(BrierRow(label, len(sub), model, market))
    return result


# ---------- data access ----------

def load_rows(db: sqlite3.Connection) -> list[dict]:
    db.row_factory = sqlite3.Row
    return [dict(r) for r in db.execute("SELECT * FROM snapshots ORDER BY ts")]


async def resolve_outcomes(
    db: sqlite3.Connection, client: httpx.AsyncClient, windows: Iterable[int], price_source: str
) -> dict[int, int]:
    """Outcome per finished window: cached -> Polymarket's resolution -> exchange candles."""
    db.execute("CREATE TABLE IF NOT EXISTS outcomes (window_start INTEGER PRIMARY KEY, outcome INTEGER, source TEXT)")
    known = {w: o for w, o in db.execute("SELECT window_start, outcome FROM outcomes")}
    now = time.time()
    for w in sorted(set(windows)):
        if w in known or w + WINDOW_SECONDS + 120 > now:
            continue
        outcome, source = None, None
        try:
            m = await polymarket.find_market(client, Window(w, w + WINDOW_SECONDS).slug)
            if m and m.closed and m.up_mid in (0.0, 1.0):
                outcome, source = int(m.up_mid == 1.0), "polymarket"
        except httpx.HTTPError:
            pass
        if outcome is None:
            try:
                candles = await prices.fetch_candles_1m(client, price_source, w, w + WINDOW_SECONDS)
                if len(candles) == 15:
                    outcome, source = int(candles[-1].close >= candles[0].open), price_source
            except httpx.HTTPError:
                pass
        if outcome is not None:
            known[w] = outcome
            db.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?)", (w, outcome, source))
            db.commit()
    return known
