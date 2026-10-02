"""Turn a model probability and Polymarket ask prices into an advisory recommendation.

Buying one share of a side at price `a` pays $1 if that side wins.
  EV per $1 staked = (p - a - fee) / (a + fee)
Kelly fraction for a binary contract bought at effective cost c: f* = (p - c) / (1 - c)
"""

from __future__ import annotations

from dataclasses import dataclass, field


def fee_per_share(price: float, fee_rate: float) -> float:
    """Assumed taker fee model: fee_rate * min(p, 1-p) per share. Verify against Polymarket docs."""
    return fee_rate * min(price, 1.0 - price)


@dataclass(frozen=True)
class SideQuote:
    side: str  # "UP" or "DOWN"
    p_model: float
    ask: float | None
    fee: float = 0.0

    @property
    def cost(self) -> float | None:
        return None if self.ask is None else self.ask + self.fee

    @property
    def ev_per_dollar(self) -> float | None:
        c = self.cost
        if c is None or c <= 0 or c >= 1:
            return None
        return (self.p_model - c) / c

    @property
    def kelly(self) -> float:
        c = self.cost
        if c is None or c >= 1:
            return 0.0
        return max(0.0, (self.p_model - c) / (1.0 - c))


@dataclass(frozen=True)
class Recommendation:
    action: str  # "BET UP", "BET DOWN", "NO BET"
    reasons: list[str] = field(default_factory=list)
    best: SideQuote | None = None
    stake: float = 0.0
    quotes: tuple[SideQuote, ...] = ()


def recommend(
    p_up: float,
    p_up_band: tuple[float, float],
    ask_up: float | None,
    ask_down: float | None,
    seconds_left: float,
    *,
    min_edge: float,
    fee_rate: float,
    kelly_fraction: float,
    bankroll: float,
    max_stake_fraction: float,
    min_seconds_left: float,
    news_shock: bool = False,
    max_disagreement: float = 0.25,
) -> Recommendation:
    up = SideQuote("UP", p_up, ask_up, fee_per_share(ask_up, fee_rate) if ask_up is not None else 0.0)
    down = SideQuote(
        "DOWN", 1.0 - p_up, ask_down, fee_per_share(ask_down, fee_rate) if ask_down is not None else 0.0
    )
    quotes = (up, down)

    reasons: list[str] = []
    if ask_up is None and ask_down is None:
        return Recommendation("NO BET", ["No Polymarket prices available"], None, 0.0, quotes)
    if seconds_left < min_seconds_left:
        reasons.append(f"Under {min_seconds_left:.0f}s left: latency/settlement-source risk")
    if news_shock:
        reasons.append("High-impact news in the last few minutes: model assumptions unreliable")

    candidates = [q for q in quotes if q.ev_per_dollar is not None]
    if not candidates:
        return Recommendation("NO BET", reasons + ["No tradable ask"], None, 0.0, quotes)
    best = max(candidates, key=lambda q: q.ev_per_dollar)

    ev = best.ev_per_dollar
    if ev < min_edge:
        reasons.append(f"Best edge {ev:+.1%} is below the {min_edge:.0%} threshold")

    # Require the edge to survive the pessimistic end of the vol-uncertainty band too.
    lo, hi = p_up_band
    p_pess = lo if best.side == "UP" else 1.0 - hi
    if best.cost is not None and p_pess <= best.cost:
        reasons.append("Edge disappears if volatility is mis-estimated by ±25%")

    # A huge gap to the market almost always means bad data (wrong open price, stale or
    # diverging feed), not free money.
    if best.ask is not None and best.p_model - best.ask > max_disagreement:
        reasons.append(
            f"Model ({best.p_model:.0%}) disagrees with the market ({best.ask:.0%}) by more than "
            f"{max_disagreement:.0%}: check the open price and price feed"
        )

    if reasons:
        return Recommendation("NO BET", reasons, best, 0.0, quotes)

    stake = min(bankroll * best.kelly * kelly_fraction, bankroll * max_stake_fraction)
    return Recommendation(
        f"BET {best.side}",
        [f"Model {best.p_model:.1%} vs cost {best.cost:.3f} → edge {ev:+.1%} per $1"],
        best,
        round(stake, 2),
        quotes,
    )
