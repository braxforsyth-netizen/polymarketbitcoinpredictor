"""15-minute window arithmetic. Polymarket's BTC Up/Down 15m markets run on aligned UTC
quarter-hours (:00, :15, :30, :45) and resolve UP if close >= open."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

WINDOW_SECONDS = 15 * 60


@dataclass(frozen=True)
class Window:
    start: int  # unix seconds, multiple of 900
    end: int

    @property
    def slug(self) -> str:
        """Polymarket event/market slug for this window, e.g. btc-updown-15m-1759420800."""
        return f"btc-updown-15m-{self.start}"

    def seconds_left(self, now: float) -> float:
        return max(0.0, self.end - now)

    def seconds_elapsed(self, now: float) -> float:
        return min(float(WINDOW_SECONDS), max(0.0, now - self.start))

    def label(self) -> str:
        fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M")
        return f"{fmt(self.start)}–{fmt(self.end)} UTC"


def window_at(ts: float) -> Window:
    start = int(ts // WINDOW_SECONDS) * WINDOW_SECONDS
    return Window(start=start, end=start + WINDOW_SECONDS)
