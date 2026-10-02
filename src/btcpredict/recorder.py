"""Append dashboard snapshots to SQLite. Over time this becomes your own dataset of
model probability vs Polymarket price vs actual outcome, which is what you need to
find out whether the edge is real."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .engine import Snapshot

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    ts REAL, window_start INTEGER, price REAL, open_price REAL, seconds_left REAL,
    sigma_annual REAL, p_up REAL, ask_up REAL, ask_down REAL, bid_up REAL, bid_down REAL,
    action TEXT, stake REAL, news_shock INTEGER
);
CREATE INDEX IF NOT EXISTS idx_snapshots_window ON snapshots(window_start);
"""


class Recorder:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def record(self, s: Snapshot) -> None:
        if s.price is None or s.open_price is None:
            return
        q, r = s.quote, s.recommendation
        self.db.execute(
            "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                s.ts, s.window.start, s.price, s.open_price, s.seconds_left, s.sigma_annual,
                s.projection.p_up if s.projection else None,
                q.ask_up if q else None, q.ask_down if q else None,
                q.up.best_bid if q else None, q.down.best_bid if q else None,
                r.action if r else None, r.stake if r else 0.0, int(s.shock),
            ),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
