"""Background loops that keep the advisor honest without manual steps:

  * startup checks  - can we reach every data source?
  * calibration     - daily backtest; tunes volatility; blocks bets if still miscalibrated
  * data check      - after each window, did our prices call the same result Polymarket did?
  * readiness       - running paper-trading review; signals stay "PAPER" until validated
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import calibration, review
from .config import Settings
from .engine import Engine
from .health import Check, run_checks
from .model.windows import WINDOW_SECONDS

log = logging.getLogger(__name__)

DATA_CHECK_SCHEMA = """
CREATE TABLE IF NOT EXISTS data_checks (
    window_start INTEGER PRIMARY KEY, ours INTEGER, polymarket INTEGER, margin_usd REAL
);
"""
NEAR_LINE_USD = 10.0  # disagreements closer than this to the line are expected noise
MAX_SERIOUS_MISMATCHES = 2  # in the last 20 checked windows
FINAL_MAX_SECONDS_LEFT = 5.0  # our "final" price must be from the last few seconds


@dataclass(frozen=True)
class DataCheckStats:
    checked: int
    matched: int
    near_line_misses: int
    serious_misses: int

    def summary(self) -> str:
        if not self.checked:
            return "waiting for the first window to resolve"
        extra = f", {self.near_line_misses} near the line" if self.near_line_misses else ""
        return f"{self.matched}/{self.checked} windows matched Polymarket's result{extra}"


class Supervisor:
    def __init__(self, settings: Settings, engine: Engine, db: sqlite3.Connection):
        self.s = settings
        self.engine = engine
        self.db = db
        self.db.executescript(DATA_CHECK_SCHEMA)
        self.client: httpx.AsyncClient = engine.client
        self.checks: list[Check] | None = None
        self.calibration: calibration.CalibrationResult | None = None
        self.calibration_note = "running…"
        self.data_check = DataCheckStats(0, 0, 0, 0)
        self.readiness: review.Readiness | None = None
        self._outcome_attempts: dict[int, int] = {}
        self._check_attempts: dict[int, int] = {}
        self._cal_path = str(Path(settings.db_path).with_name("calibration.json"))

    # ---------- startup checks ----------
    async def run_startup_checks(self) -> None:
        try:
            self.checks = await run_checks(self.s, self.client)
        except Exception as exc:
            log.exception("startup checks failed")
            self.checks = [Check("Startup checks", False, str(exc)[:80])]
        failed = [c for c in self.checks if not c.ok and c.required]
        if failed:
            self.engine.blockers["checks"] = "Data source down: " + ", ".join(c.name for c in failed)
        else:
            self.engine.blockers.pop("checks", None)

    # ---------- calibration ----------
    def _apply_calibration(self, result: calibration.CalibrationResult) -> None:
        self.calibration = result
        if result.windows >= calibration.MIN_WINDOWS:
            self.engine.vol_multiplier = result.vol_multiplier
        if result.ok:
            self.engine.blockers.pop("calibration", None)
            self.calibration_note = result.summary()
        else:
            self.engine.blockers["calibration"] = (
                f"Model miscalibrated on the last {result.days:g} days (worst bin off {result.max_gap:.0%})"
            )
            self.calibration_note = result.summary() + " — too far off, bets blocked"

    async def calibration_loop(self) -> None:
        while True:
            cached = calibration.load_cached(self._cal_path, self.s.price_source)
            if cached:
                self._apply_calibration(cached)
            else:
                try:
                    result = await calibration.run_calibration(self.client, self.s.price_source)
                    calibration.save(self._cal_path, result)
                    self._apply_calibration(result)
                except Exception as exc:
                    log.warning("calibration failed: %s", exc)
                    self.calibration_note = f"failed ({str(exc)[:60]}); retrying in 30m"
                    await asyncio.sleep(1800)
                    continue
            age = time.time() - self.calibration.ts
            await asyncio.sleep(max(600.0, calibration.MAX_AGE_S - age))

    # ---------- data check + readiness ----------
    async def check_finished_windows(self) -> None:
        now = time.time()
        done = {w for (w,) in self.db.execute("SELECT window_start FROM data_checks")}
        for w, (open_px, last_px, left) in list(self.engine.finals.items()):
            if w in done or w + WINDOW_SECONDS + 60 > now or left > FINAL_MAX_SECONDS_LEFT:
                continue
            if self._check_attempts.get(w, 0) >= 30:
                continue
            self._check_attempts[w] = self._check_attempts.get(w, 0) + 1
            try:
                pm = await review.polymarket_outcome(self.client, w)
            except httpx.HTTPError:
                continue
            if pm is None:
                continue
            ours = int(last_px >= open_px)
            self.db.execute("INSERT OR REPLACE INTO data_checks VALUES (?,?,?,?)", (w, ours, pm, last_px - open_px))
            self.db.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?)", (w, pm, "polymarket"))
            self.db.commit()
            self.engine.finals.pop(w, None)

        recent = list(self.db.execute(
            "SELECT ours, polymarket, margin_usd FROM data_checks ORDER BY window_start DESC LIMIT 20"
        ))
        misses = [(m or 0.0) for o, p, m in recent if o != p]
        serious = sum(1 for m in misses if abs(m) >= NEAR_LINE_USD)
        self.data_check = DataCheckStats(len(recent), len(recent) - len(misses), len(misses) - serious, serious)
        if serious >= MAX_SERIOUS_MISMATCHES:
            self.engine.blockers["data"] = (
                f"Our prices called {serious} of the last {len(recent)} windows differently from Polymarket"
            )
        else:
            self.engine.blockers.pop("data", None)

    async def update_readiness(self) -> None:
        rows = review.load_rows(self.db)
        outcomes = await review.resolve_outcomes(
            self.db, self.client, {r["window_start"] for r in rows}, self.s.price_source,
            attempts=self._outcome_attempts,
        )
        self.readiness = review.readiness(rows, outcomes, self.s.taker_fee_rate)
        self.engine.validated = self.readiness.validated

    async def validation_loop(self, every_s: float = 60.0) -> None:
        while True:
            for step in (self.check_finished_windows, self.update_readiness):
                try:
                    await step()
                except Exception:
                    log.exception("validation step failed")
            await asyncio.sleep(every_s)
