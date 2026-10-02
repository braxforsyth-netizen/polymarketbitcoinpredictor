"""Rich terminal dashboard. Advisory only: nothing here places orders."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .agent import Analyst
from .config import Settings
from .engine import Engine, Snapshot
from .model.windows import WINDOW_SECONDS
from .recorder import Recorder

log = logging.getLogger(__name__)

IMPACT_STYLE = {"high": "bold red", "medium": "yellow", "low": "dim"}
LEAN_MARK = {"bullish": "[green]▲[/]", "bearish": "[red]▼[/]", "neutral": "[dim]·[/]"}


def _fmt(x: float | None, spec: str, dash: str = "—") -> str:
    return dash if x is None else format(x, spec)


def market_panel(s: Snapshot) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    change = (s.price - s.open_price) if s.price and s.open_price else None
    style = "green" if change is not None and change >= 0 else "red"
    src = {"chainlink": "Chainlink (settles)", "exchange": "exchange", "exchange+basis": "exchange + basis est."}
    t.add_row("BTC", f"[bold]${_fmt(s.price, ',.2f')}[/] [dim]{src.get(s.price_source, s.price_source)}[/]")
    t.add_row("Window", s.window.label())
    t.add_row("Open (to beat)", f"${_fmt(s.open_price, ',.2f')} [dim]{src.get(s.open_source, s.open_source)}[/]")
    t.add_row(
        "Change",
        f"[{style}]{_fmt(change, '+,.2f')} ({_fmt(change / s.open_price * 100 if change is not None else None, '+.3f')}%)[/]",
    )
    m, sec = divmod(int(s.seconds_left), 60)
    t.add_row("Time left", f"{m}m {sec:02d}s")
    t.add_row("Volatility", f"{_fmt(s.sigma_annual * 100 if s.sigma_annual else None, '.0f')}% annualized")
    if s.projection:
        t.add_row("1σ move left", f"±${s.projection.expected_move_usd:,.0f}")
    bar = ProgressBar(total=WINDOW_SECONDS, completed=WINDOW_SECONDS - s.seconds_left, width=40)
    return Panel(Group(t, Text(""), bar), title="Market", border_style="cyan")


def odds_panel(s: Snapshot) -> Panel:
    t = Table(box=None, expand=True)
    for col in ("", "Model", "PM bid", "PM ask", "Edge/$"):
        t.add_column(col, justify="right" if col else "left")
    p = s.projection
    q = s.quote
    rec = s.recommendation
    edges = {sq.side: sq.ev_per_dollar for sq in rec.quotes} if rec else {}
    for side, prob, bid, ask in (
        ("UP", p.p_up if p else None, q.up.best_bid if q else None, q.ask_up if q else None),
        ("DOWN", p.p_down if p else None, q.down.best_bid if q else None, q.ask_down if q else None),
    ):
        e = edges.get(side)
        e_style = "green" if e is not None and e > 0 else "red"
        t.add_row(
            f"[bold]{side}[/]",
            _fmt(prob * 100 if prob is not None else None, ".1f") + "%",
            _fmt(bid, ".3f"),
            _fmt(ask, ".3f"),
            f"[{e_style}]{_fmt(e * 100 if e is not None else None, '+.1f')}%[/]",
        )
    band = f"P(UP) band if vol is ±25% off: {p.p_low:.1%} – {p.p_high:.1%}" if p else ""
    return Panel(Group(t, Text(band, style="dim")), title="Odds: model vs Polymarket", border_style="cyan")


def signal_panel(s: Snapshot, settings: Settings) -> Panel:
    rec = s.recommendation
    if rec is None:
        body = Text("Waiting for data…", style="dim")
        return Panel(body, title="Signal", border_style="white")
    color = "green" if rec.action == "BET UP" else "red" if rec.action == "BET DOWN" else "yellow"
    lines = [Text(rec.action, style=f"bold {color}", justify="center")]
    if rec.stake:
        lines.append(
            Text(
                f"Suggested stake ${rec.stake:.2f} (¼-Kelly, bankroll ${settings.bankroll:,.0f})",
                justify="center",
            )
        )
    for r in rec.reasons:
        lines.append(Text(f"• {r}", style="dim"))
    return Panel(Group(*lines), title="Signal (advisory)", border_style=color)


def news_panel(s: Snapshot, rows: int = 14) -> Panel:
    t = Table(box=None, expand=True, show_header=False)
    t.add_column(width=5, justify="right")
    t.add_column(width=2)
    t.add_column(width=6)
    t.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    for n in s.news[:rows]:
        age = n.age_minutes(s.ts)
        age_s = f"{age:.0f}m" if age < 60 else f"{age / 60:.0f}h"
        t.add_row(age_s, LEAN_MARK[n.lean], f"[{IMPACT_STYLE[n.impact]}]{n.impact}[/]", f"{n.title} [dim]({n.source})[/]")
    title = "News" + (" — [bold red]HIGH-IMPACT NEWS IN LAST 10m[/]" if s.shock else "")
    return Panel(t if s.news else Text("Loading headlines…", style="dim"), title=title, border_style="magenta")


def ai_panel(text: str, enabled: bool, updated: float | None) -> Panel:
    if not enabled:
        body = Text("AI briefing off (free mode). Set ANTHROPIC_API_KEY to enable.", style="dim")
    else:
        body = Text(text or "Briefing will appear ~4 minutes into each window…", style="" if text else "dim")
    sub = time.strftime("updated %H:%M:%S", time.localtime(updated)) if updated else None
    return Panel(body, title="AI analyst", subtitle=sub, border_style="blue")


def render(s: Snapshot, settings: Settings, ai_text: str, ai_updated: float | None) -> Layout:
    root = Layout()
    status = " | ".join(s.status) if s.status else "all feeds OK"
    header = Text.assemble(
        ("BTC 15m Up/Down Advisor", "bold"),
        "  ·  advisory only, not financial advice  ·  ",
        (status, "yellow" if s.status else "green"),
    )
    root.split_column(Layout(header, size=1), Layout(name="body"), Layout(name="ai", size=9))
    root["body"].split_row(Layout(name="left", ratio=5), Layout(name="right", ratio=6))
    root["left"].split_column(
        Layout(market_panel(s), size=12), Layout(odds_panel(s), size=7), Layout(signal_panel(s, settings))
    )
    root["right"].update(news_panel(s))
    root["ai"].update(ai_panel(ai_text, settings.ai_enabled, ai_updated))
    return root


class Dashboard:
    def __init__(self, settings: Settings):
        self.s = settings
        self.engine = Engine(settings)
        self.analyst = Analyst(settings, self.engine.snapshot) if settings.ai_enabled else None
        self.recorder = Recorder(settings.db_path)
        self.ai_text = ""
        self.ai_updated: float | None = None
        self._briefed: set[tuple[int, str]] = set()

    async def _ai_loop(self) -> None:
        """One briefing per window (~4 min in), plus one when high-impact news lands."""
        while True:
            await asyncio.sleep(5)
            snap = self.engine.snapshot()
            elapsed = WINDOW_SECONDS - snap.seconds_left
            key = None
            if snap.shock and (snap.window.start, "shock") not in self._briefed:
                key = (snap.window.start, "shock")
            elif elapsed >= 240 and (snap.window.start, "mid") not in self._briefed:
                key = (snap.window.start, "mid")
            if key and snap.projection:
                self._briefed.add(key)
                self.ai_text = await self.analyst.briefing()
                self.ai_updated = time.time()

    async def _record_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            self.recorder.record(self.engine.snapshot())

    async def run(self) -> None:
        with contextlib.suppress(Exception):
            await self.engine.seed_volatility()
        tasks = [
            asyncio.create_task(self.engine.run_price_stream()),
            *([asyncio.create_task(self.engine.run_chainlink_stream())] if self.s.settlement_feed else []),
            asyncio.create_task(self.engine.run_market_loop()),
            asyncio.create_task(self.engine.run_news_loop()),
            asyncio.create_task(self._record_loop()),
        ]
        if self.analyst:
            tasks.append(asyncio.create_task(self._ai_loop()))
        try:
            with Live(screen=True, refresh_per_second=4, auto_refresh=False) as live:
                while True:
                    live.update(render(self.engine.snapshot(), self.s, self.ai_text, self.ai_updated), refresh=True)
                    await asyncio.sleep(0.25)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.recorder.close()
            await self.engine.client.aclose()
