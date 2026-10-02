"""Command line entry point: `btcpredict [dashboard|snapshot|news|ask|backtest]`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

import httpx
from rich.console import Console
from rich.table import Table

from .config import load_settings

console = Console()


def cmd_dashboard(args) -> None:
    from .dashboard import Dashboard

    settings = load_settings()
    try:
        asyncio.run(Dashboard(settings).run())
    except KeyboardInterrupt:
        pass


async def _one_shot_engine():
    from .engine import Engine

    settings = load_settings()
    engine = Engine(settings)
    await engine.refresh_all()
    return settings, engine


def cmd_snapshot(args) -> None:
    async def main():
        _, engine = await _one_shot_engine()
        try:
            print(json.dumps(engine.snapshot().to_dict(), indent=2))
        finally:
            await engine.client.aclose()

    asyncio.run(main())


def cmd_news(args) -> None:
    from .data.news import fetch_news

    async def main():
        settings = load_settings()
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            items = await fetch_news(client, settings.cryptopanic_token)
        t = Table("Age", "Impact", "Lean", "Source", "Headline")
        now = time.time()
        for n in items[: args.limit]:
            t.add_row(f"{n.age_minutes(now):.0f}m", n.impact, n.lean, n.source, n.title)
        console.print(t)

    asyncio.run(main())


def cmd_ask(args) -> None:
    from .agent import Analyst

    async def main():
        settings, engine = await _one_shot_engine()
        if not settings.ai_enabled:
            console.print("[yellow]Set ANTHROPIC_API_KEY to use `ask`.[/]")
            return
        try:
            question = " ".join(args.question) or "Give me the briefing for the current window."
            answer = await Analyst(settings, engine.snapshot).ask(question)
            console.print(answer)
        finally:
            await engine.client.aclose()

    asyncio.run(main())


def cmd_backtest(args) -> None:
    from .backtest import evaluate, predict_windows
    from .data.prices import fetch_candles_1m

    async def main():
        settings = load_settings()
        source = args.source or settings.price_source
        end = int(time.time()) // 60 * 60
        start = end - int(args.days * 86400) - 3600
        console.print(f"Downloading {args.days:g} days of 1m BTC candles from {source}…")
        async with httpx.AsyncClient(timeout=20.0) as client:
            candles = await fetch_candles_1m(client, source, start, end)
        report = evaluate(predict_windows(candles))

        console.print(
            f"\n[bold]{report.windows}[/] windows, {report.predictions} predictions, "
            f"UP rate {report.up_rate:.1%}"
        )
        console.print(
            f"Brier score [bold]{report.brier:.4f}[/] (coin flip = {report.brier_coinflip:.2f}, lower is better), "
            f"log loss {report.log_loss:.4f}\n"
        )
        cal = Table("Predicted P(UP)", "N", "Avg predicted", "Actually UP", "Gap", title="Calibration")
        for b in report.bins:
            gap = b.hit_rate - b.mean_pred
            style = "green" if abs(gap) < 0.03 else "yellow" if abs(gap) < 0.06 else "red"
            cal.add_row(
                f"{b.lo:.0%}–{b.hi:.0%}", str(b.n), f"{b.mean_pred:.1%}", f"{b.hit_rate:.1%}",
                f"[{style}]{gap:+.1%}[/]",
            )
        console.print(cal)
        bm = Table("Minute into window", "N", "Brier", title="Accuracy by time elapsed")
        for m, n, brier in report.by_minute:
            bm.add_row(str(m), str(n), f"{brier:.4f}")
        console.print(bm)

        from .backtest import fit_vol_multiplier, max_calibration_gap, rescale
        preds = predict_windows(candles)
        k = fit_vol_multiplier(preds)
        tuned = evaluate(rescale(preds, k))
        console.print(
            f"\nAuto-tuning: volatility ×{k:.2f} gives Brier {tuned.brier:.4f} "
            f"(raw {report.brier:.4f}), worst calibration bin off {max_calibration_gap(tuned):.1%}. "
            "[dim]The dashboard applies this automatically each day.[/]"
        )

    asyncio.run(main())


def cmd_review(args) -> None:
    import sqlite3
    from pathlib import Path

    from . import review

    async def main():
        settings = load_settings()
        if not Path(settings.db_path).exists():
            console.print(f"[yellow]No recordings at {settings.db_path} yet. Run the dashboard for a while first.[/]")
            return
        db = sqlite3.connect(settings.db_path)
        rows = review.load_rows(db)
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            outcomes = await review.resolve_outcomes(
                db, client, {r["window_start"] for r in rows}, settings.price_source
            )
        db.close()
        windows = {r["window_start"] for r in rows if r["window_start"] in outcomes}
        console.print(f"[bold]{len(rows)}[/] snapshots across [bold]{len(windows)}[/] finished windows\n")
        if not windows:
            console.print("[yellow]No finished windows to review yet.[/]")
            return

        s = review.summarize(review.recorded_trades(rows, outcomes, settings.taker_fee_rate))
        console.print("[bold]Signals the dashboard showed[/] (first BET per window, suggested stake)")
        if s.n:
            pnl_style = "green" if s.pnl >= 0 else "red"
            console.print(
                f"  {s.n} bets, won {s.win_rate:.0%}, staked ${s.staked:,.2f}, "
                f"P&L [{pnl_style}]${s.pnl:+,.2f}[/] (ROI {s.roi:+.1%}), max drawdown ${s.max_drawdown:,.2f}\n"
            )
        else:
            console.print("  No BET signals recorded yet.\n")

        t = Table("Min edge", "Bets", "Win rate", "ROI ($1 flat)", "Max drawdown", title="What if MIN_EDGE were…")
        for th, sm in review.threshold_sweep(rows, outcomes, settings.taker_fee_rate, min_seconds_left=settings.min_seconds_left):
            style = "green" if sm.roi > 0 else "red" if sm.n else "dim"
            t.add_row(f"{th:.0%}", str(sm.n), f"{sm.win_rate:.0%}" if sm.n else "—",
                      f"[{style}]{sm.roi:+.1%}[/]" if sm.n else "—", f"${sm.max_drawdown:.2f}")
        console.print(t)

        b = Table("Time left", "Snapshots", "Model Brier", "Market Brier", "Better forecaster", title="Model vs Polymarket accuracy (lower Brier is better)")
        for row in review.brier_vs_market(rows, outcomes):
            better = "[green]model[/]" if row.model < row.market else "[red]market[/]"
            b.add_row(row.label, str(row.n), f"{row.model:.4f}", f"{row.market:.4f}", better)
        console.print(b)
        console.print(
            "[dim]If the market is the better forecaster, positive ROI above is probably luck. "
            "Collect a few hundred windows before drawing conclusions.[/]"
        )

    asyncio.run(main())


def cmd_doctor(args) -> None:
    from .health import run_checks

    async def main():
        settings = load_settings()
        console.print("Checking data sources (takes up to ~15s)…\n")
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            checks = await run_checks(settings, client)
        t = Table("", "Check", "Result", "How to fix")
        for c in checks:
            mark = "[green]✓[/]" if c.ok else ("[red]✗[/]" if c.required else "[yellow]![/]")
            t.add_row(mark, c.name, c.detail, c.fix)
        console.print(t)
        bad = [c for c in checks if not c.ok and c.required]
        if bad:
            console.print(f"[red]{len(bad)} required check(s) failed; the dashboard will show NO BET until fixed.[/]")
        else:
            console.print("[green]Ready. Run `btcpredict` to start the dashboard.[/]")

    asyncio.run(main())


def _first_run_setup() -> None:
    from pathlib import Path
    import shutil

    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        console.print("[dim]Created .env from .env.example (edit it to change settings).[/]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="btcpredict", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("dashboard", help="live terminal dashboard (default)").set_defaults(func=cmd_dashboard)
    sub.add_parser("snapshot", help="print the current state once as JSON").set_defaults(func=cmd_snapshot)
    p = sub.add_parser("news", help="show the latest scored headlines")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_news)
    p = sub.add_parser("ask", help="ask the AI analyst a question (needs ANTHROPIC_API_KEY)")
    p.add_argument("question", nargs="*")
    p.set_defaults(func=cmd_ask)
    p = sub.add_parser("backtest", help="check model calibration on historical candles")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--source", choices=["coinbase", "binance"])
    p.set_defaults(func=cmd_backtest)
    sub.add_parser("doctor", help="check every data source and show fixes").set_defaults(func=cmd_doctor)
    sub.add_parser("review", help="paper-trading report from recorded snapshots").set_defaults(func=cmd_review)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        filename="btcpredict.log" if (args.cmd in (None, "dashboard")) else None,
    )
    if not args.verbose:
        # websockets can trip an internal error when a connection drops mid-handshake;
        # it is harmless (we reconnect) but asyncio would print the traceback.
        logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    _first_run_setup()
    (args.func if args.cmd else cmd_dashboard)(args)


if __name__ == "__main__":
    sys.exit(main())
