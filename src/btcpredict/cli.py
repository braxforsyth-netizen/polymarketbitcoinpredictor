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

    asyncio.run(main())


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

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        filename="btcpredict.log" if (args.cmd in (None, "dashboard")) else None,
    )
    (args.func if args.cmd else cmd_dashboard)(args)


if __name__ == "__main__":
    sys.exit(main())
