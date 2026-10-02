"""Claude-powered news + market analyst.

The probabilities come from the deterministic model in btcpredict.model; Claude only
reads them through tools, summarizes the news, and explains the call in plain English.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import anthropic

from .config import Settings
from .engine import Snapshot

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the analyst inside an advisory terminal dashboard for Polymarket's \
"Bitcoin Up or Down - 15 minute" markets. Each market resolves UP if BTC's price at the end \
of the 15-minute window is at or above the price at the start (Chainlink BTC/USD feed), else DOWN.

You have tools that return live data from the user's dashboard:
- get_market_state: BTC price, window open price, time left, Polymarket bid/ask for UP and DOWN.
- get_projection: the quantitative model's P(UP), its uncertainty band, volatility, the \
recommendation (BET UP / BET DOWN / NO BET), the reasons, and the suggested stake.
- get_news: recent BTC and macro headlines with a keyword-based impact score.

How to work:
- Call the tools before answering; never state a price, probability or edge you did not get from a tool.
- The model's recommendation is the decision. You may add caution (for example a major headline \
the keyword scorer under-rated, or a scheduled macro release), but do not turn NO BET into a bet.
- Over 15 minutes, most news is noise. Call out only items that could plausibly move BTC within \
minutes: ETF flows/decisions, exchange hacks or halts, Fed/CPI/jobs data, large liquidations, \
regulatory actions, stablecoin problems.
- If signal_validated_by_paper_trading is false, the signal has not yet been shown to beat the \
market on recorded history: call any BET a paper bet and say not to stake real money yet.
- This is advisory only. The user places any bets themselves.
- Be concise and use plain text suited to a terminal panel (no markdown headings or tables)."""

BRIEFING_PROMPT = """Write the briefing for the current window in at most 110 words, in this shape:
Call: <the model's recommendation, with P(UP) and the best edge>
Why: <one or two sentences>
News: <the 1-3 headlines that matter right now and whether each raises risk, or "nothing market-moving">
Watch: <one thing that would change the call>"""

TOOLS = [
    {
        "name": "get_market_state",
        "description": "Live BTC price, current 15-minute window, window open price, seconds left, "
        "and Polymarket UP/DOWN best bid and ask (prices are 0-1, i.e. implied probability).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_projection",
        "description": "The quantitative model's probability that BTC finishes the window UP, its "
        "uncertainty band, volatility, per-side edge after fees, and the BET/NO BET recommendation "
        "with reasons and suggested stake.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_news",
        "description": "Recent Bitcoin and macro headlines, newest first, with source, age in "
        "minutes, keyword impact score (high/medium/low) and lean (bullish/bearish/neutral).",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max headlines to return (1-30)."},
                "min_impact": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Only return headlines at or above this impact.",
                },
            },
            "additionalProperties": False,
        },
    },
]

_MARKET_KEYS = (
    "time_utc", "window", "market_slug", "btc_price", "window_open_price", "change_usd",
    "seconds_left", "polymarket_up_ask", "polymarket_up_bid", "polymarket_down_ask",
    "polymarket_down_bid", "status",
)
_PROJ_KEYS = (
    "seconds_left", "volatility_annualized", "model_p_up", "model_p_up_band",
    "expected_1sigma_move_usd", "edge_per_dollar", "recommendation", "recommendation_reasons",
    "suggested_stake_usd", "high_impact_news_last_10m", "signal_validated_by_paper_trading",
)
_IMPACT_RANK = {"low": 0, "medium": 1, "high": 2}

# Models that accept server-side refusal fallbacks and the `effort` setting.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def run_tool(name: str, args: dict, snap: Snapshot) -> str:
    d = snap.to_dict()
    if name == "get_market_state":
        return json.dumps({k: d[k] for k in _MARKET_KEYS})
    if name == "get_projection":
        return json.dumps({k: d[k] for k in _PROJ_KEYS})
    if name == "get_news":
        limit = max(1, min(int(args.get("limit", 12)), 30))
        floor = _IMPACT_RANK.get(args.get("min_impact", "low"), 0)
        items = [
            {
                "source": n.source,
                "title": n.title,
                "age_min": round(n.age_minutes(snap.ts), 1),
                "impact": n.impact,
                "lean": n.lean,
            }
            for n in snap.news
            if _IMPACT_RANK[n.impact] >= floor
        ][:limit]
        return json.dumps(items or [{"note": "no matching headlines"}])
    raise ValueError(f"unknown tool {name}")


class Analyst:
    def __init__(self, settings: Settings, get_snapshot: Callable[[], Snapshot]):
        self.s = settings
        self.get_snapshot = get_snapshot
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)

    def _request_kwargs(self) -> dict:
        kwargs: dict = {
            "model": self.s.claude_model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "tools": TOOLS,
            "cache_control": {"type": "ephemeral"},
        }
        if self.s.claude_model in _FALLBACK_MODELS:
            kwargs["output_config"] = {"effort": self.s.claude_effort}
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def ask(self, question: str, max_turns: int = 6) -> str:
        messages: list[dict] = [{"role": "user", "content": question}]
        kwargs = self._request_kwargs()
        for _ in range(max_turns):
            try:
                resp = await self.client.beta.messages.create(messages=messages, **kwargs)
            except anthropic.AuthenticationError:
                return "AI error: invalid ANTHROPIC_API_KEY."
            except anthropic.RateLimitError:
                return "AI error: rate limited; try again shortly."
            except anthropic.APIStatusError as exc:
                return f"AI error ({exc.status_code}): {exc.message}"
            except anthropic.APIConnectionError:
                return "AI error: could not reach the Anthropic API."

            if resp.stop_reason == "refusal":
                return "AI declined to answer this request."
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                if resp.stop_reason == "max_tokens":
                    text += " [truncated]"
                return text or "(no answer)"

            messages.append({"role": "assistant", "content": resp.content})
            snap = self.get_snapshot()
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                try:
                    content, is_error = run_tool(block.name, dict(block.input or {}), snap), False
                except (ValueError, KeyError, TypeError) as exc:
                    content, is_error = f"tool error: {exc}", True
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
        return "AI stopped: too many tool calls."

    async def briefing(self) -> str:
        return await self.ask(BRIEFING_PROMPT)
