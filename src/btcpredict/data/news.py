"""Free crypto news: public RSS feeds plus (optionally) the CryptoPanic free API.

Headlines get a fast keyword score so the dashboard works without any AI. When an
Anthropic key is configured, the agent re-reads them and writes the briefing.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime

import feedparser
import httpx

log = logging.getLogger(__name__)

RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
    "The Block": "https://www.theblock.co/rss.xml",
}

BTC_TERMS = re.compile(r"\b(bitcoin|btc|crypto|spot etf|satoshi|miners?|halving)\b", re.I)
MACRO_TERMS = re.compile(r"\b(fed|fomc|powell|cpi|inflation|rate (cut|hike)s?|jobs report|nfp|tariffs?|treasury)\b", re.I)
HIGH_IMPACT = re.compile(
    r"\b(sec|etf (approv|reject|inflow|outflow)\w*|hack(ed)?|exploit|stolen|bankrupt\w*|insolven\w*|"
    r"liquidat\w*|fomc|cpi|rate (cut|hike)|ban(s|ned)?|halt(s|ed)?|tether|depeg\w*|blackrock|"
    r"strategic (bitcoin )?reserve|emergency|crash\w*|surge[sd]?|plunge[sd]?|all-time high|ath)\b",
    re.I,
)
BULLISH = re.compile(r"\b(surge[sd]?|soar\w*|rall(y|ies|ied)|inflows?|approv\w*|record high|all-time high|ath|buy(s|ing)?|adopt\w*|rate cut)\b", re.I)
BEARISH = re.compile(r"\b(plunge[sd]?|crash\w*|dump\w*|outflows?|hack(ed)?|exploit|reject\w*|ban(s|ned)?|sell-?off|liquidat\w*|lawsuit|rate hike|fears?)\b", re.I)


@dataclass(frozen=True)
class NewsItem:
    source: str
    title: str
    url: str
    published: float  # unix seconds
    impact: str  # "high" | "medium" | "low"
    lean: str  # "bullish" | "bearish" | "neutral"

    def age_minutes(self, now: float | None = None) -> float:
        return ((now or time.time()) - self.published) / 60.0


def score_headline(title: str) -> tuple[str, str] | None:
    """Return (impact, lean), or None if the headline isn't BTC/macro relevant."""
    btc, macro = bool(BTC_TERMS.search(title)), bool(MACRO_TERMS.search(title))
    if not (btc or macro):
        return None
    if HIGH_IMPACT.search(title):
        impact = "high"
    elif btc and macro:
        impact = "medium"
    else:
        impact = "medium" if btc and re.search(r"\b(price|market|traders?|whales?)\b", title, re.I) else "low"
    bull, bear = len(BULLISH.findall(title)), len(BEARISH.findall(title))
    lean = "bullish" if bull > bear else "bearish" if bear > bull else "neutral"
    return impact, lean


def _entry_time(entry) -> float:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return float(calendar.timegm(t))
    return time.time()


async def _fetch_rss(client: httpx.AsyncClient, name: str, url: str) -> list[NewsItem]:
    try:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0 btcpredict/0.1"})
        r.raise_for_status()
    except httpx.HTTPError as exc:
        log.info("news feed %s failed: %s", name, exc)
        return []
    parsed = feedparser.parse(r.content)
    items = []
    for e in parsed.entries[:40]:
        title = (e.get("title") or "").strip()
        scored = score_headline(title)
        if scored:
            items.append(NewsItem(name, title, e.get("link", ""), _entry_time(e), *scored))
    return items


async def _fetch_cryptopanic(client: httpx.AsyncClient, token: str) -> list[NewsItem]:
    try:
        r = await client.get(
            "https://cryptopanic.com/api/developer/v2/posts/",
            params={"auth_token": token, "currencies": "BTC", "public": "true"},
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except (httpx.HTTPError, ValueError) as exc:
        log.info("CryptoPanic failed: %s", exc)
        return []
    items = []
    for p in results:
        title = (p.get("title") or "").strip()
        scored = score_headline(title) or ("low", "neutral")
        ts = p.get("published_at") or p.get("created_at")
        try:
            published = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except (AttributeError, TypeError, ValueError):
            published = time.time()
        url = p.get("original_url") or p.get("url") or ""
        items.append(NewsItem("CryptoPanic", title, url, published, *scored))
    return items


async def fetch_news(client: httpx.AsyncClient, cryptopanic_token: str = "", max_age_h: float = 12) -> list[NewsItem]:
    tasks = [_fetch_rss(client, n, u) for n, u in RSS_FEEDS.items()]
    if cryptopanic_token:
        tasks.append(_fetch_cryptopanic(client, cryptopanic_token))
    batches = await asyncio.gather(*tasks)
    cutoff = time.time() - max_age_h * 3600
    seen: set[str] = set()
    items: list[NewsItem] = []
    for item in sorted((i for b in batches for i in b), key=lambda i: i.published, reverse=True):
        key = re.sub(r"\W+", "", item.title.lower())[:80]
        if item.published >= cutoff and key not in seen:
            seen.add(key)
            items.append(item)
    return items


def news_shock(items: list[NewsItem], within_minutes: float = 10.0, now: float | None = None) -> bool:
    """True if a high-impact headline appeared very recently."""
    return any(i.impact == "high" and 0 <= i.age_minutes(now) <= within_minutes for i in items)
