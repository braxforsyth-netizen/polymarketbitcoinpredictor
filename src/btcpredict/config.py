"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines, '#' comments. Never overrides real env vars."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _f(name: str, default: float) -> float:
    value = os.environ.get(name, "")
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    price_source: str = "coinbase"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "low"
    cryptopanic_token: str = ""
    min_edge: float = 0.04
    kelly_fraction: float = 0.25
    bankroll: float = 100.0
    max_stake_fraction: float = 0.05
    taker_fee_rate: float = 0.03
    min_seconds_left: float = 45.0
    db_path: str = "data/snapshots.sqlite"

    @property
    def ai_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


def load_settings() -> Settings:
    _load_dotenv()
    env = os.environ
    return Settings(
        price_source=env.get("PRICE_SOURCE", "coinbase").lower() or "coinbase",
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        claude_model=env.get("CLAUDE_MODEL", "") or "claude-opus-5-5",
        claude_effort=env.get("CLAUDE_EFFORT", "") or "low",
        cryptopanic_token=env.get("CRYPTOPANIC_TOKEN", ""),
        min_edge=_f("MIN_EDGE", 0.04),
        kelly_fraction=_f("KELLY_FRACTION", 0.25),
        bankroll=_f("BANKROLL", 100.0),
        max_stake_fraction=_f("MAX_STAKE_FRACTION", 0.05),
        taker_fee_rate=_f("TAKER_FEE_RATE", 0.03),
        min_seconds_left=_f("MIN_SECONDS_LEFT", 45.0),
        db_path=env.get("DB_PATH", "") or "data/snapshots.sqlite",
    )
