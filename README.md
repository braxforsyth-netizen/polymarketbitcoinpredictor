# btcpredict — BTC 15-minute Up/Down advisor

A terminal dashboard for Polymarket's **"Bitcoin Up or Down – 15 minute"** markets. It shows:

- the live BTC price, the window's opening price (the price to beat), and the time left
- the model's **probability that BTC finishes the window UP**, next to Polymarket's UP/DOWN prices
- the **edge after fees** on each side, and an advisory **BET UP / BET DOWN / NO BET** signal with a suggested stake
- live Bitcoin and macro **headlines**, scored for short-term impact
- optional **AI analyst** (Claude) briefings that summarize the news and explain the call

It is **advisory only**. It never places orders, and it needs no Polymarket account.

## How the projection works

A market resolves UP if BTC's price at the end of the window is at or above the price at the start. Treating price as a short-term random walk:

```
P(UP) = Φ( ln(price_now / price_open) / (σ · √seconds_left) )
```

`σ` is per-second volatility: it is seeded from the last 3 hours of 1-minute candles, then updated live with an EWMA over trades. The signal says **BET** only when all of these hold:

1. the expected value per $1, after fees, is at least `MIN_EDGE` (4% by default)
2. the edge holds even if volatility is off by ±25%
3. there are at least `MIN_SECONDS_LEFT` seconds left
4. there was no high-impact headline in the last 10 minutes (when there is one, σ is also widened 1.5×)
5. the model doesn't disagree with the market by more than 25 points; a gap that large almost always means bad data, such as a wrong open price or a stale feed

The stake is quarter-Kelly, capped at `MAX_STAKE_FRACTION` of `BANKROLL`.

**Expect NO BET most of the time.** Polymarket prices usually track this same math closely. Edges, when they exist, are small and short-lived.

## Install (laptop)

Requires Python 3.11+.

```bash
git clone <this repo> && cd polymarketbitcoinpredictor
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env                                   # optional: edit settings
```

## Use

```bash
btcpredict                 # live dashboard (Ctrl+C to quit)
btcpredict backtest --days 14   # is the probability model calibrated? (free, ~30s)
btcpredict news            # latest scored headlines
btcpredict review          # paper-trading report from what the dashboard recorded
btcpredict snapshot        # current state as JSON
btcpredict ask "should I bet this window?"   # AI analyst Q&A (needs ANTHROPIC_API_KEY)
```

**Start with `btcpredict backtest`.** If the calibration table shows the model's 70% calls coming true about 70% of the time, the probabilities can be trusted. If they don't, adjust the model before relying on the edge numbers.

The dashboard also logs a snapshot every 5 seconds to `data/snapshots.sqlite` (model P(UP), Polymarket bid/ask, and the signal). **Leave it running for a week or two, then run `btcpredict review`.** It looks up how each window resolved (Polymarket's own result, falling back to exchange candles) and reports:

- **the signals' paper P&L:** the first BET in each window at the suggested stake, with win rate, ROI and max drawdown
- **a MIN_EDGE sweep:** $1 flat-bet ROI at 0%, 2%, 4% … 15% edge thresholds, for tuning `MIN_EDGE`
- **model vs market accuracy:** Brier scores for the model's P(UP) and for Polymarket's mid-price, overall and by time left. **If the market forecasts better than the model, any positive ROI is probably luck, so don't bet real money.**

## Cost

Everything is **free by default**: Coinbase/Binance public market data, Polymarket's public APIs, and public RSS feeds (plus an optional free CryptoPanic token).

The AI analyst is optional and uses the paid Anthropic API. It runs one briefing per window (about 4 minutes in), plus one whenever high-impact news lands. That's up to about 96 calls a day, on `claude-opus-5-5` at low effort with prompt caching. Leave `ANTHROPIC_API_KEY` empty to skip it.

## Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `PRICE_SOURCE` | `coinbase` | `coinbase` (works in the US) or `binance` (blocked from US IPs) |
| `ANTHROPIC_API_KEY` | empty | enables AI briefings and `ask` |
| `CLAUDE_MODEL` / `CLAUDE_EFFORT` | `claude-opus-5-5` / `low` | AI model and effort |
| `CRYPTOPANIC_TOKEN` | empty | optional free extra news source |
| `MIN_EDGE` | `0.04` | minimum EV per $1 after fees |
| `TAKER_FEE_RATE` | `0.03` | fee per share = rate × min(p, 1−p). **Check Polymarket's current fee schedule and set this to match.** |
| `KELLY_FRACTION`, `BANKROLL`, `MAX_STAKE_FRACTION` | `0.25`, `100`, `0.05` | stake sizing |
| `MIN_SECONDS_LEFT` | `45` | no signals in the final seconds |

## Known limitations

- **Settlement source.** Polymarket resolves on Chainlink BTC/USD. The dashboard streams that price from Polymarket's free real-time feed (`SETTLEMENT_FEED=chainlink`) and uses it for the current price. It takes the price to beat from the first Chainlink tick of each window. If you start the dashboard mid-window, it uses the exchange's candle open plus the measured Chainlink–exchange gap (shown as "exchange + basis est."). If the Chainlink feed drops, it falls back to the exchange price and says so in the header. Volatility is always measured on the exchange feed. For the first few windows, check the displayed "Open (to beat)" against Polymarket's page.
- **Fees.** The fee formula is an assumption; set `TAKER_FEE_RATE` (or edit `model/edge.py`) to match Polymarket's live fee schedule.
- **Fat tails.** The normal model can understate the chance of sudden jumps. The backtest's calibration table shows where it goes wrong.
- **Access.** Check that Polymarket is legally available where you live.

## Layout

```
src/btcpredict/
  data/prices.py       Coinbase/Binance candles (REST) + live trades (WebSocket)
  data/polymarket.py   Gamma API market lookup + CLOB order books
  data/chainlink.py    Chainlink BTC/USD settlement price (Polymarket real-time feed)
  data/news.py         RSS + CryptoPanic, keyword impact scoring
  model/windows.py     15-minute window math and market slugs
  model/volatility.py  candle-seeded EWMA volatility
  model/probability.py P(UP) and uncertainty band
  model/edge.py        fees, EV, Kelly, BET/NO BET rules
  engine.py            live state -> Snapshot
  agent.py             Claude analyst with tools over the Snapshot
  dashboard.py         Rich terminal UI
  recorder.py          SQLite snapshot log
  backtest.py          calibration backtest
  review.py            paper-trading P&L, MIN_EDGE sweep, model-vs-market Brier
```

Run the tests with `pip install -e ".[dev]" && pytest`.
