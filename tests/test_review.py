import sqlite3

import pytest

from btcpredict import review
from btcpredict.data.chainlink import parse_rtds


def row(w, ts, p_up, ask_up, ask_down, left, action="NO BET", stake=0.0, bid_up=None):
    return dict(window_start=w, ts=ts, p_up=p_up, ask_up=ask_up, ask_down=ask_down, seconds_left=left,
                action=action, stake=stake, bid_up=bid_up if bid_up is not None else ask_up - 0.02)


ROWS = [
    # window 0: first BET UP at 0.60, UP wins
    row(0, 1, 0.55, 0.55, 0.47, 800),
    row(0, 2, 0.75, 0.60, 0.42, 500, "BET UP", 5.0),
    row(0, 3, 0.90, 0.80, 0.22, 300, "BET UP", 5.0),  # later signal in same window is ignored
    # window 900: BET DOWN at 0.40, but UP wins -> loss
    row(900, 4, 0.30, 0.62, 0.40, 600, "BET DOWN", 4.0),
    # window 1800: no signal
    row(1800, 5, 0.50, 0.51, 0.51, 600),
]
OUTCOMES = {0: 1, 900: 1, 1800: 0}


def test_recorded_trades_and_summary():
    trades = review.recorded_trades(ROWS, OUTCOMES, fee_rate=0.0)
    assert [(t.window_start, t.side, t.won) for t in trades] == [(0, "UP", True), (900, "DOWN", False)]
    s = review.summarize(trades)
    assert s.pnl == pytest.approx(5.0 * (1 / 0.60 - 1) - 4.0)
    assert s.win_rate == 0.5 and s.staked == 9.0


def test_threshold_sweep_counts_fall_with_threshold():
    sweep = dict(review.threshold_sweep(ROWS, OUTCOMES, fee_rate=0.0))
    assert sweep[0.0].n >= sweep[0.10].n >= sweep[0.15].n
    assert sweep[0.15].n == 2  # windows 0 (+25%) and 900 (+75% DOWN)


def test_brier_vs_market():
    rows = review.brier_vs_market(ROWS, OUTCOMES)
    assert rows[0].label == "all" and rows[0].n == len(ROWS)
    assert all(0 <= r.model <= 1 and 0 <= r.market <= 1 for r in rows)


def test_load_rows_from_recorder_schema(tmp_path):
    from btcpredict.recorder import SCHEMA
    db = sqlite3.connect(tmp_path / "x.sqlite")
    db.executescript(SCHEMA)
    db.execute("INSERT INTO snapshots VALUES (1,0,1,1,600,0.5,0.5,0.5,0.5,0.4,0.4,'NO BET',0,0,'chainlink','chainlink')")
    assert review.load_rows(db)[0]["price_source"] == "chainlink"


def test_parse_rtds_messages():
    upd = '{"topic":"crypto_prices_chainlink","type":"update","payload":{"symbol":"btc/usd","timestamp":1759420800123,"value":61234.5}}'
    assert list(parse_rtds(upd)) == [(1759420800.123, 61234.5)]
    snap = '{"topic":"crypto_prices_chainlink","payload":{"symbol":"btc/usd","data":[{"timestamp":1759420799000,"value":1},{"timestamp":1759420800000,"value":2}]}}'
    assert [p for _, p in parse_rtds(snap)] == [1.0, 2.0]
    assert list(parse_rtds("PONG")) == []
    assert list(parse_rtds('{"topic":"crypto_prices_chainlink","payload":{"symbol":"eth/usd","timestamp":1,"value":2}}')) == []
