import math
import random

from btcpredict.backtest import evaluate, predict_windows
from btcpredict.data.prices import Candle


def synthetic_candles(days: float, sigma_annual: float = 0.5, seed: int = 7) -> list[Candle]:
    rng = random.Random(seed)
    sigma_min = sigma_annual / math.sqrt(365 * 24 * 60)
    start = 1_759_000_500 // 900 * 900  # aligned
    px, out = 60_000.0, []
    for i in range(int(days * 1440)):
        o = px
        px *= math.exp(rng.gauss(0, sigma_min))
        out.append(Candle(start + 60 * i, o, max(o, px), min(o, px), px))
    return out


def test_model_is_calibrated_on_random_walk():
    preds = predict_windows(synthetic_candles(days=20))
    report = evaluate(preds)
    assert report.windows > 1800
    assert report.brier < 0.20  # far better than a coin flip once the window is underway
    for b in report.bins:
        if b.n > 300:
            assert abs(b.hit_rate - b.mean_pred) < 0.05, b
    briers = dict((m, br) for m, _, br in report.by_minute)
    assert briers[14] < briers[1]  # later in the window = more certain


def test_incomplete_windows_are_skipped():
    candles = synthetic_candles(days=1)
    del candles[700]
    preds = predict_windows(candles)
    starts = {p.window_start for p in preds}
    missing_ts = candles[0].ts + 700 * 60
    assert all(not (s <= missing_ts < s + 900) for s in starts)
