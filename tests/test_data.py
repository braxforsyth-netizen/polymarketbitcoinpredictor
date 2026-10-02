import json
import time

from btcpredict.data.news import NewsItem, news_shock, score_headline
from btcpredict.data.polymarket import parse_book, parse_market


def test_parse_market_handles_json_strings_and_order():
    m = {
        "slug": "btc-updown-15m-1759420800",
        "question": "Bitcoin Up or Down?",
        "outcomes": json.dumps(["Down", "Up"]),
        "clobTokenIds": json.dumps(["tokDown", "tokUp"]),
        "outcomePrices": json.dumps(["0.45", "0.55"]),
    }
    info = parse_market(m)
    assert info.up_token == "tokUp" and info.down_token == "tokDown"
    assert info.up_mid == 0.55 and info.down_mid == 0.45
    assert parse_market({"outcomes": '["Yes","No"]', "clobTokenIds": '["a","b"]'}) is None


def test_parse_book_best_levels_regardless_of_sort():
    book = parse_book(
        {
            "bids": [{"price": "0.40", "size": "10"}, {"price": "0.47", "size": "5"}],
            "asks": [{"price": "0.60", "size": "9"}, {"price": "0.52", "size": "7"}, {"price": "0.52", "size": "3"}],
        }
    )
    assert book.best_bid == 0.47 and book.best_ask == 0.52 and book.ask_size == 10
    empty = parse_book({"bids": [], "asks": []})
    assert empty.best_ask is None and empty.best_bid is None


def test_score_headline():
    assert score_headline("Taylor Swift releases new album") is None
    assert score_headline("Bitcoin plunges as exchange hacked for $200M") == ("high", "bearish")
    assert score_headline("Bitcoin ETF inflows surge to record") [0] == "high"
    impact, lean = score_headline("Fed's Powell hints at rate cut; bitcoin traders cheer")
    assert impact == "high" and lean == "bullish"
    assert score_headline("Bitcoin miners report quarterly earnings")[0] == "low"


def test_news_shock_window():
    now = time.time()
    fresh = NewsItem("X", "t", "", now - 120, "high", "bearish")
    old = NewsItem("X", "t", "", now - 3600, "high", "bearish")
    minor = NewsItem("X", "t", "", now - 60, "low", "neutral")
    assert news_shock([fresh], now=now)
    assert not news_shock([old, minor], now=now)
