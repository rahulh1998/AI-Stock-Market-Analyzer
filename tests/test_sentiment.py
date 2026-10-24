"""Offline tests for news entity matching, bucketing and scorer fallback.

No network. The RSS payloads here are fixtures built from titles actually observed on
Google News for these queries, including the failure cases that motivated the module.
"""
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import patch

import pandas as pd

from src.sentiment_engine import finbert_scorer
from src.sentiment_engine.finbert_scorer import HeadlineScorer, lexicon_score
from src.sentiment_engine.multi_horizon import (SentimentService, bar_return_pct,
                                                detect_divergence, label_for)
from src.sentiment_engine.news_sources import (Article, EntityMatcher, fetch_articles,
                                               normalise_title, parse_published, parse_rss)

INSTRUMENTS = {
    "RELIANCE": {"company_name": "Reliance Industries Limited",
                 "aliases": ["Reliance Industries Limited", "Reliance Industries", "RIL"]},
    "SBIN": {"company_name": "State Bank of India",
             "aliases": ["State Bank of India", "SBI", "State Bank"]},
    "SBICARD": {"company_name": "SBI Cards and Payment Services Limited",
                "aliases": ["SBI Cards and Payment Services Limited",
                            "SBI Cards and Payment Services", "SBI Cards"]},
    "SBILIFE": {"company_name": "SBI Life Insurance Company Limited",
                "aliases": ["SBI Life Insurance Company Limited",
                            "SBI Life Insurance Company", "SBI Life"]},
    "TATASTEEL": {"company_name": "Tata Steel Limited",
                  "aliases": ["Tata Steel Limited", "Tata Steel"]},
    "JSWSTEEL": {"company_name": "JSW Steel Limited",
                 "aliases": ["JSW Steel Limited", "JSW Steel"]},
}


def rss(items) -> bytes:
    """items: (title, datetime) pairs -> an RSS payload."""
    body = "".join(
        f"<item><title>{title}</title>"
        f"<pubDate>{format_datetime(when)}</pubDate>"
        f"<link>https://example.test/{index}</link>"
        f"<source>Test Wire</source></item>"
        for index, (title, when) in enumerate(items))
    return f"<rss><channel>{body}</channel></rss>".encode("utf-8")


class FakeResponse:
    def __init__(self, content, status_code=200):
        self.content, self.status_code = content, status_code


class FakeSession:
    """Returns the same payload for every query, and counts calls."""

    def __init__(self, payload, status_code=200):
        self.payload, self.status_code, self.calls = payload, status_code, []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return FakeResponse(self.payload, self.status_code)


class EntityMatchingTests(unittest.TestCase):
    def setUp(self):
        self.matcher = EntityMatcher(INSTRUMENTS)

    def test_sibling_company_is_not_matched(self):
        """The bug this module exists for: RELIANCE must not claim Reliance Power news."""
        relevant, _, reason = self.matcher.attribute(
            "Reliance Power shares surge 8% on order win", "RELIANCE")
        self.assertFalse(relevant)
        self.assertIn("no instrument alias", reason)

    def test_longest_alias_wins_within_the_sbi_family(self):
        title = "SBI Cards and Payment Services Q2 profit rises 12%"
        self.assertEqual(dict(self.matcher.matches(title)), {"SBICARD": "SBI Cards and Payment Services"})
        self.assertFalse(self.matcher.attribute(title, "SBIN")[0])
        self.assertTrue(self.matcher.attribute(title, "SBICARD")[0])

    def test_sbi_life_does_not_leak_to_sbin(self):
        title = "SBI Life Insurance reports strong APE growth"
        self.assertFalse(self.matcher.attribute(title, "SBIN")[0])
        self.assertTrue(self.matcher.attribute(title, "SBILIFE")[0])

    def test_parent_still_matches_its_own_news(self):
        self.assertTrue(self.matcher.attribute("SBI raises Rs 10,000 crore via bonds", "SBIN")[0])
        self.assertTrue(self.matcher.attribute("State Bank of India cuts MCLR", "SBIN")[0])

    def test_ticker_symbol_matches_case_sensitively(self):
        """NSE:RELIANCE is the symbol; 'Reliance' in prose is not."""
        self.assertTrue(self.matcher.attribute("Q2 review of NSE:RELIANCE earnings", "RELIANCE")[0])
        self.assertFalse(self.matcher.attribute("Reliance Power declares dividend", "RELIANCE")[0])

    def test_word_boundaries_prevent_substring_hits(self):
        matcher = EntityMatcher({"ITC": {"company_name": "ITC Limited", "aliases": ["ITC Limited"]}})
        self.assertEqual(matcher.matches("Engineers flip the switch on a new plant"), [])

    def test_boilerplate_quote_pages_are_dropped(self):
        for title in ("Reliance Industries Share Price - Live NSE",
                      "Tata Steel Limited (TATASTEEL.NS) Stock Price, News, Quote & History"):
            with self.subTest(title=title):
                self.assertFalse(self.matcher.attribute(title, "RELIANCE")[0])

    def test_round_up_headlines_are_dropped(self):
        relevant, _, reason = self.matcher.attribute(
            "Top stocks to watch today: Reliance Industries, IREDA in focus", "RELIANCE")
        self.assertFalse(relevant)
        self.assertIn("round-up", reason)

    def test_multi_company_headline_is_attributed_to_both(self):
        found = dict(self.matcher.matches("Tata Steel and JSW Steel gain on duty"))
        self.assertEqual(set(found), {"TATASTEEL", "JSWSTEEL"})


class TitleAndDateTests(unittest.TestCase):
    def test_dedup_key_ignores_publisher_suffix(self):
        self.assertEqual(normalise_title("Reliance Industries gains 3% - Mint"),
                         normalise_title("Reliance Industries gains 3% – Business Standard"))

    def test_dedup_key_ignores_punctuation_and_case(self):
        self.assertEqual(normalise_title("Tata Steel's Q2: loss widens!"),
                         normalise_title("tata steels q2 loss widens"))

    def test_rfc2822_pubdate_parses_to_the_right_instant(self):
        parsed = parse_published("Fri, 23 Oct 2026 10:39:45 GMT")
        self.assertEqual(parsed, datetime(2026, 10, 23, 10, 39, 45, tzinfo=timezone.utc))

    def test_iso_pubdate_is_accepted(self):
        self.assertEqual(parse_published("2026-10-23T10:39:45Z"),
                         datetime(2026, 10, 23, 10, 39, 45, tzinfo=timezone.utc))

    def test_missing_pubdate_raises_rather_than_inventing_one(self):
        with self.assertRaises(ValueError):
            parse_published("")

    def test_parse_rss_skips_untitled_items(self):
        payload = b"<rss><channel><item><title></title></item>" \
                  b"<item><title>Real</title><pubDate>x</pubDate></item></channel></rss>"
        self.assertEqual([t for t, *_ in parse_rss(payload)], ["Real"])


class FetchTests(unittest.TestCase):
    def test_relevant_articles_kept_and_noise_reported(self):
        now = datetime.now(timezone.utc)
        session = FakeSession(rss([
            ("Reliance Industries shares jump 3% on Jio IPO", now - timedelta(minutes=30)),
            ("Reliance Power shares surge 8%", now - timedelta(hours=2)),
            ("Reliance Industries Share Price - Live NSE", now - timedelta(hours=3)),
            ("Reliance Industries shares jump 3% on Jio IPO - Mint", now - timedelta(hours=4)),
            ("Top stocks to watch today: Reliance Industries, IREDA", now - timedelta(hours=5)),
        ]))
        articles, diagnostics = fetch_articles("RELIANCE", INSTRUMENTS, session=session)
        self.assertEqual([a.title for a in articles],
                         ["Reliance Industries shares jump 3% on Jio IPO"])
        self.assertEqual(diagnostics["kept"], 1)
        self.assertIn("duplicate title", diagnostics["dropped"])
        self.assertIn("boilerplate quote/listing page", diagnostics["dropped"])

    def test_query_never_uses_the_bare_ticker_alone(self):
        session = FakeSession(rss([]))
        _, diagnostics = fetch_articles("RELIANCE", INSTRUMENTS, session=session)
        for query in diagnostics["queries"]:
            self.assertNotEqual(query.strip(), "RELIANCE")
            self.assertIn("Reliance Industries", query)

    def test_http_error_is_reported_not_raised(self):
        session = FakeSession(b"", status_code=503)
        articles, diagnostics = fetch_articles("RELIANCE", INSTRUMENTS, session=session)
        self.assertEqual(articles, [])
        self.assertTrue(diagnostics["errors"])

    def test_malformed_payload_is_reported_not_raised(self):
        session = FakeSession(b"<<not xml")
        articles, diagnostics = fetch_articles("RELIANCE", INSTRUMENTS, session=session)
        self.assertEqual(articles, [])
        self.assertTrue(diagnostics["errors"])

    def test_articles_are_newest_first(self):
        now = datetime.now(timezone.utc)
        session = FakeSession(rss([
            ("Reliance Industries falls on crude", now - timedelta(hours=10)),
            ("Reliance Industries gains on Jio", now - timedelta(hours=1)),
        ]))
        articles, _ = fetch_articles("RELIANCE", INSTRUMENTS, session=session)
        self.assertGreater(articles[0].published, articles[1].published)


class ScorerTests(unittest.TestCase):
    def test_lexicon_direction(self):
        self.assertGreater(lexicon_score("Profit surges as company beats estimates"), 0)
        self.assertLess(lexicon_score("Loss widens as Sebi opens probe"), 0)
        self.assertEqual(lexicon_score("Company announces board meeting date"), 0.0)

    def test_lexicon_does_not_fire_on_52_week_low_as_bullish(self):
        """'high'/'low' were previously scored as sentiment words in both directions."""
        self.assertLess(lexicon_score("Stock hits 52-week low"), 0)
        self.assertGreater(lexicon_score("Stock hits record high"), 0)

    def test_unavailable_finbert_degrades_to_lexicon_and_says_so(self):
        titles = ["Profit surges as company beats estimates"]
        with patch.object(finbert_scorer, "load_finbert", return_value=(None, None)):
            scores, scorer = finbert_scorer.score_batch(titles)
        self.assertEqual(scorer, "lexicon_fallback")
        self.assertAlmostEqual(scores[0], lexicon_score(titles[0]))

    def test_inference_failure_degrades_to_lexicon(self):
        class Boom:
            def __call__(self, *a, **k):
                raise RuntimeError("inference exploded")

        with patch.object(finbert_scorer, "load_finbert", return_value=(Boom(), Boom())):
            scores, scorer = finbert_scorer.score_batch(["Loss widens as Sebi opens probe"])
        self.assertEqual(scorer, "lexicon_fallback")
        self.assertLess(scores[0], 0)

    def test_empty_batch_is_safe(self):
        self.assertEqual(finbert_scorer.score_batch([])[0], [])

    def test_cache_key_is_scorer_independent(self):
        """Keying on the title alone is what lets the cached path skip the model load."""
        self.assertEqual(HeadlineScorer.cache_key("abc"), HeadlineScorer.cache_key("abc"))
        self.assertNotEqual(HeadlineScorer.cache_key("abc"), HeadlineScorer.cache_key("abd"))


class HorizonTests(unittest.TestCase):
    def test_buckets_are_cumulative_and_absence_is_not_zero(self):
        now = datetime.now(timezone.utc)
        session = FakeSession(rss([
            ("Reliance Industries gains on Jio", now - timedelta(days=3)),
        ]))
        service = SentimentService(use_cache=False)
        try:
            with patch("src.sentiment_engine.multi_horizon.fetch_articles",
                       side_effect=lambda t, *a, **k: fetch_articles(t, INSTRUMENTS,
                                                                     session=session)):
                reading = service.reading("RELIANCE", now=now)
        finally:
            service.close()
        self.assertIsNone(reading["sentiment_1h"])
        self.assertIsNone(reading["sentiment_1d"])
        self.assertIsNotNone(reading["sentiment_1w"])
        self.assertEqual(reading["counts"], {"1h": 0, "1d": 0, "1w": 1})
        self.assertEqual(reading["label_1h"], "NO_DATA")

    def test_label_thresholds(self):
        self.assertEqual(label_for(0.25), "BULLISH")
        self.assertEqual(label_for(-0.25), "BEARISH")
        self.assertEqual(label_for(0.0), "NEUTRAL")
        self.assertEqual(label_for(None), "NO_DATA")


class DivergenceTests(unittest.TestCase):
    def test_bearish_divergence_on_positive_news_and_falling_price(self):
        self.assertIn("BEARISH_DIVERGENCE", detect_divergence(0.8, 0.5, -1.2))

    def test_bullish_absorption_on_negative_news_and_rising_price(self):
        self.assertIn("BULLISH_ABSORPTION", detect_divergence(-0.8, -0.5, 1.2))

    def test_agreement_is_congruent(self):
        self.assertEqual(detect_divergence(0.8, 0.5, 1.5), "CONGRUENT")

    def test_missing_price_is_unknown_not_congruent(self):
        self.assertEqual(detect_divergence(0.8, 0.5, None), "UNKNOWN_NO_PRICE")

    def test_bar_return_uses_completed_bars_only(self):
        frame = pd.DataFrame({"close": [100.0, 102.0]})
        self.assertAlmostEqual(bar_return_pct(frame), 2.0)
        self.assertIsNone(bar_return_pct(pd.DataFrame({"close": [100.0]})))
        self.assertIsNone(bar_return_pct(None))


if __name__ == "__main__":
    unittest.main()
