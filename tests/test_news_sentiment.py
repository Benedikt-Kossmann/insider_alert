"""Tests for the financial-domain news sentiment analyser."""
import unittest

from insider_alert.feature_engine.news_features import _financial_sentiment


class TestFinancialSentiment(unittest.TestCase):
    """Unit tests for the custom financial sentiment lexicon."""

    def test_bullish_headline(self):
        score = _financial_sentiment("Company beats earnings and raises guidance")
        self.assertGreater(score, 0.0)

    def test_strongly_bullish(self):
        score = _financial_sentiment("Stock soars on blowout quarterly results")
        self.assertGreater(score, 0.3)

    def test_bearish_headline(self):
        score = _financial_sentiment("Company misses estimates and cuts outlook")
        self.assertLess(score, 0.0)

    def test_strongly_bearish(self):
        score = _financial_sentiment("Stock crashes amid fraud investigation")
        self.assertLess(score, -0.3)

    def test_neutral_headline(self):
        score = _financial_sentiment("Company announces new product line")
        # No strong sentiment words → near zero
        self.assertAlmostEqual(score, 0.0, delta=0.3)

    def test_empty_text(self):
        self.assertEqual(_financial_sentiment(""), 0.0)

    def test_none_text(self):
        # Should not raise
        self.assertEqual(_financial_sentiment(None), 0.0)

    def test_negation_flips(self):
        pos = _financial_sentiment("strong growth reported")
        neg = _financial_sentiment("not strong growth reported")
        # Negation should flip the 'strong' contribution
        self.assertGreater(pos, neg)

    def test_range_bounded(self):
        """Sentiment should be in [-1, 1]."""
        extreme_bull = _financial_sentiment(
            "soars surges rally breakout record high blowout crushes smashes"
        )
        extreme_bear = _financial_sentiment(
            "crash plunge collapse bankruptcy fraud investigation recession crisis"
        )
        self.assertGreaterEqual(extreme_bull, -1.0)
        self.assertLessEqual(extreme_bull, 1.0)
        self.assertGreaterEqual(extreme_bear, -1.0)
        self.assertLessEqual(extreme_bear, 1.0)

    def test_mixed_headline(self):
        """Mixed sentiment should be closer to zero."""
        score = _financial_sentiment("Company beats revenue but misses on profit")
        self.assertGreater(score, -0.5)
        self.assertLess(score, 0.5)

    def test_case_insensitive(self):
        lower = _financial_sentiment("stock surges on earnings beat")
        upper = _financial_sentiment("Stock SURGES on Earnings Beat")
        self.assertAlmostEqual(lower, upper, places=2)


class TestComputeNewsFeaturesNoTextBlob(unittest.TestCase):
    """Verify that news_features no longer depends on textblob."""

    def test_no_textblob_import(self):
        import insider_alert.feature_engine.news_features as mod
        # The module should not have TextBlob in its namespace
        self.assertFalse(hasattr(mod, "TextBlob"))
        # Source should not reference textblob
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from textblob", source.lower())
        self.assertNotIn("import textblob", source.lower())


if __name__ == "__main__":
    unittest.main()
