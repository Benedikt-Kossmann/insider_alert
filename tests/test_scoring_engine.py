"""Tests for the scoring engine."""
import unittest


def _make_signals(scores: dict | None = None) -> list[dict]:
    if scores is None:
        scores = {
            "price_anomaly": 50.0,
            "volume_anomaly": 70.0,
            "orderflow_anomaly": 30.0,
            "options_anomaly": 80.0,
            "insider_signal": 60.0,
            "event_leadup": 40.0,
            "news_divergence": 20.0,
            "accumulation_pattern": 55.0,
        }
    return [
        {"signal_type": sig_type, "score": score, "flags": [f"{sig_type} flag"]}
        for sig_type, score in scores.items()
    ]


class TestComputeScore(unittest.TestCase):
    def test_ticker_score_fields(self):
        from insider_alert.scoring_engine.scorer import compute_score, TickerScore
        result = compute_score("AAPL", _make_signals())
        self.assertIsInstance(result, TickerScore)
        self.assertEqual(result.ticker, "AAPL")
        self.assertIsInstance(result.total_score, float)
        self.assertIsInstance(result.sub_scores, dict)
        self.assertIsInstance(result.flags, list)

    def test_score_in_range(self):
        from insider_alert.scoring_engine.scorer import compute_score
        result = compute_score("MSFT", _make_signals())
        self.assertGreaterEqual(result.total_score, 0.0)
        self.assertLessEqual(result.total_score, 100.0)

    def test_weighted_average(self):
        """Verify that the total score is a weighted average of sub-signals."""
        from insider_alert.scoring_engine.scorer import compute_score, DEFAULT_WEIGHTS
        signals = _make_signals()
        result = compute_score("AAPL", signals)

        total_weight = sum(DEFAULT_WEIGHTS.values())
        expected = sum(
            result.sub_scores.get(sig_type, 0.0) * w
            for sig_type, w in DEFAULT_WEIGHTS.items()
        ) / total_weight
        self.assertAlmostEqual(result.total_score, expected, places=4)

    def test_custom_weights(self):
        from insider_alert.scoring_engine.scorer import compute_score
        custom = {"price_anomaly": 1.0}
        signals = [{"signal_type": "price_anomaly", "score": 75.0, "flags": []}]
        result = compute_score("AAPL", signals, weights=custom)
        self.assertAlmostEqual(result.total_score, 75.0, places=4)

    def test_missing_signals_handled(self):
        """Score should be computed even if some signal types are missing."""
        from insider_alert.scoring_engine.scorer import compute_score
        signals = [{"signal_type": "price_anomaly", "score": 80.0, "flags": ["flag1"]}]
        result = compute_score("NVDA", signals)
        self.assertGreaterEqual(result.total_score, 0.0)
        self.assertLessEqual(result.total_score, 100.0)

    def test_empty_signals(self):
        from insider_alert.scoring_engine.scorer import compute_score
        result = compute_score("AAPL", [])
        self.assertEqual(result.total_score, 0.0)
        self.assertEqual(result.flags, [])

    def test_flags_aggregated(self):
        from insider_alert.scoring_engine.scorer import compute_score
        signals = [
            {"signal_type": "price_anomaly", "score": 50.0, "flags": ["flag_a", "flag_b"]},
            {"signal_type": "volume_anomaly", "score": 60.0, "flags": ["flag_c"]},
        ]
        result = compute_score("AAPL", signals)
        self.assertIn("flag_a", result.flags)
        self.assertIn("flag_b", result.flags)
        self.assertIn("flag_c", result.flags)

    def test_score_capped_at_100(self):
        from insider_alert.scoring_engine.scorer import compute_score
        signals = [
            {"signal_type": sig, "score": 150.0, "flags": []}
            for sig in ["price_anomaly", "volume_anomaly", "orderflow_anomaly",
                        "options_anomaly", "insider_signal", "event_leadup",
                        "news_divergence", "accumulation_pattern"]
        ]
        result = compute_score("AAPL", signals)
        self.assertLessEqual(result.total_score, 100.0)


if __name__ == "__main__":
    unittest.main()
