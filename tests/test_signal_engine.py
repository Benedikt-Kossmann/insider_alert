"""Tests for signal engine modules."""
import unittest


def _make_price_features(return_1d=0.02, return_5d=0.05, zscore=2.5):
    return {
        "return_1d": return_1d,
        "return_3d": 0.03,
        "return_5d": return_5d,
        "return_10d": 0.08,
        "daily_return_zscore": zscore,
        "gap_up_count_5d": 2,
        "gap_down_count_5d": 0,
    }


def _make_volume_features(rvol=2.5, zscore=2.0, tight_flag=0):
    return {
        "volume_rvol_20d": rvol,
        "volume_zscore_20d": zscore,
        "close_volume_ratio": 0.0001,
        "intraday_volume_acceleration": 0.3,
        "tight_range_high_volume_flag": tight_flag,
    }


def _make_orderflow_features():
    return {
        "bid_ask_imbalance": 0.5,
        "aggressive_buy_ratio": 0.7,
        "large_trade_count": 0.4,
        "iceberg_suspect_score": 0.3,
        "absorption_score": 1,
        "vwap_accumulation_score": 0.2,
    }


def _make_options_features():
    return {
        "call_volume_zscore": 3.0,
        "put_volume_zscore": 0.5,
        "put_call_ratio_change": 0.2,
        "iv_change_1d": 0.0,
        "short_dated_otm_call_score": 0.6,
        "block_trade_score": 0.4,
        "sweep_order_score": 0.5,
        "open_interest_change": 0.1,
    }


def _make_insider_features():
    return {
        "insider_buy_count_30d": 4,
        "insider_sell_count_30d": 1,
        "insider_buy_value_30d": 500000.0,
        "insider_cluster_score": 0.8,
        "insider_role_weighted_score": 0.6,
    }


def _make_event_features():
    return {
        "days_to_earnings": 5,
        "days_to_corporate_event": 5,
        "pre_event_return_score": 0.4,
        "pre_event_volume_score": 0.6,
        "pre_event_options_score": 0.5,
    }


def _make_news_features():
    return {
        "news_count_24h": 0,
        "news_sentiment_score": 0.0,
        "public_catalyst_strength": 0.0,
        "price_news_divergence_score": 0.06,
    }


def _make_accumulation_features():
    return {
        "range_compression_score": 0.5,
        "higher_lows_score": 0.6,
        "volume_under_resistance_score": 0.7,
        "wyckoff_accumulation_score": 0.6,
    }


class _SignalTestMixin:
    """Mixin with common signal assertions."""

    def _assert_signal(self, result: dict, expected_type: str):
        self.assertIn("signal_type", result)
        self.assertIn("score", result)
        self.assertIn("flags", result)
        self.assertEqual(result["signal_type"], expected_type)
        self.assertIsInstance(result["score"], float)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)
        self.assertIsInstance(result["flags"], list)


class TestPriceSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
        result = compute_price_anomaly_signal(_make_price_features())
        self._assert_signal(result, "price_anomaly")

    def test_zero_features_give_low_score(self):
        from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
        result = compute_price_anomaly_signal({
            "daily_return_zscore": 0.0, "return_5d": 0.0,
            "gap_up_count_5d": 0, "gap_down_count_5d": 0,
        })
        self.assertEqual(result["score"], 0.0)

    def test_high_features_give_high_score(self):
        from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
        result = compute_price_anomaly_signal({
            "daily_return_zscore": 10.0, "return_5d": 0.5,
            "gap_up_count_5d": 5, "gap_down_count_5d": 0,
        })
        self.assertGreater(result["score"], 50.0)


class TestVolumeSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.volume_signal import compute_volume_anomaly_signal
        result = compute_volume_anomaly_signal(_make_volume_features())
        self._assert_signal(result, "volume_anomaly")

    def test_score_bounds(self):
        from insider_alert.signal_engine.volume_signal import compute_volume_anomaly_signal
        for rvol in (0.5, 1.0, 3.0, 10.0):
            result = compute_volume_anomaly_signal(_make_volume_features(rvol=rvol))
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 100.0)


class TestOrderflowSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.orderflow_signal import compute_orderflow_anomaly_signal
        result = compute_orderflow_anomaly_signal(_make_orderflow_features())
        self._assert_signal(result, "orderflow_anomaly")


class TestOptionsSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.options_signal import compute_options_anomaly_signal
        result = compute_options_anomaly_signal(_make_options_features())
        self._assert_signal(result, "options_anomaly")

    def test_zero_features(self):
        from insider_alert.signal_engine.options_signal import compute_options_anomaly_signal
        result = compute_options_anomaly_signal({})
        self.assertEqual(result["score"], 0.0)


class TestInsiderSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.insider_signal import compute_insider_signal
        result = compute_insider_signal(_make_insider_features())
        self._assert_signal(result, "insider_signal")

    def test_no_buys_zero_score(self):
        from insider_alert.signal_engine.insider_signal import compute_insider_signal
        result = compute_insider_signal({
            "insider_buy_count_30d": 0,
            "insider_cluster_score": 0.0,
            "insider_role_weighted_score": 0.0,
        })
        self.assertEqual(result["score"], 0.0)


class TestEventSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.event_signal import compute_event_leadup_signal
        result = compute_event_leadup_signal(_make_event_features())
        self._assert_signal(result, "event_leadup")

    def test_far_earnings_low_score(self):
        from insider_alert.signal_engine.event_signal import compute_event_leadup_signal
        result = compute_event_leadup_signal({
            "days_to_earnings": 999,
            "pre_event_return_score": 0.0,
            "pre_event_volume_score": 0.0,
            "pre_event_options_score": 0.0,
        })
        self.assertEqual(result["score"], 0.0)


class TestNewsSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.news_signal import compute_news_divergence_signal
        result = compute_news_divergence_signal(_make_news_features())
        self._assert_signal(result, "news_divergence")


class TestAccumulationSignal(_SignalTestMixin, unittest.TestCase):
    def test_structure(self):
        from insider_alert.signal_engine.accumulation_signal import compute_accumulation_signal
        result = compute_accumulation_signal(_make_accumulation_features())
        self._assert_signal(result, "accumulation_pattern")

    def test_zero_features(self):
        from insider_alert.signal_engine.accumulation_signal import compute_accumulation_signal
        result = compute_accumulation_signal({
            "wyckoff_accumulation_score": 0.0,
            "higher_lows_score": 0.0,
            "range_compression_score": 0.0,
        })
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
