"""Tests for feature engine modules."""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


def _make_ohlcv(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    base = 100.0
    closes = base + np.cumsum(rng.normal(0, 1, n))
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    opens = closes - rng.normal(0, 0.5, n)
    volumes = rng.integers(500_000, 5_000_000, n).astype(float)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=pd.DatetimeIndex(dates))


class TestPriceFeatures(unittest.TestCase):
    def setUp(self):
        self.ohlcv = _make_ohlcv(30)

    def test_returns_expected_keys(self):
        from insider_alert.feature_engine.price_features import compute_price_features
        result = compute_price_features(self.ohlcv)
        expected_keys = {
            "return_1d", "return_3d", "return_5d", "return_10d",
            "daily_return_zscore", "gap_up_count_5d", "gap_down_count_5d",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_returns_are_floats(self):
        from insider_alert.feature_engine.price_features import compute_price_features
        result = compute_price_features(self.ohlcv)
        for key in ("return_1d", "return_3d", "return_5d", "return_10d"):
            self.assertIsInstance(result[key], float, f"{key} should be float")

    def test_zscore_formula(self):
        from insider_alert.feature_engine.price_features import compute_price_features
        result = compute_price_features(self.ohlcv)
        self.assertIsInstance(result["daily_return_zscore"], float)

    def test_gap_counts_non_negative(self):
        from insider_alert.feature_engine.price_features import compute_price_features
        result = compute_price_features(self.ohlcv)
        self.assertGreaterEqual(result["gap_up_count_5d"], 0)
        self.assertGreaterEqual(result["gap_down_count_5d"], 0)
        self.assertLessEqual(result["gap_up_count_5d"], 5)
        self.assertLessEqual(result["gap_down_count_5d"], 5)

    def test_empty_dataframe(self):
        from insider_alert.feature_engine.price_features import compute_price_features
        result = compute_price_features(pd.DataFrame())
        self.assertEqual(result["return_1d"], 0.0)
        self.assertEqual(result["daily_return_zscore"], 0.0)

    def test_high_return_large_ohlcv(self):
        """Ensure large price movements produce non-zero returns."""
        from insider_alert.feature_engine.price_features import compute_price_features
        df = _make_ohlcv(25)
        df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-2]["close"] * 1.1
        result = compute_price_features(df)
        self.assertNotEqual(result["return_1d"], 0.0)


class TestVolumeFeatures(unittest.TestCase):
    def setUp(self):
        self.ohlcv = _make_ohlcv(30)

    def test_returns_expected_keys(self):
        from insider_alert.feature_engine.volume_features import compute_volume_features
        result = compute_volume_features(self.ohlcv)
        expected_keys = {
            "volume_rvol_20d", "volume_zscore_20d", "close_volume_ratio",
            "intraday_volume_acceleration", "tight_range_high_volume_flag",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_rvol_positive(self):
        from insider_alert.feature_engine.volume_features import compute_volume_features
        result = compute_volume_features(self.ohlcv)
        self.assertGreater(result["volume_rvol_20d"], 0)

    def test_tight_range_flag_is_binary(self):
        from insider_alert.feature_engine.volume_features import compute_volume_features
        result = compute_volume_features(self.ohlcv)
        self.assertIn(result["tight_range_high_volume_flag"], (0, 1))

    def test_empty_dataframe(self):
        from insider_alert.feature_engine.volume_features import compute_volume_features
        result = compute_volume_features(pd.DataFrame())
        self.assertEqual(result["volume_rvol_20d"], 1.0)


class TestOrderflowFeatures(unittest.TestCase):
    def setUp(self):
        self.ohlcv = _make_ohlcv(30)

    def test_returns_expected_keys(self):
        from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
        result = compute_orderflow_features(self.ohlcv)
        expected_keys = {
            "bid_ask_imbalance", "aggressive_buy_ratio", "large_trade_count",
            "iceberg_suspect_score", "absorption_score", "vwap_accumulation_score",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_imbalance_range(self):
        from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
        result = compute_orderflow_features(self.ohlcv)
        self.assertGreaterEqual(result["bid_ask_imbalance"], -1.0)
        self.assertLessEqual(result["bid_ask_imbalance"], 1.0)

    def test_aggressive_buy_ratio_range(self):
        from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
        result = compute_orderflow_features(self.ohlcv)
        self.assertGreaterEqual(result["aggressive_buy_ratio"], 0.0)
        self.assertLessEqual(result["aggressive_buy_ratio"], 1.0)

    def test_empty_dataframe(self):
        from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
        result = compute_orderflow_features(pd.DataFrame())
        self.assertEqual(result["bid_ask_imbalance"], 0.0)


class TestAccumulationFeatures(unittest.TestCase):
    def setUp(self):
        self.ohlcv = _make_ohlcv(30)

    def test_returns_expected_keys(self):
        from insider_alert.feature_engine.accumulation_features import compute_accumulation_features
        result = compute_accumulation_features(self.ohlcv)
        expected_keys = {
            "range_compression_score", "higher_lows_score",
            "volume_under_resistance_score", "wyckoff_accumulation_score",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_scores_in_range(self):
        from insider_alert.feature_engine.accumulation_features import compute_accumulation_features
        result = compute_accumulation_features(self.ohlcv)
        for key, val in result.items():
            self.assertGreaterEqual(val, 0.0, f"{key} should be >= 0")
            self.assertLessEqual(val, 1.0, f"{key} should be <= 1")

    def test_wyckoff_is_mean_of_components(self):
        from insider_alert.feature_engine.accumulation_features import compute_accumulation_features
        result = compute_accumulation_features(self.ohlcv)
        expected_wyckoff = np.mean([
            result["range_compression_score"],
            result["higher_lows_score"],
            result["volume_under_resistance_score"],
        ])
        self.assertAlmostEqual(result["wyckoff_accumulation_score"], expected_wyckoff, places=5)

    def test_empty_dataframe(self):
        from insider_alert.feature_engine.accumulation_features import compute_accumulation_features
        result = compute_accumulation_features(pd.DataFrame())
        self.assertEqual(result["wyckoff_accumulation_score"], 0.0)


class TestZscoreCorrectness(unittest.TestCase):
    def test_zscore_high_spike(self):
        """A large upward spike on the last day should yield a high z-score."""
        from insider_alert.feature_engine.price_features import compute_price_features
        n = 25
        closes = np.ones(n) * 100.0
        closes[-1] = 110.0
        opens = closes - 0.5
        highs = closes + 0.5
        lows = closes - 0.5
        volumes = np.ones(n) * 1_000_000.0
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })
        result = compute_price_features(df)
        self.assertGreater(abs(result["daily_return_zscore"]), 1.0)


if __name__ == "__main__":
    unittest.main()
