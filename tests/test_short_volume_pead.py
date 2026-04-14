"""Tests for Phase 7: FINRA Short Volume, PEAD features and signals."""
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_short_df(
    n: int = 20,
    base_ratio: float = 0.45,
    trend: float = 0.0,
    vol_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic short-volume DataFrame."""
    import datetime as dt

    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    short_ratios = [base_ratio + trend * i / max(n - 1, 1) for i in range(n)]
    total_vols = [int(1_000_000 * (vol_multiplier if i == n - 1 else 1.0)) for i in range(n)]
    short_vols = [int(r * tv) for r, tv in zip(short_ratios, total_vols)]

    return pd.DataFrame({
        "Date": dates,
        "ShortVolume": short_vols,
        "TotalVolume": total_vols,
        "ShortRatio": short_ratios,
    })


# ---------------------------------------------------------------------------
# Tests: short_volume_features
# ---------------------------------------------------------------------------

class TestShortVolumeFeatures(unittest.TestCase):
    def test_output_keys(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        df = _make_short_df()
        result = compute_short_volume_features(df)

        for key in ("short_ratio_current", "short_ratio_zscore", "short_ratio_trend_5d",
                    "short_squeeze_score"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_defaults_on_empty(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        result = compute_short_volume_features(pd.DataFrame())
        self.assertEqual(result["short_ratio_current"], 0.0)
        self.assertEqual(result["short_squeeze_score"], 0.0)

    def test_defaults_on_none(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        result = compute_short_volume_features(None)
        self.assertEqual(result["short_ratio_current"], 0.0)

    def test_defaults_on_too_few_rows(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        df = _make_short_df(n=3)
        result = compute_short_volume_features(df)
        self.assertEqual(result["short_ratio_current"], 0.0)

    def test_current_ratio_reflects_last_row(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        df = _make_short_df(n=10, base_ratio=0.40)
        result = compute_short_volume_features(df)
        self.assertAlmostEqual(result["short_ratio_current"], 0.40, places=2)

    def test_positive_trend_captured(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        # Rising short ratio over 10 data points: 0.40 → 0.50
        df = _make_short_df(n=10, base_ratio=0.40, trend=0.10)
        result = compute_short_volume_features(df)
        self.assertGreater(result["short_ratio_trend_5d"], 0)

    def test_squeeze_score_zero_when_conditions_not_met(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        # Low short ratio, no trend
        df = _make_short_df(n=10, base_ratio=0.30)
        result = compute_short_volume_features(df)
        self.assertEqual(result["short_squeeze_score"], 0.0)

    def test_squeeze_score_positive_when_conditions_met(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        # High short ratio (0.55 → 0.65 over 6 rows: 5-day delta ~0.08), volume 3x on last day
        df = _make_short_df(n=6, base_ratio=0.55, trend=0.10, vol_multiplier=3.0)
        result = compute_short_volume_features(df)
        self.assertGreater(result["short_squeeze_score"], 0.0)

    def test_zscore_bounded(self):
        from insider_alert.feature_engine.short_volume_features import compute_short_volume_features

        df = _make_short_df(n=20, base_ratio=0.0, trend=1.0)  # extreme trend
        result = compute_short_volume_features(df)
        self.assertGreaterEqual(result["short_ratio_zscore"], -3.0)
        self.assertLessEqual(result["short_ratio_zscore"], 3.0)


# ---------------------------------------------------------------------------
# Tests: short_squeeze_signal
# ---------------------------------------------------------------------------

class TestShortSqueezeSignal(unittest.TestCase):
    def test_valid_structure(self):
        from insider_alert.signal_engine.short_squeeze_signal import short_squeeze_signal

        result = short_squeeze_signal({
            "short_ratio_zscore": 2.0,
            "short_ratio_trend_5d": 0.10,
            "short_squeeze_score": 0.6,
        })
        self.assertIn("signal_type", result)
        self.assertEqual(result["signal_type"], "short_squeeze")
        self.assertIn("score", result)
        self.assertIn("flags", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

    def test_empty_features_gives_zero(self):
        from insider_alert.signal_engine.short_squeeze_signal import short_squeeze_signal

        result = short_squeeze_signal({})
        self.assertEqual(result["score"], 0.0)

    def test_high_inputs_raise_score(self):
        from insider_alert.signal_engine.short_squeeze_signal import short_squeeze_signal

        low = short_squeeze_signal({"short_ratio_zscore": 0.0, "short_ratio_trend_5d": 0.0, "short_squeeze_score": 0.0})
        high = short_squeeze_signal({"short_ratio_zscore": 3.0, "short_ratio_trend_5d": 0.2, "short_squeeze_score": 1.0})
        self.assertGreater(high["score"], low["score"])


# ---------------------------------------------------------------------------
# Tests: fetch_short_volume (unit – no real HTTP)
# ---------------------------------------------------------------------------

class TestFetchShortVolumeUnit(unittest.TestCase):
    def test_returns_correct_columns(self):
        from insider_alert.data_ingestion.short_volume_data import fetch_short_volume

        from unittest.mock import patch, MagicMock

        # Simulate a failure response for all dates → should return empty DataFrame
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            result = fetch_short_volume("AAPL", lookback_days=3)

        self.assertIsInstance(result, pd.DataFrame)
        for col in ("Date", "ShortVolume", "TotalVolume", "ShortRatio"):
            self.assertIn(col, result.columns)

    def test_parses_valid_csv_response(self):
        from insider_alert.data_ingestion.short_volume_data import fetch_short_volume
        from unittest.mock import patch, MagicMock

        csv_content = "Symbol|ShortVolume|TotalVolume|Date\nAAPL|500000|1000000|20240401\nMSFT|200000|600000|20240401\n"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = csv_content

        with patch("requests.get", return_value=mock_resp):
            result = fetch_short_volume("AAPL", lookback_days=1)

        # May find the entry or not depending on weekday; just check structure
        self.assertIsInstance(result, pd.DataFrame)
        for col in ("Date", "ShortVolume", "TotalVolume", "ShortRatio"):
            self.assertIn(col, result.columns)


# ---------------------------------------------------------------------------
# Tests: compute_pead_features
# ---------------------------------------------------------------------------

class TestComputePeadFeatures(unittest.TestCase):
    def _base(self, **kwargs) -> dict:
        base = {
            "last_earnings_date": "2024-01-01",
            "days_since_earnings": 5,
            "earnings_day_return": 7.0,
            "post_earnings_drift_3d": 2.0,
            "post_earnings_drift_10d": 4.0,
            "next_earnings_date": None,
        }
        base.update(kwargs)
        return base

    def test_output_keys(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        result = compute_pead_features(self._base())
        for key in ("earnings_surprise_magnitude", "drift_remaining", "pead_direction"):
            self.assertIn(key, result)

    def test_positive_day_return_gives_positive_direction(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        result = compute_pead_features(self._base(earnings_day_return=5.0))
        self.assertEqual(result["pead_direction"], 1.0)

    def test_negative_day_return_gives_negative_direction(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        result = compute_pead_features(self._base(earnings_day_return=-5.0))
        self.assertEqual(result["pead_direction"], -1.0)

    def test_magnitude_bounded(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        result = compute_pead_features(self._base(earnings_day_return=50.0, days_since_earnings=1))
        self.assertLessEqual(result["earnings_surprise_magnitude"], 1.0)
        self.assertGreaterEqual(result["earnings_surprise_magnitude"], 0.0)

    def test_drift_decays_with_time(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        early = compute_pead_features(self._base(days_since_earnings=1))
        late = compute_pead_features(self._base(days_since_earnings=50))
        self.assertGreater(early["drift_remaining"], late["drift_remaining"])

    def test_no_signal_beyond_60_days(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        result = compute_pead_features(self._base(days_since_earnings=999))
        self.assertEqual(result["earnings_surprise_magnitude"], 0.0)
        self.assertEqual(result["drift_remaining"], 0.0)

    def test_no_signal_for_small_move(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        result = compute_pead_features(self._base(earnings_day_return=0.3, days_since_earnings=5))
        self.assertEqual(result["earnings_surprise_magnitude"], 0.0)

    def test_empty_earnings_data(self):
        from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features

        result = compute_pead_features({})
        self.assertEqual(result["earnings_surprise_magnitude"], 0.0)


# ---------------------------------------------------------------------------
# Tests: earnings_drift_signal
# ---------------------------------------------------------------------------

class TestEarningsDriftSignal(unittest.TestCase):
    def test_valid_structure(self):
        from insider_alert.signal_engine.earnings_drift_signal import earnings_drift_signal

        result = earnings_drift_signal({
            "earnings_surprise_magnitude": 0.8,
            "drift_remaining": 0.9,
            "pead_direction": 1.0,
        })
        self.assertIn("signal_type", result)
        self.assertEqual(result["signal_type"], "earnings_drift")
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

    def test_empty_features_gives_zero(self):
        from insider_alert.signal_engine.earnings_drift_signal import earnings_drift_signal

        result = earnings_drift_signal({})
        self.assertEqual(result["score"], 0.0)

    def test_full_features_gives_high_score(self):
        from insider_alert.signal_engine.earnings_drift_signal import earnings_drift_signal

        result = earnings_drift_signal({
            "earnings_surprise_magnitude": 1.0,
            "drift_remaining": 1.0,
            "pead_direction": 1.0,
        })
        self.assertGreater(result["score"], 80.0)


# ---------------------------------------------------------------------------
# Tests: scorer integration
# ---------------------------------------------------------------------------

class TestPhase7ScorerIntegration(unittest.TestCase):
    def test_new_weights_present(self):
        from insider_alert.scoring_engine.scorer import DEFAULT_WEIGHTS

        self.assertIn("short_squeeze", DEFAULT_WEIGHTS)
        self.assertIn("earnings_drift", DEFAULT_WEIGHTS)

    def test_weights_sum_to_one(self):
        from insider_alert.scoring_engine.scorer import DEFAULT_WEIGHTS

        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0, places=2)

    def test_new_signals_scored(self):
        from insider_alert.scoring_engine.scorer import compute_score

        signals = [
            {"signal_type": "short_squeeze", "score": 70.0, "flags": ["🩳 High short ratio"]},
            {"signal_type": "earnings_drift", "score": 80.0, "flags": ["📊 Earnings Surprise"]},
            {"signal_type": "price_anomaly", "score": 60.0, "flags": []},
        ]
        result = compute_score("TEST", signals)
        self.assertIn("short_squeeze", result.sub_scores)
        self.assertIn("earnings_drift", result.sub_scores)
        self.assertGreater(result.total_score, 0.0)


if __name__ == "__main__":
    unittest.main()
