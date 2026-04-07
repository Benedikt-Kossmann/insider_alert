"""Tests for macro regime data, features, and signal modules."""
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_macro_df(values: list[float], col: str = "close") -> pd.DataFrame:
    """Create a simple DataFrame with a close column."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({col: values}, index=idx)


def _make_macro_data(
    vix_last: float = 18.0,
    tnx_last: float = 4.5,
    irx_last: float = 5.0,
    dxy_values: list[float] | None = None,
    n: int = 30,
) -> dict[str, pd.DataFrame]:
    """Build a macro_data dict mimicking ``fetch_macro_data()`` output."""
    vix_vals = [vix_last] * n
    tnx_vals = [tnx_last] * n
    irx_vals = [irx_last] * n
    dxy_vals = dxy_values if dxy_values is not None else [100.0] * n
    return {
        "vix": _make_macro_df(vix_vals),
        "tnx": _make_macro_df(tnx_vals),
        "irx": _make_macro_df(irx_vals),
        "dxy": _make_macro_df(dxy_vals),
    }


# ---------------------------------------------------------------------------
# Tests: macro_features
# ---------------------------------------------------------------------------

class TestMacroFeatures(unittest.TestCase):
    def test_risk_on_environment(self):
        """Low VIX, normal yield curve, weak dollar → risk_on."""
        from insider_alert.feature_engine.macro_features import compute_macro_features

        data = _make_macro_data(
            vix_last=12.0,
            tnx_last=4.5,
            irx_last=3.5,          # spread = +1.0 → normal
            dxy_values=[110.0] * 15 + [104.0] * 15,  # falling: last vs -20th
        )
        f = compute_macro_features(data)

        self.assertEqual(f["vix_regime"], "low")
        self.assertEqual(f["yield_curve_regime"], "normal")
        self.assertEqual(f["dxy_trend"], "falling")
        self.assertEqual(f["risk_regime"], "risk_on")
        self.assertGreaterEqual(f["macro_score"], 65)

    def test_risk_off_environment(self):
        """High VIX, inverted yield curve, strong dollar → risk_off."""
        from insider_alert.feature_engine.macro_features import compute_macro_features

        data = _make_macro_data(
            vix_last=35.0,
            tnx_last=3.5,
            irx_last=5.0,          # spread = -1.5 → inverted
            dxy_values=[96.0] * 15 + [102.0] * 15,  # rising: last vs -20th
        )
        f = compute_macro_features(data)

        self.assertEqual(f["vix_regime"], "high")
        self.assertEqual(f["yield_curve_regime"], "inverted")
        self.assertEqual(f["dxy_trend"], "rising")
        self.assertEqual(f["risk_regime"], "risk_off")
        self.assertLessEqual(f["macro_score"], 35)

    def test_neutral_environment(self):
        """Normal VIX, flat curve, stable dollar → neutral."""
        from insider_alert.feature_engine.macro_features import compute_macro_features

        data = _make_macro_data(
            vix_last=18.0,
            tnx_last=4.5,
            irx_last=4.3,          # spread = +0.2 → flat
        )
        f = compute_macro_features(data)

        self.assertEqual(f["vix_regime"], "normal")
        self.assertEqual(f["yield_curve_regime"], "flat")
        self.assertIn(f["risk_regime"], ("neutral", "risk_on"))
        self.assertGreater(f["macro_score"], 35)
        self.assertLess(f["macro_score"], 80)

    def test_empty_data(self):
        """Empty DataFrames should return safe defaults."""
        from insider_alert.feature_engine.macro_features import compute_macro_features

        empty = {k: pd.DataFrame() for k in ("vix", "tnx", "irx", "dxy")}
        f = compute_macro_features(empty)

        self.assertEqual(f["vix_regime"], "unknown")
        self.assertEqual(f["yield_curve_regime"], "unknown")
        self.assertEqual(f["risk_regime"], "neutral")
        self.assertAlmostEqual(f["macro_score"], 50.0, delta=15)

    def test_score_bounded(self):
        """Macro score must be in [0, 100]."""
        from insider_alert.feature_engine.macro_features import compute_macro_features

        for vix in (5.0, 50.0):
            for spread in (-3.0, 3.0):
                data = _make_macro_data(vix_last=vix, tnx_last=spread + 4.0, irx_last=4.0)
                f = compute_macro_features(data)
                self.assertGreaterEqual(f["macro_score"], 0.0)
                self.assertLessEqual(f["macro_score"], 100.0)

    def test_yield_spread_computed(self):
        from insider_alert.feature_engine.macro_features import compute_macro_features

        data = _make_macro_data(tnx_last=4.5, irx_last=5.3)
        f = compute_macro_features(data)
        self.assertAlmostEqual(f["yield_spread"], 4.5 - 5.3, places=2)


# ---------------------------------------------------------------------------
# Tests: macro_signal
# ---------------------------------------------------------------------------

class TestMacroSignal(unittest.TestCase):
    def _assert_signal(self, result: dict, expected_type: str = "macro_regime"):
        self.assertIn("signal_type", result)
        self.assertIn("score", result)
        self.assertIn("flags", result)
        self.assertEqual(result["signal_type"], expected_type)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)
        self.assertIsInstance(result["flags"], list)

    def test_structure(self):
        from insider_alert.signal_engine.macro_signal import compute_macro_regime_signal

        features = {
            "vix_regime": "normal", "vix_current": 18.0,
            "yield_curve_regime": "normal", "yield_spread": 1.0,
            "dxy_trend": "flat", "dxy_return_20d": 0.0,
        }
        result = compute_macro_regime_signal(features)
        self._assert_signal(result)

    def test_max_score(self):
        """Low VIX + normal curve + falling dollar → max score = 100."""
        from insider_alert.signal_engine.macro_signal import compute_macro_regime_signal

        features = {
            "vix_regime": "low", "vix_current": 11.0,
            "yield_curve_regime": "normal", "yield_spread": 1.5,
            "dxy_trend": "falling", "dxy_return_20d": -0.03,
        }
        result = compute_macro_regime_signal(features)
        self.assertEqual(result["score"], 100.0)

    def test_min_score(self):
        """High VIX + inverted curve + rising dollar → min score = 10."""
        from insider_alert.signal_engine.macro_signal import compute_macro_regime_signal

        features = {
            "vix_regime": "high", "vix_current": 40.0,
            "yield_curve_regime": "inverted", "yield_spread": -1.0,
            "dxy_trend": "rising", "dxy_return_20d": 0.05,
        }
        result = compute_macro_regime_signal(features)
        self.assertEqual(result["score"], 10.0)

    def test_flags_populated(self):
        from insider_alert.signal_engine.macro_signal import compute_macro_regime_signal

        features = {
            "vix_regime": "high", "vix_current": 30.0,
            "yield_curve_regime": "inverted", "yield_spread": -0.8,
            "dxy_trend": "rising", "dxy_return_20d": 0.04,
        }
        result = compute_macro_regime_signal(features)
        self.assertGreater(len(result["flags"]), 0)
        # Should mention VIX and yield curve
        flags_text = " ".join(result["flags"]).lower()
        self.assertIn("vix", flags_text)
        self.assertIn("yield", flags_text)

    def test_empty_features(self):
        """Missing features should produce a valid but neutral result."""
        from insider_alert.signal_engine.macro_signal import compute_macro_regime_signal

        result = compute_macro_regime_signal({})
        self._assert_signal(result)


# ---------------------------------------------------------------------------
# Tests: macro integration in scoring
# ---------------------------------------------------------------------------

class TestMacroScoringIntegration(unittest.TestCase):
    def test_macro_weight_in_defaults(self):
        from insider_alert.scoring_engine.scorer import DEFAULT_WEIGHTS

        self.assertIn("macro_regime", DEFAULT_WEIGHTS)
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0, places=2)

    def test_macro_signal_scored(self):
        from insider_alert.scoring_engine.scorer import compute_score

        signals = [
            {"signal_type": "macro_regime", "score": 80.0, "flags": ["VIX low"]},
            {"signal_type": "price_anomaly", "score": 60.0, "flags": []},
        ]
        result = compute_score("TEST", signals)
        self.assertGreater(result.total_score, 0.0)
        self.assertIn("macro_regime", result.sub_scores)


if __name__ == "__main__":
    unittest.main()
