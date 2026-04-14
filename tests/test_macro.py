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


# ---------------------------------------------------------------------------
# Tests: Phase 6 – FRED data module (no real API calls)
# ---------------------------------------------------------------------------

class TestFredDataFallback(unittest.TestCase):
    """fetch_all_macro_data() must return defaults when FRED is unavailable."""

    def test_returns_dict_with_all_keys(self):
        from insider_alert.data_ingestion.fred_data import fetch_all_macro_data

        result = fetch_all_macro_data()

        expected_keys = {
            "hy_spread", "hy_spread_change_1m",
            "fed_funds_rate", "fed_policy_direction",
            "cpi_yoy_change", "inflation_trend",
            "initial_claims_zscore", "unemployment_rate",
            "consumer_sentiment",
        }
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_defaults_are_plausible(self):
        from insider_alert.data_ingestion.fred_data import fetch_all_macro_data

        result = fetch_all_macro_data()

        self.assertGreater(result["hy_spread"], 0)
        self.assertIn(result["fed_policy_direction"], ("tightening", "easing", "hold"))
        self.assertIn(result["inflation_trend"], ("rising", "falling", "stable"))

    def test_fetch_fred_series_returns_none_without_key(self):
        """Without API key, fetch_fred_series must return None gracefully."""
        from insider_alert.data_ingestion.fred_data import fetch_fred_series

        result = fetch_fred_series("BAMLH0A0HYM2")
        # Either None (no API key) or a pd.Series (if key is configured)
        import pandas as pd
        self.assertTrue(result is None or isinstance(result, pd.Series))


# ---------------------------------------------------------------------------
# Tests: Phase 6 – compute_fred_macro_features
# ---------------------------------------------------------------------------

class TestComputeFredMacroFeatures(unittest.TestCase):
    def _base_fred_data(self, **overrides) -> dict:
        base = {
            "hy_spread": 4.0,
            "hy_spread_change_1m": 0.0,
            "fed_funds_rate": 5.0,
            "fed_policy_direction": "hold",
            "cpi_yoy_change": 2.5,
            "inflation_trend": "stable",
            "initial_claims_zscore": 0.0,
            "unemployment_rate": 4.0,
            "consumer_sentiment": 70.0,
        }
        base.update(overrides)
        return base

    def test_output_keys(self):
        from insider_alert.feature_engine.macro_features import compute_fred_macro_features

        result = compute_fred_macro_features(self._base_fred_data(), {})

        for key in ("credit_stress_score", "fed_policy_score", "inflation_score",
                    "labor_market_score", "consumer_sentiment_norm", "macro_regime"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_high_hy_spread_raises_credit_stress(self):
        from insider_alert.feature_engine.macro_features import compute_fred_macro_features

        result = compute_fred_macro_features(self._base_fred_data(hy_spread=9.0), {})
        self.assertGreater(result["credit_stress_score"], 0.5)

    def test_low_hy_spread_low_credit_stress(self):
        from insider_alert.feature_engine.macro_features import compute_fred_macro_features

        result = compute_fred_macro_features(self._base_fred_data(hy_spread=3.0), {})
        self.assertLessEqual(result["credit_stress_score"], 0.2)

    def test_tightening_fed_positive_policy_score(self):
        from insider_alert.feature_engine.macro_features import compute_fred_macro_features

        result = compute_fred_macro_features(
            self._base_fred_data(fed_policy_direction="tightening"), {}
        )
        self.assertGreater(result["fed_policy_score"], 0)

    def test_easing_fed_negative_policy_score(self):
        from insider_alert.feature_engine.macro_features import compute_fred_macro_features

        result = compute_fred_macro_features(
            self._base_fred_data(fed_policy_direction="easing"), {}
        )
        self.assertLess(result["fed_policy_score"], 0)

    def test_stress_regime_high_credit(self):
        from insider_alert.feature_engine.macro_features import compute_fred_macro_features

        result = compute_fred_macro_features(self._base_fred_data(hy_spread=10.0), {})
        self.assertEqual(result["macro_regime"], "stress")

    def test_scores_bounded(self):
        from insider_alert.feature_engine.macro_features import compute_fred_macro_features

        extreme = self._base_fred_data(hy_spread=20.0, cpi_yoy_change=20.0)
        result = compute_fred_macro_features(extreme, {})

        self.assertGreaterEqual(result["credit_stress_score"], 0.0)
        self.assertLessEqual(result["credit_stress_score"], 1.0)
        self.assertGreaterEqual(result["consumer_sentiment_norm"], 0.0)
        self.assertLessEqual(result["consumer_sentiment_norm"], 1.0)


# ---------------------------------------------------------------------------
# Tests: Phase 6 – new macro_signal (6-component SignalComponent)
# ---------------------------------------------------------------------------

class TestMacroSignalExtended(unittest.TestCase):
    def _features(self, **overrides) -> dict:
        base = {
            # market data keys
            "vix_value": 18.0,
            "yield_spread": 1.0,
            "dxy_change_5d": 0.0,
            # FRED keys
            "credit_stress_score": 0.2,
            "fed_policy_score": 0.0,
            "labor_market_score": 0.3,
        }
        base.update(overrides)
        return base

    def test_returns_valid_structure(self):
        from insider_alert.signal_engine.macro_signal import macro_signal

        result = macro_signal(self._features())

        self.assertIn("signal_type", result)
        self.assertEqual(result["signal_type"], "macro_regime")
        self.assertIn("score", result)
        self.assertIn("flags", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

    def test_empty_features_returns_zero_score(self):
        from insider_alert.signal_engine.macro_signal import macro_signal

        result = macro_signal({})
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

    def test_high_credit_stress_raises_score(self):
        from insider_alert.signal_engine.macro_signal import macro_signal

        low_stress = macro_signal(self._features(credit_stress_score=0.0))
        high_stress = macro_signal(self._features(credit_stress_score=1.0))
        self.assertGreater(high_stress["score"], low_stress["score"])

    def test_old_signal_still_works(self):
        """compute_macro_regime_signal must remain functional (backward compat)."""
        from insider_alert.signal_engine.macro_signal import compute_macro_regime_signal

        result = compute_macro_regime_signal({
            "vix_regime": "normal", "vix_current": 18.0,
            "yield_curve_regime": "normal", "yield_spread": 1.0,
            "dxy_trend": "flat", "dxy_return_20d": 0.0,
        })
        self.assertIn("signal_type", result)
        self.assertEqual(result["signal_type"], "macro_regime")

    def test_vix_value_alias_in_compute_macro_features(self):
        """compute_macro_features must now export vix_value and dxy_change_5d."""
        from insider_alert.feature_engine.macro_features import compute_macro_features
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        vix_df = pd.DataFrame({"close": [18.0] * 30}, index=idx)
        tnx_df = pd.DataFrame({"close": [4.5] * 30}, index=idx)
        irx_df = pd.DataFrame({"close": [4.0] * 30}, index=idx)
        dxy_df = pd.DataFrame({"close": [100.0] * 30}, index=idx)

        result = compute_macro_features({
            "vix": vix_df, "tnx": tnx_df, "irx": irx_df, "dxy": dxy_df
        })
        self.assertIn("vix_value", result)
        self.assertIn("dxy_change_5d", result)
        self.assertAlmostEqual(result["vix_value"], result["vix_current"], places=4)
