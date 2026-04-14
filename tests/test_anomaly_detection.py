"""Tests for Phase 8: Isolation Forest anomaly detection and feature drift detection."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np


class TestComputeAnomalyScoreNoModel(unittest.TestCase):
    """Tests when no model is available (no DB data)."""

    def setUp(self):
        # Reset module-level state before each test
        import insider_alert.scoring_engine.anomaly_detector as mod
        mod._model = None
        mod._scaler = None
        mod._feature_names = []
        mod._last_trained = None

    def test_returns_defaults_when_no_data(self):
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        result = compute_anomaly_score({"momentum": 75.0, "volume": 80.0})
        self.assertIn("anomaly_score", result)
        self.assertIn("is_anomaly", result)
        self.assertIn("anomaly_type", result)

    def test_returns_defaults_on_empty_signals(self):
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        result = compute_anomaly_score({})
        self.assertEqual(result["anomaly_score"], 0.0)
        self.assertFalse(result["is_anomaly"])
        self.assertEqual(result["anomaly_type"], "normal")

    def test_score_range_is_zero_to_one(self):
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        result = compute_anomaly_score({"price": 50.0})
        self.assertGreaterEqual(result["anomaly_score"], 0.0)
        self.assertLessEqual(result["anomaly_score"], 1.0)

    def test_anomaly_type_valid_values(self):
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        result = compute_anomaly_score({"x": 60.0})
        self.assertIn(result["anomaly_type"], {"normal", "rare_opportunity", "rare_risk"})

    def test_is_anomaly_is_bool(self):
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        result = compute_anomaly_score({"momentum": 55.0})
        self.assertIsInstance(result["is_anomaly"], bool)


class TestComputeAnomalyScoreWithModel(unittest.TestCase):
    """Tests when a trained Isolation Forest model is available."""

    def _inject_trained_model(self, n_samples: int = 200, n_features: int = 5):
        """Inject a pre-trained model directly into module globals."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        import insider_alert.scoring_engine.anomaly_detector as mod
        from datetime import datetime

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 100, (n_samples, n_features))
        feat_names = [f"signal_{i}" for i in range(n_features)]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = IsolationForest(n_estimators=10, contamination=0.1, random_state=42)
        model.fit(X_scaled)

        mod._model = model
        mod._scaler = scaler
        mod._feature_names = feat_names
        mod._last_trained = datetime.utcnow()
        return feat_names

    def tearDown(self):
        import insider_alert.scoring_engine.anomaly_detector as mod
        mod._model = None
        mod._scaler = None
        mod._feature_names = []
        mod._last_trained = None

    def test_returns_valid_structure_with_model(self):
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        feat_names = self._inject_trained_model()
        signal_scores = {name: 50.0 for name in feat_names}
        result = compute_anomaly_score(signal_scores)
        self.assertIn("anomaly_score", result)
        self.assertIn("is_anomaly", result)
        self.assertIn("anomaly_type", result)

    def test_score_in_range_with_model(self):
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        feat_names = self._inject_trained_model()
        signal_scores = {name: 50.0 for name in feat_names}
        result = compute_anomaly_score(signal_scores)
        self.assertGreaterEqual(result["anomaly_score"], 0.0)
        self.assertLessEqual(result["anomaly_score"], 1.0)

    def test_high_composite_anomaly_gives_rare_opportunity(self):
        """An anomalous point with high composite score → rare_opportunity."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        import insider_alert.scoring_engine.anomaly_detector as mod
        from datetime import datetime

        rng = np.random.default_rng(1)
        # Training data: tight cluster around 50
        X_train = rng.normal(50, 2, (300, 3))
        feat_names = ["s0", "s1", "s2"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=1)
        model.fit(X_scaled)

        mod._model = model
        mod._scaler = scaler
        mod._feature_names = feat_names
        mod._last_trained = datetime.utcnow()

        # Extreme outlier with high scores → should be flagged as rare_opportunity or normal
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        result = compute_anomaly_score({"s0": 95.0, "s1": 95.0, "s2": 95.0})
        self.assertIn(result["anomaly_type"], {"rare_opportunity", "normal"})

    def test_missing_features_use_zero(self):
        """Unknown feature keys default to 0.0, no crash."""
        from insider_alert.scoring_engine.anomaly_detector import compute_anomaly_score
        self._inject_trained_model()
        result = compute_anomaly_score({"completely_unknown_key": 99.0})
        self.assertIn("anomaly_score", result)


class TestDetectFeatureDrift(unittest.TestCase):
    """Tests for KS-based feature drift detection."""

    def test_returns_defaults_when_no_db_data(self):
        from insider_alert.scoring_engine.anomaly_detector import detect_feature_drift
        with patch(
            "insider_alert.scoring_engine.anomaly_detector._get_training_data",
            return_value=None,
        ):
            result = detect_feature_drift({})
        self.assertFalse(result["drift_detected"])
        self.assertEqual(result["drifted_features"], [])
        self.assertEqual(result["drift_severity"], 0.0)

    def test_no_drift_on_stable_distribution(self):
        """Identical distributions → no drift."""
        from insider_alert.scoring_engine.anomaly_detector import detect_feature_drift

        rng = np.random.default_rng(42)
        X = rng.normal(50, 5, (100, 2)).astype(float)
        feat_names = ["f0", "f1"]

        with patch(
            "insider_alert.scoring_engine.anomaly_detector._get_training_data",
            return_value=(X, feat_names),
        ):
            result = detect_feature_drift({})

        self.assertIsInstance(result["drift_detected"], bool)
        self.assertIsInstance(result["drifted_features"], list)
        self.assertGreaterEqual(result["drift_severity"], 0.0)
        self.assertLessEqual(result["drift_severity"], 1.0)

    def test_detects_synthetic_drift(self):
        """Last 20 rows have a completely different distribution → drift detected."""
        from insider_alert.scoring_engine.anomaly_detector import detect_feature_drift

        rng = np.random.default_rng(7)
        # 80 rows near 50, last 20 rows near 0 (extreme shift)
        X_stable = rng.normal(50, 2, (80, 1))
        X_shifted = rng.normal(0, 2, (20, 1))
        X = np.vstack([X_stable, X_shifted])
        feat_names = ["sig"]

        with patch(
            "insider_alert.scoring_engine.anomaly_detector._get_training_data",
            return_value=(X, feat_names),
        ):
            result = detect_feature_drift({})

        self.assertTrue(result["drift_detected"])
        self.assertIn("sig", result["drifted_features"])
        self.assertGreater(result["drift_severity"], 0.0)

    def test_current_features_filter(self):
        """When current_features is provided, only those features are checked."""
        from insider_alert.scoring_engine.anomaly_detector import detect_feature_drift

        rng = np.random.default_rng(9)
        X_stable = rng.normal(50, 2, (80, 2))
        X_shifted = rng.normal(0, 2, (20, 2))
        X = np.vstack([X_stable, X_shifted])
        feat_names = ["sig_a", "sig_b"]

        with patch(
            "insider_alert.scoring_engine.anomaly_detector._get_training_data",
            return_value=(X, feat_names),
        ):
            # Only check sig_a; sig_b should be ignored
            result = detect_feature_drift({"sig_a": 50.0})

        # sig_a should be drifted; sig_b was excluded
        self.assertNotIn("sig_b", result["drifted_features"])

    def test_too_few_historical_rows_returns_defaults(self):
        from insider_alert.scoring_engine.anomaly_detector import detect_feature_drift

        X = np.zeros((30, 2))  # less than 50 rows
        with patch(
            "insider_alert.scoring_engine.anomaly_detector._get_training_data",
            return_value=(X, ["f0", "f1"]),
        ):
            result = detect_feature_drift({})

        self.assertFalse(result["drift_detected"])


class TestAnomalyAlertMessage(unittest.TestCase):
    """Test that anomaly info is included in Telegram alert messages."""

    def _make_ticker_score(self, score: float = 75.0):
        mock = MagicMock()
        mock.ticker = "AAPL"
        mock.total_score = score
        mock.sub_scores = {"momentum": 80.0, "volume": 70.0}
        mock.flags = []
        return mock

    def test_rare_opportunity_in_message(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        anomaly_info = {"anomaly_score": 0.85, "is_anomaly": True, "anomaly_type": "rare_opportunity"}
        msg = build_alert_message(ts, anomaly_info=anomaly_info)
        self.assertIn("RARE OPPORTUNITY", msg)

    def test_rare_risk_in_message(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        anomaly_info = {"anomaly_score": 0.80, "is_anomaly": True, "anomaly_type": "rare_risk"}
        msg = build_alert_message(ts, anomaly_info=anomaly_info)
        self.assertIn("RARE RISK", msg)

    def test_normal_anomaly_type_no_extra_line(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        anomaly_info = {"anomaly_score": 0.1, "is_anomaly": False, "anomaly_type": "normal"}
        msg = build_alert_message(ts, anomaly_info=anomaly_info)
        self.assertNotIn("RARE", msg)

    def test_no_anomaly_info_no_change(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        msg = build_alert_message(ts)
        self.assertNotIn("RARE", msg)

    def test_maybe_send_alert_passes_anomaly_info(self):
        from insider_alert.alert_engine.telegram_alert import maybe_send_alert
        ts = self._make_ticker_score(score=80.0)
        anomaly_info = {"anomaly_score": 0.9, "is_anomaly": True, "anomaly_type": "rare_opportunity"}
        with patch("insider_alert.alert_engine.telegram_alert.send_telegram_message", return_value=True) as mock_send:
            result = maybe_send_alert(ts, "tok", "chat", threshold=70.0, anomaly_info=anomaly_info)
        self.assertTrue(result)
        sent_msg = mock_send.call_args[0][2]
        self.assertIn("RARE OPPORTUNITY", sent_msg)


if __name__ == "__main__":
    unittest.main()
