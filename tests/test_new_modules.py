"""Tests for new modules: candlestick, S/R, sector, adaptive weights, ML scorer, bot, weekly report."""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date


def _make_ohlcv(n: int = 50, seed: int = 42) -> pd.DataFrame:
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


def _make_hammer_ohlcv() -> pd.DataFrame:
    """Create OHLCV with a hammer pattern on the last bar."""
    n = 10
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    data = {
        "open": [100.0]*n, "high": [102.0]*n, "low": [95.0]*n,
        "close": [101.0]*n, "volume": [1_000_000.0]*n,
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    # Hammer: small body at top, long lower shadow, tiny upper shadow
    df.iloc[-1] = {"open": 100.5, "high": 100.8, "low": 95.0, "close": 100.7, "volume": 2_000_000.0}
    return df


def _make_doji_ohlcv() -> pd.DataFrame:
    """Create OHLCV with a doji pattern on the last bar."""
    n = 10
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    data = {
        "open": [100.0]*n, "high": [102.0]*n, "low": [98.0]*n,
        "close": [101.0]*n, "volume": [1_000_000.0]*n,
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    # Doji: open ≈ close
    df.iloc[-1] = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.05, "volume": 1_500_000.0}
    return df


def _make_engulfing_ohlcv() -> pd.DataFrame:
    """Create OHLCV with a bullish engulfing pattern."""
    n = 10
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    data = {
        "open": [100.0]*n, "high": [102.0]*n, "low": [98.0]*n,
        "close": [101.0]*n, "volume": [1_000_000.0]*n,
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    # Prev: bearish (open > close)
    df.iloc[-2] = {"open": 102.0, "high": 103.0, "low": 99.0, "close": 99.5, "volume": 1_000_000.0}
    # Current: bullish engulfing (open < prev close, close > prev open)
    df.iloc[-1] = {"open": 99.0, "high": 104.0, "low": 98.5, "close": 103.0, "volume": 2_000_000.0}
    return df


# ===========================================================================
# Candlestick Features
# ===========================================================================

class TestCandlestickFeatures(unittest.TestCase):
    def test_returns_expected_keys(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(_make_ohlcv(50))
        self.assertIn("bullish_pattern_score", result)
        self.assertIn("bearish_pattern_score", result)
        self.assertIn("pattern_names", result)
        self.assertIn("hammer", result)
        self.assertIn("doji", result)

    def test_scores_in_range(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(_make_ohlcv(50))
        self.assertGreaterEqual(result["bullish_pattern_score"], 0)
        self.assertLessEqual(result["bullish_pattern_score"], 100)
        self.assertGreaterEqual(result["bearish_pattern_score"], 0)
        self.assertLessEqual(result["bearish_pattern_score"], 100)

    def test_empty_ohlcv(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(pd.DataFrame())
        self.assertEqual(result["bullish_pattern_score"], 0)
        self.assertEqual(result["bearish_pattern_score"], 0)

    def test_short_ohlcv(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(_make_ohlcv(2))
        self.assertIsInstance(result["bullish_pattern_score"], (int, float))

    def test_hammer_detection(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(_make_hammer_ohlcv())
        self.assertTrue(result["hammer"])

    def test_doji_detection(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(_make_doji_ohlcv())
        self.assertTrue(result["doji"])

    def test_bullish_engulfing_detection(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(_make_engulfing_ohlcv())
        self.assertTrue(result["engulfing_bullish"])

    def test_pattern_names_are_strings(self):
        from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
        result = detect_candlestick_patterns(_make_ohlcv(50))
        for name in result["pattern_names"]:
            self.assertIsInstance(name, str)


# ===========================================================================
# Support/Resistance Features
# ===========================================================================

class TestSRFeatures(unittest.TestCase):
    def test_returns_expected_keys(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(_make_ohlcv(50))
        expected = {
            "support_levels", "resistance_levels",
            "nearest_support", "nearest_resistance",
            "distance_to_support_pct", "distance_to_resistance_pct",
            "sr_zone",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_zone_valid_values(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(_make_ohlcv(50))
        self.assertIn(result["sr_zone"], ("near_support", "near_resistance", "mid_range"))

    def test_supports_are_positive(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(_make_ohlcv(50))
        for level in result["support_levels"]:
            self.assertGreater(level, 0)

    def test_resistances_are_positive(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(_make_ohlcv(50))
        for level in result["resistance_levels"]:
            self.assertGreater(level, 0)

    def test_empty_ohlcv(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(pd.DataFrame())
        self.assertEqual(result["nearest_support"], 0.0)
        self.assertEqual(result["nearest_resistance"], 0.0)
        self.assertEqual(result["sr_zone"], "mid_range")

    def test_short_ohlcv(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(_make_ohlcv(5))
        self.assertIsInstance(result["sr_zone"], str)

    def test_distances_non_negative(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(_make_ohlcv(50))
        self.assertGreaterEqual(result["distance_to_support_pct"], 0)
        self.assertGreaterEqual(result["distance_to_resistance_pct"], 0)

    def test_max_levels_limit(self):
        from insider_alert.feature_engine.sr_features import compute_support_resistance
        result = compute_support_resistance(_make_ohlcv(100), max_levels=2)
        self.assertLessEqual(len(result["support_levels"]), 2)
        self.assertLessEqual(len(result["resistance_levels"]), 2)


# ===========================================================================
# Sector Features
# ===========================================================================

class TestSectorFeatures(unittest.TestCase):
    def test_get_sector_etf_known(self):
        from insider_alert.feature_engine.sector_features import get_sector_etf
        self.assertEqual(get_sector_etf("AAPL"), "XLK")

    def test_get_sector_etf_unknown(self):
        from insider_alert.feature_engine.sector_features import get_sector_etf
        self.assertEqual(get_sector_etf("UNKNOWN_TICKER_XYZ"), "SPY")

    def test_get_sector_label(self):
        from insider_alert.feature_engine.sector_features import get_sector_label
        label = get_sector_label("AAPL")
        self.assertIsInstance(label, str)
        self.assertGreater(len(label), 0)

    def test_compute_relative_strength(self):
        from insider_alert.feature_engine.sector_features import compute_relative_strength
        ticker_ohlcv = _make_ohlcv(30, seed=42)
        sector_ohlcv = _make_ohlcv(30, seed=99)
        result = compute_relative_strength(ticker_ohlcv, sector_ohlcv)
        self.assertIn("rs_score", result)
        self.assertIn("relative_trend", result)
        self.assertIn("rs_5d", result)

    def test_rs_score_range(self):
        from insider_alert.feature_engine.sector_features import compute_relative_strength
        ticker_ohlcv = _make_ohlcv(30, seed=42)
        sector_ohlcv = _make_ohlcv(30, seed=99)
        result = compute_relative_strength(ticker_ohlcv, sector_ohlcv)
        self.assertGreaterEqual(result["rs_score"], 0)
        self.assertLessEqual(result["rs_score"], 100)

    def test_trend_valid_values(self):
        from insider_alert.feature_engine.sector_features import compute_relative_strength
        ticker_ohlcv = _make_ohlcv(30, seed=42)
        sector_ohlcv = _make_ohlcv(30, seed=99)
        result = compute_relative_strength(ticker_ohlcv, sector_ohlcv)
        self.assertIn(result["relative_trend"], ("outperforming", "inline", "underperforming"))

    def test_empty_sector_ohlcv(self):
        from insider_alert.feature_engine.sector_features import compute_relative_strength
        result = compute_relative_strength(_make_ohlcv(30), pd.DataFrame())
        self.assertEqual(result["rs_score"], 50)
        self.assertEqual(result["relative_trend"], "inline")

    def test_empty_ticker_ohlcv(self):
        from insider_alert.feature_engine.sector_features import compute_relative_strength
        result = compute_relative_strength(pd.DataFrame(), _make_ohlcv(30))
        self.assertEqual(result["rs_score"], 50)


# ===========================================================================
# Adaptive Weights
# ===========================================================================

class TestAdaptiveWeights(unittest.TestCase):
    def test_returns_defaults_when_no_data(self):
        from insider_alert.scoring_engine.adaptive_weights import compute_adaptive_weights
        default = {"price_anomaly": 0.5, "volume_anomaly": 0.5}
        with patch("insider_alert.scoring_engine.adaptive_weights._fetch_hit_rates", return_value={}):
            result = compute_adaptive_weights(default)
        self.assertEqual(result, default)

    def test_returns_dict_same_keys(self):
        from insider_alert.scoring_engine.adaptive_weights import compute_adaptive_weights
        default = {"price_anomaly": 0.5, "volume_anomaly": 0.5}
        with patch("insider_alert.scoring_engine.adaptive_weights._fetch_hit_rates", return_value={}):
            result = compute_adaptive_weights(default)
        self.assertEqual(set(result.keys()), set(default.keys()))

    def test_weights_sum_to_one(self):
        from insider_alert.scoring_engine.adaptive_weights import compute_adaptive_weights
        default = {"price_anomaly": 0.5, "volume_anomaly": 0.3, "candle_pattern": 0.2}
        hit_rates = {
            "price_anomaly": {"hit_rate_5d": 0.65, "avg_return_5d": 0.02, "count": 50},
            "volume_anomaly": {"hit_rate_5d": 0.45, "avg_return_5d": 0.005, "count": 50},
            "candle_pattern": {"hit_rate_5d": 0.55, "avg_return_5d": 0.01, "count": 50},
        }
        with patch("insider_alert.scoring_engine.adaptive_weights._fetch_hit_rates", return_value=hit_rates):
            result = compute_adaptive_weights(default)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=4)

    def test_high_hit_rate_gets_higher_weight(self):
        from insider_alert.scoring_engine.adaptive_weights import compute_adaptive_weights
        default = {"a": 0.5, "b": 0.5}
        hit_rates = {
            "a": {"hit_rate_5d": 0.80, "avg_return_5d": 0.03, "count": 100},
            "b": {"hit_rate_5d": 0.30, "avg_return_5d": -0.01, "count": 100},
        }
        with patch("insider_alert.scoring_engine.adaptive_weights._fetch_hit_rates", return_value=hit_rates):
            result = compute_adaptive_weights(default)
        self.assertGreater(result["a"], result["b"])

    def test_below_min_outcomes_keeps_default(self):
        from insider_alert.scoring_engine.adaptive_weights import compute_adaptive_weights
        default = {"a": 0.5, "b": 0.5}
        hit_rates = {
            "a": {"hit_rate_5d": 0.80, "avg_return_5d": 0.03, "count": 10},  # below min
        }
        with patch("insider_alert.scoring_engine.adaptive_weights._fetch_hit_rates", return_value=hit_rates):
            result = compute_adaptive_weights(default)
        # With insufficient data both should be equal
        self.assertAlmostEqual(result["a"], result["b"], places=3)


# ===========================================================================
# ML Scorer
# ===========================================================================

class TestMLScorer(unittest.TestCase):
    def test_is_available(self):
        from insider_alert.scoring_engine.ml_scorer import is_available
        # Should be True if sklearn is installed
        result = is_available()
        self.assertIsInstance(result, bool)

    def test_predict_without_model(self):
        from insider_alert.scoring_engine import ml_scorer
        # Reset model state
        old_model = ml_scorer._model
        ml_scorer._model = None
        try:
            result = ml_scorer.predict_score([
                {"signal_type": "price_anomaly", "score": 60.0},
            ])
            self.assertIsNone(result)
        finally:
            ml_scorer._model = old_model

    def test_train_insufficient_data(self):
        from insider_alert.scoring_engine.ml_scorer import train_model
        with patch("insider_alert.scoring_engine.ml_scorer._fetch_training_data",
                   return_value=(np.array([]), np.array([]), [])):
            result = train_model()
        self.assertFalse(result)

    def test_train_and_predict(self):
        from insider_alert.scoring_engine import ml_scorer

        # Create mock training data
        rng = np.random.default_rng(42)
        n = 100
        X = rng.random((n, 3)) * 100
        y = (X[:, 0] > 50).astype(int)  # simple rule
        names = ["price_anomaly", "volume_anomaly", "candle_pattern"]

        with patch("insider_alert.scoring_engine.ml_scorer._fetch_training_data",
                   return_value=(X, y, names)):
            success = ml_scorer.train_model()

        if ml_scorer.is_available():
            self.assertTrue(success)

            # Now predict
            signals = [
                {"signal_type": "price_anomaly", "score": 80.0},
                {"signal_type": "volume_anomaly", "score": 60.0},
                {"signal_type": "candle_pattern", "score": 30.0},
            ]
            score = ml_scorer.predict_score(signals)
            self.assertIsNotNone(score)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_maybe_retrain_fast(self):
        """maybe_retrain should not crash."""
        from insider_alert.scoring_engine import ml_scorer
        ml_scorer._last_trained = None
        with patch("insider_alert.scoring_engine.ml_scorer._fetch_training_data",
                   return_value=(np.array([]), np.array([]), [])):
            ml_scorer.maybe_retrain()


# ===========================================================================
# Telegram Bot Commands
# ===========================================================================

class TestTelegramBot(unittest.TestCase):
    def _mock_config(self):
        config = MagicMock()
        config.telegram_token = "test_token"
        config.telegram_chat_id = "12345"
        config.tickers = ["AAPL", "MSFT", "NVDA"]
        config.alert_threshold = 60.0
        config.scheduler = {"eod_hour": 17, "eod_minute": 30, "intraday_interval_minutes": 30}
        config.leveraged_etfs = {"enabled": False, "universe": []}
        return config

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_cmd_help(self, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _cmd_help
        config = self._mock_config()
        _cmd_help("", config)
        mock_reply.assert_called_once()
        msg = mock_reply.call_args[0][1]
        self.assertIn("Befehle", msg)
        self.assertIn("/status", msg)

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_cmd_status(self, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _cmd_status
        config = self._mock_config()
        _cmd_status("", config)
        mock_reply.assert_called_once()
        msg = mock_reply.call_args[0][1]
        self.assertIn("3 Ticker", msg)

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    @patch("insider_alert.persistence.storage.get_recent_scores")
    def test_cmd_scores(self, mock_scores, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _cmd_scores
        mock_scores.return_value = [{"total_score": 72.5}]
        config = self._mock_config()
        _cmd_scores("", config)
        mock_reply.assert_called_once()
        msg = mock_reply.call_args[0][1]
        self.assertIn("72.5", msg)

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    @patch("insider_alert.persistence.storage.get_recent_scores")
    def test_cmd_score_detail_no_ticker(self, mock_scores, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _cmd_score_detail
        config = self._mock_config()
        _cmd_score_detail("", config)
        mock_reply.assert_called_once()
        self.assertIn("Bitte", mock_reply.call_args[0][1])

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    @patch("insider_alert.persistence.storage.get_recent_scores")
    def test_cmd_score_detail_with_data(self, mock_scores, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _cmd_score_detail
        mock_scores.return_value = [
            {"date": "2024-06-01", "total_score": 65.0, "sub_scores": {"price_anomaly": 70.0}, "flags": []},
            {"date": "2024-05-31", "total_score": 55.0, "sub_scores": {"price_anomaly": 50.0}, "flags": []},
        ]
        config = self._mock_config()
        _cmd_score_detail("AAPL", config)
        mock_reply.assert_called_once()
        msg = mock_reply.call_args[0][1]
        self.assertIn("AAPL", msg)
        self.assertIn("65.0", msg)

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_cmd_watchlist(self, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _cmd_watchlist
        config = self._mock_config()
        _cmd_watchlist("", config)
        mock_reply.assert_called_once()
        msg = mock_reply.call_args[0][1]
        self.assertIn("3 Ticker", msg)

    @patch("insider_alert.alert_engine.telegram_bot._save_watchlist")
    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_cmd_add(self, mock_reply, mock_save):
        from insider_alert.alert_engine.telegram_bot import _cmd_add
        config = self._mock_config()
        config.tickers = ["AAPL"]
        _cmd_add("NVDA", config)
        mock_reply.assert_called_once()
        self.assertIn("NVDA", config.tickers)
        mock_save.assert_called_once()

    @patch("insider_alert.alert_engine.telegram_bot._save_watchlist")
    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_cmd_add_duplicate(self, mock_reply, mock_save):
        from insider_alert.alert_engine.telegram_bot import _cmd_add
        config = self._mock_config()
        _cmd_add("AAPL", config)
        mock_reply.assert_called_once()
        self.assertIn("bereits", mock_reply.call_args[0][1])
        mock_save.assert_not_called()

    @patch("insider_alert.alert_engine.telegram_bot._save_watchlist")
    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_cmd_remove(self, mock_reply, mock_save):
        from insider_alert.alert_engine.telegram_bot import _cmd_remove
        config = self._mock_config()
        _cmd_remove("MSFT", config)
        mock_reply.assert_called_once()
        self.assertNotIn("MSFT", config.tickers)
        mock_save.assert_called_once()

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_cmd_remove_not_found(self, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _cmd_remove
        config = self._mock_config()
        _cmd_remove("ZZZZ", config)
        self.assertIn("nicht", mock_reply.call_args[0][1])

    @patch("insider_alert.alert_engine.telegram_bot._reply")
    def test_process_update_unknown_command(self, mock_reply):
        from insider_alert.alert_engine.telegram_bot import _process_update
        config = self._mock_config()
        update = {"message": {"text": "/unknown", "chat": {"id": 12345}}}
        _process_update(update, config)
        self.assertIn("Unbekannt", mock_reply.call_args[0][1])

    def test_process_update_wrong_chat(self):
        from insider_alert.alert_engine.telegram_bot import _process_update
        config = self._mock_config()
        update = {"message": {"text": "/help", "chat": {"id": 99999}}}
        # Should not raise and should not send anything
        with patch("insider_alert.alert_engine.telegram_bot._reply") as mock_reply:
            _process_update(update, config)
            mock_reply.assert_not_called()


# ===========================================================================
# Weekly Report
# ===========================================================================

class TestWeeklyReport(unittest.TestCase):
    def test_send_weekly_report_calls_telegram(self):
        from insider_alert.alert_engine.weekly_report import send_weekly_report
        from insider_alert.persistence.storage import init_db

        # Use in-memory DB
        db_url = "sqlite:///:memory:"
        init_db(db_url)

        config = MagicMock()
        config.telegram_token = "tok"
        config.telegram_chat_id = "123"
        config.tickers = ["AAPL"]

        with patch("insider_alert.alert_engine.telegram_alert.send_telegram_message", return_value=True) as mock_send:
            with patch("insider_alert.alert_engine.weekly_report.generate_weekly_report", return_value="test report"):
                result = send_weekly_report(config)

        mock_send.assert_called_once()
        self.assertTrue(result)

    def test_generate_report_empty_db(self):
        from insider_alert.alert_engine.weekly_report import generate_weekly_report
        from insider_alert.persistence.storage import init_db

        db_url = "sqlite:///:memory:"
        init_db(db_url)

        report = generate_weekly_report(["AAPL"], db_url=db_url)

        self.assertIn("Performance-Report", report)
        self.assertIn("Analysen", report)


# ===========================================================================
# Telegram Alert – ML score in message
# ===========================================================================

class TestAlertWithMLScore(unittest.TestCase):
    def _make_ticker_score(self):
        from insider_alert.scoring_engine.scorer import TickerScore
        return TickerScore(
            ticker="AAPL",
            total_score=72.5,
            sub_scores={"price_anomaly": 80.0, "volume_anomaly": 65.0},
            flags=["test flag"],
        )

    def test_build_alert_with_ml_score(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        msg = build_alert_message(ts, ml_score=85.0)
        self.assertIn("ML Confidence", msg)
        self.assertIn("85%", msg)
        self.assertIn("🟢", msg)

    def test_build_alert_without_ml_score(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        msg = build_alert_message(ts, ml_score=None)
        self.assertNotIn("ML Confidence", msg)

    def test_build_alert_low_ml_score(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        msg = build_alert_message(ts, ml_score=30.0)
        self.assertIn("🔴", msg)

    def test_build_alert_medium_ml_score(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        msg = build_alert_message(ts, ml_score=50.0)
        self.assertIn("🟡", msg)

    def test_build_alert_with_sr_and_sector(self):
        from insider_alert.alert_engine.telegram_alert import build_alert_message
        ts = self._make_ticker_score()
        sr = {"nearest_support": 95.0, "nearest_resistance": 110.0,
              "distance_to_support_pct": 3.5, "distance_to_resistance_pct": 5.2,
              "sr_zone": "near_support"}
        sector = {"relative_trend": "outperforming", "rs_5d": 2.3,
                  "sector_label": "Technology", "sector_etf": "XLK"}
        msg = build_alert_message(ts, sr_features=sr, sector_features=sector, ml_score=75.0)
        self.assertIn("S/R Levels", msg)
        self.assertIn("Sektor", msg)
        self.assertIn("ML Confidence", msg)


if __name__ == "__main__":
    unittest.main()
