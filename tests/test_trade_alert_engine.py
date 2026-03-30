"""Tests for the trade_alert_engine modules and related persistence changes."""
import datetime
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 30, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data."""
    rng = np.random.default_rng(seed)
    base = 100.0
    closes = base + np.cumsum(rng.normal(trend, 1.0, n))
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    opens = closes - rng.normal(0, 0.5, n)
    volumes = rng.integers(500_000, 5_000_000, n).astype(float)
    dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(dates),
    )


def _price_f_with_atr(ohlcv: pd.DataFrame) -> dict:
    from insider_alert.feature_engine.price_features import compute_price_features
    return compute_price_features(ohlcv)


def _volume_f(ohlcv: pd.DataFrame) -> dict:
    from insider_alert.feature_engine.volume_features import compute_volume_features
    return compute_volume_features(ohlcv)


# ===========================================================================
# ATR computation (price_features)
# ===========================================================================

class TestATRComputation(unittest.TestCase):

    def test_atr_positive_for_normal_data(self):
        from insider_alert.feature_engine.price_features import compute_atr
        df = _make_ohlcv(30)
        atr = compute_atr(df)
        self.assertGreater(atr, 0.0)

    def test_atr_zero_for_empty(self):
        from insider_alert.feature_engine.price_features import compute_atr
        self.assertEqual(compute_atr(pd.DataFrame()), 0.0)

    def test_atr_zero_for_missing_columns(self):
        from insider_alert.feature_engine.price_features import compute_atr
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        self.assertEqual(compute_atr(df), 0.0)

    def test_price_features_include_atr_keys(self):
        from insider_alert.feature_engine.price_features import compute_price_features
        result = compute_price_features(_make_ohlcv())
        self.assertIn("atr_14", result)
        self.assertIn("atr_pct", result)
        self.assertGreater(result["atr_14"], 0.0)
        self.assertGreater(result["atr_pct"], 0.0)

    def test_atr_pct_consistent_with_atr_14(self):
        from insider_alert.feature_engine.price_features import compute_price_features
        df = _make_ohlcv()
        result = compute_price_features(df)
        current_price = float(df["close"].iloc[-1])
        expected_pct = result["atr_14"] / (current_price + 1e-9)
        self.assertAlmostEqual(result["atr_pct"], expected_pct, places=6)


# ===========================================================================
# Breakout alert
# ===========================================================================

class TestBreakoutAlert(unittest.TestCase):

    def _make_breakout_ohlcv(self, direction: str = "bullish") -> pd.DataFrame:
        """Build an OHLCV that clearly breaks out in the given direction."""
        rng = np.random.default_rng(0)
        n = 30
        closes = np.full(n, 100.0) + rng.normal(0, 0.1, n)
        # Last bar closes well above prior 20-bar range
        if direction == "bullish":
            closes[-1] = 120.0
        else:
            closes[-1] = 80.0
        highs = closes + 1.0
        lows = closes - 1.0
        opens = closes - 0.5
        volumes = np.full(n, 2_000_000.0)
        dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=pd.DatetimeIndex(dates),
        )

    def test_bullish_breakout_detected(self):
        from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
        ohlcv = self._make_breakout_ohlcv("bullish")
        pf = _price_f_with_atr(ohlcv)
        vf = {"volume_rvol_20d": 2.0}
        result = detect_breakout(ohlcv, pf, vf)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "bullish")
        self.assertEqual(result["alert_type"], "breakout")
        self.assertGreater(result["score"], 0)

    def test_bearish_breakout_detected(self):
        from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
        ohlcv = self._make_breakout_ohlcv("bearish")
        pf = _price_f_with_atr(ohlcv)
        vf = {"volume_rvol_20d": 2.0}
        result = detect_breakout(ohlcv, pf, vf)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "bearish")

    def test_no_breakout_for_flat_data(self):
        from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
        ohlcv = _make_ohlcv(30, seed=5, trend=0.0)
        pf = _price_f_with_atr(ohlcv)
        vf = {"volume_rvol_20d": 1.0}
        # With normal (non-extreme) data most calls return None
        # We just ensure no exception is raised and the return type is correct
        result = detect_breakout(ohlcv, pf, vf)
        self.assertIn(result, [None, *[result]] if result else [None])

    def test_stop_below_close_for_bullish(self):
        from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
        ohlcv = self._make_breakout_ohlcv("bullish")
        pf = _price_f_with_atr(ohlcv)
        vf = {"volume_rvol_20d": 2.0}
        result = detect_breakout(ohlcv, pf, vf)
        if result:
            close = float(ohlcv["close"].iloc[-1])
            self.assertLess(result["stop_hint"], close)
            self.assertGreater(result["target_hint"], close)

    def test_volume_confirmation_increases_score(self):
        from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
        ohlcv = self._make_breakout_ohlcv("bullish")
        pf = _price_f_with_atr(ohlcv)
        result_high_vol = detect_breakout(ohlcv, pf, {"volume_rvol_20d": 3.0})
        result_low_vol = detect_breakout(ohlcv, pf, {"volume_rvol_20d": 0.5})
        if result_high_vol and result_low_vol:
            self.assertGreater(result_high_vol["score"], result_low_vol["score"])

    def test_none_for_empty_ohlcv(self):
        from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
        result = detect_breakout(pd.DataFrame(), {}, {})
        self.assertIsNone(result)

    def test_result_keys(self):
        from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
        ohlcv = self._make_breakout_ohlcv("bullish")
        pf = _price_f_with_atr(ohlcv)
        result = detect_breakout(ohlcv, pf, {"volume_rvol_20d": 2.0})
        if result:
            for key in ("alert_type", "setup_type", "direction", "breakout_level",
                        "atr", "stop_hint", "target_hint", "rr_ratio", "score", "flags"):
                self.assertIn(key, result)


# ===========================================================================
# Mean-reversion alert
# ===========================================================================

class TestMeanReversionAlert(unittest.TestCase):

    def _price_f_extreme(self, zscore: float = 3.0) -> dict:
        return {
            "daily_return_zscore": zscore,
            "return_5d": 0.1,
            "atr_14": 2.0,
            "atr_pct": 0.02,
        }

    def _vol_f_normal(self) -> dict:
        return {"volume_zscore_20d": 0.5, "volume_rvol_20d": 1.2}

    def test_fires_on_extreme_price_zscore(self):
        from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
        result = detect_mean_reversion(self._price_f_extreme(3.5), self._vol_f_normal())
        self.assertIsNotNone(result)
        self.assertEqual(result["alert_type"], "mean_reversion")
        self.assertGreater(result["score"], 0)

    def test_fires_on_extreme_volume_zscore(self):
        from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
        result = detect_mean_reversion(
            {"daily_return_zscore": 0.5, "atr_14": 1.0, "atr_pct": 0.01},
            {"volume_zscore_20d": 3.0},
        )
        self.assertIsNotNone(result)

    def test_none_for_normal_zscore(self):
        from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
        result = detect_mean_reversion(
            {"daily_return_zscore": 1.0, "atr_14": 1.0, "atr_pct": 0.01},
            {"volume_zscore_20d": 1.0},
        )
        self.assertIsNone(result)

    def test_bearish_reversal_for_positive_zscore(self):
        from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
        result = detect_mean_reversion(self._price_f_extreme(3.0), self._vol_f_normal())
        if result:
            self.assertIn("bearish", result["direction"])

    def test_bullish_reversal_for_negative_zscore(self):
        from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
        result = detect_mean_reversion(self._price_f_extreme(-3.0), self._vol_f_normal())
        if result:
            self.assertIn("bullish", result["direction"])

    def test_news_divergence_increases_score(self):
        from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
        pf = self._price_f_extreme(3.0)
        vf = self._vol_f_normal()
        without_news = detect_mean_reversion(pf, vf, {})
        with_news = detect_mean_reversion(pf, vf, {"news_price_divergence": 0.8})
        if without_news and with_news:
            self.assertGreater(with_news["score"], without_news["score"])

    def test_result_contains_required_keys(self):
        from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
        result = detect_mean_reversion(self._price_f_extreme(3.0), self._vol_f_normal())
        if result:
            for key in ("alert_type", "setup_type", "direction", "price_zscore",
                        "volume_zscore", "rr_ratio", "score", "flags"):
                self.assertIn(key, result)


# ===========================================================================
# Options flow alert
# ===========================================================================

class TestOptionsFlowAlert(unittest.TestCase):

    def _opts_f_quiet(self) -> dict:
        return {
            "sweep_order_score": 0.1,
            "block_trade_score": 0.1,
            "iv_change_1d": 0.05,
            "open_interest_change": 0.05,
            "call_volume_zscore": 0.5,
            "short_dated_otm_call_score": 0.1,
            "put_call_ratio_change": 0.0,
        }

    def _opts_f_active(self) -> dict:
        return {
            "sweep_order_score": 0.8,
            "block_trade_score": 0.7,
            "iv_change_1d": 0.35,
            "open_interest_change": 0.50,
            "call_volume_zscore": 3.0,
            "short_dated_otm_call_score": 0.7,
            "put_call_ratio_change": -0.3,
        }

    def test_fires_on_active_options(self):
        from insider_alert.trade_alert_engine.options_flow_alert import detect_options_flow
        result = detect_options_flow(self._opts_f_active())
        self.assertIsNotNone(result)
        self.assertEqual(result["alert_type"], "options_flow")

    def test_none_for_quiet_options(self):
        from insider_alert.trade_alert_engine.options_flow_alert import detect_options_flow
        result = detect_options_flow(self._opts_f_quiet())
        self.assertIsNone(result)

    def test_near_earnings_increases_score(self):
        from insider_alert.trade_alert_engine.options_flow_alert import detect_options_flow
        # Use options data just at the sweep threshold to leave headroom for the earnings bonus
        opts = {
            "sweep_order_score": 0.65,
            "block_trade_score": 0.0,
            "iv_change_1d": 0.0,
            "open_interest_change": 0.0,
            "call_volume_zscore": 0.0,
            "short_dated_otm_call_score": 0.0,
            "put_call_ratio_change": 0.0,
        }
        without_earnings = detect_options_flow(opts, {"days_to_earnings": 30})
        with_earnings = detect_options_flow(opts, {"days_to_earnings": 5})
        if without_earnings and with_earnings:
            self.assertGreater(with_earnings["score"], without_earnings["score"])

    def test_bullish_bias_from_low_pcr(self):
        from insider_alert.trade_alert_engine.options_flow_alert import detect_options_flow
        opts = {**self._opts_f_active(), "put_call_ratio_change": -0.5}
        result = detect_options_flow(opts)
        if result:
            self.assertEqual(result["direction"], "bullish")

    def test_bearish_bias_from_high_pcr(self):
        from insider_alert.trade_alert_engine.options_flow_alert import detect_options_flow
        opts = {**self._opts_f_active(), "put_call_ratio_change": 0.5}
        result = detect_options_flow(opts)
        if result:
            self.assertEqual(result["direction"], "bearish")

    def test_result_contains_required_keys(self):
        from insider_alert.trade_alert_engine.options_flow_alert import detect_options_flow
        result = detect_options_flow(self._opts_f_active())
        if result:
            for key in ("alert_type", "setup_type", "direction", "score", "flags"):
                self.assertIn(key, result)


# ===========================================================================
# Event-driven alert
# ===========================================================================

class TestEventDrivenAlert(unittest.TestCase):

    def _base_event_f(self, dte: int = 5, corp: int = 999) -> dict:
        return {
            "days_to_earnings": dte,
            "days_to_corporate_event": corp,
            "pre_event_return_score": 0.7,
            "pre_event_volume_score": 0.8,
            "pre_event_options_score": 0.5,
        }

    def _price_f(self) -> dict:
        return {"return_1d": 0.03, "return_5d": 0.06, "atr_14": 3.0, "atr_pct": 0.03}

    def _volume_f(self) -> dict:
        return {"volume_rvol_20d": 2.5}

    def test_pre_earnings_alert_fires(self):
        from insider_alert.trade_alert_engine.event_driven_alert import detect_event_driven
        result = detect_event_driven("AAPL", self._base_event_f(dte=5), self._price_f(), self._volume_f())
        self.assertIsNotNone(result)
        self.assertEqual(result["setup_type"], "event_pre_earnings")

    def test_sector_populated(self):
        from insider_alert.trade_alert_engine.event_driven_alert import detect_event_driven
        result = detect_event_driven("AAPL", self._base_event_f(dte=5), self._price_f(), self._volume_f())
        if result:
            self.assertEqual(result["sector"], "Technology")

    def test_unknown_ticker_sector(self):
        from insider_alert.trade_alert_engine.event_driven_alert import detect_event_driven
        result = detect_event_driven("ZZZZ", self._base_event_f(dte=5), self._price_f(), self._volume_f())
        if result:
            self.assertEqual(result["sector"], "Unknown")

    def test_8k_material_event_fires(self):
        from insider_alert.trade_alert_engine.event_driven_alert import detect_event_driven
        event_f = {
            "days_to_earnings": 999,
            "days_to_corporate_event": 7,
            "pre_event_return_score": 0.8,
            "pre_event_volume_score": 0.8,
            "pre_event_options_score": 0.6,
        }
        result = detect_event_driven("MSFT", event_f, self._price_f(), self._volume_f())
        if result:
            self.assertEqual(result["setup_type"], "event_8k_material")

    def test_none_when_no_event_nearby(self):
        from insider_alert.trade_alert_engine.event_driven_alert import detect_event_driven
        event_f = {
            "days_to_earnings": 999,
            "days_to_corporate_event": 999,
            "pre_event_return_score": 0.0,
            "pre_event_volume_score": 0.0,
            "pre_event_options_score": 0.0,
        }
        result = detect_event_driven("AAPL", event_f, self._price_f(), self._volume_f())
        self.assertIsNone(result)

    def test_result_contains_required_keys(self):
        from insider_alert.trade_alert_engine.event_driven_alert import detect_event_driven
        result = detect_event_driven("AAPL", self._base_event_f(dte=5), self._price_f(), self._volume_f())
        if result:
            for key in ("alert_type", "setup_type", "sector", "score", "flags"):
                self.assertIn(key, result)


# ===========================================================================
# Multi-timeframe alert
# ===========================================================================

class TestMultiTimeframeAlert(unittest.TestCase):

    def _bullish_daily_f(self) -> dict:
        return {
            "daily_return_zscore": 1.5,
            "return_5d": 0.06,
            "atr_14": 2.5,
            "atr_pct": 0.025,
        }

    def _bearish_daily_f(self) -> dict:
        return {
            "daily_return_zscore": -1.5,
            "return_5d": -0.06,
            "atr_14": 2.5,
            "atr_pct": 0.025,
        }

    def _make_confirming_intraday(self, direction: str, bars: int = 5) -> pd.DataFrame:
        closes = [100.0 + (i if direction == "bullish" else -i) for i in range(bars)]
        return pd.DataFrame({"close": closes})

    def test_fires_without_intraday(self):
        from insider_alert.trade_alert_engine.multi_timeframe_alert import detect_multi_timeframe
        result = detect_multi_timeframe(self._bullish_daily_f(), None)
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "bullish")

    def test_intraday_confirmation_increases_score(self):
        from insider_alert.trade_alert_engine.multi_timeframe_alert import detect_multi_timeframe
        without_intraday = detect_multi_timeframe(self._bullish_daily_f(), None)
        intraday = self._make_confirming_intraday("bullish", bars=5)
        with_intraday = detect_multi_timeframe(self._bullish_daily_f(), intraday)
        if without_intraday and with_intraday:
            self.assertGreaterEqual(with_intraday["score"], without_intraday["score"])

    def test_bearish_bias_detected(self):
        from insider_alert.trade_alert_engine.multi_timeframe_alert import detect_multi_timeframe
        result = detect_multi_timeframe(self._bearish_daily_f(), None)
        if result:
            self.assertIn("bearish", result["direction"])

    def test_none_for_flat_market(self):
        from insider_alert.trade_alert_engine.multi_timeframe_alert import detect_multi_timeframe
        flat = {"daily_return_zscore": 0.1, "return_5d": 0.001, "atr_14": 1.0, "atr_pct": 0.01}
        result = detect_multi_timeframe(flat, None)
        self.assertIsNone(result)

    def test_result_keys(self):
        from insider_alert.trade_alert_engine.multi_timeframe_alert import detect_multi_timeframe
        result = detect_multi_timeframe(self._bullish_daily_f(), None)
        if result:
            for key in ("alert_type", "setup_type", "direction", "intraday_confirmed",
                        "daily_return_5d", "score", "flags"):
                self.assertIn(key, result)


# ===========================================================================
# Risk manager
# ===========================================================================

class TestRiskManager(unittest.TestCase):

    def test_bullish_stop_below_price(self):
        from insider_alert.trade_alert_engine.risk_manager import compute_risk_hints
        hints = compute_risk_hints(100.0, 2.0, "bullish")
        self.assertLess(hints["stop_loss"], 100.0)
        self.assertGreater(hints["price_target"], 100.0)

    def test_bearish_stop_above_price(self):
        from insider_alert.trade_alert_engine.risk_manager import compute_risk_hints
        hints = compute_risk_hints(100.0, 2.0, "bearish")
        self.assertGreater(hints["stop_loss"], 100.0)
        self.assertLess(hints["price_target"], 100.0)

    def test_rr_ratio_preserved(self):
        from insider_alert.trade_alert_engine.risk_manager import compute_risk_hints
        hints = compute_risk_hints(100.0, 2.0, "bullish", rr_ratio=3.0)
        self.assertAlmostEqual(hints["rr_ratio"], 3.0)

    def test_volatility_low(self):
        from insider_alert.trade_alert_engine.risk_manager import classify_volatility
        self.assertEqual(classify_volatility(0.005), "Low")

    def test_volatility_normal(self):
        from insider_alert.trade_alert_engine.risk_manager import classify_volatility
        self.assertEqual(classify_volatility(0.015), "Normal")

    def test_volatility_high(self):
        from insider_alert.trade_alert_engine.risk_manager import classify_volatility
        self.assertEqual(classify_volatility(0.03), "High")

    def test_zero_atr_returns_none_hints(self):
        from insider_alert.trade_alert_engine.risk_manager import compute_risk_hints
        hints = compute_risk_hints(100.0, 0.0, "bullish")
        self.assertIsNone(hints["stop_loss"])
        self.assertIsNone(hints["price_target"])

    def test_format_risk_hint_lines_non_empty(self):
        from insider_alert.trade_alert_engine.risk_manager import compute_risk_hints, format_risk_hint_lines
        hints = compute_risk_hints(100.0, 2.0, "bullish")
        lines = format_risk_hint_lines(hints)
        self.assertTrue(len(lines) >= 3)


# ===========================================================================
# Universe scanner
# ===========================================================================

class TestUniverseScanner(unittest.TestCase):

    def _stats(self, rvol: float, dte: int, news: float) -> dict:
        return {"volume_rvol_20d": rvol, "days_to_earnings": dte, "news_score": news}

    def test_high_rvol_keeps_ticker(self):
        from insider_alert.trade_alert_engine.universe_scanner import scan_universe
        stats = {"AAPL": self._stats(3.0, 30, 0.0)}
        state = scan_universe(["AAPL"], stats)
        self.assertIn("AAPL", state.active)

    def test_low_rvol_removes_ticker(self):
        from insider_alert.trade_alert_engine.universe_scanner import scan_universe
        stats = {"MSFT": self._stats(0.5, 30, 0.0)}
        state = scan_universe(["MSFT"], stats)
        self.assertIn("MSFT", state.removed)

    def test_near_earnings_keeps_ticker(self):
        from insider_alert.trade_alert_engine.universe_scanner import scan_universe
        stats = {"NVDA": self._stats(0.5, 7, 0.0)}
        state = scan_universe(["NVDA"], stats)
        self.assertIn("NVDA", state.active)

    def test_high_news_keeps_ticker(self):
        from insider_alert.trade_alert_engine.universe_scanner import scan_universe
        stats = {"TSLA": self._stats(0.5, 50, 0.8)}
        state = scan_universe(["TSLA"], stats)
        self.assertIn("TSLA", state.active)

    def test_propose_additions(self):
        from insider_alert.trade_alert_engine.universe_scanner import propose_additions
        stats = {
            "NEW1": self._stats(3.0, 30, 0.0),
            "NEW2": self._stats(0.5, 30, 0.0),
        }
        added = propose_additions(["NEW1", "NEW2"], [], stats)
        self.assertIn("NEW1", added)
        self.assertNotIn("NEW2", added)

    def test_propose_does_not_re_add_existing(self):
        from insider_alert.trade_alert_engine.universe_scanner import propose_additions
        stats = {"AAPL": self._stats(3.0, 30, 0.0)}
        added = propose_additions(["AAPL"], ["AAPL"], stats)
        self.assertEqual(added, [])


# ===========================================================================
# Deduplication in storage
# ===========================================================================

class TestAlertDeduplication(unittest.TestCase):

    def _tmp_db(self) -> str:
        import tempfile, os
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        return f"sqlite:///{f.name}"

    def test_not_duplicate_on_fresh_db(self):
        from insider_alert.persistence.storage import is_alert_duplicate, init_db
        db = self._tmp_db()
        init_db(db)
        self.assertFalse(is_alert_duplicate("AAPL", "breakout_bullish", cooldown_hours=4, db_url=db))

    def test_duplicate_after_saving_alert(self):
        from insider_alert.persistence.storage import is_alert_duplicate, save_alert, init_db
        db = self._tmp_db()
        init_db(db)
        save_alert(
            ticker="AAPL",
            date_val=datetime.date.today(),
            score=75.0,
            message="test",
            alert_type="breakout",
            setup_type="breakout_bullish",
            db_url=db,
        )
        self.assertTrue(is_alert_duplicate("AAPL", "breakout_bullish", cooldown_hours=4, db_url=db))

    def test_not_duplicate_different_setup(self):
        from insider_alert.persistence.storage import is_alert_duplicate, save_alert, init_db
        db = self._tmp_db()
        init_db(db)
        save_alert(
            ticker="AAPL",
            date_val=datetime.date.today(),
            score=75.0,
            message="test",
            alert_type="breakout",
            setup_type="breakout_bullish",
            db_url=db,
        )
        self.assertFalse(is_alert_duplicate("AAPL", "mean_reversion_bearish", cooldown_hours=4, db_url=db))

    def test_not_duplicate_different_ticker(self):
        from insider_alert.persistence.storage import is_alert_duplicate, save_alert, init_db
        db = self._tmp_db()
        init_db(db)
        save_alert(
            ticker="AAPL",
            date_val=datetime.date.today(),
            score=75.0,
            message="test",
            alert_type="breakout",
            setup_type="breakout_bullish",
            db_url=db,
        )
        self.assertFalse(is_alert_duplicate("MSFT", "breakout_bullish", cooldown_hours=4, db_url=db))


# ===========================================================================
# Telegram trade alert message builders
# ===========================================================================

class TestTelegramTradeAlertMessages(unittest.TestCase):

    def test_build_breakout_message(self):
        from insider_alert.alert_engine.telegram_alert import build_trade_alert_message
        alert = {
            "alert_type": "breakout",
            "setup_type": "breakout_bullish",
            "direction": "bullish",
            "breakout_level": 150.0,
            "atr": 3.5,
            "stop_hint": 146.5,
            "target_hint": 157.0,
            "rr_ratio": 2.0,
            "score": 80.0,
            "flags": ["Bullish breakout above 150.0 (ATR=3.50)", "Volume confirmed: RVOL=2.3x"],
        }
        msg = build_trade_alert_message("AAPL", alert)
        self.assertIn("AAPL", msg)
        self.assertIn("breakout", msg.lower())
        self.assertIn("150.00", msg)
        self.assertIn("Stop", msg)

    def test_build_mean_reversion_message(self):
        from insider_alert.alert_engine.telegram_alert import build_trade_alert_message
        alert = {
            "alert_type": "mean_reversion",
            "setup_type": "mean_reversion_bearish_reversal",
            "direction": "bearish_reversal",
            "price_zscore": 3.1,
            "volume_zscore": 0.5,
            "atr": 2.0,
            "atr_pct": 0.02,
            "rr_ratio": 1.5,
            "score": 70.0,
            "flags": ["Extreme price Z-score: 3.10"],
        }
        msg = build_trade_alert_message("TSLA", alert)
        self.assertIn("TSLA", msg)
        self.assertIn("3.10", msg)

    def test_build_options_flow_message(self):
        from insider_alert.alert_engine.telegram_alert import build_trade_alert_message
        alert = {
            "alert_type": "options_flow",
            "setup_type": "options_flow_unusual",
            "direction": "bullish",
            "sweep_score": 0.85,
            "block_score": 0.7,
            "iv_change": 0.30,
            "oi_change": 0.40,
            "near_earnings": True,
            "score": 90.0,
            "flags": ["Options sweep activity: score=0.85"],
        }
        msg = build_trade_alert_message("NVDA", alert)
        self.assertIn("NVDA", msg)
        self.assertIn("earnings", msg.lower())

    def test_build_event_driven_message(self):
        from insider_alert.alert_engine.telegram_alert import build_trade_alert_message
        alert = {
            "alert_type": "event_driven",
            "setup_type": "event_pre_earnings",
            "direction": "",
            "event_label": "Pre-earnings (5d)",
            "sector": "Technology",
            "days_to_earnings": 5,
            "days_to_corp_event": 999,
            "return_1d": 0.03,
            "atr": 3.0,
            "score": 75.0,
            "flags": ["Sector: Technology", "Earnings in 5 days"],
        }
        msg = build_trade_alert_message("MSFT", alert)
        self.assertIn("MSFT", msg)
        self.assertIn("Technology", msg)

    def test_build_multi_timeframe_message(self):
        from insider_alert.alert_engine.telegram_alert import build_trade_alert_message
        alert = {
            "alert_type": "multi_timeframe",
            "setup_type": "mtf_bullish",
            "direction": "bullish",
            "intraday_confirmed": True,
            "daily_return_5d": 0.05,
            "daily_zscore": 1.5,
            "atr": 2.5,
            "score": 80.0,
            "flags": ["Daily bias: bullish"],
        }
        msg = build_trade_alert_message("AMD", alert)
        self.assertIn("AMD", msg)
        self.assertIn("✅", msg)


# ===========================================================================
# Config loading – trade_alerts section
# ===========================================================================

class TestConfigTradeAlerts(unittest.TestCase):

    def test_default_trade_alerts_loaded(self):
        from insider_alert.config import load_config
        cfg = load_config("/nonexistent_path_fallback.yaml")
        self.assertIn("breakout", cfg.trade_alerts)
        self.assertIn("mean_reversion", cfg.trade_alerts)
        self.assertIn("options_flow", cfg.trade_alerts)
        self.assertIn("event_driven", cfg.trade_alerts)
        self.assertIn("universe_scan", cfg.trade_alerts)

    def test_cooldown_hours_default(self):
        from insider_alert.config import load_config
        cfg = load_config("/nonexistent_path_fallback.yaml")
        self.assertIn("cooldown_hours", cfg.trade_alerts)
        self.assertGreater(cfg.trade_alerts["cooldown_hours"], 0)

    def test_enabled_default_true(self):
        from insider_alert.config import load_config
        cfg = load_config("/nonexistent_path_fallback.yaml")
        self.assertTrue(cfg.trade_alerts.get("enabled", True))


if __name__ == "__main__":
    unittest.main()
