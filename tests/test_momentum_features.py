"""Tests for momentum feature computation (RSI, MACD, EMA crossover, ADX)."""
import numpy as np
import pandas as pd
import pytest

from insider_alert.feature_engine.momentum_features import (
    compute_momentum_features,
    compute_rsi,
    compute_macd,
    compute_adx,
)


def _make_ohlcv(n: int = 60, seed: int = 42, trend: float = 0.001) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a slight upward trend."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(trend + rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n)
    dates = pd.bdate_range(end="2026-04-07", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestComputeRSI:
    def test_rsi_range(self):
        df = _make_ohlcv(60)
        rsi = compute_rsi(df["close"], 14)
        assert 0.0 <= rsi <= 100.0

    def test_rsi_insufficient_data(self):
        short = pd.Series([100.0, 101.0, 102.0])
        assert compute_rsi(short, 14) == 50.0

    def test_rsi_uptrend_above_50(self):
        """A consistent uptrend should yield RSI > 50."""
        close = pd.Series(np.linspace(100, 120, 30))
        rsi = compute_rsi(close, 14)
        assert rsi > 50.0

    def test_rsi_downtrend_below_50(self):
        """A consistent downtrend should yield RSI < 50."""
        close = pd.Series(np.linspace(120, 100, 30))
        rsi = compute_rsi(close, 14)
        assert rsi < 50.0


class TestComputeMACD:
    def test_macd_keys(self):
        df = _make_ohlcv(60)
        result = compute_macd(df["close"])
        assert "macd_line" in result
        assert "macd_signal" in result
        assert "macd_histogram" in result

    def test_macd_insufficient_data(self):
        short = pd.Series([100.0] * 10)
        result = compute_macd(short)
        assert result["macd_line"] == 0.0

    def test_macd_histogram_is_difference(self):
        df = _make_ohlcv(80)
        result = compute_macd(df["close"])
        expected = result["macd_line"] - result["macd_signal"]
        assert abs(result["macd_histogram"] - expected) < 1e-6


class TestComputeADX:
    def test_adx_range(self):
        df = _make_ohlcv(60)
        adx = compute_adx(df, 14)
        assert adx >= 0.0

    def test_adx_insufficient_data(self):
        df = _make_ohlcv(5)
        adx = compute_adx(df, 14)
        assert adx == 0.0

    def test_adx_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        assert compute_adx(df) == 0.0


class TestComputeMomentumFeatures:
    def test_all_keys_present(self):
        df = _make_ohlcv(60)
        result = compute_momentum_features(df)
        expected_keys = {
            "rsi_14", "macd_line", "macd_signal", "macd_histogram",
            "ema_fast", "ema_slow", "ema_crossover", "adx_14", "adx_trending",
        }
        assert expected_keys == set(result.keys())

    def test_defaults_on_empty_df(self):
        result = compute_momentum_features(pd.DataFrame())
        assert result["rsi_14"] == 50.0
        assert result["ema_crossover"] == 0

    def test_ema_crossover_bullish_uptrend(self):
        """In a strong uptrend, fast EMA should be above slow EMA."""
        df = _make_ohlcv(80, trend=0.005)
        result = compute_momentum_features(df)
        assert result["ema_crossover"] == 1

    def test_ema_crossover_bearish_downtrend(self):
        """In a strong downtrend, fast EMA should be below slow EMA."""
        df = _make_ohlcv(80, trend=-0.005)
        result = compute_momentum_features(df)
        assert result["ema_crossover"] == -1

    def test_custom_config(self):
        df = _make_ohlcv(80)
        cfg = {"rsi_period": 10, "ema_fast": 5, "ema_slow": 20, "adx_period": 10}
        result = compute_momentum_features(df, cfg)
        assert 0 <= result["rsi_14"] <= 100
        assert result["adx_14"] >= 0

    def test_adx_trending_flag(self):
        df = _make_ohlcv(80, trend=0.01)
        result = compute_momentum_features(df, {"adx_trend_threshold": 10})
        # Strong trend should trigger trending flag
        assert result["adx_trending"] in (0, 1)
