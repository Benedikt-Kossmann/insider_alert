"""Tests for volatility regime feature computation."""
import numpy as np
import pandas as pd
import pytest

from insider_alert.feature_engine.volatility_regime_features import (
    compute_volatility_regime_features,
)


def _make_ohlcv(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    volume = rng.integers(1_000_000, 10_000_000, n)
    dates = pd.bdate_range(end="2026-04-07", periods=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_vix(n: int = 60, level: float = 20.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = level + rng.normal(0, 1, n)
    close = np.maximum(close, 5.0)
    dates = pd.bdate_range(end="2026-04-07", periods=n)
    return pd.DataFrame({"close": close}, index=dates)


class TestVolatilityRegimeFeatures:
    def test_all_keys_present(self):
        df = _make_ohlcv()
        result = compute_volatility_regime_features(df)
        expected = {
            "vix_current", "vix_sma_20", "vix_regime",
            "bollinger_upper", "bollinger_lower", "bollinger_pct_b",
            "atr_regime", "realized_vol_20d",
        }
        assert expected == set(result.keys())

    def test_defaults_on_empty(self):
        result = compute_volatility_regime_features(pd.DataFrame())
        assert result["vix_regime"] == "unknown"
        assert result["bollinger_pct_b"] == 0.5

    def test_vix_regime_high(self):
        df = _make_ohlcv()
        vix = _make_vix(level=35.0)
        result = compute_volatility_regime_features(df, vix, vix_high=30, vix_low=15)
        assert result["vix_regime"] == "high"

    def test_vix_regime_low(self):
        df = _make_ohlcv()
        vix = _make_vix(level=12.0)
        result = compute_volatility_regime_features(df, vix, vix_high=30, vix_low=15)
        assert result["vix_regime"] == "low"

    def test_vix_regime_normal(self):
        df = _make_ohlcv()
        vix = _make_vix(level=20.0)
        result = compute_volatility_regime_features(df, vix, vix_high=30, vix_low=15)
        assert result["vix_regime"] == "normal"

    def test_vix_regime_no_vix_data(self):
        df = _make_ohlcv()
        result = compute_volatility_regime_features(df, None)
        assert result["vix_regime"] == "unknown"

    def test_bollinger_pct_b_range(self):
        df = _make_ohlcv(60)
        result = compute_volatility_regime_features(df)
        assert 0.0 <= result["bollinger_pct_b"] <= 1.0

    def test_bollinger_upper_above_lower(self):
        df = _make_ohlcv(60)
        result = compute_volatility_regime_features(df)
        assert result["bollinger_upper"] >= result["bollinger_lower"]

    def test_atr_regime_valid(self):
        df = _make_ohlcv(60)
        result = compute_volatility_regime_features(df)
        assert result["atr_regime"] in ("contracting", "normal", "expanding")

    def test_realized_vol_positive(self):
        df = _make_ohlcv(60)
        result = compute_volatility_regime_features(df)
        assert result["realized_vol_20d"] >= 0.0

    def test_short_data_bollinger(self):
        """With fewer bars than bollinger_period, should get safe defaults."""
        df = _make_ohlcv(10)
        result = compute_volatility_regime_features(df, bollinger_period=20)
        assert result["bollinger_pct_b"] == 0.5
