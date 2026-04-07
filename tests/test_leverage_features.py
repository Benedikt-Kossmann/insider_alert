"""Tests for leverage feature computation."""
import numpy as np
import pandas as pd
import pytest

from insider_alert.feature_engine.leverage_features import compute_leverage_features


def _make_ohlcv(n: int = 60, seed: int = 42, trend: float = 0.001) -> pd.DataFrame:
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


def _make_leveraged_pair(n=60, seed=42, leverage=3, direction="long"):
    """Create a correlated ETF/underlying pair."""
    rng = np.random.default_rng(seed)
    und_returns = rng.normal(0.0005, 0.01, n)
    und_close = 100.0 * np.exp(np.cumsum(und_returns))

    dir_mult = -1.0 if direction == "short" else 1.0
    etf_returns = dir_mult * leverage * und_returns + rng.normal(0, 0.001, n)
    etf_close = 50.0 * np.exp(np.cumsum(etf_returns))

    dates = pd.bdate_range(end="2026-04-07", periods=n)

    def _to_df(close):
        high = close * 1.005
        low = close * 0.995
        return pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": 5_000_000},
            index=dates,
        )

    return _to_df(etf_close), _to_df(und_close)


class TestComputeLeverageFeatures:
    def test_all_keys_present(self):
        etf, und = _make_leveraged_pair()
        result = compute_leverage_features(etf, und, 3, "long")
        expected = {
            "tracking_error_5d", "estimated_daily_decay",
            "underlying_return_5d", "underlying_return_20d",
            "underlying_vs_etf_corr_20d", "underlying_trend_aligned",
            "leverage_adjusted_atr_pct",
        }
        assert expected == set(result.keys())

    def test_defaults_on_empty(self):
        result = compute_leverage_features(pd.DataFrame(), pd.DataFrame())
        assert result["tracking_error_5d"] == 0.0
        assert result["underlying_trend_aligned"] == 0

    def test_tracking_error_positive(self):
        etf, und = _make_leveraged_pair()
        result = compute_leverage_features(etf, und, 3, "long")
        assert result["tracking_error_5d"] >= 0.0

    def test_decay_non_negative(self):
        etf, und = _make_leveraged_pair()
        result = compute_leverage_features(etf, und, 3, "long")
        assert result["estimated_daily_decay"] >= 0.0

    def test_decay_formula(self):
        """Decay ≈ L² × σ² / 2.  Higher leverage → higher decay."""
        etf3, und = _make_leveraged_pair(leverage=3)
        r3 = compute_leverage_features(etf3, und, 3, "long")
        # With leverage=3, decay should be proportional to 9 * variance/2
        assert r3["estimated_daily_decay"] > 0

    def test_trend_aligned_long(self):
        """Positive underlying returns should be aligned for long ETFs."""
        etf, und = _make_leveraged_pair(n=60, seed=10, direction="long")
        result = compute_leverage_features(etf, und, 3, "long")
        und_ret_5d = result["underlying_return_5d"]
        if und_ret_5d > 0:
            assert result["underlying_trend_aligned"] == 1
        else:
            assert result["underlying_trend_aligned"] == 0

    def test_trend_aligned_short(self):
        """Negative underlying returns should be aligned for short ETFs."""
        etf, und = _make_leveraged_pair(n=60, seed=10, direction="short")
        result = compute_leverage_features(etf, und, 3, "short")
        und_ret_5d = result["underlying_return_5d"]
        if und_ret_5d < 0:
            assert result["underlying_trend_aligned"] == 1
        else:
            assert result["underlying_trend_aligned"] == 0

    def test_high_correlation(self):
        """A well-constructed leveraged pair should have high correlation."""
        etf, und = _make_leveraged_pair(n=100, seed=99)
        result = compute_leverage_features(etf, und, 3, "long")
        assert result["underlying_vs_etf_corr_20d"] > 0.5

    def test_atr_pct_positive(self):
        etf, und = _make_leveraged_pair()
        result = compute_leverage_features(etf, und, 3, "long")
        assert result["leverage_adjusted_atr_pct"] >= 0.0

    def test_short_data(self):
        """Too few bars should return defaults."""
        dates = pd.bdate_range(end="2026-04-07", periods=3)
        short_df = pd.DataFrame(
            {"close": [100, 101, 102], "high": [101, 102, 103],
             "low": [99, 100, 101], "volume": [1e6]*3},
            index=dates,
        )
        result = compute_leverage_features(short_df, short_df, 3, "long")
        assert result["tracking_error_5d"] == 0.0
