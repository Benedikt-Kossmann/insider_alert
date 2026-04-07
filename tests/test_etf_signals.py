"""Tests for all ETF signal modules (momentum, dip, vol regime, leverage health)."""
import numpy as np
import pytest

from insider_alert.signal_engine.momentum_signal import compute_momentum_signal
from insider_alert.signal_engine.mean_reversion_dip_signal import compute_mean_reversion_dip_signal
from insider_alert.signal_engine.volatility_regime_signal import compute_volatility_regime_signal
from insider_alert.signal_engine.leverage_signal import compute_leverage_health_signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _momentum_feats(**overrides) -> dict:
    base = {
        "rsi_14": 55.0, "macd_line": 0.5, "macd_signal": 0.3,
        "macd_histogram": 0.2, "ema_fast": 110.0, "ema_slow": 105.0,
        "ema_crossover": 1, "adx_14": 28.0, "adx_trending": 1,
    }
    base.update(overrides)
    return base


def _vol_regime_feats(**overrides) -> dict:
    base = {
        "vix_current": 18.0, "vix_sma_20": 19.0, "vix_regime": "normal",
        "bollinger_upper": 112.0, "bollinger_lower": 98.0,
        "bollinger_pct_b": 0.5, "atr_regime": "normal",
        "realized_vol_20d": 0.15,
    }
    base.update(overrides)
    return base


def _leverage_feats(**overrides) -> dict:
    base = {
        "tracking_error_5d": 0.005, "estimated_daily_decay": 0.003,
        "underlying_return_5d": 0.02, "underlying_return_20d": 0.05,
        "underlying_vs_etf_corr_20d": 0.97,
        "underlying_trend_aligned": 1, "leverage_adjusted_atr_pct": 0.03,
    }
    base.update(overrides)
    return base


def _price_feats(**overrides) -> dict:
    base = {
        "return_1d": 0.01, "return_3d": 0.03, "return_5d": 0.05,
        "return_10d": 0.08, "daily_return_zscore": 0.5,
        "gap_up_count_5d": 1, "atr_14": 1.5, "atr_pct": 0.015,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Momentum Signal
# ---------------------------------------------------------------------------
class TestMomentumSignal:
    def test_score_range(self):
        result = compute_momentum_signal(_momentum_feats())
        assert 0 <= result["score"] <= 100

    def test_signal_type(self):
        result = compute_momentum_signal(_momentum_feats())
        assert result["signal_type"] == "momentum"

    def test_flags_is_list(self):
        result = compute_momentum_signal(_momentum_feats())
        assert isinstance(result["flags"], list)

    def test_high_score_bullish(self):
        """Strong bullish features should yield high score."""
        feats = _momentum_feats(
            rsi_14=55, ema_crossover=1, macd_histogram=0.5,
            adx_14=35, adx_trending=1,
        )
        result = compute_momentum_signal(feats, direction="long")
        assert result["score"] >= 70

    def test_low_score_no_momentum(self):
        """No momentum features → low score."""
        feats = _momentum_feats(
            rsi_14=50, ema_crossover=0, macd_histogram=-0.5,
            adx_14=10, adx_trending=0,
        )
        result = compute_momentum_signal(feats, direction="long")
        assert result["score"] < 50

    def test_short_direction(self):
        """For short ETFs, bearish signals should score well."""
        feats = _momentum_feats(
            rsi_14=45, ema_crossover=-1, macd_histogram=-0.5,
            adx_14=30, adx_trending=1,
        )
        result = compute_momentum_signal(feats, direction="short")
        assert result["score"] >= 60

    def test_zero_features(self):
        feats = {k: 0 for k in _momentum_feats()}
        result = compute_momentum_signal(feats)
        assert 0 <= result["score"] <= 100


# ---------------------------------------------------------------------------
# Mean Reversion Dip Signal
# ---------------------------------------------------------------------------
class TestMeanReversionDipSignal:
    def test_score_range(self):
        result = compute_mean_reversion_dip_signal(
            _momentum_feats(), _vol_regime_feats(), _price_feats()
        )
        assert 0 <= result["score"] <= 100

    def test_signal_type(self):
        result = compute_mean_reversion_dip_signal(
            _momentum_feats(), _vol_regime_feats(), _price_feats()
        )
        assert result["signal_type"] == "mean_reversion_dip"

    def test_oversold_dip_buy(self):
        """RSI deeply oversold + low %B should give high score."""
        m = _momentum_feats(rsi_14=22, macd_histogram=0.01)
        v = _vol_regime_feats(bollinger_pct_b=0.03)
        p = _price_feats(daily_return_zscore=-3.0)
        result = compute_mean_reversion_dip_signal(m, v, p, direction="long")
        assert result["score"] >= 70

    def test_neutral_no_dip(self):
        """Neutral features → low score."""
        result = compute_mean_reversion_dip_signal(
            _momentum_feats(rsi_14=50),
            _vol_regime_feats(bollinger_pct_b=0.5),
            _price_feats(daily_return_zscore=0.0),
        )
        assert result["score"] < 30

    def test_short_direction_overbought(self):
        """For short ETFs, overbought = dip-buy opportunity."""
        m = _momentum_feats(rsi_14=78, macd_histogram=-0.5)
        v = _vol_regime_feats(bollinger_pct_b=0.98)
        p = _price_feats(daily_return_zscore=3.0)
        result = compute_mean_reversion_dip_signal(m, v, p, direction="short")
        assert result["score"] >= 60


# ---------------------------------------------------------------------------
# Volatility Regime Signal
# ---------------------------------------------------------------------------
class TestVolatilityRegimeSignal:
    def test_score_range(self):
        result = compute_volatility_regime_signal(_vol_regime_feats(), _leverage_feats())
        assert 0 <= result["score"] <= 100

    def test_signal_type(self):
        result = compute_volatility_regime_signal(_vol_regime_feats(), _leverage_feats())
        assert result["signal_type"] == "volatility_regime"

    def test_low_vix_high_score(self):
        """Low VIX + contracting ATR + low decay = ideal."""
        v = _vol_regime_feats(vix_regime="low", vix_current=12, atr_regime="contracting")
        l = _leverage_feats(estimated_daily_decay=0.002)
        result = compute_volatility_regime_signal(v, l)
        assert result["score"] >= 80

    def test_high_vix_low_score(self):
        """High VIX + expanding ATR + high decay = bad."""
        v = _vol_regime_feats(vix_regime="high", vix_current=35, atr_regime="expanding")
        l = _leverage_feats(estimated_daily_decay=0.03)
        result = compute_volatility_regime_signal(v, l)
        assert result["score"] <= 30


# ---------------------------------------------------------------------------
# Leverage Health Signal
# ---------------------------------------------------------------------------
class TestLeverageHealthSignal:
    def test_score_range(self):
        result = compute_leverage_health_signal(_leverage_feats())
        assert 0 <= result["score"] <= 100

    def test_signal_type(self):
        result = compute_leverage_health_signal(_leverage_feats())
        assert result["signal_type"] == "leverage_health"

    def test_good_health(self):
        """Low tracking error, aligned trend, high corr, low decay."""
        feats = _leverage_feats(
            tracking_error_5d=0.005, underlying_trend_aligned=1,
            underlying_vs_etf_corr_20d=0.98, estimated_daily_decay=0.002,
        )
        result = compute_leverage_health_signal(feats)
        assert result["score"] >= 80

    def test_poor_health(self):
        """High tracking error, misaligned, low corr, high decay."""
        feats = _leverage_feats(
            tracking_error_5d=0.08, underlying_trend_aligned=0,
            underlying_vs_etf_corr_20d=0.4, estimated_daily_decay=0.04,
        )
        result = compute_leverage_health_signal(feats)
        assert result["score"] <= 25

    def test_zero_features(self):
        feats = {k: 0 for k in _leverage_feats()}
        result = compute_leverage_health_signal(feats)
        assert 0 <= result["score"] <= 100
