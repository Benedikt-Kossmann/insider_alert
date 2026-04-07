"""Tests for leveraged ETF alert detectors and risk manager extensions."""
import pytest

from insider_alert.trade_alert_engine.leveraged_etf_alert import (
    detect_leveraged_etf_entry,
    detect_leveraged_etf_exit,
)
from insider_alert.trade_alert_engine.risk_manager import (
    compute_leverage_risk_hints,
    format_leverage_risk_lines,
)
from insider_alert.alert_engine.telegram_alert import build_etf_alert_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_signals(momentum=70, dip=30, vol=60, health=65, price=50, volume=50):
    return {
        "momentum": {"signal_type": "momentum", "score": momentum, "flags": []},
        "mean_reversion_dip": {"signal_type": "mean_reversion_dip", "score": dip, "flags": []},
        "volatility_regime": {"signal_type": "volatility_regime", "score": vol, "flags": []},
        "leverage_health": {"signal_type": "leverage_health", "score": health, "flags": []},
        "price_anomaly": {"signal_type": "price_anomaly", "score": price, "flags": []},
        "volume_anomaly": {"signal_type": "volume_anomaly", "score": volume, "flags": []},
    }


def _momentum_feats(**kw):
    base = {"rsi_14": 55, "macd_histogram": 0.2, "ema_crossover": 1, "adx_14": 28, "adx_trending": 1}
    base.update(kw)
    return base


def _vol_regime_feats(**kw):
    base = {"vix_regime": "normal", "vix_current": 18.0}
    base.update(kw)
    return base


def _leverage_feats(**kw):
    base = {
        "tracking_error_5d": 0.005, "estimated_daily_decay": 0.003,
        "underlying_return_5d": 0.02, "underlying_trend_aligned": 1,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Entry Detection
# ---------------------------------------------------------------------------
class TestDetectEntry:
    def test_momentum_entry(self):
        signals = _make_signals(momentum=70, health=65)
        result = detect_leveraged_etf_entry(
            "TQQQ", signals, _momentum_feats(), _vol_regime_feats(), _leverage_feats(),
            direction="long", underlying="QQQ", leverage=3,
        )
        assert result is not None
        assert result["setup_type"] == "momentum_entry"
        assert result["alert_type"] == "leveraged_etf"
        assert result["score"] > 0

    def test_dip_buy_entry(self):
        signals = _make_signals(momentum=40, dip=75)
        result = detect_leveraged_etf_entry(
            "TQQQ", signals, _momentum_feats(rsi_14=25), _vol_regime_feats(vix_current=22),
            _leverage_feats(), direction="long", underlying="QQQ", leverage=3,
        )
        assert result is not None
        assert result["setup_type"] == "dip_buy"

    def test_no_entry_low_scores(self):
        signals = _make_signals(momentum=30, dip=20, health=20)
        result = detect_leveraged_etf_entry(
            "TQQQ", signals, _momentum_feats(), _vol_regime_feats(), _leverage_feats(),
        )
        assert result is None

    def test_no_entry_high_vix(self):
        """High VIX should block momentum entry."""
        signals = _make_signals(momentum=70, health=65)
        result = detect_leveraged_etf_entry(
            "TQQQ", signals, _momentum_feats(),
            _vol_regime_feats(vix_regime="high", vix_current=35),
            _leverage_feats(),
        )
        assert result is None

    def test_no_entry_low_health(self):
        """Low health score should block momentum entry."""
        signals = _make_signals(momentum=70, health=30)
        result = detect_leveraged_etf_entry(
            "TQQQ", signals, _momentum_feats(), _vol_regime_feats(), _leverage_feats(),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Exit Detection
# ---------------------------------------------------------------------------
class TestDetectExit:
    def test_exit_on_high_decay(self):
        result = detect_leveraged_etf_exit(
            "TQQQ", _leverage_feats(estimated_daily_decay=0.03),
            _vol_regime_feats(),
            risk_cfg={"decay_warning_threshold": 0.02},
        )
        assert result is not None
        assert result["setup_type"] == "exit_warning"

    def test_exit_on_high_vix(self):
        result = detect_leveraged_etf_exit(
            "TQQQ", _leverage_feats(),
            _vol_regime_feats(vix_regime="high", vix_current=35),
        )
        assert result is not None

    def test_exit_on_misaligned_trend(self):
        result = detect_leveraged_etf_exit(
            "TQQQ", _leverage_feats(underlying_trend_aligned=0),
            _vol_regime_feats(),
        )
        assert result is not None

    def test_no_exit_when_healthy(self):
        result = detect_leveraged_etf_exit(
            "TQQQ", _leverage_feats(), _vol_regime_feats(),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Risk Manager Extensions
# ---------------------------------------------------------------------------
class TestLeverageRiskHints:
    def test_basic_hints(self):
        hints = compute_leverage_risk_hints(100.0, 2.0, "long", 3)
        assert hints["stop_loss"] is not None
        assert hints["max_holding_days"] > 0
        assert "decay_warning" in hints

    def test_high_vol_wider_stop(self):
        normal = compute_leverage_risk_hints(100.0, 2.0, "long", 3, vol_regime="normal")
        high = compute_leverage_risk_hints(100.0, 2.0, "long", 3, vol_regime="high")
        # High vol should have wider stop (further from price)
        assert abs(high["stop_loss"] - 100.0) >= abs(normal["stop_loss"] - 100.0)

    def test_high_vol_shorter_holding(self):
        normal = compute_leverage_risk_hints(100.0, 2.0, "long", 3, vol_regime="normal")
        high = compute_leverage_risk_hints(100.0, 2.0, "long", 3, vol_regime="high")
        assert high["max_holding_days"] <= normal["max_holding_days"]

    def test_decay_warning_flag(self):
        hints = compute_leverage_risk_hints(
            100.0, 2.0, "long", 3,
            estimated_decay=0.03,
            risk_cfg={"decay_warning_threshold": 0.02},
        )
        assert hints["decay_warning"] is True

    def test_format_lines(self):
        hints = compute_leverage_risk_hints(100.0, 2.0, "long", 3, estimated_decay=0.01)
        lines = format_leverage_risk_lines(hints)
        assert isinstance(lines, list)
        assert len(lines) > 0


# ---------------------------------------------------------------------------
# Telegram ETF Alert Message
# ---------------------------------------------------------------------------
class TestBuildEtfAlertMessage:
    def test_momentum_entry(self):
        alert = {
            "alert_type": "leveraged_etf", "setup_type": "momentum_entry",
            "direction": "long", "score": 72.0,
            "flags": ["Momentum score: 72/100", "VIX normal"],
        }
        msg = build_etf_alert_message("TQQQ", alert, "QQQ", 3)
        assert "TQQQ" in msg
        assert "QQQ" in msg
        assert "3x" in msg
        assert "72" in msg

    def test_exit_warning(self):
        alert = {
            "alert_type": "leveraged_etf", "setup_type": "exit_warning",
            "direction": "long", "score": 0.0,
            "flags": ["High decay"],
        }
        msg = build_etf_alert_message("TQQQ", alert, "QQQ", 3)
        assert "Exit" in msg
        assert "Score" not in msg  # exit warnings don't show score
