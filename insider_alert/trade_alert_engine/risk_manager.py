"""Risk management helpers.

Provides ATR-based stop-loss hints, volatility classification, and a
reward:risk estimate to be included in trade alert messages.
"""
import logging

logger = logging.getLogger(__name__)

# Annualised ATR-pct thresholds for volatility classification
_VOL_LOW = 0.01        # < 1 % daily ATR  → low volatility
_VOL_HIGH = 0.025      # > 2.5 % daily ATR → high volatility


def classify_volatility(atr_pct: float) -> str:
    """Return 'Low', 'Normal', or 'High' based on *atr_pct*."""
    if atr_pct < _VOL_LOW:
        return "Low"
    if atr_pct >= _VOL_HIGH:
        return "High"
    return "Normal"


def compute_risk_hints(
    current_price: float,
    atr: float,
    direction: str,
    *,
    stop_atr_multiplier: float = 1.0,
    rr_ratio: float = 2.0,
) -> dict:
    """Return a dict with stop-loss, target, and risk/reward estimates.

    Parameters
    ----------
    current_price:
        Latest closing price.
    atr:
        Average True Range (absolute price units).
    direction:
        ``'bullish'`` or ``'bearish'``.
    stop_atr_multiplier:
        ATR multiplier for the stop distance (default 1.0×ATR).
    rr_ratio:
        Reward:risk ratio used to project the price target.
    """
    if atr <= 0 or current_price <= 0:
        return {
            "stop_loss": None,
            "price_target": None,
            "risk_per_share": None,
            "rr_ratio": rr_ratio,
            "volatility": "Unknown",
        }

    risk = stop_atr_multiplier * atr
    atr_pct = atr / (current_price + 1e-9)
    volatility = classify_volatility(atr_pct)

    if direction == "bullish":
        stop_loss = round(current_price - risk, 4)
        price_target = round(current_price + rr_ratio * risk, 4)
    else:
        stop_loss = round(current_price + risk, 4)
        price_target = round(current_price - rr_ratio * risk, 4)

    return {
        "stop_loss": stop_loss,
        "price_target": price_target,
        "risk_per_share": round(risk, 4),
        "rr_ratio": rr_ratio,
        "volatility": volatility,
    }


def format_risk_hint_lines(risk_hints: dict) -> list[str]:
    """Return formatted text lines suitable for inclusion in a Telegram message."""
    lines = []
    if risk_hints.get("stop_loss") is not None:
        lines.append(f"  🛑 Stop: {risk_hints['stop_loss']:.2f}")
    if risk_hints.get("price_target") is not None:
        lines.append(f"  🎯 Target: {risk_hints['price_target']:.2f}")
    if risk_hints.get("risk_per_share") is not None:
        lines.append(f"  📐 Risk/share: {risk_hints['risk_per_share']:.2f}  |  R:R = {risk_hints['rr_ratio']:.1f}")
    vol = risk_hints.get("volatility")
    if vol and vol != "Unknown":
        lines.append(f"  📊 Volatility: {vol}")
    return lines


def compute_leverage_risk_hints(
    current_price: float,
    atr: float,
    direction: str,
    leverage: int = 3,
    vol_regime: str = "normal",
    estimated_decay: float = 0.0,
    risk_cfg: dict | None = None,
) -> dict:
    """Return extended risk hints for leveraged ETF positions.

    Parameters
    ----------
    current_price : float
        Latest closing price.
    atr : float
        Average True Range (absolute price units).
    direction : str
        ``'long'`` or ``'short'``.
    leverage : int
        Leverage factor (e.g. 3).
    vol_regime : str
        ``'low'``, ``'normal'``, or ``'high'``.
    estimated_decay : float
        Estimated daily decay from ``compute_leverage_features()``.
    risk_cfg : dict | None
        Risk config section from ``config.leveraged_etfs['risk']``.
    """
    risk_cfg = risk_cfg or {}
    stop_mult = float(risk_cfg.get("stop_atr_multiplier", 1.5))
    rr_ratio = float(risk_cfg.get("rr_ratio", 2.0))
    max_dd = float(risk_cfg.get("max_drawdown_pct", 15.0))
    decay_threshold = float(risk_cfg.get("decay_warning_threshold", 0.02))
    hold_low = int(risk_cfg.get("max_holding_days_low_vol", 15))
    hold_high = int(risk_cfg.get("max_holding_days_high_vol", 5))

    # Adjust stop multiplier for high volatility
    if vol_regime == "high":
        stop_mult *= 1.3
        max_holding_days = hold_high
    elif vol_regime == "low":
        max_holding_days = hold_low
    else:
        max_holding_days = (hold_low + hold_high) // 2

    base = compute_risk_hints(
        current_price, atr,
        "bullish" if direction == "long" else "bearish",
        stop_atr_multiplier=stop_mult,
        rr_ratio=rr_ratio,
    )

    base["max_holding_days"] = max_holding_days
    base["estimated_daily_decay"] = estimated_decay
    base["decay_warning"] = estimated_decay >= decay_threshold
    base["max_drawdown_pct"] = max_dd
    base["leverage"] = leverage
    return base


def format_leverage_risk_lines(risk_hints: dict) -> list[str]:
    """Format leverage-specific risk information for Telegram."""
    lines = format_risk_hint_lines(risk_hints)

    hold_days = risk_hints.get("max_holding_days")
    if hold_days is not None:
        lines.append(f"  ⏱️ Max Haltedauer: ~{hold_days} Tage")

    decay = risk_hints.get("estimated_daily_decay", 0)
    if decay > 0:
        lines.append(f"  📉 Geschätzter Decay: {decay:.4f}/Tag")

    if risk_hints.get("decay_warning"):
        lines.append("  ⚠️ Decay-Warnung: Volatilität zu hoch für langen Halt")

    max_dd = risk_hints.get("max_drawdown_pct")
    if max_dd is not None:
        lines.append(f"  🛡️ Max Drawdown: {max_dd:.0f}% empfohlen")

    return lines
