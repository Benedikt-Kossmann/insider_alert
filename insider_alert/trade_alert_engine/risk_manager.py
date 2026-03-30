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
