"""Risk management helpers.

Provides ATR-based stop-loss hints, volatility classification, a
reward:risk estimate, correlation cluster checks, and a drawdown guard.
"""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

logger = logging.getLogger(__name__)

# Annualised ATR-pct thresholds for volatility classification
_VOL_LOW = 0.01        # < 1 % daily ATR  → low volatility
_VOL_HIGH = 0.025      # > 2.5 % daily ATR → high volatility

# Correlation / cluster risk
_MAX_SECTOR_ALERTS = 3       # Max recent alerts per sector before warning
_LOOKBACK_HOURS = 72         # Window for counting recent sector alerts

# Drawdown guard
_DRAWDOWN_GUARD_PCT = 5.0    # Activate guard when cumulative DD exceeds this
_THRESHOLD_INCREASE = 10     # Extra score points required when guard is active


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


# ---------------------------------------------------------------------------
# Correlation / Cluster Risk (Phase 10)
# ---------------------------------------------------------------------------

def check_correlation_risk(
    ticker: str,
    sector_etf: str,
    lookback_hours: int = _LOOKBACK_HOURS,
    max_sector_alerts: int = _MAX_SECTOR_ALERTS,
) -> dict:
    """Check whether too many correlated alerts are already open.

    Parameters
    ----------
    ticker : str — Ticker being evaluated
    sector_etf : str — Sector ETF proxy (e.g. "XLK") for this ticker
    lookback_hours : int — Time window for counting recent sector alerts
    max_sector_alerts : int — Threshold before cluster warning fires

    Returns
    -------
    dict with keys:
        cluster_risk : bool — True when cluster risk exists
        same_sector_count : int — Open alerts in the same sector
        warning : str — Warning text (empty if no risk)
    """
    defaults: dict = {"cluster_risk": False, "same_sector_count": 0, "warning": ""}

    if not sector_etf:
        return defaults

    try:
        from insider_alert.persistence.storage import _get_engine, Alert
        from insider_alert.feature_engine.sector_features import _SECTOR_MAP
        from sqlalchemy.orm import sessionmaker

        engine = _get_engine("sqlite:///insider_alert.db")
        Session = sessionmaker(bind=engine)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        with Session() as session:
            recent_alerts = (
                session.query(Alert)
                .filter(Alert.sent_at >= cutoff)
                .all()
            )

        same_sector = [
            a for a in recent_alerts
            if _SECTOR_MAP.get(a.ticker.upper(), "") == sector_etf
            and a.ticker.upper() != ticker.upper()
        ]

        count = len(same_sector)
        is_risky = count >= max_sector_alerts

        warning = ""
        if is_risky:
            sample = [a.ticker for a in same_sector[:5]]
            warning = (
                f"⚠️ Cluster-Risiko: {count} offene Alerts im Sektor "
                f"({sector_etf}): {', '.join(sample)}"
            )

        return {"cluster_risk": is_risky, "same_sector_count": count, "warning": warning}

    except Exception as exc:
        logger.warning("Correlation risk check failed: %s", exc)
        return defaults


# ---------------------------------------------------------------------------
# Drawdown Guard (Phase 10)
# ---------------------------------------------------------------------------

def check_drawdown_guard(
    lookback_days: int = 30,
    guard_pct: float = _DRAWDOWN_GUARD_PCT,
    threshold_increase: int = _THRESHOLD_INCREASE,
) -> dict:
    """Detect portfolio-level drawdown and tighten alert thresholds.

    Simulates an equally-weighted portfolio from recent signal outcomes.

    Returns
    -------
    dict with keys:
        drawdown_active : bool — True when guard fires
        portfolio_drawdown_pct : float — Current drawdown (%)
        threshold_adjustment : int — Extra score points required (0 = normal)
        warning : str — Warning text (empty if not active)
    """
    defaults: dict = {
        "drawdown_active": False,
        "portfolio_drawdown_pct": 0.0,
        "threshold_adjustment": 0,
        "warning": "",
    }

    try:
        from insider_alert.persistence.storage import _get_engine, SignalOutcome
        from sqlalchemy.orm import sessionmaker

        engine = _get_engine("sqlite:///insider_alert.db")
        Session = sessionmaker(bind=engine)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        with Session() as session:
            outcomes = (
                session.query(SignalOutcome)
                .filter(SignalOutcome.created_at >= cutoff)
                .all()
            )

        returns = [
            o.return_5d / 100.0
            for o in outcomes
            if getattr(o, "return_5d", None) is not None
        ]

        if not returns:
            return defaults

        cum = np.cumprod(1.0 + np.array(returns)) - 1.0
        peak = np.maximum.accumulate(cum + 1.0)
        drawdown_series = ((cum + 1.0) - peak) / peak * 100.0

        current_dd = float(drawdown_series[-1])
        is_active = abs(current_dd) > guard_pct
        threshold_adj = threshold_increase if is_active else 0

        warning = ""
        if is_active:
            warning = (
                f"🛡️ Drawdown Guard AKTIV: {current_dd:.1f}% DD "
                f"→ Schwelle +{threshold_adj} Punkte"
            )
            logger.warning(warning)

        return {
            "drawdown_active": is_active,
            "portfolio_drawdown_pct": round(current_dd, 2),
            "threshold_adjustment": threshold_adj,
            "warning": warning,
        }

    except Exception as exc:
        logger.warning("Drawdown guard check failed: %s", exc)
        return defaults

