"""Position sizing using Kelly Criterion and risk constraints."""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Defaults — overridden by config if passed
_MAX_POSITION_PCT = 10.0   # Never more than 10% per position
_MIN_POSITION_PCT = 1.0    # Always at least 1%
_KELLY_FRACTION = 0.5      # Half-Kelly (conservative)


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Compute Kelly fraction.

    f* = (p * b - q) / b

    where:
        p = win rate
        q = 1 - p
        b = avg_win / avg_loss (reward-to-risk ratio)

    Returns
    -------
    float — Optimal portfolio fraction (0-1). Negative Kelly returns 0.0.
    """
    if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
        return 0.0

    p = win_rate
    q = 1.0 - p
    b = avg_win / avg_loss

    kelly = (p * b - q) / b
    return max(kelly, 0.0)


def compute_position_size(
    signal_scores: dict[str, float],
    composite_score: float,
    ticker: str,
    lookback_days: int = 180,
    risk_cfg: dict | None = None,
) -> dict:
    """Compute recommended position size using Kelly Criterion.

    Parameters
    ----------
    signal_scores : dict — Current signal scores for the ticker
    composite_score : float — Composite score (0-100)
    ticker : str — Ticker symbol
    lookback_days : int — Historical window for win/loss statistics
    risk_cfg : dict — Risk management config (optional)

    Returns
    -------
    dict with keys:
        position_pct : float — Recommended portfolio allocation (%)
        kelly_raw : float — Raw Kelly fraction
        kelly_half : float — Half-Kelly fraction
        confidence : str — "high" | "medium" | "low"
        reasoning : str — Human-readable explanation
    """
    risk_cfg = risk_cfg or {}
    kelly_fraction = float(risk_cfg.get("kelly_fraction", _KELLY_FRACTION))
    max_pct = float(risk_cfg.get("max_position_pct", _MAX_POSITION_PCT))
    min_pct = float(risk_cfg.get("min_position_pct", _MIN_POSITION_PCT))

    defaults = {
        "position_pct": min_pct,
        "kelly_raw": 0.0,
        "kelly_half": 0.0,
        "confidence": "low",
        "reasoning": "Insufficient historical data for position sizing",
    }

    try:
        from insider_alert.persistence.storage import _get_engine, SignalOutcome
        from datetime import datetime, timedelta, timezone
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

        if len(outcomes) < 50:
            return defaults

        # Prefer outcomes with a similar score range (±10 pts)
        similar = [
            o for o in outcomes
            if o.composite_score is not None
            and abs(o.composite_score - composite_score) < 10
        ]
        if len(similar) < 20:
            similar = outcomes

        wins = [o for o in similar if getattr(o, "return_5d", None) is not None and o.return_5d > 0]
        losses = [o for o in similar if getattr(o, "return_5d", None) is not None and o.return_5d <= 0]

        if not wins or not losses:
            return defaults

        win_rate = len(wins) / (len(wins) + len(losses))
        avg_win = float(np.mean([o.return_5d for o in wins]))
        avg_loss = float(np.mean([abs(o.return_5d) for o in losses]))

    except Exception as exc:
        logger.warning("Position sizing data fetch failed: %s", exc)
        return defaults

    raw_kelly = kelly_criterion(win_rate, avg_win, avg_loss)
    half_kelly = raw_kelly * kelly_fraction

    position_pct = float(np.clip(half_kelly * 100, min_pct, max_pct))

    # Score-based confidence adjustment
    if composite_score >= 80:
        confidence = "high"
        position_pct = min(position_pct * 1.2, max_pct)
    elif composite_score >= 65:
        confidence = "medium"
    else:
        confidence = "low"
        position_pct = max(position_pct * 0.7, min_pct)

    position_pct = round(position_pct, 1)
    rr_label = f"{avg_win / avg_loss:.1f}" if avg_loss > 0 else "N/A"
    reasoning = (
        f"Kelly: {raw_kelly:.1%} (WR={win_rate:.1%}, R:R={rr_label}) "
        f"→ Half-Kelly: {half_kelly:.1%} → Position: {position_pct:.1f}%"
    )

    return {
        "position_pct": position_pct,
        "kelly_raw": round(raw_kelly, 4),
        "kelly_half": round(half_kelly, 4),
        "confidence": confidence,
        "reasoning": reasoning,
    }
