"""Adaptive weight adjustment based on signal outcome data.

Reads past ``SignalOutcome`` rows from the DB, computes per-signal hit rates
and average returns, and produces an updated weight dict that gives more
weight to signals with proven edge and less to those that underperform.
"""
import logging
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# Minimum outcomes per signal to adjust (otherwise keep default weight)
_MIN_OUTCOMES = 30

# Blend ratio: how much of the new weights to use (0=all default, 1=all data)
_BLEND_RATIO = 0.5


def compute_adaptive_weights(
    default_weights: dict[str, float],
    *,
    lookback_days: int = 90,
    min_outcomes: int = _MIN_OUTCOMES,
    blend: float = _BLEND_RATIO,
    db_url: str = "sqlite:///insider_alert.db",
) -> dict[str, float]:
    """Compute blended weights from default weights + outcome hit rates.

    Parameters
    ----------
    default_weights : dict
        Current/default signal weights (must sum to ~1.0).
    lookback_days : int
        Only consider outcomes from the last N days.
    min_outcomes : int
        Minimum outcomes required before adjusting a signal.
    blend : float
        0.0 = use default weights entirely, 1.0 = use data-driven weights entirely.
    db_url : str
        Database URL.

    Returns
    -------
    dict
        Updated weights dict (summing to ~1.0).
    """
    try:
        hit_rates = _fetch_hit_rates(lookback_days, db_url)
    except Exception as exc:
        logger.warning("Failed to fetch outcome hit rates, using defaults: %s", exc)
        return dict(default_weights)

    if not hit_rates:
        logger.info("No outcome data for adaptive weights, using defaults.")
        return dict(default_weights)

    # Build raw score for each signal: weighted combination of hit rate and avg edge
    signal_scores: dict[str, float] = {}
    for sig_type, weight in default_weights.items():
        data = hit_rates.get(sig_type)
        if data is None or data["count"] < min_outcomes:
            # Not enough data: keep default weight
            signal_scores[sig_type] = weight
        else:
            # Score = hit_rate * 0.7 + normalised_edge * 0.3
            edge = data.get("avg_return_5d", 0.0)
            edge_norm = float(np.clip(edge * 20 + 0.5, 0.1, 1.0))  # map ±5% → [0.1, 1.0]
            perf = data["hit_rate_5d"] * 0.7 + edge_norm * 0.3
            signal_scores[sig_type] = max(perf, 0.01)

    # Normalise data-driven scores to sum to 1.0
    total = sum(signal_scores.values())
    if total <= 0:
        return dict(default_weights)

    data_weights = {k: v / total for k, v in signal_scores.items()}

    # Blend with defaults
    blended = {}
    for sig_type in default_weights:
        d = default_weights[sig_type]
        a = data_weights.get(sig_type, d)
        blended[sig_type] = d * (1 - blend) + a * blend

    # Renormalise
    total_blend = sum(blended.values())
    if total_blend > 0:
        blended = {k: v / total_blend for k, v in blended.items()}

    # Log changes
    for sig_type in default_weights:
        old = default_weights[sig_type]
        new = blended[sig_type]
        if abs(new - old) > 0.005:
            direction = "↑" if new > old else "↓"
            logger.info(
                "Adaptive weight %s: %.3f → %.3f %s",
                sig_type, old, new, direction,
            )

    return blended


def _fetch_hit_rates(
    lookback_days: int,
    db_url: str,
) -> dict[str, dict]:
    """Fetch per-signal hit rates from the signal_outcomes table.

    Returns ``{signal_type: {"hit_rate_5d": float, "avg_return_5d": float, "count": int}}``.
    """
    from insider_alert.persistence.storage import _get_engine, SignalOutcome
    from sqlalchemy.orm import sessionmaker

    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    cutoff = date.today() - timedelta(days=lookback_days)

    with Session() as session:
        rows = (
            session.query(SignalOutcome)
            .filter(
                SignalOutcome.date >= cutoff,
                SignalOutcome.hit_5d.isnot(None),
            )
            .all()
        )

    if not rows:
        return {}

    from collections import defaultdict
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r.signal_type].append(r)

    result = {}
    for sig_type, outcomes in by_type.items():
        hits = [o for o in outcomes if o.hit_5d]
        returns = [o.return_5d for o in outcomes if o.return_5d is not None]
        result[sig_type] = {
            "hit_rate_5d": len(hits) / len(outcomes) if outcomes else 0.0,
            "avg_return_5d": float(np.mean(returns)) if returns else 0.0,
            "count": len(outcomes),
        }

    return result
