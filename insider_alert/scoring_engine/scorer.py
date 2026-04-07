"""Aggregate sub-signals into a composite insider-activity score."""
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "price_anomaly": 0.15,
    "volume_anomaly": 0.15,
    "candle_pattern": 0.05,
    "options_anomaly": 0.20,
    "insider_signal": 0.20,
    "event_leadup": 0.10,
    "news_divergence": 0.05,
    "accumulation_pattern": 0.05,
    "macro_regime": 0.05,
}

DEFAULT_ETF_WEIGHTS = {
    "momentum": 0.25,
    "mean_reversion_dip": 0.20,
    "volatility_regime": 0.20,
    "leverage_health": 0.15,
    "volume_anomaly": 0.10,
    "price_anomaly": 0.10,
}


@dataclass
class TickerScore:
    ticker: str
    total_score: float
    sub_scores: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def compute_score(
    ticker: str,
    signals: list[dict],
    weights: dict | None = None,
) -> TickerScore:
    """Aggregate signal dicts into a TickerScore."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    sub_scores: dict = {}
    all_flags: list[str] = []

    for signal in signals:
        sig_type = signal.get("signal_type", "unknown")
        sig_score = float(signal.get("score", 0.0))
        sig_flags = signal.get("flags", [])
        sub_scores[sig_type] = float(np.clip(sig_score, 0.0, 100.0))
        all_flags.extend(sig_flags)

    total_weight = 0.0
    weighted_sum = 0.0
    for sig_type, weight in weights.items():
        score = sub_scores.get(sig_type, 0.0)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight > 0:
        total_score = float(np.clip(weighted_sum / total_weight, 0.0, 100.0))
    else:
        total_score = 0.0

    return TickerScore(
        ticker=ticker,
        total_score=total_score,
        sub_scores=sub_scores,
        flags=all_flags,
    )
