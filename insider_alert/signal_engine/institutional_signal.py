"""Institutional 13-F flow signal.

Scores the degree of institutional (smart-money) presence for a ticker based
on SEC 13-F holding data aggregated by `institutional_data.fetch_institutional_flows`.
"""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    # smart_money_score: fraction of checked top-20 funds that hold the stock
    SignalComponent(
        key="smart_money_score",
        max_score=60,
        normaliser=1.0,   # already normalised to [0, 1]
        flag_template="🏦 Smart Money Presence: {value:.0%} of checked funds",
        flag_threshold=0.5,
    ),
    # institutional_buy_count: raw number of funds holding (max useful ~10)
    SignalComponent(
        key="institutional_buy_count",
        max_score=40,
        normaliser=10.0,  # normalise to [0, 1] with 10 funds as ceiling
        flag_template="📊 Institutional Holders: {value:.0f} top funds",
        flag_threshold=3,
    ),
]


def institutional_signal(features: dict) -> dict:
    """Compute institutional 13-F flow signal from institutional features."""
    return compute_signal("institutional", features, _COMPONENTS)
