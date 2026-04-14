"""Insider transaction signal computation."""
from insider_alert.signal_engine.base_signal import SignalComponent, compute_signal

_COMPONENTS = [
    # Cluster: mehrere verschiedene Insider kaufen gleichzeitig
    SignalComponent(
        key="insider_cluster_score", max_score=22, normaliser=1.0,
        flag_template="Cluster-Kauf erkannt: score={value:.2f}",
    ),
    # Rolle: CEO/CFO-Käufe gewichtet stärker als Director
    SignalComponent(
        key="insider_role_weighted_score", max_score=18, normaliser=1.0,
        flag_template="Senior-Insider-Kauf: role-weighted score={value:.2f}",
    ),
    # Dollar-Volumen: $500k+ Kauf = volles Signal
    SignalComponent(
        key="insider_buy_value_30d", max_score=22, normaliser=500_000.0,
        flag_template="Insider-Kaufvolumen (30d): ${value:,.0f}",
    ),
    # Anzahl der Käufe in 30 Tagen
    SignalComponent(
        key="insider_buy_count_30d", max_score=14, normaliser=5.0,
        flag_template="Insider-Käufe in 30d: {value}",
    ),
    # Recency: Käufe in den letzten 7 Tagen — stärkstes kurzfristiges Signal
    SignalComponent(
        key="insider_recent_buy_count_7d", max_score=14, normaliser=2.0,
        flag_template="Frische Insider-Käufe (7d): {value}",
        flag_threshold=0.3,
    ),
    # Net-Buy-Score: dämpft wenn gleichzeitig viel verkauft wird
    SignalComponent(
        key="insider_net_buy_score", max_score=10, normaliser=1.0,
        flag_template="Netto-Kauf-Score: {value:.2f} (1.0 = nur Käufe)",
        flag_threshold=0.7,
    ),
]
# Total max_score: 22+18+22+14+14+10 = 100


def compute_insider_signal(features: dict) -> dict:
    """Compute insider signal from insider transaction features."""
    return compute_signal("insider_signal", features, _COMPONENTS)
