"""Insider transaction feature computation."""
import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_ROLE_WEIGHTS = {
    # Abbreviations (e.g., "CEO") and key substrings of full titles
    # (e.g., "Chief Executive Officer") are both matched via substring search.
    "CEO": 2.0,
    "CHIEF EXECUTIVE": 2.0,
    "CFO": 2.0,
    "CHIEF FINANCIAL": 2.0,
    "COO": 2.0,
    "CHIEF OPERATING": 2.0,
    "CTO": 2.0,
    "CHIEF TECHNOLOGY": 2.0,
    "PRESIDENT": 2.0,
    "DIRECTOR": 1.5,
}


def _role_weight(role: str) -> float:
    role_upper = role.upper()
    for key, weight in _ROLE_WEIGHTS.items():
        if key in role_upper:
            return weight
    return 1.0


def compute_insider_features(transactions_df: pd.DataFrame) -> dict:
    """Compute insider transaction features."""
    defaults = {
        "insider_buy_count_30d": 0,
        "insider_sell_count_30d": 0,
        "insider_buy_value_30d": 0.0,
        "insider_cluster_score": 0.0,
        "insider_role_weighted_score": 0.0,
        "insider_recent_buy_count_7d": 0,
        "insider_net_buy_score": 0.0,
    }
    if transactions_df is None or transactions_df.empty:
        return defaults

    df = transactions_df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "transaction_type" not in df.columns:
        return defaults

    buys = df[df["transaction_type"].str.lower() == "buy"]
    sells = df[df["transaction_type"].str.lower() == "sell"]

    insider_buy_count_30d = int(len(buys))
    insider_sell_count_30d = int(len(sells))

    buy_value = 0.0
    if "value" in buys.columns:
        buy_value = float(buys["value"].sum())
    insider_buy_value_30d = buy_value

    # Recency: buys in the last 7 days
    insider_recent_buy_count_7d = 0
    if "date" in buys.columns and insider_buy_count_30d > 0:
        cutoff_7d = date.today() - timedelta(days=7)
        try:
            buy_dates = pd.to_datetime(buys["date"]).dt.date
            insider_recent_buy_count_7d = int((buy_dates >= cutoff_7d).sum())
        except Exception:
            pass

    # Net buy score: penalizes when sells accompany buys (0 = only sells, 1 = only buys)
    total_activity = insider_buy_count_30d + insider_sell_count_30d
    if total_activity > 0:
        insider_net_buy_score = max(0.0, (insider_buy_count_30d - insider_sell_count_30d) / total_activity)
    else:
        insider_net_buy_score = 0.0

    if insider_buy_count_30d == 0:
        insider_cluster_score = 0.0
    elif "insider_name" in buys.columns:
        distinct_insiders = buys["insider_name"].nunique()
        if distinct_insiders > 2:
            insider_cluster_score = 1.0
        else:
            insider_cluster_score = distinct_insiders / 3.0
    else:
        insider_cluster_score = min(insider_buy_count_30d / 3.0, 1.0)

    role_weighted_sum = 0.0
    if "role" in buys.columns:
        for role in buys["role"]:
            role_weighted_sum += _role_weight(str(role))
    else:
        role_weighted_sum = float(insider_buy_count_30d)
    insider_role_weighted_score = min(role_weighted_sum / 10.0, 1.0)

    return {
        "insider_buy_count_30d": insider_buy_count_30d,
        "insider_sell_count_30d": insider_sell_count_30d,
        "insider_buy_value_30d": insider_buy_value_30d,
        "insider_cluster_score": insider_cluster_score,
        "insider_role_weighted_score": insider_role_weighted_score,
        "insider_recent_buy_count_7d": insider_recent_buy_count_7d,
        "insider_net_buy_score": insider_net_buy_score,
    }
