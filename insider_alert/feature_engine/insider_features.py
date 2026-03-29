"""Insider transaction feature computation."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_ROLE_WEIGHTS = {
    "CEO": 2.0,
    "CFO": 2.0,
    "COO": 2.0,
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
    }
