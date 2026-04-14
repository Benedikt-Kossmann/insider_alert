"""Options-based feature computation."""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum single-contract volume to classify a trade as a "block trade"
BLOCK_TRADE_VOLUME_THRESHOLD = 1000
# Short-dated options window in calendar days
SHORT_DATED_WINDOW_DAYS = 14


def compute_options_features(
    options_df: pd.DataFrame,
    current_price: float,
    iv_baseline: float = 0.0,
    risk_free_rate: float = 0.05,
) -> dict:
    """Compute options-based features.

    Parameters
    ----------
    options_df:
        Combined calls+puts options chain as returned by ``fetch_options_chain``.
    current_price:
        Current underlying price used for OTM classification.
    iv_baseline:
        Historical volatility baseline (e.g. 30-day HV) used to compute
        ``iv_change_1d``.  Pass 0.0 to skip IV-change computation.
    """
    defaults = {
        "call_volume_zscore": 0.0,
        "put_volume_zscore": 0.0,
        "put_call_ratio_change": 0.0,
        "iv_change_1d": 0.0,
        "short_dated_otm_call_score": 0.0,
        "block_trade_score": 0.0,
        "sweep_order_score": 0.0,
        "open_interest_change": 0.0,
    }
    if options_df is None or options_df.empty:
        return defaults

    df = options_df.copy()
    df.columns = [c.lower() for c in df.columns]

    if "contracttype" in df.columns and "contractType" not in df.columns:
        df.rename(columns={"contracttype": "contractType"}, inplace=True)

    for col in ["contractType", "strike", "expiration", "volume", "openinterest", "impliedvolatility"]:
        lower = col.lower()
        if lower not in df.columns and col not in df.columns:
            df[lower] = 0

    if "openinterest" not in df.columns:
        df["openinterest"] = df.get("openInterest", 0)
    if "impliedvolatility" not in df.columns:
        df["impliedvolatility"] = df.get("impliedVolatility", 0)
    if "contracttype" not in df.columns:
        df["contracttype"] = df.get("contractType", "")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["openinterest"] = pd.to_numeric(df["openinterest"], errors="coerce").fillna(0)
    df["impliedvolatility"] = pd.to_numeric(df["impliedvolatility"], errors="coerce").fillna(0)
    df["strike"] = pd.to_numeric(df.get("strike", 0), errors="coerce").fillna(0)

    calls = df[df["contracttype"].str.lower() == "call"] if "contracttype" in df.columns else pd.DataFrame()
    puts = df[df["contracttype"].str.lower() == "put"] if "contracttype" in df.columns else pd.DataFrame()

    def vol_zscore(subset: pd.DataFrame) -> float:
        if subset.empty or len(subset) < 2:
            return 0.0
        vols = subset["volume"].values.astype(float)
        mean, std = vols.mean(), vols.std()
        if std < 1e-9:
            return 0.0
        return float((vols[-1] - mean) / std)

    call_volume_zscore = vol_zscore(calls)
    put_volume_zscore = vol_zscore(puts)

    total_call_vol = float(calls["volume"].sum()) if not calls.empty else 0.0
    total_put_vol = float(puts["volume"].sum()) if not puts.empty else 0.0
    current_pcr = total_put_vol / (total_call_vol + 1e-9)
    put_call_ratio_change = current_pcr - 0.7

    iv_change_1d = 0.0
    if not df.empty and "impliedvolatility" in df.columns:
        avg_iv = float(
            df["impliedvolatility"].replace(0.0, float("nan")).dropna().mean() or 0.0
        )
        if iv_baseline > 0.0 and avg_iv > 0.0:
            iv_change_1d = (avg_iv - iv_baseline) / (iv_baseline + 1e-9)

    short_dated_otm_call_score = 0.0
    if not calls.empty and total_call_vol > 0:
        try:
            today = datetime.now().date()
            cutoff = today + timedelta(days=SHORT_DATED_WINDOW_DAYS)
            otm_calls = calls[calls["strike"] > current_price].copy()
            otm_calls["exp_date"] = pd.to_datetime(otm_calls["expiration"], errors="coerce").dt.date
            short_otm = otm_calls[otm_calls["exp_date"] <= cutoff]
            short_dated_otm_call_score = float(short_otm["volume"].sum() / (total_call_vol + 1e-9))
        except Exception as exc:
            logger.debug("short_dated_otm_call_score error: %s", exc)

    total_vol = float(df["volume"].sum())
    if total_vol > 0:
        block_trade_score = float((df["volume"] > BLOCK_TRADE_VOLUME_THRESHOLD).sum() / len(df))
    else:
        block_trade_score = 0.0

    if not calls.empty:
        total_call_oi = float(calls["openinterest"].sum())
        sweep_order_score = float(np.clip(total_call_vol / (total_call_oi + 1e-9), 0.0, 1.0))
    else:
        sweep_order_score = 0.0

    if not df.empty:
        oi_vals = df["openinterest"].values.astype(float)
        mean_oi = oi_vals.mean()
        current_oi = oi_vals[-1] if len(oi_vals) > 0 else 0
        open_interest_change = (current_oi - mean_oi) / (mean_oi + 1e-9)
    else:
        open_interest_change = 0.0

    result = {
        "call_volume_zscore": call_volume_zscore,
        "put_volume_zscore": put_volume_zscore,
        "put_call_ratio_change": put_call_ratio_change,
        "iv_change_1d": iv_change_1d,
        "short_dated_otm_call_score": float(np.clip(short_dated_otm_call_score, 0.0, 1.0)),
        "block_trade_score": float(np.clip(block_trade_score, 0.0, 1.0)),
        "sweep_order_score": float(np.clip(sweep_order_score, 0.0, 1.0)),
        "open_interest_change": open_interest_change,
    }
    greeks_feats = compute_greeks_features(df, current_price, risk_free_rate)
    result.update(greeks_feats)
    return result


def compute_greeks_features(
    options_df: pd.DataFrame,
    current_price: float,
    risk_free_rate: float = 0.05,
) -> dict:
    """Compute Greek-based options flow features.

    Returns
    -------
    dict
        net_delta_exposure : aggregated delta-weighted flow (positive = bullish)
        gamma_imbalance    : dealer gamma balance (positive = call-heavy)
        put_call_delta_ratio: direction-aware put/call ratio
        iv_skew_25d        : 25-delta put IV vs call IV skew
        iv_term_structure  : front-month IV vs back-month IV
    """
    defaults = {
        "net_delta_exposure": 0.0,
        "gamma_imbalance": 0.0,
        "put_call_delta_ratio": 0.0,
        "iv_skew_25d": 0.0,
        "iv_term_structure": 0.0,
    }

    if options_df is None or options_df.empty:
        return defaults

    from insider_alert.feature_engine.greeks import compute_chain_greeks

    try:
        chain = compute_chain_greeks(options_df, current_price, risk_free_rate)
    except Exception as exc:
        logger.debug("compute_chain_greeks failed: %s", exc)
        chain = []
    if not chain:
        return defaults

    # 1. Net Delta Exposure — sum(volume × delta × 100), normalised by total vol
    total_vol = sum(c["volume"] for c in chain) or 1.0
    net_delta = sum(c["volume"] * c["delta"] * 100 for c in chain)
    net_delta_exposure = net_delta / (total_vol * 100)

    # 2. Gamma Imbalance — call vs put gamma-weighted volume
    call_gamma = sum(c["volume"] * c["gamma"] for c in chain if c["contract_type"] == "call")
    put_gamma = sum(c["volume"] * c["gamma"] for c in chain if c["contract_type"] == "put")
    denom = call_gamma + put_gamma
    gamma_imbalance = (call_gamma - put_gamma) / max(denom, 1e-9)

    # 3. Delta-weighted Put/Call Ratio
    call_delta_vol = sum(abs(c["volume"] * c["delta"]) for c in chain if c["contract_type"] == "call")
    put_delta_vol = sum(abs(c["volume"] * c["delta"]) for c in chain if c["contract_type"] == "put")
    put_call_delta_ratio = put_delta_vol / max(call_delta_vol, 1e-9) - 1.0

    # 4. IV Skew (25-delta)
    iv_skew_25d = _compute_iv_skew(chain, target_delta=0.25)

    # 5. IV Term Structure
    iv_term_structure = _compute_iv_term_structure(chain)

    return {
        "net_delta_exposure": float(np.clip(net_delta_exposure, -2.0, 2.0)),
        "gamma_imbalance": float(np.clip(gamma_imbalance, -1.0, 1.0)),
        "put_call_delta_ratio": float(np.clip(put_call_delta_ratio, -2.0, 2.0)),
        "iv_skew_25d": float(np.clip(iv_skew_25d, -1.0, 1.0)),
        "iv_term_structure": float(np.clip(iv_term_structure, -1.0, 1.0)),
    }


def _compute_iv_skew(chain: list[dict], target_delta: float = 0.25) -> float:
    """IV skew: 25-delta put IV minus 25-delta call IV.

    Positive = puts more expensive = fear premium.
    """
    calls = [c for c in chain if c["contract_type"] == "call" and c["iv"] > 0]
    puts = [c for c in chain if c["contract_type"] == "put" and c["iv"] > 0]

    if not calls or not puts:
        return 0.0

    call_25d = min(calls, key=lambda c: abs(c["delta"] - target_delta))
    put_25d = min(puts, key=lambda c: abs(abs(c["delta"]) - target_delta))

    if call_25d["iv"] > 0:
        return float((put_25d["iv"] - call_25d["iv"]) / call_25d["iv"])
    return 0.0


def _compute_iv_term_structure(chain: list[dict]) -> float:
    """Front-month IV vs back-month IV.

    Positive = backwardation (front > back) = short-term event pricing.
    """
    from datetime import datetime
    today = datetime.now().date()

    front = [
        c for c in chain
        if c["iv"] > 0
        and (datetime.strptime(c["expiration"], "%Y-%m-%d").date() - today).days <= 30
    ]
    back = [
        c for c in chain
        if c["iv"] > 0
        and (datetime.strptime(c["expiration"], "%Y-%m-%d").date() - today).days > 30
    ]

    if not front or not back:
        return 0.0

    avg_front_iv = float(np.mean([c["iv"] for c in front]))
    avg_back_iv = float(np.mean([c["iv"] for c in back]))

    if avg_back_iv > 0:
        return float((avg_front_iv - avg_back_iv) / avg_back_iv)
    return 0.0
