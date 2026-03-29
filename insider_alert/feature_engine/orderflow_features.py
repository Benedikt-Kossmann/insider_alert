"""Order-flow feature computation (OHLCV proxies)."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_orderflow_features(ohlcv: pd.DataFrame) -> dict:
    """Compute order-flow features estimated from OHLCV data."""
    defaults = {
        "bid_ask_imbalance": 0.0,
        "aggressive_buy_ratio": 0.0,
        "large_trade_count": 0.0,
        "iceberg_suspect_score": 0.0,
        "absorption_score": 0,
        "vwap_accumulation_score": 0.0,
    }
    if ohlcv is None or ohlcv.empty:
        return defaults

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return defaults

    last = df.iloc[-1]
    o = float(last["open"])
    h = float(last["high"])
    lo = float(last["low"])
    c = float(last["close"])
    v = float(last["volume"])

    hl_range = h - lo + 1e-9
    bid_ask_imbalance = float(np.clip((c - o) / hl_range, -1.0, 1.0))
    aggressive_buy_ratio = float(np.clip((c - lo) / hl_range, 0.0, 1.0))

    volumes = df["volume"].dropna().astype(float)
    avg_vol = float(volumes.iloc[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
    rvol = v / (avg_vol + 1e-9)
    large_trade_count = float(np.clip(rvol * aggressive_buy_ratio / 5.0, 0.0, 1.0))

    range_ratio = (h - lo) / (c + 1e-9)
    iceberg_suspect_score = float(np.clip(rvol / 3.0, 0.0, 1.0)) if range_ratio < 0.01 else 0.0

    close_near_high = (h - c) / (h - lo + 1e-9) < 0.2
    absorption_score = 1 if (close_near_high and rvol > 1.5) else 0

    vwap_accumulation_score = 0.0
    if len(df) >= 5:
        df5 = df.iloc[-5:]
        vwap_5d = float(((df5["high"] + df5["low"] + df5["close"]) / 3).mean())
        atr_14 = 0.0
        if len(df) >= 14:
            atr_14 = float((df["high"].iloc[-14:] - df["low"].iloc[-14:]).mean())
        else:
            atr_14 = float((df["high"] - df["low"]).mean())
        vwap_accumulation_score = float(np.clip((c - vwap_5d) / (atr_14 + 1e-9), -1.0, 1.0))

    return {
        "bid_ask_imbalance": bid_ask_imbalance,
        "aggressive_buy_ratio": aggressive_buy_ratio,
        "large_trade_count": large_trade_count,
        "iceberg_suspect_score": iceberg_suspect_score,
        "absorption_score": absorption_score,
        "vwap_accumulation_score": vwap_accumulation_score,
    }
