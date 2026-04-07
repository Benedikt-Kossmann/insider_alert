"""Momentum-based feature computation (RSI, MACD, EMA crossover, ADX)."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (equivalent to EWM with alpha=1/period)."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Compute RSI using Wilder's smoothing.  Returns 50.0 on insufficient data."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return float(rsi.iloc[-1])


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Return MACD line, signal line, and histogram (last values)."""
    if len(close) < slow + signal:
        return {"macd_line": 0.0, "macd_signal": 0.0, "macd_histogram": 0.0}
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd_line": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_histogram": float(histogram.iloc[-1]),
    }


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ADX.  Returns 0.0 on insufficient data."""
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    required = {"high", "low", "close"}
    if not required.issubset(d.columns) or len(d) < period + 1:
        return 0.0

    high, low, close = d["high"], d["low"], d["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(pd.Series(plus_dm, index=d.index), period) / (atr + 1e-12)
    minus_di = 100.0 * _wilder_smooth(pd.Series(minus_dm, index=d.index), period) / (atr + 1e-12)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = _wilder_smooth(dx, period)
    return float(adx.iloc[-1])


def compute_momentum_features(ohlcv: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Compute momentum features from OHLCV data.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume.
    cfg : dict | None
        Optional config dict with keys: rsi_period, macd_fast, macd_slow,
        macd_signal, ema_fast, ema_slow, adx_period, adx_trend_threshold.

    Returns
    -------
    dict
        Momentum features including RSI, MACD, EMA crossover, ADX.
    """
    cfg = cfg or {}
    defaults = {
        "rsi_14": 50.0,
        "macd_line": 0.0,
        "macd_signal": 0.0,
        "macd_histogram": 0.0,
        "ema_fast": 0.0,
        "ema_slow": 0.0,
        "ema_crossover": 0,
        "adx_14": 0.0,
        "adx_trending": 0,
    }

    d = ohlcv.copy()
    d.columns = [c.lower() for c in d.columns]
    if d.empty or "close" not in d.columns:
        return defaults

    close = d["close"]
    rsi_period = int(cfg.get("rsi_period", 14))
    fast_span = int(cfg.get("ema_fast", 10))
    slow_span = int(cfg.get("ema_slow", 50))
    adx_period = int(cfg.get("adx_period", 14))
    adx_threshold = float(cfg.get("adx_trend_threshold", 25))

    rsi = compute_rsi(close, rsi_period)

    macd = compute_macd(
        close,
        fast=int(cfg.get("macd_fast", 12)),
        slow=int(cfg.get("macd_slow", 26)),
        signal=int(cfg.get("macd_signal", 9)),
    )

    ema_f = _ema(close, fast_span)
    ema_s = _ema(close, slow_span)
    ema_fast_val = float(ema_f.iloc[-1]) if len(ema_f) >= fast_span else 0.0
    ema_slow_val = float(ema_s.iloc[-1]) if len(ema_s) >= slow_span else 0.0

    # Crossover: 1=bullish (fast > slow), -1=bearish, 0=neutral/insufficient
    if ema_fast_val > 0 and ema_slow_val > 0:
        ema_crossover = 1 if ema_fast_val > ema_slow_val else -1
    else:
        ema_crossover = 0

    adx = compute_adx(ohlcv, adx_period)
    adx_trending = 1 if adx >= adx_threshold else 0

    return {
        "rsi_14": rsi,
        "macd_line": macd["macd_line"],
        "macd_signal": macd["macd_signal"],
        "macd_histogram": macd["macd_histogram"],
        "ema_fast": ema_fast_val,
        "ema_slow": ema_slow_val,
        "ema_crossover": ema_crossover,
        "adx_14": adx,
        "adx_trending": adx_trending,
    }
