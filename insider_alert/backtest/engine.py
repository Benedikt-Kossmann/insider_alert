"""Walk-forward backtesting engine for OHLCV-based signals.

Backtests the technical signals (price, volume, accumulation) against actual
forward returns.  Insider/event/news/options signals cannot be backtested
because historical API data is not available — those are tracked live via the
signal_outcomes table instead.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Signals that can be backtested from OHLCV alone
_OHLCV_SIGNAL_GENERATORS = None  # lazy-loaded


def _get_signal_generators() -> list[tuple[str, callable, callable]]:
    """Return (name, feature_fn, signal_fn) tuples, lazy-loaded."""
    global _OHLCV_SIGNAL_GENERATORS
    if _OHLCV_SIGNAL_GENERATORS is not None:
        return _OHLCV_SIGNAL_GENERATORS

    from insider_alert.feature_engine.price_features import compute_price_features
    from insider_alert.feature_engine.volume_features import compute_volume_features
    from insider_alert.feature_engine.accumulation_features import compute_accumulation_features
    from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
    from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
    from insider_alert.signal_engine.volume_signal import compute_volume_anomaly_signal
    from insider_alert.signal_engine.accumulation_signal import compute_accumulation_signal
    from insider_alert.signal_engine.orderflow_signal import compute_orderflow_anomaly_signal

    _OHLCV_SIGNAL_GENERATORS = [
        ("price_anomaly", compute_price_features, compute_price_anomaly_signal),
        ("volume_anomaly", compute_volume_features, compute_volume_anomaly_signal),
        ("accumulation_pattern", compute_accumulation_features, compute_accumulation_signal),
        ("candle_pattern", compute_orderflow_features, compute_orderflow_anomaly_signal),
    ]
    return _OHLCV_SIGNAL_GENERATORS


@dataclass
class BacktestResult:
    """Container for backtest output."""
    ticker: str
    rows: list[dict] = field(default_factory=list)
    total_days: int = 0
    error: str = ""


def backtest_ticker(
    ticker: str,
    ohlcv: pd.DataFrame,
    *,
    min_lookback: int = 30,
    forward_days: tuple[int, ...] = (1, 5, 10),
) -> BacktestResult:
    """Run walk-forward backtest for one ticker.

    For each trading day starting at ``min_lookback``, computes features from
    all data up to (and including) that day, generates signals, and records
    the actual forward return over *forward_days*.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    ohlcv : pd.DataFrame
        Full OHLCV DataFrame (daily).  Must have ``close`` column.
    min_lookback : int
        Minimum number of history bars before first signal.
    forward_days : tuple[int, ...]
        Horizons to measure forward returns.

    Returns
    -------
    BacktestResult
    """
    result = BacktestResult(ticker=ticker)

    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        result.error = "No OHLCV data"
        return result

    max_fwd = max(forward_days)
    n = len(ohlcv)

    if n < min_lookback + max_fwd:
        result.error = f"Insufficient data: {n} bars (need {min_lookback + max_fwd})"
        return result

    generators = _get_signal_generators()
    closes = ohlcv["close"].values

    for i in range(min_lookback, n - max_fwd):
        window = ohlcv.iloc[:i + 1]
        close_today = float(closes[i])

        if close_today <= 0:
            continue

        # Compute signals
        day_signals: dict[str, float] = {}
        day_flags: list[str] = []
        for name, feat_fn, sig_fn in generators:
            try:
                features = feat_fn(window)
                signal = sig_fn(features)
                day_signals[name] = signal["score"]
                day_flags.extend(signal.get("flags", []))
            except Exception:
                day_signals[name] = 0.0

        # Composite score (equal-weight for backtestable signals)
        scores = list(day_signals.values())
        composite = float(np.mean(scores)) if scores else 0.0

        # Forward returns
        fwd_returns = {}
        for d in forward_days:
            fwd_idx = i + d
            if fwd_idx < n:
                fwd_returns[f"return_{d}d"] = (float(closes[fwd_idx]) / close_today) - 1.0
            else:
                fwd_returns[f"return_{d}d"] = None

        row = {
            "date": ohlcv.index[i],
            "ticker": ticker,
            **day_signals,
            "composite": composite,
            **fwd_returns,
        }
        result.rows.append(row)

    result.total_days = len(result.rows)
    return result


def run_backtest(
    tickers: list[str],
    period: str = "1y",
) -> list[BacktestResult]:
    """Run backtest for multiple tickers, fetching data from yfinance.

    Parameters
    ----------
    tickers : list[str]
        Tickers to backtest.
    period : str
        yfinance period string (e.g. ``"1y"``, ``"2y"``).

    Returns
    -------
    list[BacktestResult]
    """
    from insider_alert.data_ingestion.market_data import fetch_ohlcv_daily

    results = []
    for ticker in tickers:
        logger.info("Backtesting %s...", ticker)
        try:
            ohlcv = fetch_ohlcv_daily(ticker, period=period)
            bt = backtest_ticker(ticker, ohlcv)
            results.append(bt)
            if bt.error:
                logger.warning("Backtest %s: %s", ticker, bt.error)
            else:
                logger.info("Backtest %s: %d days processed", ticker, bt.total_days)
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", ticker, exc)
            results.append(BacktestResult(ticker=ticker, error=str(exc)))

    return results
