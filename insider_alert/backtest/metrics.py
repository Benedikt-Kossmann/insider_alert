"""Backtest metrics and report generation."""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalMetrics:
    """Performance metrics for one signal type."""
    name: str
    total_days: int = 0
    high_signal_days: int = 0     # days where score > threshold
    hit_rate_1d: float = 0.0      # % of high-signal days with positive 1d return
    hit_rate_5d: float = 0.0
    hit_rate_10d: float = 0.0
    avg_return_1d: float = 0.0    # avg 1d return on high-signal days
    avg_return_5d: float = 0.0
    avg_return_10d: float = 0.0
    avg_return_all_1d: float = 0.0  # baseline: avg 1d return on ALL days
    edge_1d: float = 0.0          # high-signal avg return minus baseline
    edge_5d: float = 0.0


def compute_signal_metrics(
    rows: list[dict],
    signal_name: str,
    threshold: float = 50.0,
) -> SignalMetrics:
    """Compute performance metrics for one signal across backtest rows.

    Parameters
    ----------
    rows : list[dict]
        Backtest rows (from ``BacktestResult.rows``).
    signal_name : str
        Key name for the signal score in each row.
    threshold : float
        Score above which the signal is considered "active".
    """
    m = SignalMetrics(name=signal_name)
    if not rows:
        return m

    m.total_days = len(rows)

    all_ret_1d = [r["return_1d"] for r in rows if r.get("return_1d") is not None]
    m.avg_return_all_1d = float(np.mean(all_ret_1d)) if all_ret_1d else 0.0

    high = [r for r in rows if r.get(signal_name, 0) > threshold]
    m.high_signal_days = len(high)

    if not high:
        return m

    for horizon in (1, 5, 10):
        key = f"return_{horizon}d"
        returns = [r[key] for r in high if r.get(key) is not None]
        if returns:
            avg_ret = float(np.mean(returns))
            hit = sum(1 for r in returns if r > 0) / len(returns)
            setattr(m, f"avg_return_{horizon}d", avg_ret)
            setattr(m, f"hit_rate_{horizon}d", hit)

    # Edge = high-signal avg return minus baseline
    high_ret_1d = [r["return_1d"] for r in high if r.get("return_1d") is not None]
    high_ret_5d = [r["return_5d"] for r in high if r.get("return_5d") is not None]
    if high_ret_1d and all_ret_1d:
        m.edge_1d = float(np.mean(high_ret_1d)) - m.avg_return_all_1d
    all_ret_5d = [r["return_5d"] for r in rows if r.get("return_5d") is not None]
    if high_ret_5d and all_ret_5d:
        m.edge_5d = float(np.mean(high_ret_5d)) - float(np.mean(all_ret_5d))

    return m


def compute_composite_metrics(
    rows: list[dict],
    threshold: float = 50.0,
) -> SignalMetrics:
    """Compute metrics for the composite score."""
    return compute_signal_metrics(rows, "composite", threshold)


def generate_report(
    results: list,
    threshold: float = 50.0,
) -> str:
    """Generate a text report from backtest results.

    Parameters
    ----------
    results : list[BacktestResult]
        Output of ``run_backtest()``.
    threshold : float
        Score threshold for "high signal" classification.

    Returns
    -------
    str
        Formatted report text.
    """
    signal_names = ["price_anomaly", "volume_anomaly", "accumulation_pattern", "candle_pattern", "composite"]
    all_rows: list[dict] = []

    lines = ["=" * 70]
    lines.append("BACKTEST REPORT")
    lines.append("=" * 70)
    lines.append("")

    for bt in results:
        lines.append(f"--- {bt.ticker} ---")
        if bt.error:
            lines.append(f"  ERROR: {bt.error}")
            lines.append("")
            continue
        lines.append(f"  Days tested: {bt.total_days}")
        all_rows.extend(bt.rows)

        for sname in signal_names:
            m = compute_signal_metrics(bt.rows, sname, threshold)
            if m.high_signal_days == 0:
                lines.append(f"  {sname}: no high-signal days (threshold={threshold})")
                continue
            lines.append(
                f"  {sname}: {m.high_signal_days} alerts | "
                f"Hit 1d={m.hit_rate_1d:.0%} 5d={m.hit_rate_5d:.0%} 10d={m.hit_rate_10d:.0%} | "
                f"Avg ret 5d={m.avg_return_5d:+.2%} | "
                f"Edge 5d={m.edge_5d:+.2%}"
            )
        lines.append("")

    # Aggregate
    if all_rows:
        lines.append("=" * 70)
        lines.append(f"AGGREGATE ({len(results)} tickers, {len(all_rows)} total days)")
        lines.append("=" * 70)

        for sname in signal_names:
            m = compute_signal_metrics(all_rows, sname, threshold)
            if m.high_signal_days == 0:
                lines.append(f"  {sname}: no high-signal days")
                continue
            lines.append(
                f"  {sname}: {m.high_signal_days}/{m.total_days} alerts | "
                f"Hit 1d={m.hit_rate_1d:.0%} 5d={m.hit_rate_5d:.0%} 10d={m.hit_rate_10d:.0%} | "
                f"Avg ret 5d={m.avg_return_5d:+.2%} | "
                f"Edge 1d={m.edge_1d:+.2%} 5d={m.edge_5d:+.2%}"
            )

        lines.append("")
        lines.append(f"Baseline avg daily return: {m.avg_return_all_1d:+.3%}")

    lines.append("=" * 70)
    return "\n".join(lines)
