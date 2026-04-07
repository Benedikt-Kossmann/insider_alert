"""Tests for the walk-forward backtesting engine and metrics."""
import unittest

import numpy as np
import pandas as pd

from insider_alert.backtest.engine import backtest_ticker, BacktestResult
from insider_alert.backtest.metrics import (
    compute_signal_metrics,
    compute_composite_metrics,
    generate_report,
    SignalMetrics,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data with a slight upward trend."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(trend + rng.randn(n) * 0.015))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    opn = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_backtest_rows(
    n: int = 50,
    signal_score: float = 60.0,
    avg_return: float = 0.005,
    seed: int = 7,
) -> list[dict]:
    """Create synthetic backtest row dicts."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        ret_1d = avg_return + rng.randn() * 0.02
        ret_5d = avg_return * 5 + rng.randn() * 0.04
        ret_10d = avg_return * 10 + rng.randn() * 0.06
        rows.append({
            "date": pd.Timestamp("2023-06-01") + pd.Timedelta(days=i),
            "ticker": "TEST",
            "price_anomaly": signal_score + rng.randn() * 10,
            "volume_anomaly": 30.0 + rng.randn() * 10,
            "accumulation_pattern": 40.0,
            "candle_pattern": 35.0,
            "composite": signal_score * 0.8,
            "return_1d": ret_1d,
            "return_5d": ret_5d,
            "return_10d": ret_10d,
        })
    return rows


# ---------------------------------------------------------------------------
# Tests: backtest_ticker
# ---------------------------------------------------------------------------

class TestBacktestTicker(unittest.TestCase):
    def test_returns_backtest_result(self):
        ohlcv = _make_ohlcv(n=80)
        result = backtest_ticker("TEST", ohlcv, min_lookback=30)
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.ticker, "TEST")

    def test_rows_produced(self):
        ohlcv = _make_ohlcv(n=80)
        result = backtest_ticker("TEST", ohlcv, min_lookback=30)
        self.assertGreater(len(result.rows), 0)
        self.assertEqual(result.total_days, len(result.rows))
        self.assertFalse(result.error)

    def test_row_keys(self):
        ohlcv = _make_ohlcv(n=80)
        result = backtest_ticker("TEST", ohlcv, min_lookback=30)
        row = result.rows[0]
        # Expected keys
        for key in ("date", "ticker", "composite", "return_1d", "return_5d", "return_10d"):
            self.assertIn(key, row, f"Missing key: {key}")
        # Signal scores
        for sig in ("price_anomaly", "volume_anomaly", "accumulation_pattern", "candle_pattern"):
            self.assertIn(sig, row, f"Missing signal key: {sig}")

    def test_insufficient_data(self):
        ohlcv = _make_ohlcv(n=10)
        result = backtest_ticker("TEST", ohlcv, min_lookback=30)
        self.assertTrue(result.error)
        self.assertEqual(len(result.rows), 0)

    def test_empty_dataframe(self):
        result = backtest_ticker("TEST", pd.DataFrame())
        self.assertTrue(result.error)

    def test_none_dataframe(self):
        result = backtest_ticker("TEST", None)
        self.assertTrue(result.error)

    def test_forward_returns_computed(self):
        ohlcv = _make_ohlcv(n=80)
        result = backtest_ticker("TEST", ohlcv, min_lookback=30)
        # At least some rows should have non-None forward returns
        has_ret = [r for r in result.rows if r.get("return_1d") is not None]
        self.assertGreater(len(has_ret), 0)

    def test_scores_bounded(self):
        ohlcv = _make_ohlcv(n=80)
        result = backtest_ticker("TEST", ohlcv, min_lookback=30)
        for row in result.rows:
            for sig in ("price_anomaly", "volume_anomaly", "accumulation_pattern", "candle_pattern"):
                self.assertGreaterEqual(row[sig], 0.0, f"{sig} below 0")
                self.assertLessEqual(row[sig], 100.0, f"{sig} above 100")


# ---------------------------------------------------------------------------
# Tests: metrics
# ---------------------------------------------------------------------------

class TestSignalMetrics(unittest.TestCase):
    def test_empty_rows(self):
        m = compute_signal_metrics([], "price_anomaly")
        self.assertEqual(m.total_days, 0)
        self.assertEqual(m.high_signal_days, 0)

    def test_all_high_signal(self):
        rows = _make_backtest_rows(n=20, signal_score=70.0, avg_return=0.005)
        m = compute_signal_metrics(rows, "price_anomaly", threshold=40.0)
        self.assertEqual(m.total_days, 20)
        # Most rows should be above threshold since mean is 70
        self.assertGreater(m.high_signal_days, 0)

    def test_no_high_signal(self):
        rows = _make_backtest_rows(n=20, signal_score=20.0)
        m = compute_signal_metrics(rows, "price_anomaly", threshold=80.0)
        self.assertEqual(m.high_signal_days, 0)

    def test_hit_rate_bounded(self):
        rows = _make_backtest_rows(n=50, signal_score=60.0)
        m = compute_signal_metrics(rows, "price_anomaly", threshold=50.0)
        if m.high_signal_days > 0:
            self.assertGreaterEqual(m.hit_rate_1d, 0.0)
            self.assertLessEqual(m.hit_rate_1d, 1.0)
            self.assertGreaterEqual(m.hit_rate_5d, 0.0)
            self.assertLessEqual(m.hit_rate_5d, 1.0)

    def test_edge_computed(self):
        rows = _make_backtest_rows(n=50, signal_score=70.0, avg_return=0.01)
        m = compute_signal_metrics(rows, "price_anomaly", threshold=50.0)
        # edge_1d is defined as high-signal avg minus baseline avg
        self.assertIsInstance(m.edge_1d, float)

    def test_composite_metrics(self):
        rows = _make_backtest_rows(n=30)
        m = compute_composite_metrics(rows, threshold=40.0)
        self.assertEqual(m.name, "composite")
        self.assertEqual(m.total_days, 30)


# ---------------------------------------------------------------------------
# Tests: report generation
# ---------------------------------------------------------------------------

class TestGenerateReport(unittest.TestCase):
    def test_report_is_string(self):
        ohlcv = _make_ohlcv(n=80)
        bt = backtest_ticker("TEST", ohlcv, min_lookback=30)
        report = generate_report([bt], threshold=50.0)
        self.assertIsInstance(report, str)
        self.assertIn("BACKTEST REPORT", report)
        self.assertIn("TEST", report)

    def test_report_with_error(self):
        bt = BacktestResult(ticker="BAD", error="No data")
        report = generate_report([bt])
        self.assertIn("ERROR", report)
        self.assertIn("No data", report)

    def test_empty_results(self):
        report = generate_report([])
        self.assertIsInstance(report, str)


if __name__ == "__main__":
    unittest.main()
