"""Tests for the discovery scanner module."""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field

from insider_alert.trade_alert_engine.discovery_scanner import (
    _compute_signals,
    Discovery,
    DEFAULT_RVOL_THRESHOLD,
    DEFAULT_MOVE_PCT_THRESHOLD,
    DEFAULT_GAP_PCT_THRESHOLD,
    DEFAULT_CRYPTO_MOVE_PCT_THRESHOLD,
    DEFAULT_STOCK_POOL,
    DEFAULT_CRYPTO_POOL,
    run_discovery_scan,
)
from insider_alert.alert_engine.telegram_alert import (
    build_discovery_alert_message,
    send_discovery_alert,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(closes: list[float], volumes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    """Build a simple single-ticker OHLCV DataFrame."""
    if opens is None:
        opens = closes  # no gap
    return pd.DataFrame({
        "Close": closes,
        "Open": opens,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Volume": volumes,
    })


def _make_multi_ticker_df(ticker_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a MultiIndex DataFrame simulating yf.download output for multiple tickers."""
    frames = {}
    for ticker, df in ticker_data.items():
        for col in df.columns:
            frames[(col, ticker)] = df[col].values

    idx = range(len(next(iter(ticker_data.values()))))
    mi = pd.MultiIndex.from_tuples(list(frames.keys()))
    return pd.DataFrame(frames, index=idx, columns=mi)


@dataclass
class FakeConfig:
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT"])
    telegram_token: str = "test-token"
    telegram_chat_id: str = "123"
    leveraged_etfs: dict = field(default_factory=lambda: {"enabled": False, "universe": []})
    discovery: dict = field(default_factory=lambda: {"enabled": True, "scan_stocks": True, "scan_crypto": True})
    trade_alerts: dict = field(default_factory=lambda: {"enabled": False})
    scheduler: dict = field(default_factory=lambda: {"eod_hour": 17, "eod_minute": 30, "intraday_interval_minutes": 60})


# ---------------------------------------------------------------------------
# _compute_signals tests
# ---------------------------------------------------------------------------

class TestComputeSignals:
    """Tests for the core signal detection logic."""

    def test_volume_spike_detected(self):
        """High last-day volume triggers RVOL discovery."""
        # 20 days of normal volume (1000), then a spike (5000)
        closes = [100.0] * 20 + [101.0]
        volumes = [1000.0] * 20 + [5000.0]
        df = _make_ohlcv(closes, volumes)

        results = _compute_signals(
            df, ["TEST"], "stock",
            rvol_threshold=3.0, move_pct_threshold=50.0, gap_pct_threshold=50.0,
        )
        assert len(results) == 1
        assert results[0].ticker == "TEST"
        assert results[0].rvol >= 3.0
        assert any("RVOL" in r for r in results[0].reasons)

    def test_large_move_detected(self):
        """A big daily price change triggers move discovery."""
        closes = [100.0] * 20 + [110.0]  # 10% move
        volumes = [1000.0] * 21
        df = _make_ohlcv(closes, volumes)

        results = _compute_signals(
            df, ["TEST"], "stock",
            rvol_threshold=100.0, move_pct_threshold=5.0, gap_pct_threshold=50.0,
        )
        assert len(results) == 1
        assert abs(results[0].move_pct) >= 5.0
        assert any("Move" in r for r in results[0].reasons)

    def test_gap_detected(self):
        """A gap open triggers gap discovery."""
        closes = [100.0] * 20 + [104.0]
        volumes = [1000.0] * 21
        opens = [100.0] * 20 + [104.0]  # gap up from 100 -> 104 open
        df = _make_ohlcv(closes, volumes, opens)

        results = _compute_signals(
            df, ["TEST"], "stock",
            rvol_threshold=100.0, move_pct_threshold=50.0, gap_pct_threshold=3.0,
        )
        assert len(results) == 1
        assert abs(results[0].gap_pct) >= 3.0
        assert any("Gap" in r for r in results[0].reasons)

    def test_no_signal_below_thresholds(self):
        """Nothing returned when activity is below thresholds."""
        closes = [100.0] * 21
        volumes = [1000.0] * 21
        df = _make_ohlcv(closes, volumes)

        results = _compute_signals(
            df, ["TEST"], "stock",
            rvol_threshold=3.0, move_pct_threshold=5.0, gap_pct_threshold=3.0,
        )
        assert len(results) == 0

    def test_multiple_reasons(self):
        """A symbol can trigger multiple reasons at once."""
        closes = [100.0] * 20 + [108.0]  # 8% move
        volumes = [1000.0] * 20 + [5000.0]  # 5x RVOL
        opens = [100.0] * 20 + [105.0]  # 5% gap
        df = _make_ohlcv(closes, volumes, opens)

        results = _compute_signals(
            df, ["TEST"], "stock",
            rvol_threshold=3.0, move_pct_threshold=5.0, gap_pct_threshold=3.0,
        )
        assert len(results) == 1
        assert len(results[0].reasons) == 3

    def test_multi_ticker_dataframe(self):
        """Works with MultiIndex DataFrame from yf.download."""
        data_a = _make_ohlcv(
            [100.0] * 20 + [110.0],
            [1000.0] * 20 + [5000.0],
        )
        data_b = _make_ohlcv(
            [50.0] * 21,
            [500.0] * 21,
        )
        multi_df = _make_multi_ticker_df({"AAA": data_a, "BBB": data_b})

        results = _compute_signals(
            multi_df, ["AAA", "BBB"], "stock",
            rvol_threshold=3.0, move_pct_threshold=5.0, gap_pct_threshold=50.0,
        )
        tickers_found = {r.ticker for r in results}
        assert "AAA" in tickers_found
        assert "BBB" not in tickers_found

    def test_asset_class_preserved(self):
        """Asset class is correctly set."""
        closes = [100.0] * 20 + [120.0]
        volumes = [1000.0] * 21
        df = _make_ohlcv(closes, volumes)

        results = _compute_signals(
            df, ["BTC-USD"], "crypto",
            rvol_threshold=100.0, move_pct_threshold=5.0, gap_pct_threshold=50.0,
        )
        assert results[0].asset_class == "crypto"

    def test_short_data_skipped(self):
        """Tickers with less than 5 data points are skipped."""
        df = _make_ohlcv([100.0, 101.0, 102.0], [1000.0, 1000.0, 1000.0])
        results = _compute_signals(
            df, ["TEST"], "stock",
            rvol_threshold=1.0, move_pct_threshold=0.1, gap_pct_threshold=0.1,
        )
        assert len(results) == 0

    def test_negative_move_detected(self):
        """A large drop is also detected."""
        closes = [100.0] * 20 + [90.0]  # -10%
        volumes = [1000.0] * 21
        df = _make_ohlcv(closes, volumes)

        results = _compute_signals(
            df, ["TEST"], "stock",
            rvol_threshold=100.0, move_pct_threshold=5.0, gap_pct_threshold=50.0,
        )
        assert len(results) == 1
        assert results[0].move_pct < 0


# ---------------------------------------------------------------------------
# run_discovery_scan tests
# ---------------------------------------------------------------------------

class TestRunDiscoveryScan:
    """Tests for the top-level discovery function."""

    def test_disabled_returns_empty(self):
        config = FakeConfig(discovery={"enabled": False})
        result = run_discovery_scan(config)
        assert result == []

    @patch("insider_alert.trade_alert_engine.discovery_scanner._batch_download")
    def test_excludes_watched_tickers(self, mock_dl):
        """Already-watched tickers are excluded from the scan pool."""
        config = FakeConfig(
            tickers=["INTC"],
            discovery={
                "enabled": True,
                "scan_stocks": True,
                "scan_crypto": False,
                "stock_pool": ["INTC", "MU"],
            },
        )
        mock_dl.return_value = pd.DataFrame()
        run_discovery_scan(config)
        called_tickers = mock_dl.call_args[0][0]
        assert "INTC" not in called_tickers
        assert "MU" in called_tickers

    @patch("insider_alert.trade_alert_engine.discovery_scanner._batch_download")
    def test_returns_sorted_by_move(self, mock_dl):
        """Results are sorted by absolute move descending."""
        data_a = _make_ohlcv([100.0] * 20 + [105.0], [1000.0] * 20 + [5000.0])
        data_b = _make_ohlcv([100.0] * 20 + [115.0], [1000.0] * 20 + [5000.0])
        multi_df = _make_multi_ticker_df({"AAA": data_a, "BBB": data_b})

        mock_dl.return_value = multi_df
        config = FakeConfig(
            tickers=[],
            discovery={
                "enabled": True,
                "scan_stocks": True,
                "scan_crypto": False,
                "stock_pool": ["AAA", "BBB"],
                "rvol_threshold": 3.0,
                "move_pct_threshold": 3.0,
                "gap_pct_threshold": 50.0,
            },
        )
        results = run_discovery_scan(config)
        assert len(results) == 2
        assert results[0].ticker == "BBB"  # 15% move > 5% move


# ---------------------------------------------------------------------------
# Default pools
# ---------------------------------------------------------------------------

class TestDefaultPools:
    """Verify default scan pool integrity."""

    def test_stock_pool_not_empty(self):
        assert len(DEFAULT_STOCK_POOL) > 50

    def test_crypto_pool_not_empty(self):
        assert len(DEFAULT_CRYPTO_POOL) > 20

    def test_crypto_tickers_have_usd_suffix(self):
        for t in DEFAULT_CRYPTO_POOL:
            assert t.endswith("-USD"), f"{t} should end with -USD"

    def test_no_duplicates_in_stock_pool(self):
        assert len(DEFAULT_STOCK_POOL) == len(set(DEFAULT_STOCK_POOL))

    def test_no_duplicates_in_crypto_pool(self):
        assert len(DEFAULT_CRYPTO_POOL) == len(set(DEFAULT_CRYPTO_POOL))


# ---------------------------------------------------------------------------
# Telegram message tests
# ---------------------------------------------------------------------------

class TestDiscoveryTelegramMessage:
    """Tests for discovery alert message formatting."""

    def test_build_message_with_stock(self):
        d = Discovery(
            ticker="MU", asset_class="stock", rvol=4.5,
            move_pct=7.2, gap_pct=1.0, close=95.30, volume=50_000_000,
            reasons=["RVOL 4.5x", "Move +7.2%"],
        )
        msg = build_discovery_alert_message([d])
        assert "Discovery" in msg
        assert "MU" in msg
        assert "RVOL 4.5x" in msg
        assert "📊" in msg

    def test_build_message_with_crypto(self):
        d = Discovery(
            ticker="SOL-USD", asset_class="crypto", rvol=2.0,
            move_pct=12.5, gap_pct=0.5, close=185.0, volume=1_000_000,
            reasons=["Move +12.5%"],
        )
        msg = build_discovery_alert_message([d])
        assert "SOL-USD" in msg
        assert "🪙" in msg

    def test_build_message_empty(self):
        msg = build_discovery_alert_message([])
        assert msg == ""

    def test_build_message_count_line(self):
        discoveries = [
            Discovery("A", "stock", 3.0, 5.0, 0.0, 100.0, 10000, ["RVOL 3.0x"]),
            Discovery("B", "crypto", 1.0, 10.0, 0.0, 50.0, 5000, ["Move +10.0%"]),
        ]
        msg = build_discovery_alert_message(discoveries)
        assert "2 Symbol(e)" in msg

    @patch("insider_alert.alert_engine.telegram_alert.send_telegram_message", return_value=True)
    def test_send_discovery_alert_sends(self, mock_send):
        discoveries = [
            Discovery("MU", "stock", 4.0, 6.0, 0.0, 95.0, 50000, ["RVOL 4.0x"]),
        ]
        result = send_discovery_alert(discoveries, "tok", "123")
        assert result is True
        mock_send.assert_called_once()

    @patch("insider_alert.alert_engine.telegram_alert.send_telegram_message", return_value=True)
    def test_send_discovery_alert_caps_results(self, mock_send):
        discoveries = [
            Discovery(f"T{i}", "stock", 3.0, 5.0, 0.0, 100.0, 10000, ["RVOL 3.0x"])
            for i in range(20)
        ]
        send_discovery_alert(discoveries, "tok", "123", max_results=5)
        msg = mock_send.call_args[1].get("message") or mock_send.call_args[0][2]
        assert "5 Symbol(e)" in msg

    def test_send_discovery_alert_empty_returns_false(self):
        result = send_discovery_alert([], "tok", "123")
        assert result is False


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestDiscoveryConfig:
    """Test that discovery config is loaded correctly."""

    def test_config_has_discovery_field(self):
        from insider_alert.config import load_config
        config = load_config()
        assert hasattr(config, "discovery")

    def test_config_discovery_enabled(self):
        from insider_alert.config import load_config
        config = load_config()
        assert config.discovery.get("enabled") is True

    def test_config_discovery_has_thresholds(self):
        from insider_alert.config import load_config
        config = load_config()
        disc = config.discovery
        assert "rvol_threshold" in disc
        assert "move_pct_threshold" in disc
        assert "crypto_move_pct_threshold" in disc
