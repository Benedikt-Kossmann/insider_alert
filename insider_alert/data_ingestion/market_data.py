"""Market data ingestion using yfinance."""
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv_daily(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Return OHLCV daily data for a ticker."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            logger.warning("No daily OHLCV data for %s", ticker)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as exc:
        logger.warning("fetch_ohlcv_daily failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_ohlcv_intraday(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Return OHLCV intraday data."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning("No intraday OHLCV data for %s", ticker)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as exc:
        logger.warning("fetch_ohlcv_intraday failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_sector_benchmark(ticker: str) -> tuple[str, str]:
    """Return (sector, benchmark_ticker) for a given ticker."""
    sector_map = {
        "Technology": "XLK",
        "Health Care": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Utilities": "XLU",
        "Communication Services": "XLC",
    }
    try:
        t = yf.Ticker(ticker)
        info = t.info
        sector = info.get("sector", "Unknown")
        benchmark = sector_map.get(sector, "SPY")
        return sector, benchmark
    except Exception as exc:
        logger.warning("fetch_sector_benchmark failed for %s: %s", ticker, exc)
        return "Unknown", "SPY"
