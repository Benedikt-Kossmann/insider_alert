"""Macro / cross-asset data fetching via yfinance (all free)."""
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default yfinance tickers for macro indicators
_DEFAULTS = {
    "vix": "^VIX",
    "tnx": "^TNX",       # 10-Year Treasury yield
    "irx": "^IRX",       # 13-Week Treasury bill rate
    "dxy": "DX-Y.NYB",   # US Dollar Index
}


def fetch_macro_data(
    period: str = "6mo",
    *,
    vix_ticker: str = "",
    tnx_ticker: str = "",
    irx_ticker: str = "",
    dxy_ticker: str = "",
) -> dict[str, pd.DataFrame]:
    """Fetch macro OHLCV series.  Returns ``{name: DataFrame}``."""
    tickers = {
        "vix": vix_ticker or _DEFAULTS["vix"],
        "tnx": tnx_ticker or _DEFAULTS["tnx"],
        "irx": irx_ticker or _DEFAULTS["irx"],
        "dxy": dxy_ticker or _DEFAULTS["dxy"],
    }

    result: dict[str, pd.DataFrame] = {}
    for name, symbol in tickers.items():
        try:
            df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
            df.columns = [c.lower() for c in df.columns]
            result[name] = df
        except Exception as exc:
            logger.warning("fetch_macro_data failed for %s (%s): %s", name, symbol, exc)
            result[name] = pd.DataFrame()

    return result
