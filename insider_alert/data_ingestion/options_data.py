"""Options data ingestion using yfinance."""
import logging
import pandas as pd
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


def fetch_options_chain(ticker: str) -> pd.DataFrame:
    """Return a combined calls+puts options chain for next two expiration dates."""
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            logger.warning("No options expirations for %s", ticker)
            return pd.DataFrame()
        frames = []
        for exp in expirations[:2]:
            chain = t.option_chain(exp)
            calls = chain.calls.copy()
            calls["contractType"] = "call"
            calls["expiration"] = exp
            puts = chain.puts.copy()
            puts["contractType"] = "put"
            puts["expiration"] = exp
            frames.extend([calls, puts])
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        for col in ["contracttype", "strike", "expiration", "volume", "openinterest", "impliedvolatility"]:
            if col not in df.columns:
                df[col] = 0
        rename_map = {
            "contracttype": "contractType",
            "openinterest": "openInterest",
            "impliedvolatility": "impliedVolatility",
        }
        df.rename(columns=rename_map, inplace=True)
        return df
    except Exception as exc:
        logger.warning("fetch_options_chain failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_historical_iv(ticker: str) -> float:
    """Return a simple implied volatility proxy (30d HV from daily returns)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="3mo", interval="1d", auto_adjust=True)
        if len(df) < 2:
            return 0.0
        closes = df["Close"].dropna()
        log_returns = np.log(closes / closes.shift(1)).dropna()
        hv_30d = float(log_returns.tail(30).std() * np.sqrt(252))
        return hv_30d
    except Exception as exc:
        logger.warning("fetch_historical_iv failed for %s: %s", ticker, exc)
        return 0.0
