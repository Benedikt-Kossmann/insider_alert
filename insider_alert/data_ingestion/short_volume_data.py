"""FINRA RegSHO short volume data ingestion."""
import io
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://cdn.finra.org/equity/regsho/daily"
_TIMEOUT = 15


def _build_url(date: datetime) -> str:
    """Build FINRA short volume URL for a specific date."""
    date_str = date.strftime("%Y%m%d")
    return f"{_BASE_URL}/CNMSshvol{date_str}.txt"


def fetch_short_volume(ticker: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch daily short volume data for a ticker from FINRA RegSHO.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    lookback_days : int
        Number of calendar days to look back (weekends are skipped automatically).

    Returns
    -------
    pd.DataFrame
        Columns: Date, ShortVolume, TotalVolume, ShortRatio.
        Empty DataFrame on failure.
    """
    records = []
    today = datetime.now()

    for i in range(lookback_days):
        date = today - timedelta(days=i)
        # Skip weekends
        if date.weekday() >= 5:
            continue

        url = _build_url(date)
        try:
            resp = requests.get(url, timeout=_TIMEOUT)
            if resp.status_code != 200:
                continue

            df = pd.read_csv(
                io.StringIO(resp.text),
                sep="|",
                dtype=str,
            )
            df.columns = [c.strip() for c in df.columns]

            # Defensively handle column name variants
            sym_col = next(
                (c for c in df.columns if c.upper() in ("SYMBOL", "SYM")),
                None,
            )
            if sym_col is None:
                continue

            match = df[df[sym_col].str.upper() == ticker.upper()]
            if match.empty:
                continue

            row = match.iloc[0]

            # Handle column name variants
            sv_col = next((c for c in df.columns if "SHORT" in c.upper() and "EXEMPT" not in c.upper()), None)
            tv_col = next((c for c in df.columns if "TOTAL" in c.upper()), None)
            if sv_col is None or tv_col is None:
                continue

            short_vol = int(str(row.get(sv_col, "0")).replace(",", "") or 0)
            total_vol = int(str(row.get(tv_col, "0")).replace(",", "") or 0)
            short_ratio = short_vol / max(total_vol, 1)

            records.append({
                "Date": date.date(),
                "ShortVolume": short_vol,
                "TotalVolume": total_vol,
                "ShortRatio": short_ratio,
            })

            # Polite rate-limiting: ~1 req/sec
            time.sleep(0.3)

        except Exception as exc:
            logger.debug("FINRA fetch failed for %s on %s: %s", ticker, date.date(), exc)

    if not records:
        return pd.DataFrame(columns=["Date", "ShortVolume", "TotalVolume", "ShortRatio"])

    result = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    return result
