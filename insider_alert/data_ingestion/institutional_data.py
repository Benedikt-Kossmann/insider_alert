"""SEC 13-F institutional holdings data.

Fetches quarterly 13-F filings from SEC EDGAR for top institutional investors
and computes a smart-money flow score for a given ticker.

Limitations:
- 13-F data has ~45 day lag after quarter-end (expected behavior).
- Short positions are NOT reported in 13-F filings.
- Uses name-matching to find ticker holdings (no CUSIP lookup).
"""
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from insider_alert.data_ingestion.sec_utils import EDGAR_HEADERS as _HEADERS, EDGAR_REQUEST_DELAY

logger = logging.getLogger(__name__)

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore

# SEC rate limit: 10 req/s; we use 0.12s delay between requests (same as insider_data.py)
_REQ_DELAY = EDGAR_REQUEST_DELAY

# Known CIK numbers for top institutional investors.
# Only the 20 most relevant for long-only equity signals are included.
_TOP_FUNDS: dict[str, str] = {
    "Berkshire Hathaway": "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors": "0001423053",
    "Two Sigma": "0001179392",
    "DE Shaw": "0001009207",
    "Millennium Management": "0001273087",
    "Point72": "0001603466",
    "Baupost Group": "0001061768",
    "Elliott Management": "0001048445",
    "Viking Global": "0001103804",
    "Tiger Global": "0001167483",
    "Third Point": "0001040273",
    "Pershing Square": "0001336528",
    "Soros Fund Management": "0001029160",
    "AQR Capital Management": "0001167557",
    "Blackrock": "0001086364",
    "Vanguard": "0000102909",
    "State Street": "0000093751",
    "T. Rowe Price": "0001113169",
}


def _fetch_latest_13f_url(cik: str) -> Optional[str]:
    """Return the directory URL for the most recent 13-F filing of a given CIK.

    Uses the EDGAR submissions JSON API.  Returns None when no 13-F is found
    or the network request fails.
    """
    if _requests is None:
        return None

    cik_padded = cik.lstrip("0").zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        time.sleep(_REQ_DELAY)
        resp = _requests.get(submissions_url, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.debug("Submissions fetch HTTP %s for CIK %s", resp.status_code, cik)
            return None

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms: list[str] = recent.get("form", [])
        accessions: list[str] = recent.get("accessionNumber", [])
        cik_numeric = cik.lstrip("0")

        for i, form in enumerate(forms):
            if "13F" in form.upper().replace("-", ""):
                acc = accessions[i].replace("-", "")
                return f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{acc}/"

        return None
    except Exception as exc:
        logger.debug("13-F URL lookup failed for CIK %s: %s", cik, exc)
        return None


def _fetch_xml_url_from_index(filing_dir_url: str) -> Optional[str]:
    """Find the XML information-table file URL within a 13-F filing directory."""
    if _requests is None:
        return None

    try:
        time.sleep(_REQ_DELAY)
        resp = _requests.get(filing_dir_url, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        # The directory index is HTML; look for a link ending in .xml that
        # contains "infotable" or is the primary information table document.
        for line in resp.text.splitlines():
            lower = line.lower()
            if ".xml" in lower and ("infotable" in lower or "information" in lower):
                match = re.search(r'href="([^"]+\.xml)"', line, re.IGNORECASE)
                if match:
                    path = match.group(1)
                    # Make absolute if relative
                    if path.startswith("/"):
                        return f"https://www.sec.gov{path}"
                    return filing_dir_url + path

        # Fallback: any .xml link in the directory
        for line in resp.text.splitlines():
            if ".xml" in line.lower():
                match = re.search(r'href="([^"]+\.xml)"', line, re.IGNORECASE)
                if match:
                    path = match.group(1)
                    if path.startswith("/"):
                        return f"https://www.sec.gov{path}"
                    return filing_dir_url + path

        return None
    except Exception as exc:
        logger.debug("13-F index parse failed for %s: %s", filing_dir_url, exc)
        return None


def _parse_13f_xml(xml_url: str) -> list[dict]:
    """Download and parse a 13-F XML information table.

    Returns a list of dicts with keys: ``name``, ``cusip``, ``value_1000``,
    ``shares``.
    """
    if _requests is None:
        return []

    holdings: list[dict] = []
    try:
        time.sleep(_REQ_DELAY)
        resp = _requests.get(xml_url, headers=_HEADERS, timeout=20)
        if resp.status_code != 200:
            logger.debug("13-F XML fetch HTTP %s for %s", resp.status_code, xml_url)
            return holdings

        root = ET.fromstring(resp.content)

        for entry in root.iter():
            tag_local = entry.tag.split("}")[-1].lower() if "}" in entry.tag else entry.tag.lower()
            if tag_local == "infotable":
                holding: dict = {}
                for child in entry:
                    child_tag = (
                        child.tag.split("}")[-1].lower()
                        if "}" in child.tag
                        else child.tag.lower()
                    )
                    if "nameofissuer" in child_tag:
                        holding["name"] = (child.text or "").strip().upper()
                    elif "cusip" in child_tag:
                        holding["cusip"] = (child.text or "").strip()
                    elif child_tag == "value":
                        try:
                            holding["value_1000"] = int(child.text or 0)
                        except ValueError:
                            holding["value_1000"] = 0
                    elif "sshprnamt" in child_tag:
                        try:
                            holding["shares"] = int(child.text or 0)
                        except ValueError:
                            holding["shares"] = 0

                if holding.get("name"):
                    holdings.append(holding)

    except ET.ParseError as exc:
        logger.debug("13-F XML parse error for %s: %s", xml_url, exc)
    except Exception as exc:
        logger.debug("13-F XML fetch/parse failed for %s: %s", xml_url, exc)

    return holdings


def _ticker_in_holdings(ticker: str, holdings: list[dict]) -> bool:
    """Return True when the ticker appears in a list of 13-F holdings."""
    ticker_upper = ticker.upper()
    for h in holdings:
        name = h.get("name", "")
        # Exact prefix match or the ticker appears as a word boundary in the
        # issuer name (e.g. "APPLE INC" contains "AAPL" as ticker, but we
        # match on company name tokens since CUSIPs aren't resolved here).
        if name.startswith(ticker_upper) or re.search(
            r"\b" + re.escape(ticker_upper) + r"\b", name
        ):
            return True
    return False


def fetch_institutional_flows(ticker: str, max_funds: int = 20) -> dict:
    """Aggregate institutional 13-F presence for *ticker* across top funds.

    Checks at most *max_funds* (default 20) to stay within rate limits.

    Because CUSIP↔ticker mapping is not resolved, "buy" is approximated by
    the fund *holding* the stock. Without a prior-quarter baseline stored in
    the DB we cannot yet distinguish new positions from continued holds.
    The ``smart_money_score`` is the fraction of checked funds that hold the
    stock.

    Returns a dict with four keys:
        ``institutional_buy_count``   – int, funds found holding the stock
        ``institutional_sell_count``  – int (always 0 for now, placeholder)
        ``institutional_net_direction`` – "accumulation" | "distribution" | "neutral"
        ``smart_money_score``         – float [0, 1]
    """
    defaults: dict = {
        "institutional_buy_count": 0,
        "institutional_sell_count": 0,
        "institutional_net_direction": "neutral",
        "smart_money_score": 0.5,
    }

    if _requests is None:
        logger.warning("requests library not available – institutional flows skipped")
        return defaults

    buy_count = 0
    sell_count = 0
    checked = 0

    for fund_name, cik in list(_TOP_FUNDS.items())[:max_funds]:
        try:
            filing_dir = _fetch_latest_13f_url(cik)
            if not filing_dir:
                continue

            xml_url = _fetch_xml_url_from_index(filing_dir)
            if not xml_url:
                continue

            holdings = _parse_13f_xml(xml_url)
            checked += 1

            if _ticker_in_holdings(ticker, holdings):
                buy_count += 1
                logger.debug("  %s holds %s", fund_name, ticker)

        except Exception as exc:
            logger.debug("Institutional check failed for %s / %s: %s", ticker, fund_name, exc)

    if checked == 0:
        logger.debug("No 13-F filings retrieved for %s – returning defaults", ticker)
        return defaults

    net = buy_count - sell_count
    if net > 2:
        direction = "accumulation"
    elif net < -2:
        direction = "distribution"
    else:
        direction = "neutral"

    smart_score = float(np.clip(buy_count / max(checked, 1), 0.0, 1.0))

    logger.info(
        "Institutional flows %s: %d/%d funds holding, direction=%s, score=%.2f",
        ticker, buy_count, checked, direction, smart_score,
    )

    return {
        "institutional_buy_count": buy_count,
        "institutional_sell_count": sell_count,
        "institutional_net_direction": direction,
        "smart_money_score": round(smart_score, 3),
    }
