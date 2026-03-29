"""Insider transaction data.
Uses SEC EDGAR open data API (https://data.sec.gov/submissions/) for Form-4 filings.
Parses Form-4 XML documents for actual insider names, roles, transaction types,
share counts and transaction values.
Falls back to empty DataFrame if network unavailable.
"""
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# SEC EDGAR requires CIK numbers zero-padded to 10 digits in the URL path
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{document}"
_EDGAR_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{accession}-index.json"
_EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "insider_alert_bot contact@example.com",
)
_HEADERS = {"User-Agent": _EDGAR_USER_AGENT}
# Maximum number of Form-4 XML documents to parse per call (rate-limit friendly)
_MAX_FORM4_FETCH = 10
# Polite delay between EDGAR requests in seconds
_EDGAR_REQUEST_DELAY = 0.12


def _get_cik_for_ticker(ticker: str) -> str | None:
    """Resolve ticker to SEC CIK using EDGAR company search."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"])
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


def _xml_text(element, path: str) -> str:
    """Safely extract text from an XML element using a dotted path."""
    node = element.find(path)
    if node is None:
        return ""
    # Some Form-4 fields wrap the value in a <value> child element
    value_node = node.find("value")
    if value_node is not None and value_node.text:
        return value_node.text.strip()
    return (node.text or "").strip()


def _parse_role(rel_element) -> str:
    """Derive a human-readable role from a reportingOwnerRelationship element."""
    if rel_element is None:
        return "Unknown"
    officer_title = _xml_text(rel_element, "officerTitle")
    if officer_title:
        return officer_title
    if _xml_text(rel_element, "isDirector") == "1":
        return "Director"
    if _xml_text(rel_element, "isTenPercentOwner") == "1":
        return "10% Owner"
    return "Other"


def _find_xml_document(cik: str, accession: str, primary_doc: str) -> str | None:
    """Return the URL of the Form-4 XML document for a given filing.

    Tries the primaryDocument first; if that fails, looks up the filing index to
    find the first document whose documentType is '4'.
    """
    accession_nodashes = accession.replace("-", "")

    # Primary document from submissions JSON
    if primary_doc and primary_doc.lower().endswith(".xml"):
        return _EDGAR_ARCHIVES.format(
            cik=cik, accession_nodashes=accession_nodashes, document=primary_doc
        )

    # Fall back to the filing index JSON to find the XML
    try:
        index_url = _EDGAR_INDEX.format(
            cik=cik,
            accession_nodashes=accession_nodashes,
            accession=accession,
        )
        resp = requests.get(index_url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        index_data = resp.json()
        for doc in index_data.get("documents", []):
            if doc.get("documentType") == "4" and doc.get("documentName", "").endswith(".xml"):
                return _EDGAR_ARCHIVES.format(
                    cik=cik,
                    accession_nodashes=accession_nodashes,
                    document=doc["documentName"],
                )
    except Exception as exc:
        logger.debug("Filing index lookup failed for %s %s: %s", cik, accession, exc)

    return None


def _parse_form4_xml(cik: str, accession: str, primary_doc: str, filing_date) -> list[dict]:
    """Fetch and parse a Form-4 XML document.

    Returns a list of transaction dicts with keys:
    date, insider_name, role, transaction_type, shares, value.
    """
    xml_url = _find_xml_document(cik, accession, primary_doc)
    if xml_url is None:
        return []

    try:
        resp = requests.get(xml_url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:
        logger.debug("Form-4 XML fetch/parse failed for %s: %s", accession, exc)
        return []

    # Insider identity
    insider_name = _xml_text(root, ".//reportingOwner/reportingOwnerId/rptOwnerName") or "Unknown"
    rel_el = root.find(".//reportingOwner/reportingOwnerRelationship")
    role = _parse_role(rel_el)

    rows: list[dict] = []
    for txn in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        txn_date_str = _xml_text(txn, "transactionDate")
        try:
            txn_date = datetime.strptime(txn_date_str, "%Y-%m-%d").date() if txn_date_str else filing_date
        except ValueError:
            txn_date = filing_date

        try:
            shares = float(_xml_text(txn, "transactionAmounts/transactionShares") or "0")
        except ValueError:
            shares = 0.0

        try:
            price = float(_xml_text(txn, "transactionAmounts/transactionPricePerShare") or "0")
        except ValueError:
            price = 0.0

        code = _xml_text(txn, "transactionAmounts/transactionAcquiredDisposedCode")
        txn_type = "buy" if code.upper() == "A" else "sell"

        rows.append({
            "date": txn_date,
            "insider_name": insider_name,
            "role": role,
            "transaction_type": txn_type,
            "shares": shares,
            "value": shares * price,
        })

    return rows


def fetch_insider_transactions(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """Return insider buy/sell transactions for the last `days_back` days.

    Parses SEC EDGAR Form-4 XML filings to extract:
    - insider_name: reporting owner's name
    - role: officer title, Director, 10% Owner, or Other
    - transaction_type: 'buy' (Acquired) or 'sell' (Disposed)
    - shares: number of shares transacted
    - value: shares * price_per_share

    Columns: date, insider_name, role, transaction_type, value, shares.
    """
    _empty = pd.DataFrame(columns=["date", "insider_name", "role", "transaction_type", "value", "shares"])
    try:
        cik = _get_cik_for_ticker(ticker)
        if cik is None:
            logger.warning("Could not resolve CIK for %s", ticker)
            return _empty

        url = _EDGAR_SUBMISSIONS.format(cik=int(cik))
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [""] * len(forms))

        cutoff = datetime.now() - timedelta(days=days_back)
        recent_form4s = []
        for form, date_str, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
            if form != "4":
                continue
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if filing_date < cutoff:
                continue
            recent_form4s.append((filing_date.date(), accession, primary_doc))

        rows: list[dict] = []
        for filing_date, accession, primary_doc in recent_form4s[:_MAX_FORM4_FETCH]:
            try:
                txn_rows = _parse_form4_xml(cik, accession, primary_doc, filing_date)
                rows.extend(txn_rows)
            except Exception as exc:
                logger.debug("Skipping Form-4 %s: %s", accession, exc)
            time.sleep(_EDGAR_REQUEST_DELAY)

        if not rows:
            return _empty
        return pd.DataFrame(rows)[["date", "insider_name", "role", "transaction_type", "value", "shares"]]

    except Exception as exc:
        logger.warning("fetch_insider_transactions failed for %s: %s", ticker, exc)
        return _empty
