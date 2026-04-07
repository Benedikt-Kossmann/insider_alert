"""Discovery scanner – find unusual activity in stocks and crypto you don't already watch.

Uses yfinance batch download to scan a broad pool for:
* Volume spikes (RVOL vs 20-day average)
* Large price moves (absolute daily return)
* Gap opens (open vs previous close)

Returns discoveries that pass configurable thresholds, excluding tickers
already in the user's watchlist or ETF universe.
"""
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default scan pools – broad diversified sets that work on yfinance
# ---------------------------------------------------------------------------

DEFAULT_STOCK_POOL: list[str] = [
    # US mega / large-cap (beyond typical watchlist)
    "INTC", "MU", "MRVL", "PANW", "CRWD", "ZS", "DDOG", "NET",
    "SPOT", "SQ", "PYPL", "AFRM", "ROKU", "TTD", "SNAP", "PINS",
    "RBLX", "U", "SE", "MELI", "BABA", "JD", "PDD", "NIO",
    "LI", "XPEV", "BIDU", "TSM", "ASML", "LRCX", "KLAC", "AMAT",
    "TXN", "ADI", "ON", "MCHP", "WOLF", "ENPH", "SEDG", "FSLR",
    "DIS", "CMCSA", "WBD", "PARA", "T", "VZ", "TMUS",
    "WMT", "COST", "TGT", "HD", "LOW", "SBUX", "MCD", "NKE",
    "PG", "KO", "PEP",
    "CVS", "CI", "HUM", "TMO", "DHR", "ABT", "GILD", "REGN", "VRTX",
    "DE", "HON", "LMT", "RTX", "BA", "GD",
    "BLK", "SCHW", "AXP", "TFC", "USB",
    "COP", "EOG", "MPC", "VLO", "PSX",
    "AMT", "PLD", "SPG", "O", "EQIX",
    # Small / mid-cap movers
    "IONQ", "RGTI", "QUBT", "SOUN", "AI", "BBAI", "APLD", "IREN",
    "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF",
    # EU-listed (Xetra / Paris / Amsterdam)
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BMW.DE",
    "MBG.DE", "VOW3.DE", "ADS.DE", "IFX.DE",
    "MC.PA", "OR.PA", "TTE.PA", "AI.PA", "SAN.PA",
    "ASML.AS", "PHIA.AS", "INGA.AS",
]

DEFAULT_CRYPTO_POOL: list[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "AVAX-USD", "DOGE-USD", "SHIB-USD", "DOT-USD", "LINK-USD",
    "MATIC-USD", "UNI-USD", "LTC-USD", "BCH-USD", "NEAR-USD",
    "APT-USD", "ARB-USD", "OP-USD", "FIL-USD", "ATOM-USD",
    "ALGO-USD", "HBAR-USD", "ICP-USD", "INJ-USD", "SUI-USD",
    "FET-USD", "RNDR-USD", "GRT-USD", "IMX-USD", "STX-USD",
    "PEPE-USD", "WIF-USD", "BONK-USD", "FLOKI-USD",
    "AAVE-USD", "MKR-USD", "CRV-USD", "SNX-USD", "COMP-USD",
]

DEFAULT_RVOL_THRESHOLD = 3.0        # relative volume vs 20d average
DEFAULT_MOVE_PCT_THRESHOLD = 5.0    # absolute % daily move
DEFAULT_GAP_PCT_THRESHOLD = 3.0     # absolute % gap open
DEFAULT_CRYPTO_MOVE_PCT_THRESHOLD = 8.0  # crypto is more volatile


@dataclass
class Discovery:
    """A single discovered symbol with unusual activity."""
    ticker: str
    asset_class: str           # "stock" | "crypto"
    rvol: float                # relative volume
    move_pct: float            # daily price change %
    gap_pct: float             # gap open %
    close: float               # last close price
    volume: float              # last volume
    reasons: list[str] = field(default_factory=list)


def _batch_download(tickers: list[str], period: str = "1mo") -> pd.DataFrame:
    """Download OHLCV for multiple tickers via yfinance."""
    import yfinance as yf
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(tickers, period=period, interval="1d",
                         auto_adjust=True, progress=False, threads=True)
        return df
    except Exception as exc:
        logger.error("Batch download failed: %s", exc)
        return pd.DataFrame()


def _compute_signals(
    data: pd.DataFrame,
    tickers: list[str],
    asset_class: str,
    *,
    rvol_threshold: float,
    move_pct_threshold: float,
    gap_pct_threshold: float,
) -> list[Discovery]:
    """Scan downloaded data for unusual activity."""
    discoveries: list[Discovery] = []

    for ticker in tickers:
        try:
            # Handle multi-ticker df (MultiIndex columns) and single-ticker
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(1):
                    continue
                close = data[("Close", ticker)].dropna()
                volume = data[("Volume", ticker)].dropna()
                opn = data[("Open", ticker)].dropna()
            else:
                # Single ticker
                if "Close" not in data.columns:
                    # lowercase fallback
                    close = data.get("close", pd.Series(dtype=float)).dropna()
                    volume = data.get("volume", pd.Series(dtype=float)).dropna()
                    opn = data.get("open", pd.Series(dtype=float)).dropna()
                else:
                    close = data["Close"].dropna()
                    volume = data["Volume"].dropna()
                    opn = data["Open"].dropna()

            if len(close) < 5 or len(volume) < 5:
                continue

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            last_volume = float(volume.iloc[-1])
            last_open = float(opn.iloc[-1]) if len(opn) >= len(close) else last_close

            if prev_close == 0 or last_close == 0:
                continue

            # Relative volume (vs 20d avg, or available window)
            vol_window = min(20, len(volume) - 1)
            avg_vol = float(volume.iloc[-(vol_window + 1):-1].mean())
            rvol = last_volume / avg_vol if avg_vol > 0 else 1.0

            # Daily move %
            move_pct = ((last_close - prev_close) / prev_close) * 100.0

            # Gap %
            gap_pct = ((last_open - prev_close) / prev_close) * 100.0

            reasons: list[str] = []
            if rvol >= rvol_threshold:
                reasons.append(f"RVOL {rvol:.1f}x")
            if abs(move_pct) >= move_pct_threshold:
                direction = "+" if move_pct > 0 else ""
                reasons.append(f"Move {direction}{move_pct:.1f}%")
            if abs(gap_pct) >= gap_pct_threshold:
                direction = "+" if gap_pct > 0 else ""
                reasons.append(f"Gap {direction}{gap_pct:.1f}%")

            if reasons:
                discoveries.append(Discovery(
                    ticker=ticker,
                    asset_class=asset_class,
                    rvol=rvol,
                    move_pct=move_pct,
                    gap_pct=gap_pct,
                    close=last_close,
                    volume=last_volume,
                    reasons=reasons,
                ))

        except Exception as exc:
            logger.debug("Discovery scan skipped %s: %s", ticker, exc)
            continue

    return discoveries


def run_discovery_scan(
    config,
    *,
    exclude_tickers: list[str] | None = None,
) -> list[Discovery]:
    """Run the full discovery scan and return unusual findings.

    Parameters
    ----------
    config:
        Application config (uses ``config.discovery`` dict).
    exclude_tickers:
        Tickers to exclude (already watched).  If *None*, auto-builds
        the list from ``config.tickers`` + leveraged ETF universe.
    """
    disc_cfg = getattr(config, "discovery", {}) or {}
    if not disc_cfg.get("enabled", False):
        logger.debug("Discovery scanner disabled")
        return []

    # Build exclusion set
    if exclude_tickers is None:
        exclude_tickers = list(config.tickers)
        le_cfg = config.leveraged_etfs
        if le_cfg.get("enabled", False):
            for entry in le_cfg.get("universe", []):
                exclude_tickers.append(entry.get("ticker", ""))

    exclude_set = {t.upper() for t in exclude_tickers}

    # Thresholds
    rvol_thr = float(disc_cfg.get("rvol_threshold", DEFAULT_RVOL_THRESHOLD))
    move_thr = float(disc_cfg.get("move_pct_threshold", DEFAULT_MOVE_PCT_THRESHOLD))
    gap_thr = float(disc_cfg.get("gap_pct_threshold", DEFAULT_GAP_PCT_THRESHOLD))
    crypto_move_thr = float(disc_cfg.get("crypto_move_pct_threshold", DEFAULT_CRYPTO_MOVE_PCT_THRESHOLD))

    # Pools
    stock_pool = disc_cfg.get("stock_pool", DEFAULT_STOCK_POOL)
    crypto_pool = disc_cfg.get("crypto_pool", DEFAULT_CRYPTO_POOL)
    scan_stocks = disc_cfg.get("scan_stocks", True)
    scan_crypto = disc_cfg.get("scan_crypto", True)

    all_discoveries: list[Discovery] = []

    # --- Stock scan ---
    if scan_stocks and stock_pool:
        filtered_stocks = [t for t in stock_pool if t.upper() not in exclude_set]
        if filtered_stocks:
            logger.info("Discovery: scanning %d stocks", len(filtered_stocks))
            stock_data = _batch_download(filtered_stocks, period="1mo")
            if not stock_data.empty:
                all_discoveries.extend(_compute_signals(
                    stock_data, filtered_stocks, "stock",
                    rvol_threshold=rvol_thr,
                    move_pct_threshold=move_thr,
                    gap_pct_threshold=gap_thr,
                ))

    # --- Crypto scan ---
    if scan_crypto and crypto_pool:
        filtered_crypto = [t for t in crypto_pool if t.upper() not in exclude_set]
        if filtered_crypto:
            logger.info("Discovery: scanning %d crypto assets", len(filtered_crypto))
            crypto_data = _batch_download(filtered_crypto, period="1mo")
            if not crypto_data.empty:
                all_discoveries.extend(_compute_signals(
                    crypto_data, filtered_crypto, "crypto",
                    rvol_threshold=rvol_thr,
                    move_pct_threshold=crypto_move_thr,
                    gap_pct_threshold=gap_thr,
                ))

    # Sort by strongest signal (highest abs move first)
    all_discoveries.sort(key=lambda d: abs(d.move_pct), reverse=True)

    logger.info("Discovery: found %d unusual symbols", len(all_discoveries))
    return all_discoveries
