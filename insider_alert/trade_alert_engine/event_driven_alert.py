"""Event-Driven alert detection.

Generates alerts for pre/post earnings, SEC 8-K material events, and other
catalysts.  Alerts include a sector tag for filtering purposes.
"""
import logging

logger = logging.getLogger(__name__)

# Sectors mapped to a representative label (extend as needed)
_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMZN": "Consumer Cyclical", "GOOGL": "Technology", "META": "Technology",
    "TSLA": "Consumer Cyclical", "NFLX": "Communication", "AMD": "Technology",
    "AVGO": "Technology", "QCOM": "Technology", "ARM": "Technology",
    "SMCI": "Technology", "PLTR": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "SNOW": "Technology", "SHOP": "Technology",
    "UBER": "Technology", "COIN": "Financial", "HOOD": "Financial",
    "JPM": "Financial", "BAC": "Financial", "GS": "Financial",
    "MS": "Financial", "WFC": "Financial", "C": "Financial",
    "V": "Financial", "MA": "Financial",
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy", "SLB": "Energy",
    "CAT": "Industrials", "GE": "Industrials",
    "UNH": "Healthcare", "LLY": "Healthcare", "JNJ": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "ISRG": "Healthcare",
    "MRNA": "Healthcare",
    "SOFI": "Financial", "RIVN": "Consumer Cyclical", "LCID": "Consumer Cyclical",
}

DEFAULT_PRE_EARNINGS_WINDOW = 10    # days before earnings that trigger pre-event alert
DEFAULT_POST_EARNINGS_MOVE = 0.05   # 5 % move after earnings triggers post-event alert
DEFAULT_EVENT_SCORE_THRESHOLD = 50  # minimum score to emit an alert


def _get_sector(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker.upper(), "Unknown")


def detect_event_driven(
    ticker: str,
    event_features: dict,
    price_features: dict,
    volume_features: dict,
    options_features: dict | None = None,
    *,
    pre_earnings_window: int = DEFAULT_PRE_EARNINGS_WINDOW,
    post_earnings_move: float = DEFAULT_POST_EARNINGS_MOVE,
) -> dict | None:
    """Return an event-driven alert dict or *None* if no event catalyst is found."""
    days_to_earnings = event_features.get("days_to_earnings", 999)
    days_to_corp_event = event_features.get("days_to_corporate_event", 999)
    pre_return = event_features.get("pre_event_return_score", 0.0)
    pre_volume = event_features.get("pre_event_volume_score", 0.0)
    pre_options = event_features.get("pre_event_options_score", 0.0)

    return_1d = price_features.get("return_1d", 0.0)
    rvol = volume_features.get("volume_rvol_20d", 1.0)

    flags: list[str] = []
    score = 0.0
    event_label = ""
    setup_type = ""

    # Pre-earnings: N days before earnings + unusual activity
    if days_to_earnings <= pre_earnings_window:
        event_label = f"Pre-earnings ({days_to_earnings}d)"
        setup_type = "event_pre_earnings"
        score += (pre_earnings_window - days_to_earnings) / pre_earnings_window * 30.0
        score += pre_return * 25.0
        score += pre_volume * 25.0
        score += pre_options * 20.0
        flags.append(f"Earnings in {days_to_earnings} days")
        if rvol >= 1.5:
            flags.append(f"Pre-earnings volume elevated: RVOL={rvol:.2f}x")

    # Post-earnings: large move on the day of or day after earnings
    elif abs(return_1d) >= post_earnings_move and pre_return > 0:
        event_label = "Post-earnings move"
        setup_type = "event_post_earnings"
        score = 50.0 + min(abs(return_1d) / post_earnings_move, 2.0) * 25.0
        direction = "up" if return_1d > 0 else "down"
        flags.append(f"Post-earnings move {direction}: {return_1d * 100:.1f}%")
        if rvol >= 1.5:
            flags.append(f"Volume confirmation: RVOL={rvol:.2f}x")

    # Material 8-K corporate event
    elif days_to_corp_event <= 10 and days_to_corp_event != days_to_earnings:
        event_label = f"Material 8-K event ({days_to_corp_event}d)"
        setup_type = "event_8k_material"
        score = 40.0 + pre_return * 30.0 + pre_volume * 30.0
        flags.append(f"SEC 8-K filing: material event in {days_to_corp_event} days")

    else:
        return None

    if score < DEFAULT_EVENT_SCORE_THRESHOLD:
        return None

    sector = _get_sector(ticker)
    flags.insert(0, f"Sector: {sector}")

    atr_14 = price_features.get("atr_14", 0.0)

    return {
        "alert_type": "event_driven",
        "setup_type": setup_type,
        "event_label": event_label,
        "sector": sector,
        "days_to_earnings": days_to_earnings,
        "days_to_corp_event": days_to_corp_event,
        "return_1d": round(return_1d, 4),
        "atr": round(atr_14, 4),
        "score": float(min(max(score, 0.0), 100.0)),
        "flags": flags,
    }
