"""Universe scanner – dynamic ticker add/remove based on RVOL, news, and earnings.

Provides helpers to:
* Filter the current ticker universe to only those with interesting activity.
* Propose new tickers to add based on RVOL or earnings proximity.
* Mark tickers as inactive when they no longer meet any criterion.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_RVOL_ADD_THRESHOLD = 2.0       # add ticker when RVOL >= this value
DEFAULT_RVOL_REMOVE_THRESHOLD = 0.8    # remove/disable ticker when RVOL < this value
DEFAULT_EARNINGS_ADD_WINDOW = 14       # add ticker when earnings within this many days
DEFAULT_NEWS_SCORE_ADD = 0.5           # add ticker when news score >= this value


@dataclass
class UniverseState:
    """Mutable state of the ticker universe."""

    active: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def scan_universe(
    tickers: list[str],
    ticker_stats: dict[str, dict],
    *,
    rvol_add_threshold: float = DEFAULT_RVOL_ADD_THRESHOLD,
    rvol_remove_threshold: float = DEFAULT_RVOL_REMOVE_THRESHOLD,
    earnings_add_window: int = DEFAULT_EARNINGS_ADD_WINDOW,
    news_score_add: float = DEFAULT_NEWS_SCORE_ADD,
) -> UniverseState:
    """Filter and update the ticker universe.

    Parameters
    ----------
    tickers:
        Current list of active tickers (from config).
    ticker_stats:
        Mapping of ticker → dict with keys:
        ``volume_rvol_20d``, ``days_to_earnings``, ``news_score``.
        Any missing key defaults to a neutral value.
    rvol_add_threshold / rvol_remove_threshold:
        RVOL boundaries for add/remove decisions.
    earnings_add_window:
        Days-to-earnings threshold for keeping/adding a ticker.
    news_score_add:
        Minimum news score (0–1) to add/keep a ticker.

    Returns
    -------
    UniverseState
        Lists the tickers that should remain *active*, those freshly *added*,
        and those *removed* in this scan cycle.
    """
    active: list[str] = []
    added: list[str] = []
    removed: list[str] = []

    for ticker in tickers:
        stats = ticker_stats.get(ticker, {})
        rvol = float(stats.get("volume_rvol_20d", 1.0))
        dte = int(stats.get("days_to_earnings", 999))
        news = float(stats.get("news_score", 0.0))

        keep = (
            rvol >= rvol_add_threshold
            or dte <= earnings_add_window
            or news >= news_score_add
        )

        if keep:
            active.append(ticker)
        else:
            removed.append(ticker)
            logger.debug("Universe: removing %s (RVOL=%.2f, DTE=%d, news=%.2f)", ticker, rvol, dte, news)

    return UniverseState(active=active, added=added, removed=removed)


def propose_additions(
    candidate_tickers: list[str],
    current_tickers: list[str],
    ticker_stats: dict[str, dict],
    *,
    rvol_add_threshold: float = DEFAULT_RVOL_ADD_THRESHOLD,
    earnings_add_window: int = DEFAULT_EARNINGS_ADD_WINDOW,
    news_score_add: float = DEFAULT_NEWS_SCORE_ADD,
) -> list[str]:
    """Return tickers from *candidate_tickers* that merit being added to the universe."""
    new_tickers = []
    current_set = set(current_tickers)
    for ticker in candidate_tickers:
        if ticker in current_set:
            continue
        stats = ticker_stats.get(ticker, {})
        rvol = float(stats.get("volume_rvol_20d", 1.0))
        dte = int(stats.get("days_to_earnings", 999))
        news = float(stats.get("news_score", 0.0))
        if rvol >= rvol_add_threshold or dte <= earnings_add_window or news >= news_score_add:
            new_tickers.append(ticker)
            logger.info("Universe: proposing addition of %s (RVOL=%.2f, DTE=%d, news=%.2f)", ticker, rvol, dte, news)
    return new_tickers
