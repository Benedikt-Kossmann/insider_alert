"""Weekly performance report sent via Telegram.

Aggregates signal outcomes, hit rates, and score distributions for the past
7 days and sends a summary message.
"""
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def generate_weekly_report(
    tickers: list[str],
    db_url: str = "sqlite:///insider_alert.db",
) -> str:
    """Build a Markdown-formatted weekly performance report.

    Parameters
    ----------
    tickers : list[str]
        List of watched tickers.
    db_url : str
        Database URL.

    Returns
    -------
    str
        Telegram-ready Markdown message.
    """
    from insider_alert.persistence.storage import (
        _get_engine, SignalOutcome, Score, Alert,
    )
    from sqlalchemy.orm import sessionmaker

    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    cutoff = date.today() - timedelta(days=7)

    with Session() as session:
        # Scores from last 7 days
        scores = (
            session.query(Score)
            .filter(Score.date >= cutoff)
            .all()
        )
        # Alerts sent last 7 days
        alert_count = (
            session.query(Alert)
            .filter(Alert.sent_at is not None, Alert.date >= cutoff)
            .count()
        )
        # Signal outcomes with results
        outcomes = (
            session.query(SignalOutcome)
            .filter(
                SignalOutcome.date >= cutoff,
                SignalOutcome.hit_5d.isnot(None),
            )
            .all()
        )

    # --- Aggregate metrics ---
    total_analyses = len(scores)
    if scores:
        avg_score = sum(s.total_score for s in scores) / len(scores)
        max_score_row = max(scores, key=lambda s: s.total_score)
        high_scores = [s for s in scores if s.total_score >= 60]
    else:
        avg_score = 0.0
        max_score_row = None
        high_scores = []

    # Hit rates from outcomes
    if outcomes:
        hits_1d = [o for o in outcomes if o.hit_1d]
        hits_5d = [o for o in outcomes if o.hit_5d]
        hit_rate_1d = len(hits_1d) / len(outcomes) * 100
        hit_rate_5d = len(hits_5d) / len(outcomes) * 100
        returns_5d = [o.return_5d for o in outcomes if o.return_5d is not None]
        avg_return_5d = sum(returns_5d) / len(returns_5d) * 100 if returns_5d else 0.0
    else:
        hit_rate_1d = 0.0
        hit_rate_5d = 0.0
        avg_return_5d = 0.0

    # Per-signal breakdown
    from collections import defaultdict
    by_signal: dict[str, list] = defaultdict(list)
    for o in outcomes:
        by_signal[o.signal_type].append(o)

    # --- Build message ---
    lines = [
        "📊 *Wöchentlicher Performance-Report*",
        f"_Zeitraum: {cutoff.isoformat()} bis {date.today().isoformat()}_",
        "",
        "*Übersicht:*",
        f"  📈 Analysen durchgeführt: {total_analyses}",
        f"  🚨 Alerts gesendet: {alert_count}",
        f"  📊 Ø Composite Score: {avg_score:.1f}",
        f"  🏆 Höchster Score: {max_score_row.total_score:.1f} ({max_score_row.ticker})" if max_score_row else "  🏆 Höchster Score: –",
        f"  🔥 High-Score Alerts (≥60): {len(high_scores)}",
    ]

    if outcomes:
        lines += [
            "",
            "*Signal-Performance:*",
            f"  Hit-Rate 1d: {hit_rate_1d:.1f}%",
            f"  Hit-Rate 5d: {hit_rate_5d:.1f}%",
            f"  Ø Return 5d: {avg_return_5d:+.2f}%",
        ]

        # Top / worst signals
        if by_signal:
            lines.append("")
            lines.append("*Signal-Breakdown:*")
            for sig_type, sig_outcomes in sorted(by_signal.items()):
                if not sig_outcomes:
                    continue
                sig_hits = sum(1 for o in sig_outcomes if o.hit_5d)
                sig_rate = sig_hits / len(sig_outcomes) * 100
                emoji = "✅" if sig_rate >= 55 else ("⚠️" if sig_rate >= 45 else "❌")
                lines.append(f"  {emoji} {sig_type}: {sig_rate:.0f}% ({len(sig_outcomes)} Signale)")

    else:
        lines += [
            "",
            "_Noch keine Signal-Ergebnisse für diese Woche._",
        ]

    # Top tickers by score
    if scores:
        from collections import defaultdict as dd
        by_ticker: dict[str, list[float]] = dd(list)
        for s in scores:
            by_ticker[s.ticker].append(s.total_score)

        avg_by_ticker = {t: sum(v) / len(v) for t, v in by_ticker.items()}
        top_tickers = sorted(avg_by_ticker.items(), key=lambda x: -x[1])[:5]

        if top_tickers:
            lines.append("")
            lines.append("*Top Ticker (Ø Score):*")
            for i, (t, avg) in enumerate(top_tickers, 1):
                medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
                lines.append(f"  {medal} {t}: {avg:.1f}")

    lines.append("")
    lines.append(f"_Watchlist: {len(tickers)} Ticker aktiv_")
    lines.append("_Gute Woche!_ 🚀")

    return "\n".join(lines)


def send_weekly_report(config) -> bool:
    """Generate and send the weekly performance report via Telegram.

    Returns True if sent successfully.
    """
    from insider_alert.alert_engine.telegram_alert import send_telegram_message

    try:
        report = generate_weekly_report(config.tickers)
        return send_telegram_message(config.telegram_token, config.telegram_chat_id, report)
    except Exception as exc:
        logger.error("Failed to send weekly report: %s", exc)
        return False
