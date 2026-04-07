"""Send Telegram alerts via Bot API."""
import logging

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not token or not chat_id:
        logger.warning("Telegram token or chat_id not configured; skipping alert")
        return False
    try:
        url = _TELEGRAM_API.format(token=token)
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            logger.info("Telegram alert sent for chat_id=%s", chat_id)
            return True
        logger.warning("Telegram API returned not-ok: %s", data)
        return False
    except Exception as exc:
        logger.error("Failed to send Telegram message: %s", exc)
        return False


def send_welcome_message(config) -> bool:
    """Send a startup welcome message via Telegram summarising the config."""
    token = config.telegram_token
    chat_id = config.telegram_chat_id
    if not token or not chat_id:
        logger.info("Telegram not configured; skipping welcome message")
        return False

    n_tickers = len(config.tickers)
    le_cfg = config.leveraged_etfs
    etf_enabled = le_cfg.get("enabled", False)
    n_etfs = len(le_cfg.get("universe", [])) if etf_enabled else 0

    eod_h = config.scheduler.get("eod_hour", 17)
    eod_m = config.scheduler.get("eod_minute", 30)
    intra = config.scheduler.get("intraday_interval_minutes", 30)

    lines = [
        "🟢 *Insider Alert Bot gestartet*",
        "",
        f"📊 Aktien: {n_tickers}",
    ]
    if etf_enabled:
        lines.append(f"⚡ Leveraged ETFs: {n_etfs}")
    lines += [
        "",
        f"⏰ EOD-Analyse: täglich {eod_h:02d}:{eod_m:02d} UTC",
        f"🔄 Intraday-Scan: alle {intra} Min",
    ]

    trade_cfg = config.trade_alerts
    if trade_cfg.get("enabled", False):
        lines.append("📈 Trade-Alerts: aktiv")

    disc_cfg = getattr(config, "discovery", {}) or {}
    if disc_cfg.get("enabled", False):
        scan_parts = []
        if disc_cfg.get("scan_stocks", True):
            scan_parts.append("Aktien")
        if disc_cfg.get("scan_crypto", True):
            scan_parts.append("Krypto")
        lines.append(f"🔍 Discovery: {' + '.join(scan_parts)}")

    lines.append("")
    lines.append("_Bot läuft – viel Erfolg!_ 🚀")

    message = "\n".join(lines)
    return send_telegram_message(token, chat_id, message)


def build_alert_message(ticker_score) -> str:
    """Format a TickerScore into a human-readable Telegram message."""
    lines = [
        f"🚨 *Insider Alert: {ticker_score.ticker}*",
        f"Composite Score: *{ticker_score.total_score:.1f}/100*",
        "",
        "*Sub-scores:*",
    ]
    for sig_type, score in ticker_score.sub_scores.items():
        lines.append(f"  • {sig_type}: {score:.1f}")
    if ticker_score.flags:
        lines.append("")
        lines.append("*Flags:*")
        for flag in ticker_score.flags[:10]:
            lines.append(f"  ⚠️ {flag}")
    return "\n".join(lines)


def maybe_send_alert(ticker_score, token: str, chat_id: str, threshold: float = 60.0) -> bool:
    """Send alert if total_score >= threshold. Returns True if sent."""
    if ticker_score.total_score >= threshold:
        message = build_alert_message(ticker_score)
        return send_telegram_message(token, chat_id, message)
    return False


# ---------------------------------------------------------------------------
# Trade-alert message builders (Breakout / Mean-Reversion / Options / Event)
# ---------------------------------------------------------------------------

_ALERT_EMOJI = {
    "breakout": "📈",
    "mean_reversion": "🔄",
    "options_flow": "🎯",
    "event_driven": "📅",
    "multi_timeframe": "🔭",
    "leveraged_etf": "⚡",
}


def build_trade_alert_message(ticker: str, alert: dict) -> str:
    """Format a trade-alert dict into a Telegram-ready Markdown message.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    alert:
        Dict returned by one of the ``detect_*`` functions in
        ``trade_alert_engine``.  Must contain at minimum ``alert_type``,
        ``score``, and ``flags``.
    """
    alert_type = alert.get("alert_type", "trade")
    setup_type = alert.get("setup_type", alert_type)
    score = alert.get("score", 0.0)
    direction = alert.get("direction", "")
    emoji = _ALERT_EMOJI.get(alert_type, "🚨")

    title = f"{emoji} *{alert_type.replace('_', ' ').title()} Alert: {ticker}*"
    lines = [title, f"Setup: `{setup_type}`  |  Score: *{score:.0f}/100*"]

    if direction:
        dir_emoji = "🟢" if "bull" in direction else "🔴"
        lines.append(f"Direction: {dir_emoji} {direction.replace('_', ' ').title()}")

    # Alert-type-specific fields
    if alert_type == "breakout":
        lines.append(f"Breakout Level: `{alert.get('breakout_level', 0):.2f}`")
        lines.append(f"ATR(14): `{alert.get('atr', 0):.2f}`")
        lines.append(f"Stop Hint: `{alert.get('stop_hint', 0):.2f}`  |  Target: `{alert.get('target_hint', 0):.2f}`")
        lines.append(f"R:R ≈ {alert.get('rr_ratio', 2.0):.1f}")

    elif alert_type == "mean_reversion":
        lines.append(f"Price Z-score: `{alert.get('price_zscore', 0):.2f}`")
        lines.append(f"Volume Z-score: `{alert.get('volume_zscore', 0):.2f}`")
        if alert.get("atr_pct", 0) > 0:
            lines.append(f"ATR%: `{alert.get('atr_pct', 0) * 100:.2f}%`")
        lines.append(f"R:R ≈ {alert.get('rr_ratio', 1.5):.1f}")

    elif alert_type == "options_flow":
        if alert.get("sweep_score", 0) > 0:
            lines.append(f"Sweep Score: `{alert.get('sweep_score', 0):.2f}`")
        if alert.get("block_score", 0) > 0:
            lines.append(f"Block Score: `{alert.get('block_score', 0):.2f}`")
        if alert.get("iv_change", 0) != 0:
            lines.append(f"IV Change: `{alert.get('iv_change', 0) * 100:.1f}%`")
        if alert.get("near_earnings"):
            lines.append("⚡ Near earnings catalyst")

    elif alert_type == "event_driven":
        lines.append(f"Event: *{alert.get('event_label', '')}*")
        if alert.get("sector"):
            lines.append(f"Sector: {alert.get('sector')}")
        if alert.get("days_to_earnings", 999) <= 10:
            lines.append(f"Earnings in: {alert.get('days_to_earnings')}d")
        if alert.get("atr", 0) > 0:
            lines.append(f"ATR(14): `{alert.get('atr', 0):.2f}`")

    elif alert_type == "multi_timeframe":
        confirmed = alert.get("intraday_confirmed", False)
        lines.append(f"Intraday confirmed: {'✅' if confirmed else '⏳'}")
        lines.append(f"Daily 5d return: `{alert.get('daily_return_5d', 0) * 100:.1f}%`")
        if alert.get("atr", 0) > 0:
            lines.append(f"ATR(14): `{alert.get('atr', 0):.2f}`")

    flags = alert.get("flags", [])
    if flags:
        lines.append("")
        lines.append("*Flags:*")
        for flag in flags[:8]:
            lines.append(f"  ⚠️ {flag}")

    return "\n".join(lines)


def maybe_send_trade_alert(
    ticker: str,
    alert: dict,
    token: str,
    chat_id: str,
    *,
    score_threshold: float = 55.0,
    cooldown_hours: float = 4.0,
    db_url: str = "sqlite:///insider_alert.db",
) -> bool:
    """Send a trade alert via Telegram if it passes the score threshold and
    deduplication check.  Returns True if the message was sent.

    The function persists the sent alert and prevents the same
    ticker/setup_type combination from being re-sent within *cooldown_hours*.
    """
    from insider_alert.persistence.storage import is_alert_duplicate, save_alert
    from datetime import date

    score = alert.get("score", 0.0)
    setup_type = alert.get("setup_type", alert.get("alert_type", ""))

    if score < score_threshold:
        return False

    if is_alert_duplicate(ticker, setup_type, cooldown_hours=cooldown_hours, db_url=db_url):
        logger.info("Skipping duplicate trade alert: %s / %s", ticker, setup_type)
        return False

    message = build_trade_alert_message(ticker, alert)
    sent = send_telegram_message(token, chat_id, message)
    if sent:
        save_alert(
            ticker=ticker,
            date_val=date.today(),
            score=score,
            message=message,
            alert_type=alert.get("alert_type", ""),
            setup_type=setup_type,
            db_url=db_url,
        )
    return sent


# ---------------------------------------------------------------------------
# Leveraged-ETF alert message builders
# ---------------------------------------------------------------------------

def build_etf_alert_message(ticker: str, alert: dict, underlying: str = "", leverage: int = 3) -> str:
    """Format a leveraged-ETF alert for Telegram."""
    setup_type = alert.get("setup_type", "")
    score = alert.get("score", 0.0)
    direction = alert.get("direction", "long")
    dir_label = "Long" if direction == "long" else "Short/Inverse"

    setup_labels = {
        "momentum_entry": "📈 Momentum Entry",
        "dip_buy": "🔄 Dip-Buy Signal",
        "exit_warning": "🚨 Exit-Warnung",
    }
    setup_label = setup_labels.get(setup_type, setup_type)

    lines = [
        f"⚡ *Leveraged ETF Alert: {ticker}*",
        f"{leverage}x {underlying} {dir_label}",
        f"Setup: {setup_label}",
    ]
    if setup_type != "exit_warning":
        lines.append(f"Score: *{score:.0f}/100*")

    # Risk lines (if present)
    risk_lines = alert.get("risk_lines", [])
    if risk_lines:
        lines.append("")
        lines.append("*Risk:*")
        lines.extend(risk_lines)

    flags = alert.get("flags", [])
    if flags:
        lines.append("")
        lines.append("*Details:*")
        for flag in flags[:10]:
            lines.append(f"  • {flag}")

    return "\n".join(lines)


def maybe_send_etf_alert(
    ticker: str,
    alert: dict,
    token: str,
    chat_id: str,
    *,
    score_threshold: float = 55.0,
    cooldown_hours: float = 4.0,
    underlying: str = "",
    leverage: int = 3,
    db_url: str = "sqlite:///insider_alert.db",
) -> bool:
    """Send a leveraged-ETF alert via Telegram with deduplication."""
    from insider_alert.persistence.storage import is_alert_duplicate, save_alert
    from datetime import date

    score = alert.get("score", 0.0)
    setup_type = alert.get("setup_type", "")

    # Exit warnings always sent (score=0), entries need threshold
    if setup_type != "exit_warning" and score < score_threshold:
        return False

    if is_alert_duplicate(ticker, setup_type, cooldown_hours=cooldown_hours, db_url=db_url):
        logger.info("Skipping duplicate ETF alert: %s / %s", ticker, setup_type)
        return False

    message = build_etf_alert_message(ticker, alert, underlying, leverage)
    sent = send_telegram_message(token, chat_id, message)
    if sent:
        save_alert(
            ticker=ticker,
            date_val=date.today(),
            score=score,
            message=message,
            alert_type="leveraged_etf",
            setup_type=setup_type,
            db_url=db_url,
        )
    return sent


# ---------------------------------------------------------------------------
# Discovery scanner alert
# ---------------------------------------------------------------------------

def build_discovery_alert_message(discoveries: list) -> str:
    """Format a list of Discovery objects into a single Telegram message."""
    if not discoveries:
        return ""

    asset_emoji = {"stock": "📊", "crypto": "🪙"}
    lines = [
        "🔍 *Discovery Scanner – ungewöhnliche Aktivität*",
        "",
    ]

    for d in discoveries:
        emoji = asset_emoji.get(d.asset_class, "📊")
        reasons_str = " | ".join(d.reasons)
        lines.append(
            f"{emoji} *{d.ticker}*  ${d.close:,.2f}  →  {reasons_str}"
        )

    lines.append("")
    lines.append(f"_{len(discoveries)} Symbol(e) entdeckt_")
    return "\n".join(lines)


def send_discovery_alert(discoveries: list, token: str, chat_id: str, *, max_results: int = 15) -> bool:
    """Send a discovery alert via Telegram. Returns True if sent."""
    if not discoveries:
        return False
    capped = discoveries[:max_results]
    message = build_discovery_alert_message(capped)
    return send_telegram_message(token, chat_id, message)
