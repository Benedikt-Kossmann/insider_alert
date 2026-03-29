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
