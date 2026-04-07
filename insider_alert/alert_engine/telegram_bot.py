"""Telegram Bot command handler with long-polling.

Supports interactive commands:
    /status   – Show bot status and ticker count
    /scores   – Show latest composite scores for all tickers
    /score TICKER – Show detailed latest score for one ticker
    /backtest TICKER – Run and show minibacktest result
    /watchlist – Show current watchlist
    /add TICKER – Add a ticker to the watchlist
    /remove TICKER – Remove a ticker from the watchlist
    /help     – Show available commands

Uses the raw Telegram Bot API via requests (no extra dependencies).
Designed to run in a background thread alongside the scheduler.
"""
import logging
import threading
import time

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}"
_POLL_TIMEOUT = 30  # seconds for long polling
_running = False
_thread: threading.Thread | None = None


def start_bot(config) -> None:
    """Start the Telegram bot polling in a background thread."""
    global _running, _thread
    token = config.telegram_token
    chat_id = config.telegram_chat_id
    if not token or not chat_id:
        logger.info("Telegram not configured; bot commands disabled.")
        return

    if _running:
        logger.warning("Telegram bot already running.")
        return

    _running = True
    _thread = threading.Thread(
        target=_poll_loop,
        args=(config,),
        daemon=True,
        name="telegram-bot",
    )
    _thread.start()
    logger.info("Telegram bot polling started.")


def stop_bot() -> None:
    """Stop the polling loop."""
    global _running
    _running = False
    logger.info("Telegram bot polling stopped.")


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------

def _poll_loop(config) -> None:
    """Long-polling loop that processes incoming messages."""
    token = config.telegram_token
    offset = 0

    while _running:
        try:
            url = f"{_TELEGRAM_API.format(token=token)}/getUpdates"
            params = {"timeout": _POLL_TIMEOUT, "offset": offset}
            resp = requests.get(url, params=params, timeout=_POLL_TIMEOUT + 5)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("ok"):
                logger.warning("Telegram getUpdates not ok: %s", data)
                time.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                _process_update(update, config)

        except requests.exceptions.Timeout:
            continue
        except Exception as exc:
            logger.error("Telegram polling error: %s", exc)
            time.sleep(10)


def _process_update(update: dict, config) -> None:
    """Route an incoming Telegram update to the right handler."""
    message = update.get("message", {})
    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))

    # Only respond to the configured chat
    if chat_id != config.telegram_chat_id:
        return

    if not text.startswith("/"):
        return

    parts = text.split(maxsplit=1)
    command = parts[0].lower().split("@")[0]  # strip @botname
    args = parts[1].strip().upper() if len(parts) > 1 else ""

    handlers = {
        "/status": _cmd_status,
        "/scores": _cmd_scores,
        "/score": _cmd_score_detail,
        "/backtest": _cmd_backtest,
        "/watchlist": _cmd_watchlist,
        "/add": _cmd_add,
        "/remove": _cmd_remove,
        "/help": _cmd_help,
        "/start": _cmd_help,
    }

    handler = handlers.get(command)
    if handler:
        try:
            handler(args, config)
        except Exception as exc:
            logger.error("Command %s failed: %s", command, exc)
            _reply(config, f"❌ Fehler bei {command}: {exc}")
    else:
        _reply(config, f"❓ Unbekannter Befehl: `{command}`\nSiehe /help")


def _reply(config, text: str) -> None:
    """Send a reply message."""
    from insider_alert.alert_engine.telegram_alert import send_telegram_message
    send_telegram_message(config.telegram_token, config.telegram_chat_id, text)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_help(_args: str, config) -> None:
    lines = [
        "🤖 *Insider Alert Bot – Befehle*",
        "",
        "/status – Bot-Status & Ticker-Anzahl",
        "/scores – Letzte Composite-Scores aller Ticker",
        "/score TICKER – Detail-Score für einen Ticker",
        "/backtest TICKER – Mini-Backtest (1y)",
        "/watchlist – Aktuelle Watchlist anzeigen",
        "/add TICKER – Ticker zur Watchlist hinzufügen",
        "/remove TICKER – Ticker von Watchlist entfernen",
        "/help – Diese Hilfe anzeigen",
    ]
    _reply(config, "\n".join(lines))


def _cmd_status(_args: str, config) -> None:
    n_tickers = len(config.tickers)
    le_cfg = config.leveraged_etfs
    n_etfs = len(le_cfg.get("universe", [])) if le_cfg.get("enabled") else 0

    from insider_alert.scoring_engine.ml_scorer import is_available as ml_available

    lines = [
        "📊 *Bot Status*",
        "",
        f"Aktien-Watchlist: {n_tickers} Ticker",
    ]
    if n_etfs:
        lines.append(f"Leveraged ETFs: {n_etfs}")
    lines.append(f"Alert-Schwelle: {config.alert_threshold}")
    lines.append(f"ML-Scoring: {'✅ aktiv' if ml_available() else '❌ inaktiv'}")

    eod_h = config.scheduler.get("eod_hour", 17)
    eod_m = config.scheduler.get("eod_minute", 30)
    intra = config.scheduler.get("intraday_interval_minutes", 30)
    lines.append(f"EOD: {eod_h:02d}:{eod_m:02d} UTC | Intraday: alle {intra} Min")

    _reply(config, "\n".join(lines))


def _cmd_scores(_args: str, config) -> None:
    from insider_alert.persistence.storage import get_recent_scores

    lines = ["📈 *Letzte Scores*", ""]
    found = False

    for ticker in config.tickers:
        scores = get_recent_scores(ticker, days=1)
        if scores:
            s = scores[0]
            emoji = "🟢" if s["total_score"] >= 60 else ("🟡" if s["total_score"] >= 40 else "⚪")
            lines.append(f"{emoji} *{ticker}*: {s['total_score']:.1f}")
            found = True

    if not found:
        lines.append("_Keine aktuellen Scores vorhanden._")
    _reply(config, "\n".join(lines))


def _cmd_score_detail(ticker: str, config) -> None:
    if not ticker:
        _reply(config, "Bitte Ticker angeben: `/score AAPL`")
        return

    from insider_alert.persistence.storage import get_recent_scores

    scores = get_recent_scores(ticker, days=7)
    if not scores:
        _reply(config, f"Keine Scores für *{ticker}* gefunden.")
        return

    latest = scores[0]
    lines = [
        f"📊 *Score-Detail: {ticker}*",
        f"Datum: {latest['date']}",
        f"Composite: *{latest['total_score']:.1f}/100*",
        "",
        "*Sub-Scores:*",
    ]
    for sig, val in latest.get("sub_scores", {}).items():
        lines.append(f"  • {sig}: {val:.1f}")

    flags = latest.get("flags", [])
    if flags:
        lines.append("")
        lines.append("*Flags:*")
        for f in flags[:8]:
            lines.append(f"  ⚠️ {f}")

    # Trend over last 7 days
    if len(scores) > 1:
        trend_scores = [s["total_score"] for s in scores]
        delta = trend_scores[0] - trend_scores[-1]
        arrow = "📈" if delta > 0 else ("📉" if delta < 0 else "➡️")
        lines.append("")
        lines.append(f"7d Trend: {arrow} {delta:+.1f}")

    _reply(config, "\n".join(lines))


def _cmd_backtest(ticker: str, config) -> None:
    if not ticker:
        _reply(config, "Bitte Ticker angeben: `/backtest AAPL`")
        return

    _reply(config, f"⏳ Backtest für *{ticker}* läuft...")

    try:
        from insider_alert.backtest.engine import run_backtest
        from insider_alert.backtest.metrics import compute_composite_metrics

        results = run_backtest([ticker], period="1y")
        if not results:
            _reply(config, f"❌ Kein Backtest-Ergebnis für *{ticker}*.")
            return

        bt = results[0]
        if bt.error:
            _reply(config, f"❌ Backtest-Fehler: {bt.error}")
            return

        m = compute_composite_metrics(bt.rows, threshold=50.0)
        lines = [
            f"📊 *Backtest: {ticker}* (1 Jahr)",
            "",
            f"Tage getestet: {m.total_days}",
            f"Signal-Tage (>50): {m.high_signal_days}",
            "",
            f"Hit-Rate 1d: {m.hit_rate_1d * 100:.1f}%",
            f"Hit-Rate 5d: {m.hit_rate_5d * 100:.1f}%",
            f"Hit-Rate 10d: {m.hit_rate_10d * 100:.1f}%",
            "",
            f"Avg Return 5d: {m.avg_return_5d * 100:.2f}%",
            f"Edge 5d: {m.edge_5d * 100:.2f}%",
        ]
        _reply(config, "\n".join(lines))

    except Exception as exc:
        _reply(config, f"❌ Backtest fehlgeschlagen: {exc}")


def _cmd_watchlist(_args: str, config) -> None:
    tickers = config.tickers
    if not tickers:
        _reply(config, "📋 Watchlist ist leer.")
        return

    lines = [f"📋 *Watchlist* ({len(tickers)} Ticker)", ""]
    for t in sorted(tickers):
        lines.append(f"  • {t}")
    _reply(config, "\n".join(lines))


def _cmd_add(ticker: str, config) -> None:
    if not ticker:
        _reply(config, "Bitte Ticker angeben: `/add NVDA`")
        return

    ticker = ticker.upper().strip()
    if ticker in config.tickers:
        _reply(config, f"*{ticker}* ist bereits in der Watchlist.")
        return

    config.tickers.append(ticker)
    _save_watchlist(config)
    _reply(config, f"✅ *{ticker}* zur Watchlist hinzugefügt ({len(config.tickers)} Ticker).")


def _cmd_remove(ticker: str, config) -> None:
    if not ticker:
        _reply(config, "Bitte Ticker angeben: `/remove TSLA`")
        return

    ticker = ticker.upper().strip()
    if ticker not in config.tickers:
        _reply(config, f"*{ticker}* ist nicht in der Watchlist.")
        return

    config.tickers.remove(ticker)
    _save_watchlist(config)
    _reply(config, f"🗑️ *{ticker}* von Watchlist entfernt ({len(config.tickers)} Ticker).")


def _save_watchlist(config) -> None:
    """Persist the updated watchlist back to config.yaml."""
    try:
        from pathlib import Path
        import yaml

        config_path = Path("config.yaml")
        if not config_path.exists():
            raw = {}
        else:
            with open(config_path, "r") as fh:
                raw = yaml.safe_load(fh) or {}

        raw["tickers"] = config.tickers

        with open(config_path, "w") as fh:
            yaml.dump(raw, fh, default_flow_style=False, allow_unicode=True)

        logger.info("Watchlist saved to config.yaml: %s", config.tickers)
    except Exception as exc:
        logger.error("Failed to save watchlist: %s", exc)
