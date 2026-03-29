"""APScheduler-based job runner for EOD and intraday analysis."""
import logging
from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

logger = logging.getLogger(__name__)


def run_analysis_for_ticker(ticker: str, config) -> None:
    """Run full analysis pipeline for one ticker."""
    from insider_alert.data_ingestion.market_data import fetch_ohlcv_daily
    from insider_alert.data_ingestion.options_data import fetch_options_chain
    from insider_alert.data_ingestion.insider_data import fetch_insider_transactions
    from insider_alert.data_ingestion.event_data import days_to_next_earnings
    from insider_alert.data_ingestion.news_data import fetch_news
    from insider_alert.feature_engine.price_features import compute_price_features
    from insider_alert.feature_engine.volume_features import compute_volume_features
    from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
    from insider_alert.feature_engine.options_features import compute_options_features
    from insider_alert.feature_engine.insider_features import compute_insider_features
    from insider_alert.feature_engine.event_features import compute_event_features
    from insider_alert.feature_engine.news_features import compute_news_features
    from insider_alert.feature_engine.accumulation_features import compute_accumulation_features
    from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
    from insider_alert.signal_engine.volume_signal import compute_volume_anomaly_signal
    from insider_alert.signal_engine.orderflow_signal import compute_orderflow_anomaly_signal
    from insider_alert.signal_engine.options_signal import compute_options_anomaly_signal
    from insider_alert.signal_engine.insider_signal import compute_insider_signal
    from insider_alert.signal_engine.event_signal import compute_event_leadup_signal
    from insider_alert.signal_engine.news_signal import compute_news_divergence_signal
    from insider_alert.signal_engine.accumulation_signal import compute_accumulation_signal
    from insider_alert.scoring_engine.scorer import compute_score
    from insider_alert.alert_engine.telegram_alert import maybe_send_alert, build_alert_message
    from insider_alert.persistence.storage import save_signal, save_score, save_alert, init_db

    logger.info("Running analysis for %s", ticker)

    try:
        ohlcv = fetch_ohlcv_daily(ticker)
        options = fetch_options_chain(ticker)
        insider_txns = fetch_insider_transactions(ticker)
        dte = days_to_next_earnings(ticker)
        news = fetch_news(ticker)

        current_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty and "close" in ohlcv.columns else 100.0
        price_f = compute_price_features(ohlcv)
        volume_f = compute_volume_features(ohlcv)
        orderflow_f = compute_orderflow_features(ohlcv)
        options_f = compute_options_features(options, current_price)
        insider_f = compute_insider_features(insider_txns)
        event_f = compute_event_features(dte, price_f, volume_f, options_f)
        news_f = compute_news_features(news, price_f.get("return_1d", 0.0))
        accumulation_f = compute_accumulation_features(ohlcv)

        signals = [
            compute_price_anomaly_signal(price_f),
            compute_volume_anomaly_signal(volume_f),
            compute_orderflow_anomaly_signal(orderflow_f),
            compute_options_anomaly_signal(options_f),
            compute_insider_signal(insider_f),
            compute_event_leadup_signal(event_f),
            compute_news_divergence_signal(news_f),
            compute_accumulation_signal(accumulation_f),
        ]

        ticker_score = compute_score(ticker, signals, config.weights)

        today = date.today()
        init_db()

        for signal in signals:
            save_signal(
                ticker=ticker,
                date_val=today,
                signal_type=signal["signal_type"],
                score=signal["score"],
                flags=signal["flags"],
            )
        save_score(ticker, today, ticker_score)

        sent = maybe_send_alert(ticker_score, config.telegram_token, config.telegram_chat_id, config.alert_threshold)
        if sent:
            message = build_alert_message(ticker_score)
            save_alert(ticker, today, ticker_score.total_score, message)

        logger.info(
            "Analysis complete for %s: score=%.1f, alert_sent=%s",
            ticker, ticker_score.total_score, sent
        )

    except Exception as exc:
        logger.error("Analysis failed for %s: %s", ticker, exc, exc_info=True)


def run_eod_job(config) -> None:
    """End-of-day batch: analyze all tickers in config."""
    logger.info("Running EOD job for %d tickers", len(config.tickers))
    for ticker in config.tickers:
        run_analysis_for_ticker(ticker, config)


def run_intraday_job(config) -> None:
    """Intraday job: quick scan."""
    logger.info("Running intraday job for %d tickers", len(config.tickers))
    for ticker in config.tickers:
        run_analysis_for_ticker(ticker, config)


def start_scheduler(config, blocking: bool = True) -> None:
    """Configure and start APScheduler."""
    eod_hour = config.scheduler.get("eod_hour", 17)
    eod_minute = config.scheduler.get("eod_minute", 30)
    intraday_interval = config.scheduler.get("intraday_interval_minutes", 30)

    if blocking:
        scheduler = BlockingScheduler()
    else:
        scheduler = BackgroundScheduler()

    scheduler.add_job(
        run_eod_job,
        trigger="cron",
        hour=eod_hour,
        minute=eod_minute,
        args=[config],
        id="eod_job",
        name="EOD Analysis",
    )
    scheduler.add_job(
        run_intraday_job,
        trigger="interval",
        minutes=intraday_interval,
        args=[config],
        id="intraday_job",
        name="Intraday Scan",
    )

    logger.info(
        "Scheduler starting: EOD at %02d:%02d, intraday every %dm",
        eod_hour, eod_minute, intraday_interval
    )
    scheduler.start()
