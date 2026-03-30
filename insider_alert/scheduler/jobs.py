"""APScheduler-based job runner for EOD and intraday analysis."""
import logging
from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

logger = logging.getLogger(__name__)


def run_trade_alerts_for_ticker(
    ticker: str,
    config,
    ohlcv,
    price_features: dict,
    volume_features: dict,
    options_features: dict,
    event_features: dict,
    news_features: dict,
) -> None:
    """Run all trade-alert detectors and send Telegram messages as appropriate."""
    from insider_alert.trade_alert_engine.breakout_alert import detect_breakout
    from insider_alert.trade_alert_engine.mean_reversion_alert import detect_mean_reversion
    from insider_alert.trade_alert_engine.options_flow_alert import detect_options_flow
    from insider_alert.trade_alert_engine.event_driven_alert import detect_event_driven
    from insider_alert.trade_alert_engine.multi_timeframe_alert import detect_multi_timeframe
    from insider_alert.alert_engine.telegram_alert import maybe_send_trade_alert

    ta_cfg = config.trade_alerts
    if not ta_cfg.get("enabled", True):
        return

    score_threshold = float(ta_cfg.get("score_threshold", 55))
    cooldown_hours = float(ta_cfg.get("cooldown_hours", 4))

    bo_cfg = ta_cfg.get("breakout", {})
    mr_cfg = ta_cfg.get("mean_reversion", {})
    of_cfg = ta_cfg.get("options_flow", {})
    ev_cfg = ta_cfg.get("event_driven", {})

    detectors = [
        lambda: detect_breakout(
            ohlcv, price_features, volume_features,
            breakout_window=int(bo_cfg.get("window", 20)),
            volume_confirmation=float(bo_cfg.get("volume_confirmation", 1.5)),
            impulse_factor=float(bo_cfg.get("impulse_factor", 1.0)),
            rr_ratio=float(bo_cfg.get("rr_ratio", 2.0)),
        ),
        lambda: detect_mean_reversion(
            price_features, volume_features, news_features,
            zscore_threshold=float(mr_cfg.get("zscore_threshold", 2.5)),
            rr_ratio=float(mr_cfg.get("rr_ratio", 1.5)),
        ),
        lambda: detect_options_flow(
            options_features, event_features,
            sweep_threshold=float(of_cfg.get("sweep_threshold", 0.6)),
            block_threshold=float(of_cfg.get("block_threshold", 0.5)),
            iv_jump_threshold=float(of_cfg.get("iv_jump_threshold", 0.20)),
            oi_change_threshold=float(of_cfg.get("oi_change_threshold", 0.30)),
            call_zscore_threshold=float(of_cfg.get("call_zscore_threshold", 2.0)),
        ),
        lambda: detect_event_driven(
            ticker, event_features, price_features, volume_features, options_features,
            pre_earnings_window=int(ev_cfg.get("pre_earnings_window", 10)),
            post_earnings_move=float(ev_cfg.get("post_earnings_move", 0.05)),
        ),
        lambda: detect_multi_timeframe(price_features, None),
    ]

    for detect_fn in detectors:
        try:
            alert = detect_fn()
            if alert is None:
                continue
            maybe_send_trade_alert(
                ticker,
                alert,
                config.telegram_token,
                config.telegram_chat_id,
                score_threshold=score_threshold,
                cooldown_hours=cooldown_hours,
            )
        except Exception as exc:
            logger.error("Trade alert detector failed for %s: %s", ticker, exc, exc_info=True)


def run_analysis_for_ticker(ticker: str, config) -> None:
    """Run full analysis pipeline for one ticker."""
    from insider_alert.data_ingestion.market_data import fetch_ohlcv_daily
    from insider_alert.data_ingestion.options_data import fetch_options_chain, fetch_historical_iv
    from insider_alert.data_ingestion.insider_data import fetch_insider_transactions
    from insider_alert.data_ingestion.event_data import days_to_next_earnings, fetch_recent_corporate_events
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
        iv_baseline = fetch_historical_iv(ticker)
        insider_txns = fetch_insider_transactions(ticker)
        dte = days_to_next_earnings(ticker)
        corporate_events = fetch_recent_corporate_events(ticker)
        news = fetch_news(ticker)

        current_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty and "close" in ohlcv.columns else 100.0
        price_f = compute_price_features(ohlcv)
        volume_f = compute_volume_features(ohlcv)
        orderflow_f = compute_orderflow_features(ohlcv)
        options_f = compute_options_features(options, current_price, iv_baseline=iv_baseline)
        insider_f = compute_insider_features(insider_txns)

        # Derive days-to-next-corporate-event from recent 8-K filings.
        # A recent 8-K with a known future date would be surfaced here;
        # if none found, pass None so event_features falls back to earnings only.
        days_to_corp_event: int | None = None
        if not corporate_events.empty and "date" in corporate_events.columns:
            import datetime as _dt
            today = _dt.date.today()
            for ev_date in corporate_events["date"]:
                try:
                    d = ev_date if isinstance(ev_date, _dt.date) else _dt.date.fromisoformat(str(ev_date))
                    delta = (d - today).days
                    if 0 <= delta <= 30:
                        if days_to_corp_event is None or delta < days_to_corp_event:
                            days_to_corp_event = delta
                except Exception:
                    continue

        event_f = compute_event_features(dte, price_f, volume_f, options_f, days_to_corp_event)
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

        # Run trade-alert detectors (Breakout, Mean-Reversion, Options, Event, MTF)
        run_trade_alerts_for_ticker(
            ticker, config, ohlcv, price_f, volume_f, options_f, event_f, news_f
        )

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
