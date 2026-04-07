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

    mtf_cfg = ta_cfg.get("multi_timeframe", {})

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
        lambda: detect_multi_timeframe(
            price_features, None,
            score_threshold=float(mtf_cfg.get("score_threshold", 55)),
            confirmation_bars=int(mtf_cfg.get("confirmation_bars", 3)),
        ),
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


def run_analysis_for_ticker(ticker: str, config, macro_features: dict | None = None) -> None:
    """Run full analysis pipeline for one ticker."""
    from insider_alert.scheduler.pipeline import (
        _fetch_stock_data, _compute_stock_features,
        _compute_stock_signals, _persist_signals_and_score,
    )
    from insider_alert.scoring_engine.scorer import compute_score
    from insider_alert.alert_engine.telegram_alert import maybe_send_alert, build_alert_message
    from insider_alert.persistence.storage import save_alert, save_signal_outcomes

    logger.info("Running analysis for %s", ticker)

    try:
        data = _fetch_stock_data(ticker)
        features = _compute_stock_features(data)
        signals = _compute_stock_signals(features, macro_features)
        ticker_score = compute_score(ticker, signals, config.weights)

        # ML scoring overlay
        ml_score = _get_ml_score(signals)

        _persist_signals_and_score(ticker, signals, ticker_score)
        save_signal_outcomes(ticker, date.today(), signals, ticker_score.total_score)

        sr_features = features.get("sr")
        sector_features = features.get("sector")
        sent = maybe_send_alert(ticker_score, config.telegram_token, config.telegram_chat_id, config.alert_threshold, sr_features=sr_features, sector_features=sector_features, ml_score=ml_score)
        if sent:
            message = build_alert_message(ticker_score, sr_features, sector_features, ml_score=ml_score)
            save_alert(ticker, date.today(), ticker_score.total_score, message)

        # Run trade-alert detectors
        run_trade_alerts_for_ticker(
            ticker, config, data["ohlcv"],
            features["price"], features["volume"],
            features["options"], features["event"], features["news"],
        )

        logger.info(
            "Analysis complete for %s: score=%.1f, alert_sent=%s",
            ticker, ticker_score.total_score, sent
        )

    except Exception as exc:
        logger.error("Analysis failed for %s: %s", ticker, exc, exc_info=True)


def run_etf_analysis_for_ticker(etf_entry: dict, config) -> None:
    """Run leveraged-ETF analysis pipeline for one ETF."""
    from insider_alert.scheduler.pipeline import (
        _fetch_etf_data, _compute_etf_features_and_signals,
        _persist_signals_and_score,
    )
    from insider_alert.scoring_engine.scorer import compute_score
    from insider_alert.trade_alert_engine.leveraged_etf_alert import (
        detect_leveraged_etf_entry, detect_leveraged_etf_exit,
    )
    from insider_alert.trade_alert_engine.risk_manager import (
        compute_leverage_risk_hints, format_leverage_risk_lines,
    )
    from insider_alert.alert_engine.telegram_alert import maybe_send_etf_alert

    ticker = etf_entry.get("ticker", "")
    underlying = etf_entry.get("underlying", "")
    direction = etf_entry.get("direction", "long")
    leverage = int(etf_entry.get("leverage", 3))
    le_cfg = config.leveraged_etfs

    logger.info("Running ETF analysis for %s (underlying=%s, %dx %s)", ticker, underlying, leverage, direction)

    try:
        vix_ticker = le_cfg.get("volatility", {}).get("vix_ticker", "^VIX")
        data = _fetch_etf_data(ticker, underlying, vix_ticker)

        if data["etf_ohlcv"].empty:
            logger.warning("No OHLCV data for ETF %s, skipping", ticker)
            return

        result = _compute_etf_features_and_signals(data, le_cfg, leverage, direction)
        signals_list = result["signals"]

        # Scoring
        scoring_cfg = le_cfg.get("scoring", {})
        etf_weights = scoring_cfg.get("weights", None)
        etf_threshold = float(scoring_cfg.get("alert_threshold", 55))
        ticker_score = compute_score(ticker, signals_list, etf_weights)

        _persist_signals_and_score(ticker, signals_list, ticker_score)

        # Risk hints
        price_f = result["price_f"]
        atr = float(price_f.get("atr_14", 0))
        current_price = float(data["etf_ohlcv"]["close"].iloc[-1]) if "close" in data["etf_ohlcv"].columns else 0.0
        risk_cfg = le_cfg.get("risk", {})
        risk_hints = compute_leverage_risk_hints(
            current_price, atr, direction, leverage,
            vol_regime=result["vol_regime_f"].get("vix_regime", "normal"),
            estimated_decay=result["leverage_f"].get("estimated_daily_decay", 0),
            risk_cfg=risk_cfg,
        )
        risk_lines = format_leverage_risk_lines(risk_hints)

        # Signal dict for alert detectors
        signals_map = {s["signal_type"]: s for s in signals_list}

        # Entry detection
        entry_cfg = le_cfg.get("entry", {})
        entry = detect_leveraged_etf_entry(
            ticker, signals_map, result["momentum_f"], result["vol_regime_f"], result["leverage_f"],
            risk_cfg=risk_cfg, entry_cfg=entry_cfg,
            direction=direction, underlying=underlying, leverage=leverage,
        )
        if entry:
            entry["risk_lines"] = risk_lines
            cooldown = float(risk_cfg.get("cooldown_hours", 4.0))
            maybe_send_etf_alert(
                ticker, entry, config.telegram_token, config.telegram_chat_id,
                score_threshold=etf_threshold, cooldown_hours=cooldown,
                underlying=underlying, leverage=leverage,
            )

        # Exit detection
        exit_alert = detect_leveraged_etf_exit(
            ticker, result["leverage_f"], result["vol_regime_f"],
            risk_cfg=risk_cfg, direction=direction, underlying=underlying, leverage=leverage,
        )
        if exit_alert:
            exit_alert["risk_lines"] = risk_lines
            maybe_send_etf_alert(
                ticker, exit_alert, config.telegram_token, config.telegram_chat_id,
                score_threshold=0, cooldown_hours=cooldown,
                underlying=underlying, leverage=leverage,
            )

        logger.info("ETF analysis complete for %s: score=%.1f", ticker, ticker_score.total_score)

    except Exception as exc:
        logger.error("ETF analysis failed for %s: %s", ticker, exc, exc_info=True)


def run_discovery_scan_job(config) -> None:
    """Run the discovery scanner and send Telegram alert for findings."""
    disc_cfg = getattr(config, "discovery", {}) or {}
    if not disc_cfg.get("enabled", False):
        return

    from insider_alert.trade_alert_engine.discovery_scanner import run_discovery_scan
    from insider_alert.alert_engine.telegram_alert import send_discovery_alert

    try:
        discoveries = run_discovery_scan(config)
        if discoveries:
            max_results = int(disc_cfg.get("max_results", 15))
            send_discovery_alert(
                discoveries,
                config.telegram_token,
                config.telegram_chat_id,
                max_results=max_results,
            )
    except Exception as exc:
        logger.error("Discovery scan failed: %s", exc, exc_info=True)


def _fetch_macro_features(config) -> dict | None:
    """Fetch macro data and compute features. Returns None on failure."""
    macro_cfg = getattr(config, "macro", None) or {}
    if not macro_cfg.get("enabled", False):
        return None
    try:
        from insider_alert.data_ingestion.macro_data import fetch_macro_data
        from insider_alert.feature_engine.macro_features import compute_macro_features

        macro_data = fetch_macro_data(
            vix_ticker=macro_cfg.get("vix_ticker", ""),
            tnx_ticker=macro_cfg.get("tnx_ticker", ""),
            irx_ticker=macro_cfg.get("irx_ticker", ""),
            dxy_ticker=macro_cfg.get("dxy_ticker", ""),
        )
        features = compute_macro_features(macro_data)
        logger.info(
            "Macro regime: %s (VIX=%.1f, yield spread=%.2f%%, DXY %s)",
            features["risk_regime"], features["vix_current"],
            features["yield_spread"], features["dxy_trend"],
        )
        return features
    except Exception as exc:
        logger.warning("Macro data fetch failed: %s", exc)
        return None


def _run_outcome_backfill() -> None:
    """Backfill forward returns for past signal outcomes."""
    try:
        from insider_alert.persistence.storage import backfill_signal_outcomes
        count = backfill_signal_outcomes()
        if count:
            logger.info("Backfilled %d signal outcomes", count)
    except Exception as exc:
        logger.warning("Outcome backfill failed: %s", exc)


def _try_adaptive_weights(config) -> None:
    """Adjust scoring weights based on outcome data if enough samples exist."""
    try:
        from insider_alert.scoring_engine.adaptive_weights import compute_adaptive_weights
        new_weights = compute_adaptive_weights(config.weights)
        if new_weights != config.weights:
            config.weights = new_weights
            logger.info("Adaptive weights applied for this EOD run.")
    except Exception as exc:
        logger.warning("Adaptive weight adjustment failed: %s", exc)


def _get_ml_score(signals: list[dict]) -> float | None:
    """Get ML model prediction for a set of signals. Returns None if unavailable."""
    try:
        from insider_alert.scoring_engine.ml_scorer import predict_score
        return predict_score(signals)
    except Exception:
        return None


def _try_ml_retrain() -> None:
    """Retrain the ML model if due."""
    try:
        from insider_alert.scoring_engine.ml_scorer import maybe_retrain
        maybe_retrain()
    except Exception as exc:
        logger.warning("ML model retrain failed: %s", exc)


def _run_weekly_report(config) -> None:
    """Send the weekly performance report via Telegram."""
    try:
        from insider_alert.alert_engine.weekly_report import send_weekly_report
        send_weekly_report(config)
    except Exception as exc:
        logger.error("Weekly report failed: %s", exc)


def run_eod_job(config) -> None:
    """End-of-day batch: analyze all tickers in config."""
    # Fetch macro once for all tickers
    macro_features = _fetch_macro_features(config)

    # Backfill past signal outcomes
    _run_outcome_backfill()

    # Try adaptive weight adjustment
    _try_adaptive_weights(config)

    # Retrain ML model if due
    _try_ml_retrain()

    logger.info("Running EOD job for %d tickers", len(config.tickers))
    for ticker in config.tickers:
        run_analysis_for_ticker(ticker, config, macro_features)

    # Leveraged-ETF analysis
    le_cfg = config.leveraged_etfs
    if le_cfg.get("enabled", False):
        universe = le_cfg.get("universe", [])
        logger.info("Running EOD leveraged-ETF job for %d ETFs", len(universe))
        for etf_entry in universe:
            run_etf_analysis_for_ticker(etf_entry, config)

    # Discovery scanner
    run_discovery_scan_job(config)


def run_intraday_job(config) -> None:
    """Intraday job: quick scan."""
    macro_features = _fetch_macro_features(config)

    logger.info("Running intraday job for %d tickers", len(config.tickers))
    for ticker in config.tickers:
        run_analysis_for_ticker(ticker, config, macro_features)

    # Leveraged-ETF analysis
    le_cfg = config.leveraged_etfs
    if le_cfg.get("enabled", False):
        universe = le_cfg.get("universe", [])
        logger.info("Running intraday leveraged-ETF job for %d ETFs", len(universe))
        for etf_entry in universe:
            run_etf_analysis_for_ticker(etf_entry, config)


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

    scheduler.add_job(
        _run_weekly_report,
        trigger="cron",
        day_of_week="sun",
        hour=18,
        minute=0,
        args=[config],
        id="weekly_report",
        name="Weekly Performance Report",
    )

    logger.info(
        "Scheduler starting: EOD at %02d:%02d, intraday every %dm",
        eod_hour, eod_minute, intraday_interval
    )

    # Start Telegram bot command listener
    try:
        from insider_alert.alert_engine.telegram_bot import start_bot
        start_bot(config)
    except Exception as exc:
        logger.warning("Telegram bot commands not started: %s", exc)

    scheduler.start()
