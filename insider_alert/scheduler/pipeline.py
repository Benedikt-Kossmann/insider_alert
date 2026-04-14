"""Analysis pipeline runners – thin wrappers that orchestrate data→features→signals→score."""
import logging
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data ingestion helpers
# ---------------------------------------------------------------------------

def _fetch_stock_data(ticker: str) -> dict:
    """Fetch all raw data for a single stock ticker. Returns a dict of DataFrames/values."""
    from insider_alert.data_ingestion.market_data import fetch_ohlcv_daily
    from insider_alert.data_ingestion.options_data import fetch_options_chain, fetch_historical_iv
    from insider_alert.data_ingestion.insider_data import fetch_insider_transactions
    from insider_alert.data_ingestion.event_data import days_to_next_earnings, fetch_recent_corporate_events
    from insider_alert.data_ingestion.news_data import fetch_news
    from insider_alert.feature_engine.sector_features import get_sector_etf

    sector_etf = get_sector_etf(ticker)

    # Short volume and earnings data (Phase 7)
    try:
        from insider_alert.data_ingestion.short_volume_data import fetch_short_volume
        short_vol_df = fetch_short_volume(ticker, lookback_days=30)
    except Exception:  # noqa: BLE001
        import pandas as _pd
        short_vol_df = _pd.DataFrame(columns=["Date", "ShortVolume", "TotalVolume", "ShortRatio"])

    try:
        from insider_alert.data_ingestion.earnings_data import fetch_earnings_data
        earnings_raw = fetch_earnings_data(ticker)
    except Exception:
        earnings_raw = {}

    return {
        "ohlcv": fetch_ohlcv_daily(ticker),
        "options": fetch_options_chain(ticker),
        "iv_baseline": fetch_historical_iv(ticker),
        "insider_txns": fetch_insider_transactions(ticker),
        "dte": days_to_next_earnings(ticker),
        "corporate_events": fetch_recent_corporate_events(ticker),
        "news": fetch_news(ticker),
        "sector_ohlcv": fetch_ohlcv_daily(sector_etf),
        "sector_etf": sector_etf,
        "ticker": ticker,
        "short_vol_df": short_vol_df,
        "earnings_raw": earnings_raw,
    }


def _fetch_etf_data(ticker: str, underlying: str, vix_ticker: str) -> dict:
    """Fetch OHLCV for ETF, underlying, and VIX."""
    from insider_alert.data_ingestion.market_data import fetch_ohlcv_daily

    return {
        "etf_ohlcv": fetch_ohlcv_daily(ticker),
        "und_ohlcv": fetch_ohlcv_daily(underlying),
        "vix_ohlcv": fetch_ohlcv_daily(vix_ticker),
    }


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _compute_stock_features(data: dict, risk_free_rate: float = 0.05) -> dict:
    """Compute all stock features from raw data. Returns keyed dict."""
    from insider_alert.feature_engine.price_features import compute_price_features
    from insider_alert.feature_engine.volume_features import compute_volume_features
    from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
    from insider_alert.feature_engine.options_features import compute_options_features
    from insider_alert.feature_engine.insider_features import compute_insider_features
    from insider_alert.feature_engine.event_features import compute_event_features
    from insider_alert.feature_engine.news_features import compute_news_features
    from insider_alert.feature_engine.accumulation_features import compute_accumulation_features
    from insider_alert.feature_engine.candlestick_features import detect_candlestick_patterns
    from insider_alert.feature_engine.sr_features import compute_support_resistance
    from insider_alert.feature_engine.sector_features import compute_relative_strength, get_sector_label
    from insider_alert.feature_engine.volatility_forecast import compute_volatility_forecast
    from insider_alert.signal_engine.sector_rotation_signal import compute_sector_rotation_features

    ohlcv = data["ohlcv"]
    current_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty and "close" in ohlcv.columns else 100.0

    price_f = compute_price_features(ohlcv)
    volume_f = compute_volume_features(ohlcv)
    orderflow_f = compute_orderflow_features(ohlcv)

    # Merge candlestick pattern features into orderflow dict
    candle_patterns = detect_candlestick_patterns(ohlcv)
    orderflow_f.update(candle_patterns)
    options_f = compute_options_features(
        data["options"], current_price,
        iv_baseline=data["iv_baseline"],
        risk_free_rate=risk_free_rate,
    )
    insider_f = compute_insider_features(data["insider_txns"])

    # Nearest corporate event
    days_to_corp_event = _nearest_corp_event(data["corporate_events"])

    event_f = compute_event_features(data["dte"], price_f, volume_f, options_f, days_to_corp_event)
    news_f = compute_news_features(data["news"], price_f.get("return_1d", 0.0))
    accumulation_f = compute_accumulation_features(ohlcv)
    sr_f = compute_support_resistance(ohlcv)

    # Sector relative strength
    sector_ohlcv = data.get("sector_ohlcv", pd.DataFrame())
    sector_f = compute_relative_strength(ohlcv, sector_ohlcv)
    sector_f["sector_etf"] = data.get("sector_etf", "SPY")
    sector_f["sector_label"] = get_sector_label(data.get("sector_etf", "SPY"))

    ticker = data.get("ticker", "")
    vol_forecast_f = compute_volatility_forecast(ohlcv, ticker=ticker)

    sector_etf_ticker = data.get("sector_etf", "SPY")
    sector_rotation_f = compute_sector_rotation_features(sector_etf_ticker)

    # Short Volume features (Phase 7)
    from insider_alert.feature_engine.short_volume_features import compute_short_volume_features
    short_vol_f = compute_short_volume_features(data.get("short_vol_df"))

    # PEAD features (Phase 7)
    from insider_alert.signal_engine.earnings_drift_signal import compute_pead_features
    pead_f = compute_pead_features(data.get("earnings_raw", {}))

    return {
        "price": price_f,
        "volume": volume_f,
        "orderflow": orderflow_f,
        "options": options_f,
        "insider": insider_f,
        "event": event_f,
        "news": news_f,
        "accumulation": accumulation_f,
        "sr": sr_f,
        "sector": sector_f,
        "vol_forecast": vol_forecast_f,
        "sector_rotation": sector_rotation_f,
        "short_volume": short_vol_f,
        "pead": pead_f,
    }


def _nearest_corp_event(corporate_events) -> int | None:
    """Find nearest future corporate event within 30 days."""
    if corporate_events.empty or "date" not in corporate_events.columns:
        return None
    import datetime as _dt
    today = _dt.date.today()
    best = None
    for ev_date in corporate_events["date"]:
        try:
            d = ev_date if isinstance(ev_date, _dt.date) else _dt.date.fromisoformat(str(ev_date))
            delta = (d - today).days
            if 0 <= delta <= 30:
                if best is None or delta < best:
                    best = delta
        except Exception:
            continue
    return best


# ---------------------------------------------------------------------------
# Market context builder (shared across stock + ETF pipelines)
# ---------------------------------------------------------------------------

def build_market_context(config) -> dict:
    """Build a shared market-context dict with macro, news, and options data.

    Fetches macro data once, then for each unique ``news_proxy`` in the ETF
    universe fetches news and (where available) options data.  The result is
    a dict::

        {
            "macro": {…}  or None,
            "news":    {"SPY": {…}, "QQQ": {…}, …},
            "options": {"SPY": {…}, "QQQ": {…}, …},
        }
    """
    from insider_alert.data_ingestion.macro_data import fetch_macro_data
    from insider_alert.feature_engine.macro_features import compute_macro_features
    from insider_alert.data_ingestion.news_data import fetch_news
    from insider_alert.feature_engine.news_features import compute_news_features
    from insider_alert.data_ingestion.options_data import fetch_options_chain, fetch_historical_iv
    from insider_alert.feature_engine.options_features import compute_options_features
    from insider_alert.data_ingestion.market_data import fetch_ohlcv_daily

    ctx: dict = {"macro": None, "news": {}, "options": {}, "irx_rate": 0.05, "cross_asset": {}}

    # --- Macro (once) ---
    macro_cfg = getattr(config, "macro", None) or {}
    if macro_cfg.get("enabled", False):
        try:
            macro_data = fetch_macro_data(
                vix_ticker=macro_cfg.get("vix_ticker", ""),
                tnx_ticker=macro_cfg.get("tnx_ticker", ""),
                irx_ticker=macro_cfg.get("irx_ticker", ""),
                dxy_ticker=macro_cfg.get("dxy_ticker", ""),
            )
            ctx["macro"] = compute_macro_features(macro_data)
            mf = ctx["macro"]
            ctx["irx_rate"] = mf.get("irx_rate", 0.05)
            logger.info(
                "Macro regime: %s (VIX=%.1f, yield spread=%.2f%%, DXY %s)",
                mf["risk_regime"], mf["vix_current"],
                mf["yield_spread"], mf["dxy_trend"],
            )
            # --- Enrich with FRED macro data (Phase 6) ---
            try:
                from insider_alert.data_ingestion.fred_data import fetch_all_macro_data
                from insider_alert.feature_engine.macro_features import compute_fred_macro_features
                fred_data = fetch_all_macro_data()
                fred_features = compute_fred_macro_features(fred_data, mf)
                ctx["macro"].update(fred_features)
                logger.info(
                    "FRED macro: credit_stress=%.2f, fed=%s, inflation=%s",
                    fred_features.get("credit_stress_score", 0.0),
                    fred_features.get("fed_policy_direction", "?"),
                    fred_features.get("macro_regime", "?"),
                )
            except Exception as fred_exc:
                logger.warning("FRED macro enrichment failed: %s", fred_exc)
        except Exception as exc:
            logger.warning("Macro data fetch failed: %s", exc)

    # --- Cross-asset correlation (once per job run) ---
    try:
        from insider_alert.feature_engine.cross_asset_features import compute_cross_asset_features
        ctx["cross_asset"] = compute_cross_asset_features()
        regime = ctx["cross_asset"].get("equity_correlation_regime", "normal")
        anomaly = ctx["cross_asset"].get("correlation_anomaly_score", 0.0)
        logger.info("Cross-asset regime: %s (anomaly=%.2f)", regime, anomaly)
    except Exception as exc:
        logger.warning("Cross-asset features failed: %s", exc)

    # --- Collect unique proxy tickers ---
    le_cfg = getattr(config, "leveraged_etfs", None) or {}
    universe = le_cfg.get("universe", [])
    proxies: set[str] = set()
    for entry in universe:
        proxy = entry.get("news_proxy", "")
        if proxy:
            proxies.add(proxy)

    # --- News + options per proxy ---
    for proxy in proxies:
        # News
        try:
            news_df = fetch_news(proxy)
            ohlcv = fetch_ohlcv_daily(proxy, period="1mo")
            return_1d = 0.0
            if not ohlcv.empty and "close" in ohlcv.columns and len(ohlcv) >= 2:
                return_1d = float(ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[-2] - 1)
            ctx["news"][proxy] = compute_news_features(news_df, return_1d)
            logger.info("News features for proxy %s: sentiment=%.2f, count=%d",
                        proxy, ctx["news"][proxy].get("news_sentiment_score", 0),
                        ctx["news"][proxy].get("news_count_24h", 0))
        except Exception as exc:
            logger.warning("News fetch failed for proxy %s: %s", proxy, exc)

        # Options (may not be available for all proxies)
        try:
            opts_df = fetch_options_chain(proxy)
            if not opts_df.empty:
                iv_baseline = fetch_historical_iv(proxy)
                current_price = 0.0
                if not ohlcv.empty and "close" in ohlcv.columns:
                    current_price = float(ohlcv["close"].iloc[-1])
                ctx["options"][proxy] = compute_options_features(
                    opts_df, current_price, iv_baseline,
                    risk_free_rate=ctx.get("irx_rate", 0.05),
                )
                logger.info("Options features for proxy %s loaded", proxy)
        except Exception as exc:
            logger.warning("Options fetch failed for proxy %s: %s", proxy, exc)

    return ctx


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------

def _compute_stock_signals(features: dict, macro_features: dict | None = None, market_ctx: dict | None = None) -> list[dict]:
    """Run all stock signal generators (core + macro + sector_rotation if available)."""
    from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
    from insider_alert.signal_engine.volume_signal import compute_volume_anomaly_signal
    from insider_alert.signal_engine.orderflow_signal import compute_orderflow_anomaly_signal
    from insider_alert.signal_engine.options_signal import compute_options_anomaly_signal
    from insider_alert.signal_engine.insider_signal import compute_insider_signal
    from insider_alert.signal_engine.event_signal import compute_event_leadup_signal
    from insider_alert.signal_engine.news_signal import compute_news_divergence_signal
    from insider_alert.signal_engine.accumulation_signal import compute_accumulation_signal

    from insider_alert.signal_engine.volatility_forecast_signal import compute_volatility_forecast_signal
    from insider_alert.signal_engine.sector_rotation_signal import compute_sector_rotation_signal

    signals = [
        compute_price_anomaly_signal(features["price"]),
        compute_volume_anomaly_signal(features["volume"]),
        compute_orderflow_anomaly_signal(features["orderflow"]),
        compute_options_anomaly_signal(features["options"]),
        compute_insider_signal(features["insider"]),
        compute_event_leadup_signal(features["event"]),
        compute_news_divergence_signal(features["news"]),
        compute_accumulation_signal(features["accumulation"]),
        compute_volatility_forecast_signal(features.get("vol_forecast", {})),
        compute_sector_rotation_signal(features.get("sector_rotation", {})),
    ]

    if macro_features is not None:
        from insider_alert.signal_engine.macro_signal import macro_signal
        signals.append(macro_signal(macro_features))

    # Short Squeeze signal (Phase 7)
    from insider_alert.signal_engine.short_squeeze_signal import short_squeeze_signal
    signals.append(short_squeeze_signal(features.get("short_volume", {})))

    # Earnings Drift (PEAD) signal (Phase 7)
    from insider_alert.signal_engine.earnings_drift_signal import earnings_drift_signal
    signals.append(earnings_drift_signal(features.get("pead", {})))

    return signals


def _compute_etf_features_and_signals(
    data: dict, le_cfg: dict, leverage: int, direction: str,
    market_ctx: dict | None = None, news_proxy: str = "",
) -> dict:
    """Compute ETF features and signals. Returns dict with features, signals, and maps."""
    from insider_alert.feature_engine.price_features import compute_price_features
    from insider_alert.feature_engine.volume_features import compute_volume_features
    from insider_alert.feature_engine.momentum_features import compute_momentum_features
    from insider_alert.feature_engine.leverage_features import compute_leverage_features
    from insider_alert.feature_engine.volatility_regime_features import compute_volatility_regime_features
    from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
    from insider_alert.signal_engine.volume_signal import compute_volume_anomaly_signal
    from insider_alert.signal_engine.momentum_signal import compute_momentum_signal
    from insider_alert.signal_engine.mean_reversion_dip_signal import compute_mean_reversion_dip_signal
    from insider_alert.signal_engine.volatility_regime_signal import compute_volatility_regime_signal
    from insider_alert.signal_engine.leverage_signal import compute_leverage_health_signal

    etf_ohlcv = data["etf_ohlcv"]
    und_ohlcv = data["und_ohlcv"]
    vix_ohlcv = data["vix_ohlcv"]

    price_f = compute_price_features(etf_ohlcv)
    volume_f = compute_volume_features(etf_ohlcv)
    momentum_f = compute_momentum_features(etf_ohlcv, le_cfg.get("momentum", {}))
    leverage_f = compute_leverage_features(etf_ohlcv, und_ohlcv, leverage, direction)
    vol_regime_f = compute_volatility_regime_features(
        etf_ohlcv, vix_ohlcv,
        bollinger_period=int(le_cfg.get("mean_reversion", {}).get("bollinger_period", 20)),
        bollinger_std=float(le_cfg.get("mean_reversion", {}).get("bollinger_std", 2.0)),
        atr_regime_window=int(le_cfg.get("volatility", {}).get("atr_regime_window", 20)),
        vix_high=float(le_cfg.get("volatility", {}).get("vix_high", 30)),
        vix_low=float(le_cfg.get("volatility", {}).get("vix_low", 15)),
    )

    signals = [
        compute_momentum_signal(momentum_f, direction=direction),
        compute_mean_reversion_dip_signal(momentum_f, vol_regime_f, price_f, direction=direction),
        compute_volatility_regime_signal(vol_regime_f, leverage_f),
        compute_leverage_health_signal(leverage_f),
        compute_price_anomaly_signal(price_f),
        compute_volume_anomaly_signal(volume_f),
    ]

    # --- Market-context signals (macro, news, options) ---
    if market_ctx:
        macro_f = market_ctx.get("macro")
        if macro_f:
            from insider_alert.signal_engine.macro_signal import macro_signal
            signals.append(macro_signal(macro_f))

        news_f = market_ctx.get("news", {}).get(news_proxy) if news_proxy else None
        if news_f:
            from insider_alert.signal_engine.etf_news_signal import compute_etf_news_sentiment_signal
            signals.append(compute_etf_news_sentiment_signal(news_f, direction=direction))

        opts_f = market_ctx.get("options", {}).get(news_proxy) if news_proxy else None
        if opts_f:
            from insider_alert.signal_engine.options_signal import compute_options_anomaly_signal
            signals.append(compute_options_anomaly_signal(opts_f))

    return {
        "price_f": price_f,
        "momentum_f": momentum_f,
        "leverage_f": leverage_f,
        "vol_regime_f": vol_regime_f,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _persist_signals_and_score(ticker: str, signals: list[dict], ticker_score) -> None:
    """Save signals and score to DB."""
    from insider_alert.persistence.storage import save_signal, save_score

    today = date.today()
    for signal in signals:
        save_signal(
            ticker=ticker,
            date_val=today,
            signal_type=signal["signal_type"],
            score=signal["score"],
            flags=signal["flags"],
        )
    save_score(ticker, today, ticker_score)
