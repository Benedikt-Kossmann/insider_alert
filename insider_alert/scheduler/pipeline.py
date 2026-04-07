"""Analysis pipeline runners – thin wrappers that orchestrate data→features→signals→score."""
import logging
from datetime import date

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

    return {
        "ohlcv": fetch_ohlcv_daily(ticker),
        "options": fetch_options_chain(ticker),
        "iv_baseline": fetch_historical_iv(ticker),
        "insider_txns": fetch_insider_transactions(ticker),
        "dte": days_to_next_earnings(ticker),
        "corporate_events": fetch_recent_corporate_events(ticker),
        "news": fetch_news(ticker),
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

def _compute_stock_features(data: dict) -> dict:
    """Compute all stock features from raw data. Returns keyed dict."""
    from insider_alert.feature_engine.price_features import compute_price_features
    from insider_alert.feature_engine.volume_features import compute_volume_features
    from insider_alert.feature_engine.orderflow_features import compute_orderflow_features
    from insider_alert.feature_engine.options_features import compute_options_features
    from insider_alert.feature_engine.insider_features import compute_insider_features
    from insider_alert.feature_engine.event_features import compute_event_features
    from insider_alert.feature_engine.news_features import compute_news_features
    from insider_alert.feature_engine.accumulation_features import compute_accumulation_features

    ohlcv = data["ohlcv"]
    current_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty and "close" in ohlcv.columns else 100.0

    price_f = compute_price_features(ohlcv)
    volume_f = compute_volume_features(ohlcv)
    orderflow_f = compute_orderflow_features(ohlcv)
    options_f = compute_options_features(data["options"], current_price, iv_baseline=data["iv_baseline"])
    insider_f = compute_insider_features(data["insider_txns"])

    # Nearest corporate event
    days_to_corp_event = _nearest_corp_event(data["corporate_events"])

    event_f = compute_event_features(data["dte"], price_f, volume_f, options_f, days_to_corp_event)
    news_f = compute_news_features(data["news"], price_f.get("return_1d", 0.0))
    accumulation_f = compute_accumulation_features(ohlcv)

    return {
        "price": price_f,
        "volume": volume_f,
        "orderflow": orderflow_f,
        "options": options_f,
        "insider": insider_f,
        "event": event_f,
        "news": news_f,
        "accumulation": accumulation_f,
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
# Signal computation helpers
# ---------------------------------------------------------------------------

def _compute_stock_signals(features: dict, macro_features: dict | None = None) -> list[dict]:
    """Run all stock signal generators (8 core + macro if available)."""
    from insider_alert.signal_engine.price_signal import compute_price_anomaly_signal
    from insider_alert.signal_engine.volume_signal import compute_volume_anomaly_signal
    from insider_alert.signal_engine.orderflow_signal import compute_orderflow_anomaly_signal
    from insider_alert.signal_engine.options_signal import compute_options_anomaly_signal
    from insider_alert.signal_engine.insider_signal import compute_insider_signal
    from insider_alert.signal_engine.event_signal import compute_event_leadup_signal
    from insider_alert.signal_engine.news_signal import compute_news_divergence_signal
    from insider_alert.signal_engine.accumulation_signal import compute_accumulation_signal

    signals = [
        compute_price_anomaly_signal(features["price"]),
        compute_volume_anomaly_signal(features["volume"]),
        compute_orderflow_anomaly_signal(features["orderflow"]),
        compute_options_anomaly_signal(features["options"]),
        compute_insider_signal(features["insider"]),
        compute_event_leadup_signal(features["event"]),
        compute_news_divergence_signal(features["news"]),
        compute_accumulation_signal(features["accumulation"]),
    ]

    if macro_features is not None:
        from insider_alert.signal_engine.macro_signal import compute_macro_regime_signal
        signals.append(compute_macro_regime_signal(macro_features))

    return signals


def _compute_etf_features_and_signals(data: dict, le_cfg: dict, leverage: int, direction: str) -> dict:
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
