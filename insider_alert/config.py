"""Configuration loader for insider_alert."""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_config_singleton: Optional["Config"] = None

_DEFAULT_TRADE_ALERTS: dict = {
    "enabled": True,
    "score_threshold": 55,
    "cooldown_hours": 4,
    "breakout": {
        "window": 20,
        "volume_confirmation": 1.5,
        "impulse_factor": 1.0,
        "rr_ratio": 2.0,
    },
    "mean_reversion": {
        "zscore_threshold": 2.5,
        "rr_ratio": 1.5,
    },
    "options_flow": {
        "sweep_threshold": 0.6,
        "block_threshold": 0.5,
        "iv_jump_threshold": 0.20,
        "oi_change_threshold": 0.30,
        "call_zscore_threshold": 2.0,
    },
    "event_driven": {
        "pre_earnings_window": 10,
        "post_earnings_move": 0.05,
    },
    "universe_scan": {
        "rvol_add_threshold": 2.0,
        "rvol_remove_threshold": 0.8,
        "earnings_add_window": 14,
        "news_score_add": 0.5,
    },
}


@dataclass
class Config:
    tickers: list[str]
    alert_threshold: float
    weights: dict
    feature_engine: dict
    scheduler: dict
    telegram_token: str
    telegram_chat_id: str
    alpha_vantage_key: str
    trade_alerts: dict = field(default_factory=lambda: dict(_DEFAULT_TRADE_ALERTS))


def load_config(path: str = "config.yaml") -> "Config":
    """Load configuration from config.yaml and .env."""
    config_path = Path(path)
    env_path = config_path.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    load_dotenv(override=False)

    if not config_path.exists():
        logger.warning("config.yaml not found at %s, using defaults", path)
        raw = {}
    else:
        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

    tickers = raw.get("tickers", ["AAPL"])
    scoring = raw.get("scoring", {})
    alert_threshold = float(scoring.get("alert_threshold", 60))
    weights = scoring.get("weights", {
        "price_anomaly": 0.15,
        "volume_anomaly": 0.15,
        "orderflow_anomaly": 0.10,
        "options_anomaly": 0.20,
        "insider_signal": 0.20,
        "event_leadup": 0.10,
        "news_divergence": 0.05,
        "accumulation_pattern": 0.05,
    })
    feature_engine = raw.get("feature_engine", {
        "rolling_window": 20,
        "zscore_window": 20,
        "rvol_window": 20,
    })
    scheduler = raw.get("scheduler", {
        "eod_hour": 17,
        "eod_minute": 30,
        "intraday_interval_minutes": 30,
    })

    # Merge trade_alerts from config file on top of defaults
    trade_alerts = dict(_DEFAULT_TRADE_ALERTS)
    raw_ta = raw.get("trade_alerts", {})
    for key, value in raw_ta.items():
        if isinstance(value, dict) and isinstance(trade_alerts.get(key), dict):
            trade_alerts[key] = {**trade_alerts[key], **value}
        else:
            trade_alerts[key] = value

    return Config(
        tickers=tickers,
        alert_threshold=alert_threshold,
        weights=weights,
        feature_engine=feature_engine,
        scheduler=scheduler,
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
        trade_alerts=trade_alerts,
    )


def get_config() -> "Config":
    """Return the singleton Config instance, loading it if necessary."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = load_config()
    return _config_singleton
