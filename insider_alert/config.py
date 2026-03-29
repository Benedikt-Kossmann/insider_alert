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

    return Config(
        tickers=tickers,
        alert_threshold=alert_threshold,
        weights=weights,
        feature_engine=feature_engine,
        scheduler=scheduler,
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    )


def get_config() -> "Config":
    """Return the singleton Config instance, loading it if necessary."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = load_config()
    return _config_singleton
