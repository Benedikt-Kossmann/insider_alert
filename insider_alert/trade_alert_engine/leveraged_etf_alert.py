"""Leveraged-ETF entry/exit alert detectors."""
import logging

logger = logging.getLogger(__name__)


def detect_leveraged_etf_entry(
    ticker: str,
    signals: dict,
    momentum_features: dict,
    vol_regime_features: dict,
    leverage_features: dict,
    risk_cfg: dict | None = None,
    entry_cfg: dict | None = None,
    *,
    direction: str = "long",
    underlying: str = "",
    leverage: int = 3,
) -> dict | None:
    """Detect a leveraged-ETF entry opportunity.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol.
    signals : dict
        Mapping of signal_type → signal dict (must include ``score``).
    momentum_features : dict
        Output of ``compute_momentum_features()``.
    vol_regime_features : dict
        Output of ``compute_volatility_regime_features()``.
    leverage_features : dict
        Output of ``compute_leverage_features()``.
    risk_cfg : dict | None
        Risk config dict (``config.leveraged_etfs['risk']``).
    entry_cfg : dict | None
        Entry config dict (``config.leveraged_etfs['entry']``).
    direction : str
        ``'long'`` or ``'short'``.
    underlying : str
        Underlying ticker symbol (for display).
    leverage : int
        Leverage factor.

    Returns
    -------
    dict | None
        Alert dict or ``None`` if no entry signal detected.
    """
    entry_cfg = entry_cfg or {}
    momentum_min = float(entry_cfg.get("momentum_min_score", 60))
    health_min = float(entry_cfg.get("health_min_score", 50))
    dip_min = float(entry_cfg.get("dip_min_score", 70))
    vix_max = float(entry_cfg.get("vix_max", 30))
    momentum_score = float(signals.get("momentum", {}).get("score", 0))
    dip_score = float(signals.get("mean_reversion_dip", {}).get("score", 0))
    vol_score = float(signals.get("volatility_regime", {}).get("score", 0))
    health_score = float(signals.get("leverage_health", {}).get("score", 0))

    vix_regime = vol_regime_features.get("vix_regime", "normal")
    vix_current = float(vol_regime_features.get("vix_current", 20))

    flags: list[str] = []

    # --- Momentum entry ---
    if momentum_score >= momentum_min and vix_regime != "high" and health_score >= health_min:
        flags.append(f"Momentum score: {momentum_score:.0f}/100")
        flags.append(f"Leverage health: {health_score:.0f}/100")
        flags.append(f"Volatility regime: {vix_regime} (VIX={vix_current:.1f})")

        und_ret = leverage_features.get("underlying_return_5d", 0)
        if und_ret:
            flags.append(f"Underlying 5d return: {und_ret * 100:.1f}%")

        combined = momentum_score * 0.5 + vol_score * 0.25 + health_score * 0.25
        return {
            "alert_type": "leveraged_etf",
            "setup_type": "momentum_entry",
            "direction": direction,
            "score": combined,
            "underlying": underlying,
            "leverage": leverage,
            "flags": flags,
        }

    # --- Dip-buy entry ---
    if dip_score >= dip_min and vix_current < vix_max:
        flags.append(f"Dip-buy score: {dip_score:.0f}/100")
        flags.append(f"VIX: {vix_current:.1f}")
        flags.append(f"Leverage health: {health_score:.0f}/100")

        rsi = float(momentum_features.get("rsi_14", 50))
        flags.append(f"RSI: {rsi:.0f}")

        combined = dip_score * 0.6 + vol_score * 0.2 + health_score * 0.2
        return {
            "alert_type": "leveraged_etf",
            "setup_type": "dip_buy",
            "direction": direction,
            "score": combined,
            "underlying": underlying,
            "leverage": leverage,
            "flags": flags,
        }

    return None


def detect_leveraged_etf_exit(
    ticker: str,
    leverage_features: dict,
    vol_regime_features: dict,
    risk_cfg: dict | None = None,
    *,
    direction: str = "long",
    underlying: str = "",
    leverage: int = 3,
) -> dict | None:
    """Detect a leveraged-ETF exit / risk warning.

    Returns
    -------
    dict | None
        Alert dict or ``None`` if no exit signal detected.
    """
    risk_cfg = risk_cfg or {}
    decay_threshold = float(risk_cfg.get("decay_warning_threshold", 0.02))

    decay = float(leverage_features.get("estimated_daily_decay", 0))
    vix_regime = vol_regime_features.get("vix_regime", "normal")
    vix_current = float(vol_regime_features.get("vix_current", 20))
    trend_aligned = int(leverage_features.get("underlying_trend_aligned", 0))

    flags: list[str] = []
    should_exit = False

    if decay >= decay_threshold:
        flags.append(f"Decay-Warnung: {decay:.4f}/Tag >= Schwelle {decay_threshold}")
        should_exit = True

    if vix_regime == "high":
        flags.append(f"VIX hoch ({vix_current:.1f}) — Leverage-Risiko erhöht")
        should_exit = True

    if not trend_aligned:
        flags.append("Underlying-Trend nicht aligned mit ETF-Richtung")
        should_exit = True

    if not should_exit:
        return None

    return {
        "alert_type": "leveraged_etf",
        "setup_type": "exit_warning",
        "direction": direction,
        "score": 0.0,
        "underlying": underlying,
        "leverage": leverage,
        "flags": flags,
    }
