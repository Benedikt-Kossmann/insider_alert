"""Mean-reversion / dip-buy signal for leveraged ETFs."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_mean_reversion_dip_signal(
    momentum_features: dict,
    vol_regime_features: dict,
    price_features: dict,
    *,
    direction: str = "long",
) -> dict:
    """Score a dip-buy (long) or overbought-fade (short) opportunity.

    Parameters
    ----------
    momentum_features : dict
        Output of ``compute_momentum_features()``.
    vol_regime_features : dict
        Output of ``compute_volatility_regime_features()``.
    price_features : dict
        Output of ``compute_price_features()``.
    direction : str
        ``'long'`` or ``'short'``.

    Returns
    -------
    dict
        ``{"signal_type": "mean_reversion_dip", "score": float, "flags": list[str]}``
    """
    score = 0.0
    flags: list[str] = []

    rsi = float(momentum_features.get("rsi_14", 50.0))
    macd_hist = float(momentum_features.get("macd_histogram", 0.0))
    bollinger_pct_b = float(vol_regime_features.get("bollinger_pct_b", 0.5))
    return_zscore = float(price_features.get("daily_return_zscore", 0.0))

    if direction == "long":
        # --- RSI oversold (up to 30 pts) ---
        if rsi < 25:
            score += 30.0
            flags.append(f"RSI deeply oversold ({rsi:.0f}) — dip-buy candidate")
        elif rsi < 30:
            score += 25.0
            flags.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 35:
            score += 15.0
            flags.append(f"RSI approaching oversold ({rsi:.0f})")

        # --- Bollinger %B near lower band (up to 25 pts) ---
        if bollinger_pct_b < 0.05:
            score += 25.0
            flags.append(f"Price below lower Bollinger Band (%B={bollinger_pct_b:.2f})")
        elif bollinger_pct_b < 0.10:
            score += 20.0
            flags.append(f"Price near lower Bollinger Band (%B={bollinger_pct_b:.2f})")
        elif bollinger_pct_b < 0.20:
            score += 10.0
            flags.append(f"Price in lower Bollinger zone (%B={bollinger_pct_b:.2f})")

        # --- MACD positive divergence (up to 20 pts) ---
        # Histogram turning positive after being negative = bullish divergence
        if macd_hist > 0 and rsi < 40:
            score += 20.0
            flags.append("MACD bullish divergence (histogram turning positive while oversold)")
        elif macd_hist > -0.01 and rsi < 35:
            score += 10.0
            flags.append("MACD flattening near zero (potential reversal)")

        # --- Extreme z-score (up to 25 pts) ---
        if return_zscore < -2.5:
            score += 25.0
            flags.append(f"Extreme negative z-score ({return_zscore:.1f})")
        elif return_zscore < -2.0:
            score += 20.0
            flags.append(f"Strong negative z-score ({return_zscore:.1f})")
        elif return_zscore < -1.5:
            score += 10.0
            flags.append(f"Negative z-score ({return_zscore:.1f})")
    else:
        # Short/inverse ETFs: overbought = dip-buy for inverse
        if rsi > 75:
            score += 30.0
            flags.append(f"RSI overbought ({rsi:.0f}) — inverse dip-buy")
        elif rsi > 70:
            score += 25.0
            flags.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 65:
            score += 15.0
            flags.append(f"RSI approaching overbought ({rsi:.0f})")

        if bollinger_pct_b > 0.95:
            score += 25.0
            flags.append(f"Price above upper Bollinger Band (%B={bollinger_pct_b:.2f})")
        elif bollinger_pct_b > 0.90:
            score += 20.0
            flags.append(f"Price near upper Bollinger Band (%B={bollinger_pct_b:.2f})")

        if macd_hist < 0 and rsi > 60:
            score += 20.0
            flags.append("MACD bearish divergence (histogram negative while overbought)")

        if return_zscore > 2.5:
            score += 25.0
            flags.append(f"Extreme positive z-score ({return_zscore:.1f})")
        elif return_zscore > 2.0:
            score += 20.0
            flags.append(f"Strong positive z-score ({return_zscore:.1f})")

    score = float(np.clip(score, 0.0, 100.0))

    return {
        "signal_type": "mean_reversion_dip",
        "score": score,
        "flags": flags,
    }
