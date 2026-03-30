"""Mean-Reversion alert detection.

Fires when price or volume Z-score reaches an extreme, indicating a potential
counter-move.  Optionally combines with news divergence to filter noise.
"""
import logging

logger = logging.getLogger(__name__)

DEFAULT_ZSCORE_THRESHOLD = 2.5   # |Z-score| that triggers an alert
DEFAULT_RR_RATIO = 1.5           # typical reward:risk for mean-reversion setups


def detect_mean_reversion(
    price_features: dict,
    volume_features: dict,
    news_features: dict | None = None,
    *,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    rr_ratio: float = DEFAULT_RR_RATIO,
) -> dict | None:
    """Return a mean-reversion alert dict or *None* if conditions are not met.

    Conditions (at least one required):
    * |price Z-score| >= zscore_threshold
    * |volume Z-score| >= zscore_threshold

    Optionally amplified by news divergence (price moved sharply but news
    sentiment is neutral/opposite).
    """
    price_zscore = price_features.get("daily_return_zscore", 0.0)
    volume_zscore = volume_features.get("volume_zscore_20d", 0.0)
    news_divergence = (news_features or {}).get("news_price_divergence", 0.0)

    price_extreme = abs(price_zscore) >= zscore_threshold
    volume_extreme = abs(volume_zscore) >= zscore_threshold

    if not price_extreme and not volume_extreme:
        return None

    flags: list[str] = []
    score = 50.0

    direction = "bullish_reversal" if price_zscore < 0 else "bearish_reversal"

    if price_extreme:
        component = min(abs(price_zscore) / zscore_threshold, 2.0) * 20.0
        score += component
        flags.append(f"Extreme price Z-score: {price_zscore:.2f}")

    if volume_extreme:
        component = min(abs(volume_zscore) / zscore_threshold, 2.0) * 15.0
        score += component
        flags.append(f"Extreme volume Z-score: {volume_zscore:.2f}")

    if abs(news_divergence) >= 0.5:
        score += 10.0
        flags.append(f"News-price divergence detected: {news_divergence:.2f}")

    atr_pct = price_features.get("atr_pct", 0.0)
    atr_14 = price_features.get("atr_14", 0.0)

    return {
        "alert_type": "mean_reversion",
        "setup_type": f"mean_reversion_{direction}",
        "direction": direction,
        "price_zscore": round(price_zscore, 3),
        "volume_zscore": round(volume_zscore, 3),
        "atr": round(atr_14, 4),
        "atr_pct": round(atr_pct, 4),
        "rr_ratio": rr_ratio,
        "score": float(min(max(score, 0.0), 100.0)),
        "flags": flags,
    }
