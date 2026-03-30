"""Multi-Timeframe alert detection.

Combines a Daily-timeframe setup signal with an Intraday confirmation trigger.
A daily setup is confirmed when the intraday price action (shorter-window OHLCV)
aligns with the daily directional bias.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INTRADAY_CONFIRMATION_BARS = 3   # intraday bars that must confirm direction
DEFAULT_SCORE_THRESHOLD = 55.0


def detect_multi_timeframe(
    daily_features: dict,
    intraday_ohlcv: pd.DataFrame | None,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    confirmation_bars: int = DEFAULT_INTRADAY_CONFIRMATION_BARS,
) -> dict | None:
    """Return a multi-timeframe alert dict or *None* if there is no confirmation.

    Parameters
    ----------
    daily_features:
        Price features computed from daily OHLCV (output of
        ``compute_price_features``).
    intraday_ohlcv:
        Intraday OHLCV DataFrame (e.g. 60-minute bars).  May be ``None`` when
        intraday data is unavailable; in that case the check is skipped.
    """
    daily_zscore = daily_features.get("daily_return_zscore", 0.0)
    daily_return_5d = daily_features.get("return_5d", 0.0)
    atr_14 = daily_features.get("atr_14", 0.0)

    # Daily setup requires a meaningful directional bias
    daily_bias: str | None = None
    if daily_return_5d > 0.02 and daily_zscore > 0.5:
        daily_bias = "bullish"
    elif daily_return_5d < -0.02 and daily_zscore < -0.5:
        daily_bias = "bearish"

    if daily_bias is None:
        return None

    intraday_confirmed = False
    intraday_note = "Intraday data unavailable; daily setup only"

    if intraday_ohlcv is not None and not intraday_ohlcv.empty:
        df = intraday_ohlcv.copy()
        df.columns = [c.lower() for c in df.columns]
        if "close" in df.columns and len(df) >= confirmation_bars + 1:
            closes = df["close"].dropna()
            recent = closes.iloc[-confirmation_bars:]
            if daily_bias == "bullish" and all(
                recent.iloc[i] >= recent.iloc[i - 1] for i in range(1, len(recent))
            ):
                intraday_confirmed = True
                intraday_note = f"Intraday confirmed: {confirmation_bars} consecutive higher closes"
            elif daily_bias == "bearish" and all(
                recent.iloc[i] <= recent.iloc[i - 1] for i in range(1, len(recent))
            ):
                intraday_confirmed = True
                intraday_note = f"Intraday confirmed: {confirmation_bars} consecutive lower closes"
            else:
                intraday_note = "Intraday not yet confirmed"

    score = 40.0
    if intraday_confirmed:
        score += 40.0
    score += min(abs(daily_return_5d) / 0.05, 1.0) * 20.0

    if score < score_threshold:
        return None

    flags = [
        f"Daily bias: {daily_bias} (5d return={daily_return_5d * 100:.1f}%, Z={daily_zscore:.2f})",
        intraday_note,
    ]
    if atr_14 > 0:
        flags.append(f"ATR(14)={atr_14:.2f}")

    return {
        "alert_type": "multi_timeframe",
        "setup_type": f"mtf_{daily_bias}",
        "direction": daily_bias,
        "intraday_confirmed": intraday_confirmed,
        "daily_return_5d": round(daily_return_5d, 4),
        "daily_zscore": round(daily_zscore, 3),
        "atr": round(atr_14, 4),
        "score": float(min(max(score, 0.0), 100.0)),
        "flags": flags,
    }
