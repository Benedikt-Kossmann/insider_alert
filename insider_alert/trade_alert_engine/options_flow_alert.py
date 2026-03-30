"""Options Flow alert detection.

Fires on unusual options activity: sweeps/blocks on short-dated OTM contracts,
significant IV jumps, large open interest changes, and earnings-driven unusual
options behaviour.
"""
import logging

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_THRESHOLD = 0.6        # sweep_order_score above which we flag
DEFAULT_BLOCK_THRESHOLD = 0.5        # block_trade_score threshold
DEFAULT_IV_JUMP_THRESHOLD = 0.20     # 20 % relative IV jump
DEFAULT_OI_CHANGE_THRESHOLD = 0.30   # 30 % open-interest change
DEFAULT_CALL_ZSCORE_THRESHOLD = 2.0  # call volume z-score threshold


def detect_options_flow(
    options_features: dict,
    event_features: dict | None = None,
    *,
    sweep_threshold: float = DEFAULT_SWEEP_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
    iv_jump_threshold: float = DEFAULT_IV_JUMP_THRESHOLD,
    oi_change_threshold: float = DEFAULT_OI_CHANGE_THRESHOLD,
    call_zscore_threshold: float = DEFAULT_CALL_ZSCORE_THRESHOLD,
) -> dict | None:
    """Return an options-flow alert dict or *None* if no unusual activity is found."""
    sweep = options_features.get("sweep_order_score", 0.0)
    block = options_features.get("block_trade_score", 0.0)
    iv_change = options_features.get("iv_change_1d", 0.0)
    oi_change = options_features.get("open_interest_change", 0.0)
    call_zscore = options_features.get("call_volume_zscore", 0.0)
    short_otm = options_features.get("short_dated_otm_call_score", 0.0)
    put_call_ratio_change = options_features.get("put_call_ratio_change", 0.0)

    days_to_earnings = (event_features or {}).get("days_to_earnings", 999)
    near_earnings = days_to_earnings <= 10

    flags: list[str] = []
    score = 0.0

    triggered = False

    if sweep >= sweep_threshold:
        triggered = True
        score += sweep * 30.0
        flags.append(f"Options sweep activity: score={sweep:.2f}")

    if block >= block_threshold:
        triggered = True
        score += block * 25.0
        flags.append(f"Block trade detected: score={block:.2f}")

    if abs(iv_change) >= iv_jump_threshold:
        triggered = True
        score += min(abs(iv_change) / iv_jump_threshold, 2.0) * 15.0
        direction = "spike" if iv_change > 0 else "crush"
        flags.append(f"IV {direction}: {iv_change * 100:.1f}%")

    if abs(oi_change) >= oi_change_threshold:
        triggered = True
        score += min(abs(oi_change) / oi_change_threshold, 2.0) * 10.0
        flags.append(f"Open interest change: {oi_change * 100:.1f}%")

    if call_zscore >= call_zscore_threshold:
        triggered = True
        score += min(call_zscore / call_zscore_threshold, 2.0) * 10.0
        flags.append(f"Elevated call volume z-score: {call_zscore:.2f}")

    if short_otm >= 0.5:
        triggered = True
        score += short_otm * 15.0
        flags.append(f"Short-dated OTM call activity: {short_otm:.2f}")

    if near_earnings:
        score += 10.0
        flags.append(f"Near earnings ({days_to_earnings}d): unusual options activity amplified")

    if not triggered:
        return None

    # Infer directional bias from put/call ratio change
    direction = "bullish" if put_call_ratio_change < 0 else "bearish"

    return {
        "alert_type": "options_flow",
        "setup_type": "options_flow_unusual",
        "direction": direction,
        "sweep_score": round(sweep, 3),
        "block_score": round(block, 3),
        "iv_change": round(iv_change, 4),
        "oi_change": round(oi_change, 4),
        "near_earnings": near_earnings,
        "score": float(min(max(score, 0.0), 100.0)),
        "flags": flags,
    }
