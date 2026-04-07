"""Base signal computation framework.

All signal modules follow the same pattern: extract components from features,
apply scoring rules, collect flags, clip to [0, 100]. This module provides a
declarative helper so each signal only needs to define its *components* rather
than repeating the boilerplate.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalComponent:
    """Definition of one scoring component within a signal.

    Parameters
    ----------
    key : str
        Feature dict key to read.
    max_score : float
        Maximum score this component can contribute.
    normaliser : float
        Divisor to map the raw feature value to [0, 1].  Ignored when
        *score_fn* is provided.
    flag_template : str
        Python format-string with ``{value}`` placeholder.
        A flag is emitted when the component exceeds ``flag_threshold``.
    flag_threshold : float
        Fraction of *max_score* above which to emit a flag (default 0.5 = 50%).
    abs_value : bool
        If True, use ``abs(feature_value)`` before normalising.
    offset : float
        Subtracted from the raw value before normalising (useful for RVOL
        where baseline is 1.0 not 0.0).
    binary : bool
        If True, treat feature as 0/1 flag and multiply directly with max_score.
    score_fn : callable | None
        Optional custom ``(raw_value) -> score`` override.  Must return a
        float in ``[0, max_score]``.
    """
    key: str
    max_score: float
    normaliser: float = 1.0
    flag_template: str = ""
    flag_threshold: float = 0.5
    abs_value: bool = False
    offset: float = 0.0
    binary: bool = False
    score_fn: object = None  # callable | None


def compute_signal(
    signal_type: str,
    features: dict,
    components: list[SignalComponent],
) -> dict:
    """Generic signal computation from a list of component definitions.

    Returns ``{"signal_type": ..., "score": 0-100, "flags": [...]}``.
    """
    total = 0.0
    flags: list[str] = []

    for comp in components:
        raw = features.get(comp.key, 0.0)
        if isinstance(raw, (int, float)):
            raw = float(raw)
        else:
            raw = 0.0

        if comp.score_fn is not None:
            score = float(comp.score_fn(raw))
            score = max(0.0, min(score, comp.max_score))
        elif comp.binary:
            score = raw * comp.max_score
        else:
            val = (abs(raw) if comp.abs_value else raw) - comp.offset
            score = min(max(val, 0.0) / comp.normaliser, 1.0) * comp.max_score

        if score > comp.max_score * comp.flag_threshold and comp.flag_template:
            flags.append(comp.flag_template.format(value=raw))

        total += score

    return {
        "signal_type": signal_type,
        "score": float(np.clip(total, 0.0, 100.0)),
        "flags": flags,
    }
