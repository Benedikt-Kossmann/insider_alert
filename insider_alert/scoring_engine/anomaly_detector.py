"""Anomaly detection using Isolation Forest on feature vectors."""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_model: Optional[IsolationForest] = None
_scaler: Optional[StandardScaler] = None
_feature_names: list[str] = []
_last_trained: Optional[datetime] = None
_RETRAIN_DAYS = 7
_MIN_SAMPLES = 100


def _get_training_data() -> Optional[tuple[np.ndarray, list[str]]]:
    """Fetch historische Signal-Scores aus der DB als Feature-Matrix."""
    try:
        from insider_alert.persistence.storage import _session, Signal
        with _session() as session:
            cutoff = datetime.utcnow() - timedelta(days=180)
            rows = session.query(Signal).filter(
                Signal.computed_at >= cutoff
            ).all()

            if len(rows) < _MIN_SAMPLES:
                return None

            from collections import defaultdict
            daily_data: dict = defaultdict(dict)
            for r in rows:
                date_key = r.computed_at.date() if r.computed_at else None
                if date_key:
                    key = (r.ticker, date_key)
                    daily_data[key][r.signal_key] = float(r.score)

            if len(daily_data) < _MIN_SAMPLES:
                return None

            all_keys = sorted({k for d in daily_data.values() for k in d})
            matrix = [
                [scores.get(sk, 0.0) for sk in all_keys]
                for scores in daily_data.values()
            ]

            X = np.array(matrix, dtype=float)
            return X, all_keys
    except Exception as exc:
        logger.warning("Anomaly training data fetch failed: %s", exc)
        return None


def _maybe_retrain() -> bool:
    """Retrain wenn nötig. Returns True wenn Modell bereit."""
    global _model, _scaler, _feature_names, _last_trained

    if _model is not None and _last_trained:
        if (datetime.utcnow() - _last_trained).days < _RETRAIN_DAYS:
            return True

    result = _get_training_data()
    if result is None:
        return False

    X, feat_names = result

    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
    )
    _model.fit(X_scaled)
    _feature_names = feat_names
    _last_trained = datetime.utcnow()

    logger.info("Anomaly detector trained on %d samples, %d features", len(X), len(feat_names))
    return True


def compute_anomaly_score(signal_scores: dict[str, float]) -> dict:
    """Berechne Anomaly Score für aktuelle Signal-Konstellation.

    Parameters
    ----------
    signal_scores : dict — {signal_key: score} für den aktuellen Ticker/Tag

    Returns
    -------
    dict mit Keys:
        anomaly_score: float — 0 (normal) bis 1 (extrem anomal)
        is_anomaly: bool — True wenn als Anomalie klassifiziert
        anomaly_type: str — "rare_opportunity" | "rare_risk" | "normal"
    """
    defaults: dict = {
        "anomaly_score": 0.0,
        "is_anomaly": False,
        "anomaly_type": "normal",
    }

    if not signal_scores:
        return defaults

    if not _maybe_retrain():
        return defaults

    try:
        feature_vec = np.array([
            signal_scores.get(k, 0.0) for k in _feature_names
        ]).reshape(1, -1)

        scaled = _scaler.transform(feature_vec)  # type: ignore[union-attr]

        # decision_function gibt negative Werte für Anomalien
        raw_score = -float(_model.decision_function(scaled)[0])  # type: ignore[union-attr]
        anomaly_score = float(np.clip(raw_score * 2 + 0.5, 0, 1))

        is_anomaly = bool(_model.predict(scaled)[0] == -1)  # type: ignore[union-attr]

        composite = float(np.mean(list(signal_scores.values())))
        if is_anomaly and composite > 60:
            anomaly_type = "rare_opportunity"
        elif is_anomaly and composite < 40:
            anomaly_type = "rare_risk"
        else:
            anomaly_type = "normal"

        return {
            "anomaly_score": round(anomaly_score, 3),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        }
    except Exception as exc:
        logger.warning("Anomaly scoring failed: %s", exc)
        return defaults


# ---------------------------------------------------------------------------
# Feature Drift Detection
# ---------------------------------------------------------------------------

_DRIFT_THRESHOLD_PVALUE = 0.01  # p < 0.01 → signifikanter Drift


def detect_feature_drift(current_features: dict[str, float], lookback_days: int = 90) -> dict:
    """Vergleiche aktuelle Feature-Verteilung mit historischen Werten (KS-Test).

    Returns
    -------
    dict mit Keys:
        drift_detected: bool
        drifted_features: list[str]
        drift_severity: float — 0 (kein Drift) bis 1 (alle Features driften)
    """
    from scipy.stats import ks_2samp  # lazy import

    defaults: dict = {
        "drift_detected": False,
        "drifted_features": [],
        "drift_severity": 0.0,
    }

    result = _get_training_data()
    if result is None:
        return defaults

    X_hist, feat_names = result
    if len(X_hist) < 50:
        return defaults

    drifted: list[str] = []
    for i, name in enumerate(feat_names):
        # If current_features is non-empty, only check features present in it
        if current_features and name not in current_features:
            continue

        # Vergleiche letzte 20 Werte (aktuelle Periode) vs. ältere Werte
        recent = X_hist[-20:, i]
        older = X_hist[:-20, i]

        if len(recent) < 10 or len(older) < 30:
            continue

        _, pval = ks_2samp(recent, older)
        if pval < _DRIFT_THRESHOLD_PVALUE:
            drifted.append(name)

    severity = len(drifted) / max(len(feat_names), 1)

    return {
        "drift_detected": len(drifted) > 0,
        "drifted_features": drifted,
        "drift_severity": round(severity, 3),
    }
