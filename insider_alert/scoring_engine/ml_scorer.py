"""Lightweight ML scoring using Gradient Boosting on signal outcomes.

Trains a ``GradientBoostingClassifier`` on historical ``SignalOutcome`` rows
to predict whether a ticker will have a positive 5-day return.  The model
is retrained periodically and stored in memory.

Falls back gracefully when scikit-learn is not installed or when not enough
training data is available.
"""
import logging
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.info("scikit-learn not installed; ML scoring disabled.")


# Singleton model cache
_model = None
_scaler = None
_feature_names: list[str] = []
_last_trained: date | None = None
_RETRAIN_DAYS = 7  # retrain at most every N days
_MIN_SAMPLES = 80  # minimum outcome rows to train on


def is_available() -> bool:
    """Return True if ML scoring can run (sklearn installed)."""
    return _HAS_SKLEARN


def _fetch_training_data(
    lookback_days: int = 180,
    db_url: str = "sqlite:///insider_alert.db",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load training data from SignalOutcome table.

    Returns (X, y, feature_names).
    X shape: (n_days, n_signal_types) — each row is one date/ticker combo
    with sub-signal scores as features.
    y shape: (n_days,) — whether composite hit_5d was True.
    """
    from insider_alert.persistence.storage import _get_engine, SignalOutcome
    from sqlalchemy.orm import sessionmaker
    from collections import defaultdict

    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    cutoff = date.today() - timedelta(days=lookback_days)

    with Session() as session:
        rows = (
            session.query(SignalOutcome)
            .filter(
                SignalOutcome.date >= cutoff,
                SignalOutcome.hit_5d.isnot(None),
            )
            .all()
        )

    if not rows:
        return np.array([]), np.array([]), []

    # Group by (ticker, date) to create one sample per ticker-day
    by_key: dict[tuple, dict] = defaultdict(dict)
    labels: dict[tuple, bool] = {}
    signal_types: set[str] = set()

    for r in rows:
        key = (r.ticker, r.date)
        by_key[key][r.signal_type] = r.score
        signal_types.add(r.signal_type)
        # Use the conjunction: majority of signals hit
        if key not in labels:
            labels[key] = r.hit_5d
        else:
            # Any hit is a hit for target
            labels[key] = labels[key] or r.hit_5d

    feature_names = sorted(signal_types)
    X_rows = []
    y_rows = []

    for key, scores in by_key.items():
        row = [float(scores.get(ft, 0.0)) for ft in feature_names]
        X_rows.append(row)
        y_rows.append(1 if labels.get(key, False) else 0)

    return np.array(X_rows), np.array(y_rows), feature_names


def train_model(
    lookback_days: int = 180,
    db_url: str = "sqlite:///insider_alert.db",
) -> bool:
    """Train the GBM model on historical outcomes.

    Returns True if training succeeded.
    """
    global _model, _scaler, _feature_names, _last_trained

    if not _HAS_SKLEARN:
        return False

    try:
        X, y, feat_names = _fetch_training_data(lookback_days, db_url)
    except Exception as exc:
        logger.warning("Failed to fetch ML training data: %s", exc)
        return False

    n = len(y)
    if n < _MIN_SAMPLES:
        logger.info(
            "Not enough training data for ML scorer (%d < %d samples).",
            n, _MIN_SAMPLES,
        )
        return False

    # Need both classes
    if len(set(y)) < 2:
        logger.info("ML training data has only one class, skipping.")
        return False

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_scaled, y)

    _model = model
    _scaler = scaler
    _feature_names = feat_names
    _last_trained = date.today()

    # Log feature importances
    importances = model.feature_importances_
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        logger.info("ML feature importance: %s = %.3f", name, imp)

    accuracy = model.score(X_scaled, y)
    logger.info(
        "ML model trained: %d samples, %d features, train accuracy=%.1f%%",
        n, len(feat_names), accuracy * 100,
    )
    return True


def predict_score(signals: list[dict]) -> float | None:
    """Predict hit probability from a list of signal dicts.

    Returns a score 0-100, or None if model is unavailable.
    """
    if not _HAS_SKLEARN or _model is None:
        return None

    # Build feature vector
    scores_map = {s.get("signal_type", ""): float(s.get("score", 0.0)) for s in signals}
    features = [scores_map.get(fn, 0.0) for fn in _feature_names]

    if not features:
        return None

    X = np.array([features])
    X_scaled = _scaler.transform(X)
    proba = _model.predict_proba(X_scaled)[0]

    # Probability of class 1 (hit)
    hit_prob = float(proba[1]) if len(proba) > 1 else 0.0
    return round(hit_prob * 100, 1)


def maybe_retrain(db_url: str = "sqlite:///insider_alert.db") -> None:
    """Retrain if enough time has passed since last training."""
    if not _HAS_SKLEARN:
        return

    if _last_trained is not None:
        days_since = (date.today() - _last_trained).days
        if days_since < _RETRAIN_DAYS:
            return

    train_model(db_url=db_url)
