"""FinBERT-based financial sentiment analysis with in-memory and SQLite caching.

Falls back gracefully to the keyword lexicon in news_features.py when
``transformers`` / ``torch`` are not installed.
"""
import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded pipeline singleton
# ---------------------------------------------------------------------------
_pipeline = None

# In-memory cache:  headline_hash → result dict
_cache: dict[str, dict] = {}

# Label → polarity mapping (FinBERT outputs "positive", "negative", "neutral")
_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _get_pipeline():
    """Lazy-load the FinBERT pipeline (downloads ~420 MB on first call)."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,          # CPU only
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT pipeline loaded successfully")
        except Exception as exc:
            logger.warning("FinBERT unavailable, falling back to lexicon: %s", exc)
            _pipeline = None
    return _pipeline


def is_available() -> bool:
    """Return True if transformers is importable (model may not be pre-loaded)."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _headline_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# DB cache helpers (imported lazily to avoid circular imports at module load)
# ---------------------------------------------------------------------------

def _db_get(headline_hash: str) -> dict | None:
    try:
        from insider_alert.persistence.storage import get_cached_sentiment
        return get_cached_sentiment(headline_hash)
    except Exception:
        return None


def _db_save(headline_hash: str, text: str, result: dict) -> None:
    try:
        from insider_alert.persistence.storage import save_sentiment_cache
        save_sentiment_cache(
            headline_hash=headline_hash,
            headline_text=text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            label=result["label"],
        )
    except Exception as exc:
        logger.debug("Could not persist sentiment cache: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_single(text: str) -> dict:
    """Analyse a single headline.

    Returns ``{"sentiment": float[-1,1], "confidence": float[0,1], "label": str}``.
    """
    if not text or not text.strip():
        return {"sentiment": 0.0, "confidence": 0.0, "label": "neutral"}

    h = _headline_hash(text)

    # 1. in-memory cache
    if h in _cache:
        return _cache[h]

    # 2. DB cache
    db_result = _db_get(h)
    if db_result is not None:
        _cache[h] = db_result
        return db_result

    # 3. FinBERT inference
    pipe = _get_pipeline()
    if pipe is not None:
        try:
            out = pipe(text)[0]
            label = out["label"].lower()
            confidence = float(out["score"])
            result = {
                "sentiment": _LABEL_MAP.get(label, 0.0) * confidence,
                "confidence": confidence,
                "label": label,
            }
        except Exception as exc:
            logger.warning("FinBERT inference failed for '%.50s': %s", text, exc)
            result = {"sentiment": 0.0, "confidence": 0.0, "label": "neutral"}
    else:
        # 4. Lexicon fallback
        from insider_alert.feature_engine.news_features import _financial_sentiment
        score = _financial_sentiment(text)
        label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        result = {"sentiment": score, "confidence": 0.5, "label": label}

    _cache[h] = result
    _db_save(h, text, result)
    return result


def analyze_batch(texts: list[str]) -> list[dict]:
    """Batch-analyse headlines.  More efficient than repeated ``analyze_single`` calls.

    Returns a list of ``{"sentiment", "confidence", "label"}`` dicts in the same
    order as the input.
    """
    if not texts:
        return []

    pipe = _get_pipeline()
    if pipe is None:
        return [analyze_single(t) for t in texts]

    results: list[dict | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            results[i] = {"sentiment": 0.0, "confidence": 0.0, "label": "neutral"}
            continue
        h = _headline_hash(text)
        # Check in-memory then DB
        cached = _cache.get(h) or _db_get(h)
        if cached is not None:
            _cache[h] = cached
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        try:
            batch_out = pipe(uncached_texts, batch_size=16)
            for idx, out in zip(uncached_indices, batch_out):
                label = out["label"].lower()
                confidence = float(out["score"])
                result = {
                    "sentiment": _LABEL_MAP.get(label, 0.0) * confidence,
                    "confidence": confidence,
                    "label": label,
                }
                results[idx] = result
                h = _headline_hash(texts[idx])
                _cache[h] = result
                _db_save(h, texts[idx], result)
        except Exception as exc:
            logger.warning("FinBERT batch failed: %s", exc)
            for idx in uncached_indices:
                if results[idx] is None:
                    results[idx] = analyze_single(texts[idx])

    # Fill any remaining None slots (safety net)
    return [r if r is not None else {"sentiment": 0.0, "confidence": 0.0, "label": "neutral"}
            for r in results]
