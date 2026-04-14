# Changelog

All notable changes to insider_alert will be documented in this file.
Format: Phases from the upgrade plan + ad-hoc changes.

Legend:
- 🗄️ **DB** — New/modified database tables (auto-created on restart)
- 📦 **Deps** — New/changed dependencies → run `pip install -r requirements.txt`
- ⚙️ **Config** — Changes to `config.yaml` format or defaults
- 🔄 **Restart** — Service restart required (`sudo systemctl restart insider-alert`)

---

## Phase 2 — FinBERT Sentiment (2026-04-14)

### Added
- `insider_alert/nlp/finbert_sentiment.py` — lazy-loaded FinBERT pipeline, `analyze_single()` + `analyze_batch()`, in-memory + SQLite cache, lexicon fallback
- `insider_alert/nlp/filing_sentiment.py` — SEC filing text analysis (8-K sections) with FinBERT
- `insider_alert/nlp/__init__.py`
- `finbert:` config block in `config.yaml` (enabled, cache_db, batch_size, model)
- `Config.finbert` field in `config.py`

### Changed
- `feature_engine/news_features.py` — uses FinBERT confidence-weighted averaging; new `news_confidence` feature key
- `signal_engine/news_signal.py` — score scaled by `confidence_multiplier = 0.5 + 0.5 * news_confidence`
- `persistence/storage.py` — new `SentimentCache` ORM table + `get_cached_sentiment()` + `save_sentiment_cache()`

### Migration Notes
- 📦 **Deps**: `pip install -r requirements.txt` (added: `transformers>=4.35`, `torch>=2.1`)
  - CPU-only server: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- 🗄️ **DB**: New table `sentiment_cache` (auto-created on restart)
- ⚙️ **Config**: New `finbert:` section in `config.yaml`
- 🔄 **Restart**: `systemctl restart insider-alert`

---

## Phase 1 — Fundament-Reparaturen (2026-04-14)

### Fixed
- `backtest/engine.py` — look-ahead bias: `ohlcv.iloc[:i+1]` → `ohlcv.iloc[:i]` (Ergebnisse waren 20-30% zu optimistisch)
- `scoring_engine/ml_scorer.py` — ersetzt Train-Accuracy durch TimeSeriesSplit CV (AUC + F1); balanced class weights via `compute_sample_weight`
- `scoring_engine/adaptive_weights.py` — dynamischer Blend-Ratio (0.3–0.8 je nach Datenmenge); Sharpe-like Edge-Metrik; `std_return_5d` in Hit-Rate-Berechnung; `_MIN_OUTCOMES` 30→50; Default-Lookback 90→180 Tage
- `data_ingestion/insider_data.py` — `_MAX_FORM4_FETCH` 20→40; `days_back` Default 30→90

### Migration Notes
- 🔄 **Restart**: `systemctl restart insider-alert`

---

## [Unreleased]

### Pre-Phase Cleanup
- Created `.github/copilot-instructions.md` for VS Code Copilot context
- Created `deploy/upgrade.sh` for automated server upgrades
- Created this CHANGELOG.md

---

<!-- Template for future entries:

## Phase XX — Title (YYYY-MM-DD)

### Added
- ...

### Changed
- ...

### Migration Notes
- 📦 **Deps**: `pip install -r requirements.txt` (added: xxx)
- 🗄️ **DB**: New table `xxx` (auto-created on restart)
- ⚙️ **Config**: New key `xxx:` in config.yaml
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

-->
