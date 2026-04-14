# Changelog

All notable changes to insider_alert will be documented in this file.
Format: Phases from the upgrade plan + ad-hoc changes.

Legend:
- 🗄️ **DB** — New/modified database tables (auto-created on restart)
- 📦 **Deps** — New/changed dependencies → run `pip install -r requirements.txt`
- ⚙️ **Config** — Changes to `config.yaml` format or defaults
- 🔄 **Restart** — Service restart required (`sudo systemctl restart insider-alert`)

---

## Phase 5 — Cross-Asset Correlation & Sector Rotation (2026-04-14)

### Added
- `insider_alert/feature_engine/cross_asset_features.py` — 20-day rolling correlation matrix over SPY, QQQ, GLD, TLT, UUP; `compute_cross_asset_features()` returns 5 keys: `equity_correlation_regime` ("normal"|"decorrelation"|"panic"), `spy_qqq_correlation`, `spy_gld_correlation`, `spy_tlt_correlation`, `correlation_anomaly_score`
- `insider_alert/signal_engine/sector_rotation_signal.py` — `compute_sector_rotation_features(sector_etf)` (10d momentum rank, reversal score, capital flow proxy); `compute_sector_rotation_signal()` with 3 components (40+30+30 pts)

### Changed
- `scheduler/pipeline.py`:
  - `build_market_context()`: adds `cross_asset` key (computed once per job run); ctx dict now includes `"cross_asset": {}`
  - `_compute_stock_features()`: imports and calls `compute_sector_rotation_features` using the ticker's sector ETF; adds `sector_rotation` sub-dict to returned features
  - `_compute_stock_signals()`: new `market_ctx` parameter; appends `compute_sector_rotation_signal` to signal list
  - `_fetch_stock_data()`: already added `ticker` key in Phase 4
- `scheduler/jobs.py`: `run_analysis_for_ticker()` gains `market_ctx` parameter; both EOD and intraday jobs pass `market_ctx` down; `_compute_stock_signals` called with `market_ctx`
- `scoring_engine/scorer.py` `DEFAULT_WEIGHTS`: `sector_rotation: 0.07` added; `price_anomaly`/`volume_anomaly` trimmed to 0.14, `options_anomaly`/`insider_signal` to 0.18; sum = 1.00
- `config.yaml` scoring weights updated to match
- `trade_alert_engine/leveraged_etf_alert.py` `detect_leveraged_etf_entry()`: when `market_ctx["cross_asset"]["equity_correlation_regime"] == "panic"`, raises `momentum_min_score` and `dip_min_score` by +10 and prepends `"⚠️ Correlation Regime: PANIC"` flag to all entry alerts

### Migration Notes
- 🔄 **Restart**: Service restart required
- No new dependencies

---

## Phase 4 — GARCH Volatility Forecasting (2026-04-14)

### Added
- `insider_alert/feature_engine/volatility_forecast.py` — GARCH(1,1) via `arch` library; `compute_volatility_forecast(ohlcv, ticker)` with module-level per-ticker cache (refit every 7 days or on >5% daily return shock); graceful fallback when `arch` not installed
- 6 new feature keys: `garch_forecast_1d/5d/10d` (annualised), `vol_surprise_ratio`, `vol_regime_forecast` ("expanding"|"contracting"|"stable"), `vol_of_vol`
- `insider_alert/signal_engine/volatility_forecast_signal.py` — `compute_volatility_forecast_signal()` using 3 Greek-like components (`vol_surprise_ratio` 40pts, `vol_of_vol` 30pts, `garch_forecast_1d` 30pts)
- `garch:` config block in `config.yaml` (`min_observations: 252`, `refit_interval_days: 7`)
- `Config.garch` field + `_DEFAULT_GARCH` defaults in `config.py`

### Changed
- `scheduler/pipeline.py` — `_fetch_stock_data()` adds `ticker` key to returned dict; `_compute_stock_features()` computes `vol_forecast` sub-dict via `compute_volatility_forecast`; `_compute_stock_signals()` appends `compute_volatility_forecast_signal`
- `scoring_engine/scorer.py` — `DEFAULT_WEIGHTS` updated: `volatility_forecast: 0.08` added; `candle_pattern`, `news_divergence`, `accumulation_pattern`, `macro_regime` reduced to keep sum = 1.00
- `config.yaml` scoring weights updated to match new `DEFAULT_WEIGHTS`

### Migration Notes
- 📦 **Deps**: `pip install -r requirements.txt` (added: `arch>=6.0`)
- 🔄 **Restart**: Service restart required

---

## Phase 3 — Options Greeks (2026-04-14)

### Added
- `insider_alert/feature_engine/greeks.py` — Black-Scholes Greeks via `scipy.stats.norm`; `compute_greeks()` + `compute_chain_greeks()`; graceful fallback when scipy unavailable
- 5 new Greek-based feature keys in `compute_options_features()`:
  - `net_delta_exposure` — delta-weighted aggregate flow (positive = bullish)
  - `gamma_imbalance` — call vs put gamma balance
  - `put_call_delta_ratio` — direction-aware put/call ratio
  - `iv_skew_25d` — 25-delta put IV vs call IV skew
  - `iv_term_structure` — front-month vs back-month IV ratio

### Changed
- `feature_engine/options_features.py` — new `risk_free_rate` parameter (default 0.05); Greek features merged into the return dict; helper functions `compute_greeks_features()`, `_compute_iv_skew()`, `_compute_iv_term_structure()` added
- `signal_engine/options_signal.py` — completely replaced volume-proxy components with 4 Greek-based `SignalComponent`s (`net_delta_exposure`, `gamma_imbalance`, `iv_skew_25d`, `iv_term_structure`)
- `feature_engine/macro_features.py` — `irx_rate` (decimal, e.g. 0.045) now included in the `compute_macro_features()` return dict
- `scheduler/pipeline.py` — `build_market_context()` stores `irx_rate` in context dict; proxy options fetch passes real risk-free rate; `_compute_stock_features()` accepts `risk_free_rate` parameter
- `scheduler/jobs.py` — `run_analysis_for_ticker()` extracts `irx_rate` from `macro_features` and forwards to `_compute_stock_features()`

### Migration Notes
- 🔄 **Restart**: Service restart required
- 📦 **Deps**: `scipy` must be installed on server (`pip install scipy` — usually already present)
  - If not: `pip install -r requirements.txt`

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
