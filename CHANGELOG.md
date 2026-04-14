# Changelog

All notable changes to insider_alert will be documented in this file.
Format: Phases from the upgrade plan + ad-hoc changes.

Legend:
- 🗄️ **DB** — New/modified database tables (auto-created on restart)
- 📦 **Deps** — New/changed dependencies → run `pip install -r requirements.txt`
- ⚙️ **Config** — Changes to `config.yaml` format or defaults
- 🔄 **Restart** — Service restart required (`sudo systemctl restart insider-alert`)

---

## Phase 8 — Anomaly Detection (Isolation Forest + Feature Drift) (2026-04-14)

### Added
- `insider_alert/scoring_engine/anomaly_detector.py` — Isolation Forest–based anomaly detection:
  - `compute_anomaly_score(signal_scores)` — trains on historical signal scores from DB, returns `{anomaly_score, is_anomaly, anomaly_type}`. Types: `rare_opportunity` (anomal + bullish), `rare_risk` (anomal + bearish), `normal`.
  - `detect_feature_drift(current_features)` — KS-test comparing last 20 observations vs. older history; returns `{drift_detected, drifted_features, drift_severity}`.
  - Modell re-trainsiert automatisch alle 7 Tage wenn ≥ 100 Samples in der DB vorhanden.
- `tests/test_anomaly_detection.py` — 16 neue Tests für alle Phase-8-Komponenten.

### Changed
- `insider_alert/scheduler/jobs.py`:
  - `run_analysis_for_ticker()`: berechnet `anomaly_info` nach Signal-Scoring und gibt es an `maybe_send_alert` + `build_alert_message` weiter.
  - `run_eod_job()`: ruft `detect_feature_drift({})` einmal pro Job auf und loggt Drift als WARNING.
- `insider_alert/alert_engine/telegram_alert.py`:
  - `build_alert_message()`: neuer Parameter `anomaly_info`; fügt 🔥 *RARE OPPORTUNITY* oder ⚠️ *RARE RISK* Zeile ein wenn relevant.
  - `maybe_send_alert()`: neuer Parameter `anomaly_info`, wird an `build_alert_message` durchgereicht.

### Migration
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

---

## Phase 7 — FINRA Short Volume & Earnings Drift (PEAD) (2026-04-14)

### Added
- `insider_alert/data_ingestion/short_volume_data.py` — `fetch_short_volume(ticker, lookback_days=30)`: downloads FINRA RegSHO daily short-volume CSVs (consolidated NMS), parses pipe-delimited format defensively, returns DataFrame `[Date, ShortVolume, TotalVolume, ShortRatio]`.
- `insider_alert/feature_engine/short_volume_features.py` — `compute_short_volume_features(short_df)`: 4 features: `short_ratio_current`, `short_ratio_zscore`, `short_ratio_trend_5d`, `short_squeeze_score`.
- `insider_alert/signal_engine/short_squeeze_signal.py` — `short_squeeze_signal(features)` with 3 `SignalComponent`s (40+30+30 pts), returns `{"signal_type": "short_squeeze", ...}`.
- `insider_alert/data_ingestion/earnings_data.py` — `fetch_earnings_data(ticker)`: fetches earnings dates and computes earnings-day return, 3d/10d post-earnings drift via yfinance. Graceful fallback to defaults on errors.
- `insider_alert/signal_engine/earnings_drift_signal.py` — `compute_pead_features(earnings_data)` + `earnings_drift_signal(features)`: PEAD 3-component signal (40+30+30 pts) with exponential time decay.
- `tests/test_short_volume_pead.py` — 28 new tests covering all Phase 7 modules.

### Changed
- `insider_alert/scheduler/pipeline.py`:
  - `_fetch_stock_data()`: fetches `short_vol_df` and `earnings_raw` per ticker (with exception isolation)
  - `_compute_stock_features()`: computes `short_volume` and `pead` feature sub-dicts, adds them to returned features
  - `_compute_stock_signals()`: appends `short_squeeze_signal` and `earnings_drift_signal` to the signal list
- `insider_alert/scoring_engine/scorer.py` `DEFAULT_WEIGHTS`: added `short_squeeze: 0.05` and `earnings_drift: 0.06`; adjusted existing weights to keep sum = 1.00 (`options_anomaly`/`insider_signal` 0.18→0.16, `price_anomaly`/`volume_anomaly` 0.14→0.12, `event_leadup` 0.10→0.09, `news_divergence` 0.04→0.03, `sector_rotation` 0.07→0.06)
- `config.yaml` scoring weights updated to match

### Migration
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

---



### Added
- `insider_alert/data_ingestion/fred_data.py` — FRED API client (`fetch_fred_series()`) + `fetch_all_macro_data()` returning 9 pre-processed macro features (HY spread, Fed funds, CPI, initial claims, unemployment, consumer sentiment). Gracefully falls back to sane defaults when no API key is configured.
- `insider_alert/feature_engine/macro_features.py` → `compute_fred_macro_features(fred_data, market_data)` — derives `credit_stress_score`, `fed_policy_score`, `inflation_score`, `labor_market_score`, `consumer_sentiment_norm`, `macro_regime`

### Changed
- `insider_alert/feature_engine/macro_features.py` → `compute_macro_features()` now also emits `vix_value` (alias for `vix_current`) and `dxy_change_5d` (20d % return ×100) for use by the new signal.
- `insider_alert/signal_engine/macro_signal.py` — new `macro_signal(features)` function using `SignalComponent` pattern with 6 components (100 pts total: VIX 25, Yield 15, DXY 10, Credit Stress 25, Fed Policy 15, Labor Market 10). Old `compute_macro_regime_signal()` kept for backward compatibility.
- `insider_alert/scheduler/pipeline.py`:
  - `build_market_context()`: after fetching VIX/yield/DXY macro, enriches `ctx["macro"]` with FRED features via `fetch_all_macro_data()` + `compute_fred_macro_features()`
  - `_compute_stock_signals()`: uses new `macro_signal` instead of `compute_macro_regime_signal`
  - `_compute_etf_features_and_signals()`: same switch to `macro_signal`
- `insider_alert/config.py` — `Config` dataclass gains `fred_api_key: str = ""`; `load_config()` reads from `config.yaml` key `fred_api_key` with `FRED_API_KEY` env-var fallback
- `config.yaml` — new `fred_api_key: ""` key under FRED API section
- `requirements.txt` + `pyproject.toml` — `fredapi>=0.5` added

### Migration
- 📦 **Deps**: `pip install fredapi>=0.5`
- ⚙️ **Config**: Add `fred_api_key: "YOUR_KEY"` to `config.yaml` **or** `FRED_API_KEY=YOUR_KEY` to `.env`. Free key at https://fred.stlouisfed.org/docs/api/api_key.html  
  Without a key the system operates normally using yfinance-based macro features and default FRED fallback values.
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

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
