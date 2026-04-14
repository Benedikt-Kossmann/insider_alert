# Changelog

All notable changes to insider_alert will be documented in this file.
Format: Phases from the upgrade plan + ad-hoc changes.

Legend:
- 🗄️ **DB** — New/modified database tables (auto-created on restart)
- 📦 **Deps** — New/changed dependencies → run `pip install -r requirements.txt`
- ⚙️ **Config** — Changes to `config.yaml` format or defaults
- 🔄 **Restart** — Service restart required (`sudo systemctl restart insider-alert`)

---

## REVIEW_01 — Critical Bug Fixes (2026-04-14)

### Fixed
- **insider_signal.py** — `_COMPONENTS` max_score summed to 110 instead of 100; redistributed to 22+18+22+14+14+10=100. Scores now properly fill the 0–100 range without premature clipping.
- **short_volume_data.py** — `int()` parse for short/total volume wrapped in individual `try/except (ValueError, TypeError)` with WARNING logging; prevents crash propagation on malformed FINRA data.
- **orderflow_features.py** — Iceberg detection logic was inverted (fired on high RVOL, not normal RVOL). Fixed: `iceberg_suspect_score` now fires on tight range (<1%) + normal/low RVOL (<3×). High RVOL + tight range is absorption, not iceberg.
- **accumulation_features.py** — `higher_lows_score` guarded with `min(..., 1.0)` clip for defensive correctness.
- **options_data.py** — `fetch_historical_iv()` could return NaN when `std()` is called on <2 returns; added `if pd.isna(hv_30d): return 0.0`.
- **news_features.py** — `price_news_divergence_score` incorrectly fired when there were NO news + big move. Fixed: score only fires when news exist AND price moves opposite to sentiment direction; empty-news early-return also corrected.
- **macro_features.py** — Added clear docstring documenting units: `yield_spread` in percentage-point terms; `irx_rate` in decimal (for Black-Scholes); `vix_current` in index points. All downstream consumers (macro_signal, jobs.py, tests) already consistent.

### Tests
- `tests/test_signal_engine.py` — `TestInsiderSignalMaxScore`: verifies sum=100, full-feature score=100, no overflow.
- `tests/test_feature_engine.py` — `TestAccumulationHigherLowsClip`, `TestIcebergLogicFixed` (4 cases), `TestNewsDivergenceLogicFixed` (3 cases).

### Migration
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

---

## Phase 13 — Persistenz-Erweiterung (2026-04-14)

### Added
- `insider_alert/persistence/storage.py` — 4 neue ORM-Tabellen:
  - `OHLCVCache` — Tagespreise je Ticker; vermeidet redundante yfinance-Calls.
  - `OptionsArchive` — Archiviert täglich Top-40 Options-Zeilen (nach Volume) je Ticker.
  - `FeatureSnapshot` — Täglicher Feature-Snapshot je Ticker (serialisiert als JSON).
  - `MacroSnapshot` — Täglicher Makro-Feature-Snapshot (unique per Datum).
- Neue Storage-Funktionen: `get_ohlcv_with_cache()`, `save_ohlcv_cache()`, `get_cached_ohlcv()`, `save_options_archive()`, `get_archived_options()`, `save_feature_snapshot()`, `get_feature_snapshots()`, `save_macro_snapshot()`, `get_macro_history()`, `cleanup_old_data()`.

### Changed
- `insider_alert/data_ingestion/market_data.py`:
  - Neue Funktion `fetch_ohlcv_daily_cached()` — Smart-Fetch mit SQLite-Cache; fällt auf `fetch_ohlcv_daily()` zurück wenn DB nicht verfügbar.
- `insider_alert/scheduler/pipeline.py`:
  - `_fetch_stock_data()`: Verwendet `fetch_ohlcv_daily_cached()` für den Haupt-Ticker (Sektor-ETFs weiter direkt via yfinance).
  - `build_market_context()`: Speichert Makro-Features via `save_macro_snapshot()` nach jedem EOD-Lauf.
  - Neue Funktion `_persist_phase13_data()` — Archiviert Options-Chain + Feature-Snapshot je Ticker.
- `insider_alert/scheduler/jobs.py`:
  - `run_analysis_for_ticker()`: Ruft `_persist_phase13_data()` nach Feature-Berechnung auf.
  - `run_eod_job()`: Ruft montags `cleanup_old_data(max_age_days=365)` auf.

### Migration
- 🗄️ **DB**: 4 neue Tabellen (`ohlcv_cache`, `options_archive`, `feature_snapshots`, `macro_snapshots`) — automatisch beim nächsten Start angelegt.
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

---

## Phase 12 — Seasonality (2026-04-14)

### Added
- `insider_alert/feature_engine/seasonality_features.py` — Neu:
  - `compute_seasonality_features(ohlcv, current_date)` — 6 Keys: `monthly_bias`, `weekday_bias`, `quad_witching`, `sell_in_may_active`, `seasonal_score`, `month_strength`.
  - Basiert auf historischen S&P-500-Monatsbias-Werten + Quad-Witching-Erkennung (3. Freitag in März/Juni/September/Dezember).

### Changed
- `insider_alert/scheduler/pipeline.py`:
  - `_compute_stock_features()`: Fügt `seasonality`-Feature-Gruppe hinzu.
  - `_compute_stock_signals()`: Emittiert saisonale Flags (Quad-Witching, bullisch/bearisch) als Zero-Score-Signal in die Signalliste.
- `insider_alert/alert_engine/weekly_report.py`:
  - `generate_weekly_report()`: Neuer Abschnitt "Saisonaler Ausblick (nächste Woche)" mit Monats-Bias, Sell-in-May-Status, Quad-Witching und Seasonal Score.

### Migration
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

---

## Phase 11 — 13-F Institutional Flows (2026-04-14)

### Added
- `insider_alert/data_ingestion/institutional_data.py` — Neu:
  - `fetch_institutional_flows(ticker)` — Prüft Top-20 Institutionen (SEC 13-F) via EDGAR Submissions API + XML-Parsing; gibt `institutional_buy_count`, `institutional_sell_count`, `institutional_net_direction`, `smart_money_score` zurück.
  - `_fetch_latest_13f_url(cik)` — Findet neueste 13-F Filing-URL für ein CIK.
  - `_parse_13f_xml(xml_url)` — Parst XML Information Table.
  - Rate-Limit: 0.12s zwischen Requests (gleich wie insider_data.py); SEC User-Agent gesetzt.
- `insider_alert/signal_engine/institutional_signal.py` — Neu:
  - `institutional_signal(features)` — SignalComponent-Pattern; score 0–100 aus `smart_money_score` (max 60) + `institutional_buy_count` (max 40).

### Changed
- `insider_alert/persistence/storage.py`:
  - Neue ORM-Klasse `InstitutionalCache` — 7-Tage-Cache für 13-F-Ergebnisse je Ticker.
  - Neue Funktionen: `get_cached_institutional()`, `save_institutional_cache()`, `should_refresh_institutional()`.
- `insider_alert/scheduler/pipeline.py`:
  - `_fetch_stock_data()`: ruft `fetch_institutional_flows()` auf (mit 7-Tage-Cache); Ergebnis unter Key `"institutional"` im Data-Dict.
  - `_compute_stock_features()`: reicht `data["institutional"]` direkt als Feature-Group `"institutional"` weiter.
  - `_compute_stock_signals()`: fügt `institutional_signal()` zur Signalliste hinzu.
- `insider_alert/scoring_engine/scorer.py`:
  - `DEFAULT_WEIGHTS`: `"institutional": 0.05` hinzugefügt; `price_anomaly`, `volume_anomaly`, `options_anomaly`, `insider_signal`, `event_leadup` jeweils leicht reduziert (Summe bleibt 1.0).

### Migration
- 🗄️ **DB**: Neue Tabelle `institutional_cache` — wird automatisch beim nächsten Start angelegt (`Base.metadata.create_all()`).
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

---

## Phase 10 — Risk Management & Position Sizing (2026-04-14)

### Added
- `insider_alert/trade_alert_engine/position_sizer.py` — Neu:
  - `kelly_criterion(win_rate, avg_win, avg_loss)` — Berechnet Kelly-Fraction.
  - `compute_position_size(signal_scores, composite_score, ticker, ...)` — Empfohlener Portfolio-Anteil (%) basierend auf historischen Win/Loss-Daten aus der DB. Gibt `position_pct`, `kelly_raw`, `kelly_half`, `confidence`, `reasoning` zurück.
- `insider_alert/trade_alert_engine/risk_manager.py` — Erweitert um:
  - `check_correlation_risk(ticker, sector_etf, ...)` — Warnt bei ≥3 offenen Alerts im gleichen Sektor (72h-Fenster).
  - `check_drawdown_guard(lookback_days, ...)` — Simuliert Portfolio-Return aus DB-Outcomes; erhöht Score-Schwelle um 10 Punkte wenn DD > 5%.

### Changed
- `insider_alert/scheduler/jobs.py`:
  - `run_analysis_for_ticker()`: neuer optionaler Parameter `dd_guard`; führt Korrelations-Check + Position-Sizing durch wenn Alert gesendet wird; angepasste `effective_threshold = base + dd_adj + corr_adj`.
  - `run_eod_job()`: ruft `check_drawdown_guard()` einmal auf und gibt `dd_guard`-Dict an alle Ticker-Analysen weiter.
- `insider_alert/alert_engine/telegram_alert.py`:
  - `build_alert_message()`: neue Parameter `position_info` und `risk_warnings`; zeigt Position-Size + Confidence-Emoji und Drawdown-/Cluster-Warnungen am Ende der Nachricht.
  - `maybe_send_alert()`: Signatur unverändert (threshold kommt now vorher aus jobs.py angepasst).
- `insider_alert/config.py`: neues Feld `risk_management: dict` in `Config`-Dataclass; geladen aus YAML mit Defaults.
- `config.yaml`: neuer Abschnitt `risk_management:` mit `kelly_fraction`, `max_position_pct`, `min_position_pct`, `max_sector_alerts`, `drawdown_guard_pct`, `threshold_increase`.

### Migration
- ⚙️ **Config**: `config.yaml` enthält neuen `risk_management:`-Block (optional, Defaults greifen automatisch).
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

---

## Phase 9 — Chart-Generation für Telegram (2026-04-14)

### Added
- `insider_alert/alert_engine/chart_generator.py` — Neu:
  - `generate_ticker_chart(ohlcv, ticker, score, ...)` — Candlestick-PNG mit EMA10/50, S/R-Levels als horizontale Linien, Volume-Panel. Speichert in `/tmp/insider_alert_charts/`.
  - `generate_macro_dashboard(market_ctx, days=90)` — 3-Panel PNG: VIX, Yield Curve (10Y-3M), Dollar (UUP). Für den Weekly Report.
  - `cleanup_old_charts(max_age_days=7)` — Löscht PNGs älter als N Tage.
- `insider_alert/alert_engine/telegram_alert.py`:
  - `send_telegram_photo(token, chat_id, image_path, caption)` — sendet lokale PNG-Datei via Telegram `sendPhoto`-API.

### Changed
- `insider_alert/scheduler/jobs.py`:
  - `run_analysis_for_ticker()`: sendet nach einem Text-Alert automatisch den Ticker-Chart via `send_telegram_photo()` (wenn `charts.enabled: true`).
  - `run_eod_job()`: ruft `cleanup_old_charts()` am Ende auf.
- `insider_alert/alert_engine/weekly_report.py`:
  - `send_weekly_report()`: sendet nach dem Textbericht das Makro-Dashboard als Bild.
- `insider_alert/config.py`:
  - `Config` Dataclass: neues Feld `charts: dict` mit Defaults `{enabled: true, style: "charles", days: 30}`.
  - `load_config()`: liest `charts:` aus config.yaml.
- `config.yaml`: neuer Abschnitt `charts:` mit `enabled`, `style`, `days`, `include_ema`, `include_sr`.

### Migration
- 📦 **Deps**: `pip install -r requirements.txt` (neu: `mplfinance>=0.12`)
- 🔄 **Restart**: `sudo systemctl restart insider-alert`

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
