# Copilot Instructions for insider_alert

## Project Overview
Financial signal detection system that monitors 45 US stocks and 10 leveraged ETFs, computing multi-signal composite scores and sending Telegram alerts when thresholds are met. Runs as a systemd service on Ubuntu (`/root/insider_alert`).

## Architecture

```
main.py (CLI: scan | schedule | backtest)
  └─ scheduler/jobs.py          — APScheduler EOD (17:30 UTC) + Intraday (every 60 min)
       ├─ scheduler/pipeline.py  — build_market_context(), per-ticker feature+signal pipeline
       │    ├─ data_ingestion/   — yfinance (OHLCV, options, news), SEC EDGAR (Form-4, 8-K)
       │    ├─ feature_engine/   — 15 feature modules → dict of floats
       │    ├─ nlp/              — finbert_sentiment.py (FinBERT), filing_sentiment.py (SEC 8-K)
       │    ├─ signal_engine/    — 9+ signals, each returns (score 0-100, flags[])
       │    │    └─ base_signal.py — SignalComponent dataclass + compute_signal() helper
       │    └─ scoring_engine/   — scorer.py (weighted sum), ml_scorer.py (GBM), adaptive_weights.py
       ├─ trade_alert_engine/    — 5 alert types (breakout, mean-reversion, options, event, multi-tf)
       ├─ alert_engine/          — telegram_alert.py (send), weekly_report.py
       └─ persistence/storage.py — SQLite via SQLAlchemy (signals, scores, alerts, signal_outcomes)
```

## Key Patterns

### Signal Pattern
All signals use the declarative `SignalComponent` system from `signal_engine/base_signal.py`:
```python
COMPONENTS = [
    SignalComponent(key="feature_key", max_score=40, normaliser=2.0, flag_template="...", flag_threshold=0.5),
    ...
]
def my_signal(features: dict) -> tuple[float, list[str]]:
    return compute_signal(COMPONENTS, features)
```
**New signals must follow this pattern.** Total max_score across components should sum to 100.

### Config
- `config.yaml` (YAML) + `.env` (secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
- Loaded via `insider_alert/config.py` → `get_config()` singleton returns `Config` dataclass
- ETF config under `leveraged_etfs:` key with `universe:` list and per-ETF `news_proxy:`

### Database (SQLite)
- Tables: `signals`, `scores`, `alerts`, `signal_outcomes`, `sentiment_cache`
- All functions in `persistence/storage.py` accept optional `db_url` parameter
- `init_db()` creates tables via `Base.metadata.create_all()`
- New tables are automatically created on startup — no migration tool needed

### Scoring Weights
- **Stocks**: `DEFAULT_WEIGHTS` in `scoring_engine/scorer.py` — 9 signals, threshold 60
- **ETFs**: `DEFAULT_ETF_WEIGHTS` in `scoring_engine/scorer.py` — 6+3 signals, threshold from config (~68)

## Deployment Context

- **Server**: Ubuntu, systemd service at `/root/insider_alert`
- **Venv**: `/root/insider_alert/.venv`
- **Process**: `python main.py schedule` (long-running blocking scheduler)
- **Restarting**: `systemctl restart insider-alert`
- **Update flow**: Cronjob ruft `deploy/upgrade.sh` auf → `git pull → pip install (bei Änderung) → systemctl restart`

### When adding new dependencies
1. Add to `requirements.txt` **and** `pyproject.toml [project] dependencies`
2. Note in CHANGELOG.md that a `pip install` is needed on deploy

### When adding new DB tables
- Just add new ORM class to `storage.py` inheriting `Base`
- `Base.metadata.create_all()` handles creation on next startup (restart)
- No ALTER TABLE support — if modifying existing columns, note manually in CHANGELOG

## Code Conventions
- Python 3.10+, type hints on function signatures
- Logging via `logging.getLogger(__name__)` — no print()
- Feature functions return `dict[str, float]` with sensible defaults on error
- Signal functions return `tuple[float, list[str]]` — (score 0-100, flag strings)
- Import inside functions when needed to avoid circular imports (common in jobs.py, pipeline.py)
- SEC EDGAR requests MUST include `User-Agent` header

## Testing
```bash
python -m pytest tests/ -v
```
Tests are in `tests/` with `test_` prefix. Use `pytest` fixtures, mock external APIs.

## NLP / Sentiment
- `nlp/finbert_sentiment.py` — `analyze_single()` + `analyze_batch()`, lazy-loads ProsusAI/finbert
- Falls back to keyword lexicon (`news_features._financial_sentiment`) when `transformers` not installed
- Results cached in-memory + SQLite (`sentiment_cache` table)
- `nlp/filing_sentiment.py` — chunks SEC filing text, runs FinBERT per section
- `news_features.compute_news_features()` uses confidence-weighted FinBERT scoring + `news_confidence` feature
- `news_signal` scales score by `confidence_multiplier = 0.5 + 0.5 * news_confidence`
- FinBERT config under `finbert:` key in `config.yaml`, accessible via `Config.finbert`

## File Naming
- Feature modules: `insider_alert/feature_engine/{name}_features.py`
- Signal modules: `insider_alert/signal_engine/{name}_signal.py`
- Data modules: `insider_alert/data_ingestion/{name}_data.py`
- Alert modules: `insider_alert/trade_alert_engine/{name}_alert.py`

## Upgrade Plan
There is a 13-phase upgrade plan in `.plans/PHASE-{01..13}_*.md` (git-ignored).
Each phase document is self-contained with exact file paths, code examples, and acceptance criteria.
Phases should be implemented sequentially (Phase 1 first — it fixes critical bugs).

**After implementing any phase or significant change, ALWAYS update `CHANGELOG.md` immediately.**
Use the existing legend (🗄️ DB, 📦 Deps, ⚙️ Config, 🔄 Restart) and note every migration action required on the server.
