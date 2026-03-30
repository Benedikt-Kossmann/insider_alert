# insider_alert

**Insider-Activity Detection Bot** – identifies potentially informed market activity and scores it across eight signal categories. Sends Telegram alerts when a composite score exceeds a configurable threshold.

---

## Features

| Signal Category | What is detected |
|---|---|
| **Price Anomaly** | Unusual returns / z-scores / gap patterns |
| **Volume Anomaly** | Relative volume spikes, tight-range+high-volume |
| **Orderflow Anomaly** | Bid-ask imbalance, absorption, VWAP accumulation |
| **Options Anomaly** | OTM call surges, sweep/block trades, IV changes |
| **Insider Signal** | SEC Form-4 buy clusters, CEO/CFO weighting |
| **Event Lead-up** | Abnormal activity in the 10 days before earnings/M&A |
| **News Divergence** | Price move without a public news catalyst |
| **Accumulation Pattern** | Wyckoff accumulation, higher lows, range compression |

---

## Architecture

```
insider_alert/
├── config.py                  # Config loader (config.yaml + .env)
├── data_ingestion/            # yfinance, SEC EDGAR, news endpoints
├── feature_engine/            # Pure feature computation (8 modules)
├── signal_engine/             # 0–100 signal scoring (8 modules)
├── scoring_engine/            # Weighted composite score aggregation
├── alert_engine/              # Telegram Bot API integration
├── persistence/               # SQLAlchemy / SQLite storage
└── scheduler/                 # APScheduler EOD + intraday jobs
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
ALPHA_VANTAGE_API_KEY=optional
```

Edit `config.yaml` to set your tickers and alert threshold:

```yaml
tickers:
  - AAPL
  - MSFT

scoring:
  alert_threshold: 60   # 0–100; alerts are sent when score >= this value
```

### 3. Run a one-off scan

```bash
python main.py scan
# or for a single ticker:
python main.py scan --ticker NVDA
```

### 4. Start the scheduler (EOD + intraday)

```bash
python main.py schedule
```

For Ubuntu server operation, use `systemd` instead of `screen`. See [docs/ubuntu-deployment.md](/opt/insider_alert/docs/ubuntu-deployment.md).

---

## Output

Each scan produces:

- A **composite score** (0–100) per ticker
- **Sub-scores** for each signal category
- Human-readable **flags** explaining why a ticker is suspicious
- Persistent storage in `insider_alert.db` (SQLite)
- **Telegram alert** when score ≥ `alert_threshold`

Example Telegram message:
```
🚨 Insider Alert: AAPL
Composite Score: 72.4/100

Sub-scores:
  • price_anomaly: 55.0
  • volume_anomaly: 80.0
  • options_anomaly: 90.0
  ...

Flags:
  ⚠️ Elevated relative volume: 3.2x
  ⚠️ Short-dated OTM call surge detected
  ⚠️ Price moved without news catalyst
```

---

## Running tests

```bash
python -m unittest discover tests -v
```

---

## Deployment

For a persistent Ubuntu server setup:

- use a virtual environment
- run `python main.py schedule` via `systemd`
- restart the service after pulling repo changes

Reference files:

- [docs/ubuntu-deployment.md](/opt/insider_alert/docs/ubuntu-deployment.md)
- [deploy/systemd/insider-alert.service](/opt/insider_alert/deploy/systemd/insider-alert.service)

---

## Data sources

| Data | Source |
|---|---|
| OHLCV (daily & intraday) | [yfinance](https://github.com/ranaroussi/yfinance) |
| Options chains | yfinance |
| Earnings dates | yfinance |
| News | yfinance |
| Insider transactions (Form 4) | [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) |

No paid API keys are required for basic operation.
