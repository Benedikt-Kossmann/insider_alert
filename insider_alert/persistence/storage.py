"""SQLAlchemy-based persistence for signals, scores, alerts, and outcomes."""
import json
import logging
from datetime import datetime, date, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Date, Boolean,
    create_engine, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()
_engines: dict = {}


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False)
    date = Column(Date, nullable=False)
    signal_type = Column(String(64), nullable=False)
    score = Column(Float, nullable=False)
    flags = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Score(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False)
    date = Column(Date, nullable=False)
    total_score = Column(Float, nullable=False)
    sub_scores = Column(Text, nullable=True)
    flags = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False)
    date = Column(Date, nullable=False)
    score = Column(Float, nullable=False)
    alert_type = Column(String(64), nullable=True)
    setup_type = Column(String(64), nullable=True)
    message = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SignalOutcome(Base):
    """Tracks signal predictions vs actual future returns for validation."""
    __tablename__ = "signal_outcomes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    signal_type = Column(String(64), nullable=False)
    score = Column(Float, nullable=False)
    composite_score = Column(Float, nullable=True)
    return_1d = Column(Float, nullable=True)
    return_5d = Column(Float, nullable=True)
    return_10d = Column(Float, nullable=True)
    hit_1d = Column(Boolean, nullable=True)
    hit_5d = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SentimentCache(Base):
    """Persistent cache for FinBERT (or lexicon) sentiment results."""
    __tablename__ = "sentiment_cache"
    id = Column(Integer, primary_key=True)
    headline_hash = Column(String(16), unique=True, index=True, nullable=False)
    headline_text = Column(String(500), nullable=False)
    sentiment = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    label = Column(String(16), nullable=False)
    model_version = Column(String(32), default="finbert-v1")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class InstitutionalCache(Base):
    """Persistent cache for SEC 13-F institutional flow results (TTL: 7 days)."""
    __tablename__ = "institutional_cache"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), unique=True, index=True, nullable=False)
    institutional_buy_count = Column(Integer, nullable=False, default=0)
    institutional_sell_count = Column(Integer, nullable=False, default=0)
    institutional_net_direction = Column(String(16), nullable=False, default="neutral")
    smart_money_score = Column(Float, nullable=False, default=0.5)
    fetched_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class OHLCVCache(Base):
    """Cached daily OHLCV data per ticker (Phase 13)."""
    __tablename__ = "ohlcv_cache"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adj_close = Column(Float)


class OptionsArchive(Base):
    """Archived daily options chain data per ticker (Phase 13)."""
    __tablename__ = "options_archive"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    contract_type = Column(String(8))
    strike = Column(Float)
    expiration = Column(Date)
    volume = Column(Integer)
    open_interest = Column(Integer)
    implied_volatility = Column(Float)
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)


class FeatureSnapshot(Base):
    """Daily flattened feature snapshot per ticker (Phase 13)."""
    __tablename__ = "feature_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    features_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MacroSnapshot(Base):
    """Daily macro indicator snapshot (Phase 13)."""
    __tablename__ = "macro_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    features_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def _get_engine(db_url: str):
    if db_url not in _engines:
        _engines[db_url] = create_engine(db_url, echo=False)
    return _engines[db_url]


def init_db(db_url: str = "sqlite:///insider_alert.db") -> None:
    """Create all tables if they don't exist."""
    engine = _get_engine(db_url)
    Base.metadata.create_all(engine)
    logger.info("Database initialized at %s", db_url)


def get_cached_sentiment(
    headline_hash: str,
    db_url: str = "sqlite:///insider_alert.db",
) -> dict | None:
    """Return cached sentiment result for a headline hash, or None if not found."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        row = session.query(SentimentCache).filter_by(headline_hash=headline_hash).first()
        if row:
            return {"sentiment": row.sentiment, "confidence": row.confidence, "label": row.label}
    return None


def save_sentiment_cache(
    headline_hash: str,
    headline_text: str,
    sentiment: float,
    confidence: float,
    label: str,
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Persist a sentiment result.  Silently skips if the hash already exists."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        existing = session.query(SentimentCache).filter_by(headline_hash=headline_hash).first()
        if not existing:
            session.add(SentimentCache(
                headline_hash=headline_hash,
                headline_text=headline_text[:500],
                sentiment=sentiment,
                confidence=confidence,
                label=label,
            ))
            session.commit()


def get_cached_institutional(
    ticker: str,
    db_url: str = "sqlite:///insider_alert.db",
) -> dict | None:
    """Return cached institutional flow data for *ticker*, or None if not cached."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        row = session.query(InstitutionalCache).filter_by(ticker=ticker.upper()).first()
        if row:
            return {
                "institutional_buy_count": row.institutional_buy_count,
                "institutional_sell_count": row.institutional_sell_count,
                "institutional_net_direction": row.institutional_net_direction,
                "smart_money_score": row.smart_money_score,
            }
    return None


def save_institutional_cache(
    ticker: str,
    data: dict,
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Upsert institutional flow data for *ticker* (replaces existing row)."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        existing = session.query(InstitutionalCache).filter_by(ticker=ticker.upper()).first()
        if existing:
            existing.institutional_buy_count = data.get("institutional_buy_count", 0)
            existing.institutional_sell_count = data.get("institutional_sell_count", 0)
            existing.institutional_net_direction = data.get("institutional_net_direction", "neutral")
            existing.smart_money_score = data.get("smart_money_score", 0.5)
            existing.fetched_at = datetime.now(timezone.utc)
        else:
            session.add(InstitutionalCache(
                ticker=ticker.upper(),
                institutional_buy_count=data.get("institutional_buy_count", 0),
                institutional_sell_count=data.get("institutional_sell_count", 0),
                institutional_net_direction=data.get("institutional_net_direction", "neutral"),
                smart_money_score=data.get("smart_money_score", 0.5),
                fetched_at=datetime.now(timezone.utc),
            ))
        session.commit()


def should_refresh_institutional(
    ticker: str,
    ttl_days: int = 7,
    db_url: str = "sqlite:///insider_alert.db",
) -> bool:
    """Return True when institutional cache for *ticker* is absent or expired."""
    from datetime import timedelta
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        row = session.query(InstitutionalCache).filter_by(ticker=ticker.upper()).first()
        if row is None:
            return True
        age = datetime.now(timezone.utc) - row.fetched_at.replace(tzinfo=timezone.utc)
        return age > timedelta(days=ttl_days)


# ---------------------------------------------------------------------------
# Phase 13: OHLCV Cache
# ---------------------------------------------------------------------------

def get_cached_ohlcv(
    ticker: str,
    lookback_days: int = 400,
    db_url: str = "sqlite:///insider_alert.db",
):
    """Return cached OHLCV rows as a DataFrame (DatetimeIndex, lowercase columns).

    Returns empty DataFrame when no data is cached.
    """
    import pandas as pd
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=lookback_days)
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        rows = (
            session.query(OHLCVCache)
            .filter(OHLCVCache.ticker == ticker.upper(), OHLCVCache.date >= cutoff)
            .order_by(OHLCVCache.date)
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "Date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
        df = pd.DataFrame(data).set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df


def save_ohlcv_cache(
    ticker: str,
    ohlcv,
    db_url: str = "sqlite:///insider_alert.db",
) -> int:
    """Insert new OHLCV rows (skips existing ticker+date pairs). Returns count inserted."""
    if ohlcv is None or ohlcv.empty:
        return 0

    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    saved = 0

    # Normalise column names to lowercase
    ohlcv = ohlcv.copy()
    ohlcv.columns = [c.lower() for c in ohlcv.columns]

    with Session() as session:
        for idx, row in ohlcv.iterrows():
            d = idx.date() if hasattr(idx, "date") else idx
            existing = (
                session.query(OHLCVCache)
                .filter_by(ticker=ticker.upper(), date=d)
                .first()
            )
            if not existing:
                session.add(OHLCVCache(
                    ticker=ticker.upper(),
                    date=d,
                    open=float(row.get("open", 0) or 0),
                    high=float(row.get("high", 0) or 0),
                    low=float(row.get("low", 0) or 0),
                    close=float(row.get("close", 0) or 0),
                    volume=float(row.get("volume", 0) or 0),
                    adj_close=float(row.get("adj close", row.get("close", 0)) or 0),
                ))
                saved += 1
        session.commit()
    return saved


def get_ohlcv_with_cache(
    ticker: str,
    period: str = "1y",
    db_url: str = "sqlite:///insider_alert.db",
):
    """Smart-fetch OHLCV: serve from cache, refresh only missing days via yfinance.

    Falls back to direct yfinance download when cache is empty.
    Returns a DataFrame with lowercase column names and DatetimeIndex.
    """
    import pandas as pd
    import yfinance as yf
    from datetime import timedelta

    cached = get_cached_ohlcv(ticker, lookback_days=400, db_url=db_url)

    if not cached.empty:
        last_cached = cached.index[-1].date()
        today = date.today()

        if last_cached >= today - timedelta(days=1):
            return cached

        # Fetch only the days since the last cached date
        start = (last_cached + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            new_raw = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
            if not new_raw.empty:
                new_raw.columns = [c.lower() for c in new_raw.columns]
                save_ohlcv_cache(ticker, new_raw, db_url=db_url)
                return pd.concat([cached, new_raw])
        except Exception as exc:
            logger.debug("Incremental OHLCV fetch failed for %s: %s", ticker, exc)
        return cached

    # No cache — full download
    try:
        raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if not raw.empty:
            raw.columns = [c.lower() for c in raw.columns]
            save_ohlcv_cache(ticker, raw, db_url=db_url)
        return raw
    except Exception as exc:
        logger.warning("OHLCV download failed for %s: %s", ticker, exc)
        import pandas as pd
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 13: Options Archive
# ---------------------------------------------------------------------------

def save_options_archive(
    ticker: str,
    options_df,
    db_url: str = "sqlite:///insider_alert.db",
) -> int:
    """Archive today's top-40 options rows (by volume). Returns count inserted."""
    if options_df is None or options_df.empty:
        return 0

    df = options_df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Keep top 40 by volume around ATM
    vol_col = "volume" if "volume" in df.columns else None
    df = df.nlargest(40, vol_col) if vol_col else df.head(40)

    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    today = date.today()
    saved = 0

    with Session() as session:
        for _, row in df.iterrows():
            try:
                exp = row.get("expiration")
                if exp is not None and hasattr(exp, "date"):
                    exp = exp.date()
                session.add(OptionsArchive(
                    ticker=ticker.upper(),
                    date=today,
                    contract_type=str(row.get("contracttype", row.get("contract_type", ""))),
                    strike=float(row.get("strike", 0) or 0),
                    expiration=exp,
                    volume=int(row.get("volume", 0) or 0),
                    open_interest=int(row.get("openinterest", row.get("open_interest", 0)) or 0),
                    implied_volatility=float(row.get("impliedvolatility", row.get("implied_volatility", 0)) or 0),
                    last_price=float(row.get("lastprice", row.get("last_price", 0)) or 0),
                    bid=float(row.get("bid", 0) or 0),
                    ask=float(row.get("ask", 0) or 0),
                ))
                saved += 1
            except Exception as exc:
                logger.debug("OptionsArchive row skipped: %s", exc)
        session.commit()
    return saved


def get_archived_options(
    ticker: str,
    lookback_days: int = 30,
    db_url: str = "sqlite:///insider_alert.db",
):
    """Return archived options rows for *ticker* as a DataFrame."""
    import pandas as pd
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=lookback_days)
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        rows = (
            session.query(OptionsArchive)
            .filter(OptionsArchive.ticker == ticker.upper(), OptionsArchive.date >= cutoff)
            .order_by(OptionsArchive.date)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "date": r.date,
            "contract_type": r.contract_type,
            "strike": r.strike,
            "expiration": r.expiration,
            "volume": r.volume,
            "open_interest": r.open_interest,
            "implied_volatility": r.implied_volatility,
        } for r in rows])


# ---------------------------------------------------------------------------
# Phase 13: Feature Snapshot Store
# ---------------------------------------------------------------------------

def save_feature_snapshot(
    ticker: str,
    features: dict,
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Upsert a flattened feature snapshot for *ticker* / today.

    Only primitive values (int, float, str, bool, None) are serialised.
    """
    today = date.today()
    safe: dict = {}
    for k, v in features.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, tuple)):
            safe[k] = list(v)

    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        existing = (
            session.query(FeatureSnapshot)
            .filter_by(ticker=ticker.upper(), date=today)
            .first()
        )
        if existing:
            existing.features_json = json.dumps(safe)
        else:
            session.add(FeatureSnapshot(
                ticker=ticker.upper(),
                date=today,
                features_json=json.dumps(safe),
            ))
        session.commit()


def get_feature_snapshots(
    ticker: str,
    lookback_days: int = 90,
    db_url: str = "sqlite:///insider_alert.db",
) -> list[dict]:
    """Return historical feature snapshots for *ticker* as a list of dicts."""
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=lookback_days)
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        rows = (
            session.query(FeatureSnapshot)
            .filter(FeatureSnapshot.ticker == ticker.upper(), FeatureSnapshot.date >= cutoff)
            .order_by(FeatureSnapshot.date)
            .all()
        )
        return [{"date": r.date, **json.loads(r.features_json)} for r in rows]


# ---------------------------------------------------------------------------
# Phase 13: Macro Snapshot History
# ---------------------------------------------------------------------------

def save_macro_snapshot(
    macro_features: dict,
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Upsert today's macro feature snapshot."""
    today = date.today()
    safe = {
        k: v
        for k, v in macro_features.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        existing = session.query(MacroSnapshot).filter_by(date=today).first()
        if existing:
            existing.features_json = json.dumps(safe)
        else:
            session.add(MacroSnapshot(date=today, features_json=json.dumps(safe)))
        session.commit()


def get_macro_history(
    lookback_days: int = 365,
    db_url: str = "sqlite:///insider_alert.db",
):
    """Return historical macro snapshots as a DataFrame indexed by date."""
    import pandas as pd
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=lookback_days)
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        rows = (
            session.query(MacroSnapshot)
            .filter(MacroSnapshot.date >= cutoff)
            .order_by(MacroSnapshot.date)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        data = [{"date": r.date, **json.loads(r.features_json)} for r in rows]
        return pd.DataFrame(data).set_index("date")


# ---------------------------------------------------------------------------
# Phase 13: DB Maintenance
# ---------------------------------------------------------------------------

def cleanup_old_data(
    max_age_days: int = 365,
    db_url: str = "sqlite:///insider_alert.db",
) -> dict:
    """Delete rows older than *max_age_days* from all time-series tables.

    Returns a dict ``{table_name: deleted_count}``.
    """
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=max_age_days)
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    deleted: dict = {}

    with Session() as session:
        for model, date_col in [
            (OHLCVCache, OHLCVCache.date),
            (OptionsArchive, OptionsArchive.date),
            (FeatureSnapshot, FeatureSnapshot.date),
            (MacroSnapshot, MacroSnapshot.date),
            (Signal, Signal.date),
            (Score, Score.date),
        ]:
            count = session.query(model).filter(date_col < cutoff).delete()
            deleted[model.__tablename__] = count
        session.commit()

    logger.info("DB cleanup (max_age=%dd): %s", max_age_days, deleted)
    return deleted


def save_signal(
    ticker: str,
    date_val: date,
    signal_type: str,
    score: float,
    flags: list,
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Persist a signal result."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        record = Signal(
            ticker=ticker,
            date=date_val,
            signal_type=signal_type,
            score=score,
            flags=json.dumps(flags),
        )
        session.add(record)
        session.commit()


def save_score(
    ticker: str,
    date_val: date,
    ticker_score,
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Persist a composite score."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        record = Score(
            ticker=ticker,
            date=date_val,
            total_score=ticker_score.total_score,
            sub_scores=json.dumps(ticker_score.sub_scores),
            flags=json.dumps(ticker_score.flags),
        )
        session.add(record)
        session.commit()


def save_alert(
    ticker: str,
    date_val: date,
    score: float,
    message: str,
    alert_type: str = "",
    setup_type: str = "",
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Persist a sent alert."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        record = Alert(
            ticker=ticker,
            date=date_val,
            score=score,
            alert_type=alert_type,
            setup_type=setup_type,
            message=message,
            sent_at=datetime.now(timezone.utc),
        )
        session.add(record)
        session.commit()


def get_recent_scores(
    ticker: str,
    days: int = 30,
    db_url: str = "sqlite:///insider_alert.db",
) -> list[dict]:
    """Return recent composite scores for a ticker."""
    from datetime import timedelta
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    cutoff = date.today() - timedelta(days=days)
    with Session() as session:
        records = (
            session.query(Score)
            .filter(Score.ticker == ticker, Score.date >= cutoff)
            .order_by(Score.date.desc())
            .all()
        )
        return [
            {
                "ticker": r.ticker,
                "date": r.date.isoformat() if r.date else None,
                "total_score": r.total_score,
                "sub_scores": json.loads(r.sub_scores) if r.sub_scores else {},
                "flags": json.loads(r.flags) if r.flags else [],
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]


def is_alert_duplicate(
    ticker: str,
    setup_type: str,
    cooldown_hours: float = 4.0,
    db_url: str = "sqlite:///insider_alert.db",
) -> bool:
    """Return True if an alert for *ticker* / *setup_type* was already sent within
    the cooldown window, preventing duplicate Telegram messages."""
    from datetime import timedelta
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
    with Session() as session:
        count = (
            session.query(Alert)
            .filter(
                Alert.ticker == ticker,
                Alert.setup_type == setup_type,
                Alert.sent_at >= cutoff,
            )
            .count()
        )
    return count > 0


def save_signal_outcomes(
    ticker: str,
    date_val: date,
    signals: list[dict],
    composite_score: float,
    db_url: str = "sqlite:///insider_alert.db",
) -> None:
    """Persist signal outcome rows (returns filled later by backfill)."""
    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        for sig in signals:
            record = SignalOutcome(
                ticker=ticker,
                date=date_val,
                signal_type=sig.get("signal_type", "unknown"),
                score=float(sig.get("score", 0.0)),
                composite_score=composite_score,
            )
            session.add(record)
        session.commit()


def backfill_signal_outcomes(
    db_url: str = "sqlite:///insider_alert.db",
) -> int:
    """Fill return_1d/5d/10d for past signal outcomes using stored scores.

    Looks up closing prices from OHLCV data and computes actual forward returns.
    Returns the number of rows updated.
    """
    from datetime import timedelta

    engine = _get_engine(db_url)
    Session = sessionmaker(bind=engine)
    updated = 0

    with Session() as session:
        # Find outcomes missing returns that are old enough (at least 10 trading days)
        cutoff = date.today() - timedelta(days=14)
        pending = (
            session.query(SignalOutcome)
            .filter(
                SignalOutcome.return_1d.is_(None),
                SignalOutcome.date <= cutoff,
            )
            .all()
        )

        if not pending:
            return 0

        # Group by ticker to batch OHLCV fetches
        from collections import defaultdict
        by_ticker: dict[str, list] = defaultdict(list)
        for row in pending:
            by_ticker[row.ticker].append(row)

        from insider_alert.data_ingestion.market_data import fetch_ohlcv_daily

        for ticker, rows in by_ticker.items():
            try:
                ohlcv = fetch_ohlcv_daily(ticker, period="3mo")
                if ohlcv.empty or "close" not in ohlcv.columns:
                    continue

                closes = ohlcv["close"]
                date_index = {d.date() if hasattr(d, "date") else d: i
                              for i, d in enumerate(closes.index)}

                for row in rows:
                    sig_date = row.date
                    idx = date_index.get(sig_date)
                    if idx is None:
                        # Find nearest trading date
                        for offset in range(0, 3):
                            d = sig_date + timedelta(days=offset)
                            if d in date_index:
                                idx = date_index[d]
                                break
                    if idx is None:
                        continue

                    close_0 = float(closes.iloc[idx])
                    if close_0 == 0:
                        continue

                    for horizon, attr, hit_attr in [
                        (1, "return_1d", "hit_1d"),
                        (5, "return_5d", "hit_5d"),
                        (10, "return_10d", None),
                    ]:
                        fwd_idx = idx + horizon
                        if fwd_idx < len(closes):
                            ret = (float(closes.iloc[fwd_idx]) / close_0) - 1
                            setattr(row, attr, ret)
                            if hit_attr is not None:
                                setattr(row, hit_attr, row.score > 50 and ret > 0)

                    updated += 1

            except Exception as exc:
                logger.warning("Backfill failed for %s: %s", ticker, exc)

        session.commit()

    logger.info("Backfilled %d signal outcome rows", updated)
    return updated
