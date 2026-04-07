"""SQLAlchemy-based persistence for signals, scores, and alerts."""
import json
import logging
from datetime import datetime, date, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Date,
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


def _get_engine(db_url: str):
    if db_url not in _engines:
        _engines[db_url] = create_engine(db_url, echo=False)
    return _engines[db_url]


def init_db(db_url: str = "sqlite:///insider_alert.db") -> None:
    """Create all tables if they don't exist."""
    engine = _get_engine(db_url)
    Base.metadata.create_all(engine)
    logger.info("Database initialized at %s", db_url)


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
