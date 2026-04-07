"""Tests for SignalOutcome persistence (save + model)."""
import unittest
from datetime import date

from insider_alert.persistence.storage import (
    init_db,
    save_signal_outcomes,
    Base,
    SignalOutcome,
    _get_engine,
    _engines,
)
from sqlalchemy.orm import sessionmaker

_TEST_DB = "sqlite://"  # in-memory


class TestSignalOutcomes(unittest.TestCase):
    def setUp(self):
        """Create a fresh in-memory test DB."""
        # Clear cached engines to get a fresh DB
        _engines.pop(_TEST_DB, None)
        init_db(_TEST_DB)

    def tearDown(self):
        _engines.pop(_TEST_DB, None)

    def test_save_creates_rows(self):
        signals = [
            {"signal_type": "price_anomaly", "score": 72.0, "flags": ["flag1"]},
            {"signal_type": "volume_anomaly", "score": 55.0, "flags": []},
        ]
        save_signal_outcomes("AAPL", date.today(), signals, composite_score=63.5, db_url=_TEST_DB)

        engine = _get_engine(_TEST_DB)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            rows = session.query(SignalOutcome).filter(SignalOutcome.ticker == "AAPL").all()
            self.assertEqual(len(rows), 2)
            types = {r.signal_type for r in rows}
            self.assertEqual(types, {"price_anomaly", "volume_anomaly"})
            for r in rows:
                self.assertAlmostEqual(r.composite_score, 63.5)
                self.assertIsNone(r.return_1d)  # not yet backfilled
                self.assertIsNone(r.hit_1d)

    def test_save_empty_signals(self):
        """Saving empty signal list should not raise."""
        save_signal_outcomes("XYZ", date.today(), [], composite_score=0.0, db_url=_TEST_DB)

        engine = _get_engine(_TEST_DB)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            count = session.query(SignalOutcome).filter(SignalOutcome.ticker == "XYZ").count()
            self.assertEqual(count, 0)

    def test_model_fields(self):
        """Verify the SignalOutcome model has all expected columns."""
        cols = {c.name for c in SignalOutcome.__table__.columns}
        expected = {
            "id", "ticker", "date", "signal_type", "score",
            "composite_score", "return_1d", "return_5d", "return_10d",
            "hit_1d", "hit_5d", "created_at",
        }
        self.assertTrue(expected.issubset(cols), f"Missing: {expected - cols}")


if __name__ == "__main__":
    unittest.main()
