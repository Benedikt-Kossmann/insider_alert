"""Tests for new/completed features: insider parsing, IV change, corporate events."""
import datetime
import unittest
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_options_df(n_calls: int = 5, n_puts: int = 5, avg_iv: float = 0.30) -> pd.DataFrame:
    """Build a synthetic options chain DataFrame."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(n_calls):
        rows.append({
            "contractType": "call",
            "strike": 100.0 + i * 5,
            "expiration": "2024-02-16",
            "volume": float(rng.integers(100, 2000)),
            "openInterest": float(rng.integers(500, 5000)),
            "impliedVolatility": avg_iv + rng.normal(0, 0.02),
        })
    for i in range(n_puts):
        rows.append({
            "contractType": "put",
            "strike": 100.0 - i * 5,
            "expiration": "2024-02-16",
            "volume": float(rng.integers(100, 2000)),
            "openInterest": float(rng.integers(500, 5000)),
            "impliedVolatility": avg_iv + rng.normal(0, 0.02),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IV change computation
# ---------------------------------------------------------------------------

class TestIVChangeComputation(unittest.TestCase):
    """Tests for iv_change_1d in compute_options_features."""

    def test_iv_change_zero_when_no_baseline(self):
        from insider_alert.feature_engine.options_features import compute_options_features
        df = _make_options_df(avg_iv=0.30)
        result = compute_options_features(df, current_price=100.0, iv_baseline=0.0)
        self.assertEqual(result["iv_change_1d"], 0.0)

    def test_iv_change_positive_when_iv_elevated(self):
        """Options IV well above historical baseline → positive iv_change_1d."""
        from insider_alert.feature_engine.options_features import compute_options_features
        df = _make_options_df(avg_iv=0.50)
        result = compute_options_features(df, current_price=100.0, iv_baseline=0.20)
        self.assertGreater(result["iv_change_1d"], 0.0)

    def test_iv_change_negative_when_iv_compressed(self):
        """Options IV well below historical baseline → negative iv_change_1d."""
        from insider_alert.feature_engine.options_features import compute_options_features
        df = _make_options_df(avg_iv=0.10)
        result = compute_options_features(df, current_price=100.0, iv_baseline=0.40)
        self.assertLess(result["iv_change_1d"], 0.0)

    def test_iv_change_zero_when_equal(self):
        """When avg IV ≈ baseline the change should be near zero."""
        from insider_alert.feature_engine.options_features import compute_options_features
        df = _make_options_df(avg_iv=0.25)
        result = compute_options_features(df, current_price=100.0, iv_baseline=0.25)
        self.assertAlmostEqual(result["iv_change_1d"], 0.0, places=1)

    def test_all_other_keys_still_present(self):
        """Ensure the full feature dict is returned with iv_baseline set."""
        from insider_alert.feature_engine.options_features import compute_options_features
        df = _make_options_df()
        result = compute_options_features(df, current_price=100.0, iv_baseline=0.20)
        for key in ("call_volume_zscore", "put_volume_zscore", "put_call_ratio_change",
                    "iv_change_1d", "short_dated_otm_call_score", "block_trade_score",
                    "sweep_order_score", "open_interest_change"):
            self.assertIn(key, result)

    def test_empty_df_returns_zero_iv_change(self):
        """Empty options df → iv_change_1d = 0."""
        from insider_alert.feature_engine.options_features import compute_options_features
        result = compute_options_features(pd.DataFrame(), current_price=100.0, iv_baseline=0.30)
        self.assertEqual(result["iv_change_1d"], 0.0)


# ---------------------------------------------------------------------------
# Insider data parsing helpers
# ---------------------------------------------------------------------------

class TestInsiderDataHelpers(unittest.TestCase):
    """Unit tests for Form-4 XML parsing helpers in insider_data."""

    def _make_form4_xml(
        self,
        owner_name: str = "Jane Smith",
        is_officer: str = "1",
        officer_title: str = "Chief Financial Officer",
        is_director: str = "0",
        is_ten_pct: str = "0",
        txn_date: str = "2024-01-15",
        shares: str = "5000",
        price: str = "150.00",
        code: str = "A",
    ) -> ET.Element:
        xml_str = f"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>{owner_name}</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>{is_director}</isDirector>
      <isOfficer>{is_officer}</isOfficer>
      <isTenPercentOwner>{is_ten_pct}</isTenPercentOwner>
      <officerTitle>{officer_title}</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle>Common Stock</securityTitle>
      <transactionDate>
        <value>{txn_date}</value>
      </transactionDate>
      <transactionAmounts>
        <transactionShares>
          <value>{shares}</value>
        </transactionShares>
        <transactionPricePerShare>
          <value>{price}</value>
        </transactionPricePerShare>
        <transactionAcquiredDisposedCode>
          <value>{code}</value>
        </transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        return ET.fromstring(xml_str)

    def test_xml_text_extracts_value_child(self):
        from insider_alert.data_ingestion.insider_data import _xml_text
        root = self._make_form4_xml(txn_date="2024-03-10")
        result = _xml_text(root, ".//nonDerivativeTransaction/transactionDate")
        self.assertEqual(result, "2024-03-10")

    def test_xml_text_direct_text(self):
        from insider_alert.data_ingestion.insider_data import _xml_text
        root = self._make_form4_xml(owner_name="John Doe")
        result = _xml_text(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
        self.assertEqual(result, "John Doe")

    def test_parse_role_officer(self):
        from insider_alert.data_ingestion.insider_data import _parse_role
        root = self._make_form4_xml(officer_title="Chief Executive Officer")
        rel_el = root.find(".//reportingOwner/reportingOwnerRelationship")
        role = _parse_role(rel_el)
        self.assertEqual(role, "Chief Executive Officer")

    def test_parse_role_director_fallback(self):
        from insider_alert.data_ingestion.insider_data import _parse_role
        root = self._make_form4_xml(is_officer="0", officer_title="", is_director="1")
        rel_el = root.find(".//reportingOwner/reportingOwnerRelationship")
        role = _parse_role(rel_el)
        self.assertEqual(role, "Director")

    def test_parse_role_ten_pct_owner(self):
        from insider_alert.data_ingestion.insider_data import _parse_role
        root = self._make_form4_xml(is_officer="0", officer_title="", is_director="0", is_ten_pct="1")
        rel_el = root.find(".//reportingOwner/reportingOwnerRelationship")
        role = _parse_role(rel_el)
        self.assertEqual(role, "10% Owner")

    def test_parse_role_none_element(self):
        from insider_alert.data_ingestion.insider_data import _parse_role
        role = _parse_role(None)
        self.assertEqual(role, "Unknown")


# ---------------------------------------------------------------------------
# Insider features with parsed data
# ---------------------------------------------------------------------------

class TestInsiderFeaturesWithRealData(unittest.TestCase):
    """Test compute_insider_features with realistic parsed data."""

    def _make_txns(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": datetime.date(2024, 1, 10), "insider_name": "Alice CEO",
             "role": "Chief Executive Officer", "transaction_type": "buy",
             "shares": 10000, "value": 1_500_000.0},
            {"date": datetime.date(2024, 1, 11), "insider_name": "Bob CFO",
             "role": "Chief Financial Officer", "transaction_type": "buy",
             "shares": 5000, "value": 750_000.0},
            {"date": datetime.date(2024, 1, 12), "insider_name": "Carol Director",
             "role": "Director", "transaction_type": "sell",
             "shares": 2000, "value": 300_000.0},
        ])

    def test_buy_count(self):
        from insider_alert.feature_engine.insider_features import compute_insider_features
        result = compute_insider_features(self._make_txns())
        self.assertEqual(result["insider_buy_count_30d"], 2)

    def test_sell_count(self):
        from insider_alert.feature_engine.insider_features import compute_insider_features
        result = compute_insider_features(self._make_txns())
        self.assertEqual(result["insider_sell_count_30d"], 1)

    def test_buy_value(self):
        from insider_alert.feature_engine.insider_features import compute_insider_features
        result = compute_insider_features(self._make_txns())
        self.assertAlmostEqual(result["insider_buy_value_30d"], 2_250_000.0, places=0)

    def test_cluster_score_two_buyers(self):
        from insider_alert.feature_engine.insider_features import compute_insider_features
        result = compute_insider_features(self._make_txns())
        # 2 distinct buyers → 2/3 cluster score
        self.assertAlmostEqual(result["insider_cluster_score"], 2 / 3.0, places=5)

    def test_role_weighted_score_senior_buyers(self):
        from insider_alert.feature_engine.insider_features import compute_insider_features
        result = compute_insider_features(self._make_txns())
        # CEO weight=2.0 + CFO weight=2.0 → sum=4.0 → capped at min(4/10, 1)=0.4
        self.assertAlmostEqual(result["insider_role_weighted_score"], 0.4, places=5)


# ---------------------------------------------------------------------------
# Event features with corporate events
# ---------------------------------------------------------------------------

class TestEventFeaturesWithCorporateEvent(unittest.TestCase):
    """Test that days_to_corporate_event properly influences event features."""

    def _base_price_f(self):
        return {"return_5d": 0.04, "return_1d": 0.01}

    def _base_volume_f(self):
        return {"volume_rvol_20d": 2.0}

    def _base_options_f(self):
        return {"call_volume_zscore": 1.5}

    def test_corporate_event_within_window_activates_scores(self):
        from insider_alert.feature_engine.event_features import compute_event_features
        result = compute_event_features(
            days_to_earnings=None,
            price_features=self._base_price_f(),
            volume_features=self._base_volume_f(),
            options_features=self._base_options_f(),
            days_to_corporate_event=5,
        )
        # Effective DTE = min(999, 5) = 5 ≤ 10 → scores activated
        self.assertGreater(result["pre_event_volume_score"], 0.0)
        self.assertGreater(result["pre_event_return_score"], 0.0)
        self.assertEqual(result["days_to_corporate_event"], 5)

    def test_no_corporate_event_and_no_earnings_zero_scores(self):
        from insider_alert.feature_engine.event_features import compute_event_features
        result = compute_event_features(
            days_to_earnings=None,
            price_features=self._base_price_f(),
            volume_features=self._base_volume_f(),
            options_features=self._base_options_f(),
            days_to_corporate_event=None,
        )
        self.assertEqual(result["pre_event_return_score"], 0.0)
        self.assertEqual(result["pre_event_volume_score"], 0.0)
        self.assertEqual(result["days_to_corporate_event"], 999)

    def test_earnings_closer_than_8k_uses_earnings(self):
        from insider_alert.feature_engine.event_features import compute_event_features
        result = compute_event_features(
            days_to_earnings=3,
            price_features=self._base_price_f(),
            volume_features=self._base_volume_f(),
            options_features=self._base_options_f(),
            days_to_corporate_event=20,
        )
        # Effective DTE = min(3, 20) = 3 ≤ 10 → scores activated
        self.assertGreater(result["pre_event_return_score"], 0.0)

    def test_corporate_event_closer_than_earnings_uses_event(self):
        from insider_alert.feature_engine.event_features import compute_event_features
        result = compute_event_features(
            days_to_earnings=30,
            price_features=self._base_price_f(),
            volume_features=self._base_volume_f(),
            options_features=self._base_options_f(),
            days_to_corporate_event=4,
        )
        # Effective DTE = min(30, 4) = 4 ≤ 10 → scores activated
        self.assertGreater(result["pre_event_return_score"], 0.0)

    def test_returns_expected_keys(self):
        from insider_alert.feature_engine.event_features import compute_event_features
        result = compute_event_features(None, {}, {}, {}, None)
        self.assertIn("days_to_earnings", result)
        self.assertIn("days_to_corporate_event", result)
        self.assertIn("pre_event_return_score", result)
        self.assertIn("pre_event_volume_score", result)
        self.assertIn("pre_event_options_score", result)


# ---------------------------------------------------------------------------
# Event signal with corporate event flag
# ---------------------------------------------------------------------------

class TestEventSignalCorporateEventFlag(unittest.TestCase):
    """Test that the event signal surfaces corporate-event flags."""

    def test_8k_flag_added_when_event_close_and_different_from_earnings(self):
        from insider_alert.signal_engine.event_signal import compute_event_leadup_signal
        features = {
            "days_to_earnings": 30,
            "days_to_corporate_event": 5,
            "pre_event_return_score": 0.4,
            "pre_event_volume_score": 0.5,
            "pre_event_options_score": 0.3,
        }
        result = compute_event_leadup_signal(features)
        flag_texts = " ".join(result["flags"])
        self.assertIn("8-K", flag_texts)

    def test_no_8k_flag_when_event_same_as_earnings(self):
        from insider_alert.signal_engine.event_signal import compute_event_leadup_signal
        features = {
            "days_to_earnings": 5,
            "days_to_corporate_event": 5,
            "pre_event_return_score": 0.0,
            "pre_event_volume_score": 0.0,
            "pre_event_options_score": 0.0,
        }
        result = compute_event_leadup_signal(features)
        flag_texts = " ".join(result["flags"])
        self.assertNotIn("8-K", flag_texts)


# ---------------------------------------------------------------------------
# fetch_recent_corporate_events (offline / structural tests)
# ---------------------------------------------------------------------------

class TestFetchRecentCorporateEventsStructure(unittest.TestCase):
    """Test the structure of the corporate events DataFrame without network calls."""

    def test_returns_dataframe_with_expected_columns(self):
        """The empty fallback DataFrame has the right columns."""
        from insider_alert.data_ingestion.event_data import fetch_recent_corporate_events
        # This test is structural only; the ticker won't resolve without network
        result = fetch_recent_corporate_events("__INVALID_TICKER__9999")
        self.assertIsInstance(result, pd.DataFrame)
        for col in ("date", "form_type", "items", "description"):
            self.assertIn(col, result.columns)

    def test_material_item_codes_constant_has_known_items(self):
        """The material items dict contains key financial event codes."""
        from insider_alert.data_ingestion.event_data import _MATERIAL_8K_ITEMS
        self.assertIn("1.01", _MATERIAL_8K_ITEMS)  # Material Definitive Agreement
        self.assertIn("2.01", _MATERIAL_8K_ITEMS)  # Acquisition/Disposition
        self.assertIn("5.01", _MATERIAL_8K_ITEMS)  # Change of Control


if __name__ == "__main__":
    unittest.main()
