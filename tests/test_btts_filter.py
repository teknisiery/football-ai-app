# tests/test_btts_filter.py
import pytest
from services.btts_filter import evaluate_btts_filter, BTTS_FILTER_VERSION

class TestBttsFilter:
    # Helper untuk pemanggilan standar
    def call_filter(self, prob, market_yes, market_no, ev_yes, ev_no):
        return evaluate_btts_filter(prob, market_yes, market_no, ev_yes, ev_no)

    # 1. Normal value: kedua odds >= 1.35, EV YES > 0.05, EV NO rendah -> YES
    def test_normal_value_yes(self):
        res = self.call_filter(0.55, 1.90, 2.10, 0.12, -0.02)
        assert res["recommendation"] == "YES"
        assert res["stake_side"] == "YES"
        assert res["reason"] == "NORMAL_VALUE"
        assert res["filtered"] == False

    def test_normal_value_no(self):
        res = self.call_filter(0.45, 2.10, 1.95, -0.01, 0.08)
        assert res["recommendation"] == "NO"
        assert res["reason"] == "NORMAL_VALUE"

    def test_neutral_ev_below_threshold(self):
        res = self.call_filter(0.55, 1.90, 2.10, 0.03, 0.02)
        assert res["recommendation"] == "NO BET"
        assert res["reason"] == "NEUTRAL_OR_EV_TOO_LOW"
        assert res["filtered"] == True

    # 2. Low odds: YES odds < 1.35, NO odds normal, EV NO > 0.30 -> bet NO
    def test_low_odds_yes_opposite_value(self):
        res = self.call_filter(0.70, 1.30, 2.20, 0.50, 0.35)  # ev_no = 0.35 > 0.30
        assert res["recommendation"] == "NO"
        assert res["reason"] == "LOW_ODDS_OPPOSITE_VALUE"
        assert res["filtered"] == True

    def test_low_odds_yes_opposite_ev_too_low(self):
        res = self.call_filter(0.70, 1.30, 2.20, 0.50, 0.25)
        assert res["recommendation"] == "NO BET"
        assert res["reason"] == "LOW_ODDS_OPPOSITE_EV_TOO_LOW"
        assert res["filtered"] == True

    def test_low_odds_no_opposite_value(self):
        res = self.call_filter(0.30, 2.50, 1.25, 0.40, 0.10)
        assert res["recommendation"] == "YES"
        assert res["reason"] == "LOW_ODDS_OPPOSITE_VALUE"
        assert res["filtered"] == True

    def test_low_odds_no_opposite_ev_too_low(self):
        res = self.call_filter(0.30, 2.50, 1.25, 0.25, 0.10)
        assert res["recommendation"] == "NO BET"
        assert res["reason"] == "LOW_ODDS_OPPOSITE_EV_TOO_LOW"

    # 3. Both low odds
    def test_both_low_odds(self):
        res = self.call_filter(0.55, 1.30, 1.25, 0.60, 0.55)
        assert res["recommendation"] == "NO BET"
        assert res["reason"] == "BOTH_SIDES_LOW_ODDS"
        assert res["filtered"] == True

    # 4. Edge cases: None values
    def test_none_prob(self):
        res = evaluate_btts_filter(None, 1.90, 2.10, 0.12, 0.08)
        assert res["recommendation"] == "NO BET"
        assert res["reason"] == "NO_POSITIVE_VALUE"

    def test_missing_odds_and_ev(self):
        res = evaluate_btts_filter(0.55, None, None, None, None)
        assert res["recommendation"] == "NO BET"
        assert res["reason"] == "NO_POSITIVE_VALUE"

    def test_version_included(self):
        res = evaluate_btts_filter(0.55, 1.90, 2.10, 0.12, 0.08)
        assert res["version"] == BTTS_FILTER_VERSION
