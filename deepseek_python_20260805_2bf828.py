# test_settlement.py
import pytest
from services.settlement import SettlementEngine
from utils import split_quarter_line


class TestSplitQuarterLine:
    def test_bulat_2_0(self):
        assert split_quarter_line(2.0) == [(2.0, 1.0)]

    def test_bulat_2_5(self):
        assert split_quarter_line(2.5) == [(2.5, 1.0)]

    def test_quarter_2_25(self):
        res = split_quarter_line(2.25)
        assert len(res) == 2
        assert res == [(2.0, 0.5), (2.5, 0.5)]

    def test_quarter_2_75(self):
        res = split_quarter_line(2.75)
        assert len(res) == 2
        assert res == [(2.5, 0.5), (3.0, 0.5)]


class TestSettlementOverUnder:

    def test_over_225_full_win(self):
        row = {
            'current_ou': 2.25,
            'current_over_odds': 1.90,
            'recommendation': 'TARUHAN OVER',
            'prediction': 'OVER 2.25',
            'stake': 100000
        }
        result = SettlementEngine.evaluate(row, 2, 1)  # total 3
        assert result['result'] == 'FULL WIN'
        assert result['profit'] == pytest.approx(90000.0)

    def test_over_225_half_lose(self):
        row = {
            'current_ou': 2.25,
            'current_over_odds': 1.90,
            'recommendation': 'TARUHAN OVER',
            'prediction': 'OVER 2.25',
            'stake': 100000
        }
        result = SettlementEngine.evaluate(row, 1, 1)  # total 2
        assert result['result'] == 'HALF LOSE'
        assert result['profit'] == pytest.approx(-50000.0)

    def test_over_225_full_lose(self):
        row = {
            'current_ou': 2.25,
            'current_over_odds': 1.90,
            'recommendation': 'TARUHAN OVER',
            'prediction': 'OVER 2.25',
            'stake': 100000
        }
        result = SettlementEngine.evaluate(row, 0, 0)  # total 0
        assert result['result'] == 'FULL LOSE'
        assert result['profit'] == pytest.approx(-100000.0)

    def test_under_25_full_win(self):
        row = {
            'current_ou': 2.5,
            'current_under_odds': 1.85,
            'recommendation': 'TARUHAN UNDER',
            'prediction': 'UNDER 2.5',
            'stake': 100000
        }
        result = SettlementEngine.evaluate(row, 1, 1)  # total 2 < 2.5
        assert result['result'] == 'FULL WIN'
        assert result['profit'] == pytest.approx(85000.0)

    def test_over_20_push(self):
        row = {
            'current_ou': 2.0,
            'current_over_odds': 1.90,
            'recommendation': 'TARUHAN OVER',
            'prediction': 'OVER 2.0',
            'stake': 100000
        }
        result = SettlementEngine.evaluate(row, 1, 1)  # total 2 == line
        assert result['result'] == 'PUSH'
        assert result['profit'] == 0.0

    def test_no_bet(self):
        row = {
            'current_ou': 2.5,
            'current_over_odds': 1.90,
            'recommendation': 'NO BET',
            'prediction': 'OVER 2.5',
            'stake': 0
        }
        result = SettlementEngine.evaluate(row, 2, 1)
        assert result['result'] == 'NO BET'
        assert result['profit'] == 0.0