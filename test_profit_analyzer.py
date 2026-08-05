# test_profit_analyzer.py
import pandas as pd
import numpy as np
import json
import pytest
from pathlib import Path
from services.profit_analyzer import (
    compute_detailed_profits,
    compute_profits_by_league,
    save_profit_summary,
    load_profit_summary
)
# Mock streamlit agar fungsi dapat dijalankan tanpa Streamlit runtime
import sys
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

# Fixture data history dummy
@pytest.fixture
def history_df():
    return pd.DataFrame([
        {
            'home_team': 'Team A', 'away_team': 'Team B',
            'league_name': 'Liga 1', 'kickoff_time': pd.Timestamp('2024-01-05 20:00:00'),
            'home_goals': 2, 'away_goals': 1, 'totalgol_ft': 3,
            'current_ou': 2.5, 'current_over_odds': 1.90, 'current_under_odds': 1.85,
            'recommendation': 'TARUHAN OVER', 'prediction': 'OVER 2.5', 'stake': 100000,
        },
        {
            'home_team': 'Team C', 'away_team': 'Team D',
            'league_name': 'Liga 1', 'kickoff_time': pd.Timestamp('2024-02-15 22:00:00'),
            'home_goals': 1, 'away_goals': 1, 'totalgol_ft': 2,
            'current_ou': 2.25, 'current_over_odds': 1.88, 'current_under_odds': 1.92,
            'recommendation': 'TARUHAN OVER', 'prediction': 'OVER 2.25', 'stake': 100000,
        },
        {
            'home_team': 'Team E', 'away_team': 'Team F',
            'league_name': 'Liga 2', 'kickoff_time': pd.Timestamp('2024-02-20 19:30:00'),
            'home_goals': 0, 'away_goals': 0, 'totalgol_ft': 0,
            'current_ou': 2.5, 'current_over_odds': 1.95, 'current_under_odds': 1.80,
            'recommendation': 'TARUHAN UNDER', 'prediction': 'UNDER 2.5', 'stake': 100000,
        },
        {
            'home_team': 'Team G', 'away_team': 'Team H',
            'league_name': 'Liga 2', 'kickoff_time': pd.Timestamp('2024-01-25 21:00:00'),
            'home_goals': 2, 'away_goals': 2, 'totalgol_ft': 4,
            'current_ou': 2.0, 'current_over_odds': 1.90, 'current_under_odds': 1.85,
            'recommendation': 'TARUHAN OVER', 'prediction': 'OVER 2.0', 'stake': 100000,
        }
    ])


def test_compute_detailed_profits(history_df):
    total_profit, profit_df, summary, monthly_data, profit_by_status = compute_detailed_profits(history_df)

    # Total profit harus berupa float
    assert isinstance(total_profit, float)
    # Summary harus memiliki kunci yang diharapkan
    for key in ['total_bets', 'full_win', 'half_win', 'push', 'half_lose', 'full_lose']:
        assert key in summary

    # Monthly data harus ada (Januari dan Februari)
    assert len(monthly_data) == 2  # Jan 2024 dan Feb 2024
    assert '2024-01' in monthly_data
    assert '2024-02' in monthly_data


def test_compute_profits_by_league(history_df):
    league_data = compute_profits_by_league(history_df)
    # Harus mengelompokkan 2 liga
    assert len(league_data) == 2
    assert 'Liga 1' in league_data
    assert 'Liga 2' in league_data
    # Setiap liga punya profit dan summary
    for league, data in league_data.items():
        assert 'profit' in data
        assert 'summary' in data
        assert 'total_bets' in data['summary']


def test_save_and_load_profit_summary(tmp_path, monkeypatch):
    # Override PROFIT_SUMMARY_FILE untuk menggunakan tmp_path
    import services.profit_analyzer as pa
    monkeypatch.setattr(pa, 'PROFIT_SUMMARY_FILE', tmp_path / "profit_summary.json")

    dummy = {
        'Liga A': {'profit': 150000.0, 'summary': {'total_bets': 10, 'full_win': 5, 'half_win': 2,
                                                    'push': 1, 'half_lose': 1, 'full_lose': 1}},
        'Liga B': {'profit': -50000.0, 'summary': {'total_bets': 8, 'full_win': 3, 'half_win': 1,
                                                    'push': 0, 'half_lose': 2, 'full_lose': 2}}
    }
    save_profit_summary(dummy)
    loaded = load_profit_summary()
    assert loaded == dummy
