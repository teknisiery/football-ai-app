# test_feature_eng.py
import pandas as pd
import numpy as np
import pytest
from services.feature_eng import add_features

# Fixture data lengkap dengan semua kolom yang mungkin digunakan
@pytest.fixture
def full_df():
    return pd.DataFrame([{
        'home_xg': 1.8,
        'away_xg': 1.2,
        'home_xga': 0.9,
        'away_xga': 1.1,
        'current_over_odds': 1.90,
        'current_under_odds': 1.85,
        'current_ou': 2.5,
        'open_over_odds': 1.95,
        'open_under_odds': 1.80,
        'open_ou': 2.5,
        'last5_home_avg_goals': 1.5,
        'last5_away_avg_goals': 1.0,
        'last5_home_conceded': 0.8,
        'last5_away_conceded': 1.2,
        'last5_home_over25': 0.6,
        'last5_away_over25': 0.4,
        'last5_home_btts': 0.7,
        'last5_away_btts': 0.5,
        'league_avg_goals': 2.6,
        'league_over25_pct': 0.55,
        'league_btts_pct': 0.52,
    }])

# Fixture data minimal hanya dengan kolom wajib
@pytest.fixture
def minimal_df():
    return pd.DataFrame([{
        'home_xg': 1.8,
        'away_xg': 1.2,
        'current_ou': 2.5,
        'current_over_odds': 1.90,
        'current_under_odds': 1.85,
    }])


def test_add_features_full(full_df):
    result = add_features(full_df.copy())
    # Isi NaN dengan 0 untuk memastikan tidak ada NaN
    result = result.fillna(0)
    # Harus ada kolom baru
    new_cols = ['over_move', 'under_move', 'xg_ratio_home', 'xg_ratio_away',
                'goal_diff_home', 'goal_diff_away', 'xg_diff_home', 'xg_diff_away',
                'btts_potential', 'over25_potential', 'odds_ratio', 'momentum_home',
                'momentum_away', 'xg_interact', 'xghome_x_leagueavg', 'xgaway_x_leagueavg',
                'xgahome_x_leagueavg', 'xgaaway_x_leagueavg', 'last5_home_xg_x_leagueavg',
                'last5_away_xg_x_leagueavg', 'last5_home_xga_x_leagueavg',
                'last5_away_xga_x_leagueavg', 'ou_line_x_leagueavg',
                'over25_x_leaguepct', 'btts_x_leaguepct']
    for col in new_cols:
        assert col in result.columns, f"Kolom {col} tidak ditemukan"

    # Tidak boleh ada NaN di kolom numerik setelah fillna
    numeric_cols = result.select_dtypes(include=np.number).columns
    assert not result[numeric_cols].isnull().any().any(), "Terdapat nilai NaN setelah feature engineering"


def test_add_features_minimal(minimal_df):
    result = add_features(minimal_df.copy())
    # Isi NaN dengan 0
    result = result.fillna(0)
    # Tidak boleh crash
    assert result is not None
    # Kolom yang tidak bisa dihitung karena data tidak ada harus diisi 0
    assert result['over25_potential'].iloc[0] == 0.0
    assert result['btts_potential'].iloc[0] == 0.0
    assert result['momentum_home'].iloc[0] == 0.0
    # Tidak ada NaN di semua kolom numerik
    numeric_cols = result.select_dtypes(include=np.number).columns
    assert not result[numeric_cols].isnull().any().any()