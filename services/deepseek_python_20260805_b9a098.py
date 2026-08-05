"""
Feature engineering untuk Football AI V2.
Menghasilkan fitur tambahan dari data mentah pertandingan.
"""
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    if 'open_over_odds' in cols and 'current_over_odds' in cols:
        df['over_move'] = (df['current_over_odds'] - df['open_over_odds']).round(2)
    if 'open_under_odds' in cols and 'current_under_odds' in cols:
        df['under_move'] = (df['current_under_odds'] - df['open_under_odds']).round(2)
    if 'open_ou' in cols and 'current_ou' in cols:
        df['ou_movement'] = (df['current_ou'] - df['open_ou']).round(2)

    # Perbaikan: gunakan pd.Series sebagai default agar .replace() tidak error
    home_xga = df.get('home_xga', pd.Series(1, index=df.index))
    df['xg_ratio_home'] = (df.get('home_xg', 0) / home_xga.replace(0, np.nan)).fillna(0)

    away_xga = df.get('away_xga', pd.Series(1, index=df.index))
    df['xg_ratio_away'] = (df.get('away_xg', 0) / away_xga.replace(0, np.nan)).fillna(0)

    home_avg = df.get('last5_home_avg_goals', pd.Series(0, index=df.index))
    away_avg = df.get('last5_away_avg_goals', pd.Series(0, index=df.index))
    home_con = df.get('last5_home_conceded', pd.Series(0, index=df.index))
    away_con = df.get('last5_away_conceded', pd.Series(0, index=df.index))
    df['goal_diff_home'] = home_avg - home_con
    df['goal_diff_away'] = away_avg - away_con

    # Perbaikan: gunakan df.get() agar tidak KeyError ketika kolom tidak ada
    df['xg_diff_home'] = df['home_xg'] - df.get('home_xga', pd.Series(0, index=df.index))
    df['xg_diff_away'] = df['away_xg'] - df.get('away_xga', pd.Series(0, index=df.index))

    home_btts = df.get('last5_home_btts', pd.Series(0, index=df.index))
    away_btts = df.get('last5_away_btts', pd.Series(0, index=df.index))
    df['btts_potential'] = (home_btts + away_btts) / 2
    home_over = df.get('last5_home_over25', pd.Series(0, index=df.index))
    away_over = df.get('last5_away_over25', pd.Series(0, index=df.index))
    df['over25_potential'] = (home_over + away_over) / 2

    df['odds_ratio'] = (df['current_over_odds'] / df['current_under_odds'].replace(0, np.nan)).fillna(0)
    df['momentum_home'] = home_avg.fillna(0)
    df['momentum_away'] = away_avg.fillna(0)
    df['xg_interact'] = df['home_xg'] * df['away_xg']
    df['odds_momentum'] = df['over_move'] * df['momentum_home'] if 'over_move' in cols else 0

    interactions = [
        ('home_xg','league_avg_goals','xghome_x_leagueavg'),
        ('away_xg','league_avg_goals','xgaway_x_leagueavg'),
        ('home_xga','league_avg_goals','xgahome_x_leagueavg'),
        ('away_xga','league_avg_goals','xgaaway_x_leagueavg'),
        ('last5_home_xg','league_avg_goals','last5_home_xg_x_leagueavg'),
        ('last5_away_xg','league_avg_goals','last5_away_xg_x_leagueavg'),
        ('last5_home_xga','league_avg_goals','last5_home_xga_x_leagueavg'),
        ('last5_away_xga','league_avg_goals','last5_away_xga_x_leagueavg'),
        ('current_ou','league_avg_goals','ou_line_x_leagueavg'),
        ('last5_home_over25','league_over25_pct','over25_x_leaguepct'),
        ('last5_home_btts','league_btts_pct','btts_x_leaguepct')
    ]
    for c1, c2, name in interactions:
        if c1 in cols and c2 in cols: df[name] = df[c1] * df[c2]
        else: df[name] = 0.0

    df[df.select_dtypes(include=np.number).columns] = df.select_dtypes(include=np.number).fillna(0)
    return df