# services/league_profile.py
"""Manajemen profil liga: load, attach, update, dan penambahan liga baru."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

import config
from config import BASE_DIR, load_league_round_config
from services.resource_registry import ResourceRegistry
from services.storage import StorageProvider


def get_league_profile(
    storage: StorageProvider,
    session: 'SessionManager',  # type: ignore[name-defined] # forward ref
) -> pd.DataFrame:
    """Ambil profil liga dari storage, dengan cache di session."""
    state = session.state
    if state.league_profile_cache is not None:
        return state.league_profile_cache
    if storage.exists(ResourceRegistry.LEAGUE_PROFILE):
        state.league_profile_cache = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
        for col, default in [('home_win_pct', 0.40), ('away_win_pct', 0.30), ('draw_pct', 0.30)]:
            if col not in state.league_profile_cache.columns:
                state.league_profile_cache[col] = default
    else:
        state.league_profile_cache = pd.DataFrame()
    return state.league_profile_cache


def attach_league_profile(
    storage: StorageProvider,
    df: pd.DataFrame,
    session: 'SessionManager',
) -> pd.DataFrame:
    """Gabungkan DataFrame pertandingan dengan profil liga."""
    profile = get_league_profile(storage, session)
    if not profile.empty and 'league_code' in profile.columns and 'league_code' in df.columns:
        df = df.merge(profile, on='league_code', how='left', suffixes=('', '_profile'))
    defaults = {
        'league_avg_goals': 2.5, 'league_over25_pct': 0.5, 'league_btts_pct': 0.5,
        'league_under35_pct': 0.7, 'league_name': 'Unknown League',
        'home_win_pct': 0.40, 'away_win_pct': 0.30, 'draw_pct': 0.30
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)
    return df


def update_league_profile(
    storage: StorageProvider,
    league_code: int,
    session: Optional['SessionManager'] = None,
) -> None:
    """Perbarui profil liga berdasarkan data pertandingan yang sudah selesai."""
    if not storage.exists(ResourceRegistry.DATASET_WITH_GOAL):
        return
    df = storage.load_dataframe(ResourceRegistry.DATASET_WITH_GOAL)
    if 'league_name' in df.columns:
        df['league_name'] = df['league_name'].str.title().str.strip()

    if 'totalgol_ft' not in df.columns:
        if 'home_goals' in df.columns:
            df['totalgol_ft'] = df['home_goals'] + df['away_goals']
        else:
            return
    if 'totalgol_ht' not in df.columns:
        df['totalgol_ht'] = df['home_ht_goals'] + df['away_ht_goals'] if 'home_ht_goals' in df.columns else 0
    df = df.dropna(subset=['totalgol_ft'])
    df_league = df[df['league_code'] == league_code]
    if df_league.empty:
        return

    # Gunakan config.LEAGUE_ROUND_CONFIG agar selalu membaca nilai terbaru
    config_item = config.load_league_round_config().get(league_code)
    if config_item and len(df_league) % config_item['matches_per_round'] != 0:
        return

    df_league['btts'] = ((df_league['home_goals'] > 0) & (df_league['away_goals'] > 0)).astype(int)
    df_league['ht0'] = (df_league['totalgol_ht'] == 0).astype(int)
    total = len(df_league)

    profile_df = (
        storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
        if storage.exists(ResourceRegistry.LEAGUE_PROFILE)
        else pd.DataFrame()
    )

    existing_name = None
    if not profile_df.empty and 'league_code' in profile_df.columns and league_code in profile_df['league_code'].values:
        existing_name = profile_df[profile_df['league_code'] == league_code]['league_name'].iloc[0]

    if existing_name:
        league_name_final = existing_name
    elif 'league_name' in df_league.columns and not str(df_league['league_name'].iloc[0]).startswith('League '):
        league_name_final = df_league['league_name'].iloc[0]
    else:
        league_name_final = f'League {league_code}'

    new_row = {
        'league_code': league_code,
        'league_name': league_name_final,
        'league_avg_goals': df_league['totalgol_ft'].mean(),
        'league_over25_pct': (df_league['totalgol_ft'] > 2.5).mean(),
        'league_btts_pct': df_league['btts'].mean(),
        'league_under35_pct': (df_league['totalgol_ft'] < 3.5).mean(),
        'eg_p25': df_league['totalgol_ft'].quantile(0.25),
        'eg_p75': df_league['totalgol_ft'].quantile(0.75),
        'btts_p25': 0.0, 'btts_p75': 0.0, 'ht0_p25': 0.0, 'ht0_p75': 0.0,
        'ev_over_threshold': 0.01, 'ev_under_threshold': 0.02, 'total_matches': total,
        'btts_threshold': 0.22,
        'home_win_pct': (
            profile_df.loc[profile_df['league_code'] == league_code, 'home_win_pct'].iloc[0]
            if (league_code in profile_df['league_code'].values and 'home_win_pct' in profile_df.columns)
            else 0.40
        ),
        'away_win_pct': (
            profile_df.loc[profile_df['league_code'] == league_code, 'away_win_pct'].iloc[0]
            if (league_code in profile_df['league_code'].values and 'away_win_pct' in profile_df.columns)
            else 0.30
        ),
        'draw_pct': (
            profile_df.loc[profile_df['league_code'] == league_code, 'draw_pct'].iloc[0]
            if (league_code in profile_df['league_code'].values and 'draw_pct' in profile_df.columns)
            else 0.30
        ),
    }

    if league_code not in profile_df['league_code'].values:
        profile_df = pd.concat([profile_df, pd.DataFrame([new_row])])
    else:
        idx = profile_df[profile_df['league_code'] == league_code].index[0]
        for k, v in new_row.items():
            if k != 'btts_threshold' or 'btts_threshold' not in profile_df.columns:
                profile_df.at[idx, k] = v

    storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profile_df)
    if session:
        session.invalidate_league_profile_cache()

    history_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'league_code': league_code,
        'league_name': league_name_final,
        'league_avg_goals': new_row['league_avg_goals'],
        'league_over25_pct': new_row['league_over25_pct'],
        'league_btts_pct': new_row['league_btts_pct'],
        'league_under35_pct': new_row['league_under35_pct'],
        'total_matches': total,
    }
    history_df = pd.DataFrame([history_row])
    if storage.exists(ResourceRegistry.LEAGUE_PROFILE_HISTORY):
        existing_hist = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE_HISTORY)
        history_df = pd.concat([existing_hist, history_df], ignore_index=True)
    storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE_HISTORY, history_df)


def add_new_league(
    league_code: int,
    league_name: str,
    avg_goals: float,
    over25_pct: float,
    btts_pct: float,
    under35_pct: float,
    teams: int,
    matches_per_round: int,
    db_storage: StorageProvider,
    app_storage: StorageProvider,
) -> Tuple[bool, str]:
    """Tambahkan liga baru ke profil dan konfigurasi round."""
    try:
        profil = db_storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
    except Exception:
        profil = pd.DataFrame()

    if league_code in profil['league_code'].values:
        return False, "Kode liga sudah ada di profil."

    new_row = {
        'league_code': league_code,
        'league_name': league_name,
        'league_avg_goals': avg_goals,
        'league_over25_pct': over25_pct,
        'league_btts_pct': btts_pct,
        'league_under35_pct': under35_pct,
        'eg_p25': 0.0, 'eg_p75': 0.0,
        'btts_p25': 0.0, 'btts_p75': 0.0,
        'ht0_p25': 0.0, 'ht0_p75': 0.0,
        'ev_over_threshold': 0.01, 'ev_under_threshold': 0.02,
        'total_matches': 0,
        'btts_threshold': 0.22,
        'home_win_pct': 0.40,
        'away_win_pct': 0.30,
        'draw_pct': 0.30,
    }
    profil = pd.concat([profil, pd.DataFrame([new_row])], ignore_index=True)
    db_storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profil)

    config_file = BASE_DIR / "league_round_config.json"
    config_dict = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                config_dict = json.load(f)
        except json.JSONDecodeError:
            config_dict = {}

    config_dict[str(league_code)] = {'teams': teams, 'matches_per_round': matches_per_round}
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Reload konfigurasi global agar semua modul mendapatkan data terbaru
    import config as cfg
    cfg.LEAGUE_ROUND_CONFIG = load_league_round_config()

    return True, "Liga baru berhasil ditambahkan."
