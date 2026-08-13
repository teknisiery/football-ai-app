# services/league_profile.py
"""Manajemen profil liga: load, attach, update inkremental, dan penambahan liga baru."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
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
    match_data: Dict[str, Any],
    session: Optional['SessionManager'] = None,
) -> None:
    """
    Perbarui profil liga secara inkremental berdasarkan satu pertandingan yang baru selesai.

    match_data minimal harus berisi:
        home_goals, away_goals, totalgol_ft
    Opsional:
        home_ht_goals, away_ht_goals, totalgol_ht
    """
    # Muat profil liga
    if not storage.exists(ResourceRegistry.LEAGUE_PROFILE):
        return
    profile_df = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)

    # Cari baris liga
    mask = profile_df['league_code'] == league_code
    if not mask.any():
        return  # Tidak buat otomatis jika liga belum ada

    idx = profile_df[mask].index[0]
    row = profile_df.loc[idx]

    # Ambil nilai saat ini
    total_matches = int(row.get('total_matches', 0) or 0)
    avg_goals = float(row.get('league_avg_goals', 2.5) or 2.5)
    over25_pct = float(row.get('league_over25_pct', 0.5) or 0.5)
    btts_pct = float(row.get('league_btts_pct', 0.5) or 0.5)
    under35_pct = float(row.get('league_under35_pct', 0.7) or 0.7)
    home_win_pct = float(row.get('home_win_pct', 0.40) or 0.40)
    draw_pct = float(row.get('draw_pct', 0.30) or 0.30)
    away_win_pct = float(row.get('away_win_pct', 0.30) or 0.30)

    # Total kejadian lama
    total_goals_old = avg_goals * total_matches
    total_over25_old = over25_pct * total_matches
    total_btts_old = btts_pct * total_matches
    total_under35_old = under35_pct * total_matches
    total_home_win_old = home_win_pct * total_matches
    total_draw_old = draw_pct * total_matches
    total_away_win_old = away_win_pct * total_matches

    # Ambil data pertandingan baru
    home_goals = int(match_data.get('home_goals', 0) or 0)
    away_goals = int(match_data.get('away_goals', 0) or 0)
    totalgol_ft = int(match_data.get('totalgol_ft', home_goals + away_goals) or 0)

    # Indikator pertandingan baru
    is_over25 = 1 if totalgol_ft > 2.5 else 0
    is_btts = 1 if home_goals > 0 and away_goals > 0 else 0
    is_under35 = 1 if totalgol_ft < 3.5 else 0
    home_win = 1 if home_goals > away_goals else 0
    draw = 1 if home_goals == away_goals else 0
    away_win = 1 if home_goals < away_goals else 0

    # Total baru
    total_matches_new = total_matches + 1
    total_goals_new = total_goals_old + totalgol_ft
    total_over25_new = total_over25_old + is_over25
    total_btts_new = total_btts_old + is_btts
    total_under35_new = total_under35_old + is_under35
    total_home_win_new = total_home_win_old + home_win
    total_draw_new = total_draw_old + draw
    total_away_win_new = total_away_win_old + away_win

    # Persentase baru
    avg_goals_new = total_goals_new / total_matches_new if total_matches_new > 0 else avg_goals
    over25_pct_new = total_over25_new / total_matches_new if total_matches_new > 0 else over25_pct
    btts_pct_new = total_btts_new / total_matches_new if total_matches_new > 0 else btts_pct
    under35_pct_new = total_under35_new / total_matches_new if total_matches_new > 0 else under35_pct
    home_win_pct_new = total_home_win_new / total_matches_new if total_matches_new > 0 else home_win_pct
    draw_pct_new = total_draw_new / total_matches_new if total_matches_new > 0 else draw_pct
    away_win_pct_new = total_away_win_new / total_matches_new if total_matches_new > 0 else away_win_pct

    # Update baris liga
    profile_df.at[idx, 'total_matches'] = total_matches_new
    profile_df.at[idx, 'league_avg_goals'] = avg_goals_new
    profile_df.at[idx, 'league_over25_pct'] = over25_pct_new
    profile_df.at[idx, 'league_btts_pct'] = btts_pct_new
    profile_df.at[idx, 'league_under35_pct'] = under35_pct_new
    profile_df.at[idx, 'home_win_pct'] = home_win_pct_new
    profile_df.at[idx, 'draw_pct'] = draw_pct_new
    profile_df.at[idx, 'away_win_pct'] = away_win_pct_new

    # Simpan profil
    storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profile_df)

    # Invalidasi cache session jika ada
    if session:
        session.invalidate_league_profile_cache()

    # Catat history
    history_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'league_code': league_code,
        'league_name': row.get('league_name', 'Unknown'),
        'league_avg_goals': avg_goals_new,
        'league_over25_pct': over25_pct_new,
        'league_btts_pct': btts_pct_new,
        'league_under35_pct': under35_pct_new,
        'total_matches': total_matches_new,
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

    # Reload konfigurasi global
    import config as cfg
    cfg.LEAGUE_ROUND_CONFIG = load_league_round_config()

    return True, "Liga baru berhasil ditambahkan."
