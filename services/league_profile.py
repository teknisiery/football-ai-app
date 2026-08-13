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
        # Pastikan kolom default selalu ada
        for col, default in [
            ('home_win_pct', 0.40),
            ('away_win_pct', 0.30),
            ('draw_pct', 0.30),
            ('home_avg_goals', 1.20),
            ('away_avg_goals', 0.90),
            ('score_combination_distribution', '{}'),
        ]:
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
        'home_win_pct': 0.40, 'away_win_pct': 0.30, 'draw_pct': 0.30,
        'home_avg_goals': 1.20, 'away_avg_goals': 0.90,
        'score_combination_distribution': '{}',
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
    """
    if not storage.exists(ResourceRegistry.LEAGUE_PROFILE):
        return
    profile_df = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)

    mask = profile_df['league_code'] == league_code
    if not mask.any():
        return

    idx = profile_df[mask].index[0]
    row = profile_df.loc[idx]

    total_matches = int(row.get('total_matches', 0) or 0)
    avg_goals = float(row.get('league_avg_goals', 2.5) or 2.5)
    over25_pct = float(row.get('league_over25_pct', 0.5) or 0.5)
    btts_pct = float(row.get('league_btts_pct', 0.5) or 0.5)
    under35_pct = float(row.get('league_under35_pct', 0.7) or 0.7)
    home_win_pct = float(row.get('home_win_pct', 0.40) or 0.40)
    draw_pct = float(row.get('draw_pct', 0.30) or 0.30)
    away_win_pct = float(row.get('away_win_pct', 0.30) or 0.30)
    home_avg_goals = float(row.get('home_avg_goals', 1.20) or 1.20)
    away_avg_goals = float(row.get('away_avg_goals', 0.90) or 0.90)

    # Distribusi kombinasi skor
    raw_comb_dist = row.get('score_combination_distribution', '{}')
    if pd.isna(raw_comb_dist) or raw_comb_dist is None:
        raw_comb_dist = '{}'
    try:
        comb_dist = json.loads(raw_comb_dist)
    except Exception:
        comb_dist = {}

    home_goals = int(match_data.get('home_goals', 0) or 0)
    away_goals = int(match_data.get('away_goals', 0) or 0)
    totalgol_ft = int(match_data.get('totalgol_ft', home_goals + away_goals) or 0)

    is_over25 = 1 if totalgol_ft > 2.5 else 0
    is_btts = 1 if home_goals > 0 and away_goals > 0 else 0
    is_under35 = 1 if totalgol_ft < 3.5 else 0
    is_home_win = 1 if home_goals > away_goals else 0
    is_draw = 1 if home_goals == away_goals else 0
    is_away_win = 1 if home_goals < away_goals else 0

    new_total_matches = total_matches + 1

    new_avg_goals = (avg_goals * total_matches + totalgol_ft) / new_total_matches
    new_over25_pct = (over25_pct * total_matches + is_over25) / new_total_matches
    new_btts_pct = (btts_pct * total_matches + is_btts) / new_total_matches
    new_under35_pct = (under35_pct * total_matches + is_under35) / new_total_matches
    new_home_win_pct = (home_win_pct * total_matches + is_home_win) / new_total_matches
    new_draw_pct = (draw_pct * total_matches + is_draw) / new_total_matches
    new_away_win_pct = (away_win_pct * total_matches + is_away_win) / new_total_matches
    new_home_avg_goals = (home_avg_goals * total_matches + home_goals) / new_total_matches
    new_away_avg_goals = (away_avg_goals * total_matches + away_goals) / new_total_matches

    # Tentukan kunci kombinasi; gabungkan skor 5+ ke "Other"
    if home_goals >= 5 or away_goals >= 5:
        comb_key = "Other"
    elif home_goals > away_goals:
        comb_key = f"{home_goals}:{away_goals}"
    elif home_goals < away_goals:
        comb_key = f"{away_goals}:{home_goals}"
    else:
        comb_key = f"{home_goals}:{away_goals}"

    # Skalakan semua probabilitas lama
    scale_factor = total_matches / new_total_matches
    for key in comb_dist.keys():
        comb_dist[key] = comb_dist[key] * scale_factor

    # Tambahkan kunci baru
    comb_dist[comb_key] = comb_dist.get(comb_key, 0.0) + (1.0 / new_total_matches)

    # Update baris liga
    profile_df.at[idx, 'total_matches'] = new_total_matches
    profile_df.at[idx, 'league_avg_goals'] = new_avg_goals
    profile_df.at[idx, 'league_over25_pct'] = new_over25_pct
    profile_df.at[idx, 'league_btts_pct'] = new_btts_pct
    profile_df.at[idx, 'league_under35_pct'] = new_under35_pct
    profile_df.at[idx, 'home_win_pct'] = new_home_win_pct
    profile_df.at[idx, 'draw_pct'] = new_draw_pct
    profile_df.at[idx, 'away_win_pct'] = new_away_win_pct
    profile_df.at[idx, 'home_avg_goals'] = new_home_avg_goals
    profile_df.at[idx, 'away_avg_goals'] = new_away_avg_goals
    profile_df.at[idx, 'score_combination_distribution'] = json.dumps(comb_dist)

    storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profile_df)

    if session:
        session.invalidate_league_profile_cache()

    history_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'league_code': league_code,
        'league_name': row.get('league_name', 'Unknown'),
        'league_avg_goals': new_avg_goals,
        'league_over25_pct': new_over25_pct,
        'league_btts_pct': new_btts_pct,
        'league_under35_pct': new_under35_pct,
        'total_matches': new_total_matches,
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
        'home_avg_goals': 1.20,
        'away_avg_goals': 0.90,
        'score_combination_distribution': '{}',
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

    import config as cfg
    cfg.LEAGUE_ROUND_CONFIG = load_league_round_config()

    return True, "Liga baru berhasil ditambahkan."
