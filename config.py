# config.py
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

APP_TITLE = "Football AI V2"
APP_VERSION = "2.3.0"

EXPECTED_FEATURES = [
    'league_code', 'home_xg', 'away_xg', 'home_xga', 'away_xga',
    'last5_home_xg', 'last5_away_xg', 'last5_home_xga', 'last5_away_xga',
    'last5_home_avg_goals', 'last5_away_avg_goals',
    'last5_home_conceded', 'last5_away_conceded',
    'last5_home_over25', 'last5_away_over25',
    'last5_home_btts', 'last5_away_btts',
    'open_over_odds', 'open_under_odds',
    'current_over_odds', 'current_under_odds',
    'open_ou', 'current_ou', 'over_move', 'under_move',
    'xg_ratio_home', 'xg_ratio_away',
    'goal_diff_home', 'goal_diff_away',
    'xg_diff_home', 'xg_diff_away',
    'btts_potential', 'over25_potential',
    'odds_ratio', 'momentum_home', 'momentum_away',
    'xg_interact', 'odds_momentum',
    'league_avg_goals', 'league_over25_pct', 'league_btts_pct', 'league_under35_pct',
    'xghome_x_leagueavg', 'xgaway_x_leagueavg',
    'xgahome_x_leagueavg', 'xgaaway_x_leagueavg',
    'last5_home_xg_x_leagueavg', 'last5_away_xg_x_leagueavg',
    'last5_home_xga_x_leagueavg', 'last5_away_xga_x_leagueavg',
    'ou_line_x_leagueavg', 'over25_x_leaguepct', 'btts_x_leaguepct'
]

def load_league_round_config():
    config_file = BASE_DIR / "league_round_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
        except:
            pass
    return {
        390: {'teams': 20, 'matches_per_round': 10},
        40: {'teams': 16, 'matches_per_round': 8},
        22: {'teams': 16, 'matches_per_round': 8},
        11653: {'teams': 15, 'matches_per_round': 7},
        682: {'teams': 15, 'matches_per_round': 7},
        649: {'teams': 14, 'matches_per_round': 7},
        782: {'teams': 13, 'matches_per_round': 6},
        188: {'teams': 12, 'matches_per_round': 6},
        41: {'teams': 12, 'matches_per_round': 6},
        55: {'teams': 10, 'matches_per_round': 5},
        178: {'teams': 9, 'matches_per_round': 4},
        198: {'teams': 9, 'matches_per_round': 4},
        197: {'teams': 9, 'matches_per_round': 4},
        192: {'teams': 10, 'matches_per_round': 5},
        1275: {'teams': 8, 'matches_per_round': 3},
        20: {'teams': 16, 'matches_per_round': 8},
        46: {'teams': 16, 'matches_per_round': 8},
        16736: {'teams': 13, 'matches_per_round': 6},
        278: {'teams': 16, 'matches_per_round': 8},
        240: {'teams': 15, 'matches_per_round': 7},
        242: {'teams': 28, 'matches_per_round': 14},
        1240: {'teams': 16, 'matches_per_round': 8},
        11539: {'teams': 20, 'matches_per_round': 10},
        352: {'teams': 17, 'matches_per_round': 8},
        1260: {'teams': 9, 'matches_per_round': 4},
        1274: {'teams': 11, 'matches_per_round': 4},
    }