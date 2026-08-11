# services/live_bet.py
"""Kalkulasi rekomendasi live betting untuk Football AI V2."""
from typing import Dict, Any
from scipy.stats import poisson
from utils import calc_kelly
from services.resource_registry import ResourceRegistry
from app import ThresholdService  # tetap gunakan ThresholdService dari app.py untuk sementara


def calculate_live_recommendation(
    lam_total: float,
    home_xg: float,
    away_xg: float,
    menit_berjalan: float,
    home_goals_live: int,
    away_goals_live: int,
    current_ou: float,
    current_over_odds: float,
    current_under_odds: float,
    storage,
    max_goals: int = 7,
) -> Dict[str, Any]:
    """Hitung rekomendasi live betting berdasarkan xG dan menit berjalan."""
    total_xg = home_xg + away_xg
    if total_xg > 0:
        lam_home = lam_total * (home_xg / total_xg)
        lam_away = lam_total * (away_xg / total_xg)
    else:
        lam_home = lam_total / 2
        lam_away = lam_total / 2

    menit_tersisa = max(0.0, 90.0 - menit_berjalan)
    lam_home_sisa = lam_home * (menit_tersisa / 90.0)
    lam_away_sisa = lam_away * (menit_tersisa / 90.0)

    # Momentum adjustment
    faktor_home = 1.0
    faktor_away = 1.0
    if home_goals_live < away_goals_live:
        if home_xg > away_xg:
            faktor_home, faktor_away = 1.15, 0.95
        else:
            faktor_home, faktor_away = 0.95, 1.10
    elif home_goals_live > away_goals_live:
        if home_xg < away_xg:
            faktor_home, faktor_away = 0.90, 1.10
        else:
            faktor_home, faktor_away = 1.0, 0.95
    lam_home_adj = lam_home_sisa * faktor_home
    lam_away_adj = lam_away_sisa * faktor_away

    rho = -0.1
    prob_over_win = 0.0
    prob_under_win = 0.0
    prob_push = 0.0
    for h in range(0, max_goals + 1):
        for a in range(0, max_goals + 1):
            prob = poisson.pmf(h, lam_home_adj) * poisson.pmf(a, lam_away_adj)
            if h == 0 and a == 0:
                prob *= (1 - lam_home_adj * lam_away_adj * rho)
            elif h == 1 and a == 0:
                prob *= (1 + lam_away_adj * rho)
            elif h == 0 and a == 1:
                prob *= (1 + lam_home_adj * rho)
            elif h == 1 and a == 1:
                prob *= (1 - rho)
            total_gol = h + a
            if total_gol > current_ou:
                prob_over_win += prob
            elif total_gol < current_ou:
                prob_under_win += prob
            else:
                prob_push += prob

    prob_over = prob_over_win
    if current_ou % 1 == 0:
        ev_over = (prob_over_win * current_over_odds) + (prob_push * 1) - 1
        ev_under = (prob_under_win * current_under_odds) + (prob_push * 1) - 1
    else:
        ev_over = (prob_over_win * current_over_odds) - 1
        ev_under = (prob_under_win * current_under_odds) - 1

    kelly_over = calc_kelly(prob_over, current_over_odds)
    kelly_under = calc_kelly(1 - prob_over, current_under_odds)

    ev_th_over, ev_th_under = ThresholdService.get_thresholds(storage)
    if prob_over >= 0.10 and ev_over > ev_th_over:
        rec = "OVER"
    elif (1 - prob_over) >= 0.10 and ev_under > ev_th_under:
        rec = "UNDER"
    else:
        rec = "NO BET"

    prob_home_score = 1 - poisson.pmf(0, lam_home_adj)
    prob_away_score = 1 - poisson.pmf(0, lam_away_adj)
    prob_btts_yes = prob_home_score * prob_away_score
    confidence_btts = max(prob_btts_yes, 1 - prob_btts_yes)

    return {
        "prob_over": prob_over,
        "ev_over": ev_over,
        "ev_under": ev_under,
        "kelly_over": kelly_over,
        "kelly_under": kelly_under,
        "recommendation": rec,
        "prob_btts_yes": prob_btts_yes,
        "btts_pred": "YES" if prob_btts_yes >= 0.5 else "NO",
        "confidence_btts": confidence_btts,
    }
