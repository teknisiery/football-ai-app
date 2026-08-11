# services/shadow_predictor.py
"""
Pipeline prediksi paralel (shadow mode) berbasis Probability Fusion.
Menghasilkan prediksi alternatif tanpa memengaruhi output production.
"""
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from scipy.stats import poisson

from services.probability_fusion import (
    fuse_score_distributions,
    normalize_score_distribution,
    prob_over,
    prob_under,
    marginalize,
)
from services.market_reconciliation import (
    de_vig_correct_score,
    reconcile_cs_with_1x2,
)
from utils import calculate_fair_probs


def _build_model_distribution(score_probs: List[Tuple[int, int, float]]) -> Dict[Tuple[int, int], float]:
    """Ubah list (h,a,prob) model menjadi dictionary ter-normalisasi."""
    dist = {}
    for h, a, p in score_probs:
        dist[(int(h), int(a))] = float(p)
    return normalize_score_distribution(dist)


def _build_league_distribution(league_profile: Dict[str, float]) -> Dict[Tuple[int, int], float]:
    """Buat distribusi Poisson independen Home & Away dari profil liga."""
    avg_goals = float(league_profile.get('league_avg_goals', 2.5))
    home_win_pct = float(league_profile.get('home_win_pct', 0.40))
    away_win_pct = float(league_profile.get('away_win_pct', 0.30))
    draw_pct = float(league_profile.get('draw_pct', 0.30))

    home_exp = avg_goals * (home_win_pct + 0.5 * draw_pct)
    away_exp = avg_goals * (away_win_pct + 0.5 * draw_pct)

    max_goals = 7
    dist = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            dist[(h, a)] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)

    return normalize_score_distribution(dist)


def compute_shadow_prediction(
    r: Dict[str, Any],
    df: pd.Series,
    odds_1x2_dict: Optional[Dict[str, float]],
    odds_dict: Optional[Dict[str, float]],
    league_profile_dict: Dict[str, float],
    storage: Any,
) -> Dict[str, Any]:
    """
    Jalankan pipeline prediksi baru berbasis probability fusion.

    Parameters
    ----------
    r : dict hasil prediksi model saat ini (berisi score_probs, prob_over, dll)
    df : pd.Series data pertandingan yang sudah di-feature-engineer
    odds_1x2_dict : dict | None, odds 1X2 mentah (home/draw/away)
    odds_dict : dict | None, odds Correct Score mentah
    league_profile_dict : dict, statistik liga (league_avg_goals, home_win_pct, dll)
    storage : StorageProvider (tidak digunakan langsung, disediakan untuk konsistensi)

    Returns
    -------
    dict dengan kunci:
        shadow_prob_over, shadow_prob_under, shadow_prob_home,
        shadow_prob_draw, shadow_prob_away, shadow_prob_btts
    """
    # 1. P_MODEL dari score_probs
    score_probs = r.get('score_probs')
    if not score_probs:
        raise ValueError("score_probs tidak tersedia di prediction result")
    P_MODEL = _build_model_distribution(score_probs)

    # 2. P_MARKET
    P_MARKET = None
    if odds_dict:
        # Konversi odds CS ke probabilitas (fair)
        fair_probs_cs = calculate_fair_probs(odds_dict)
        if fair_probs_cs:
            P_CS = de_vig_correct_score(
                odds_dict,  # menggunakan odds mentah, fungsi internal akan konversi
                method='poisson_tail',
                model_score_probs=score_probs,
            )
            # Rekonsiliasi dengan 1X2 jika tersedia
            if odds_1x2_dict and P_CS:
                implied_1x2 = {k: 1.0 / v for k, v in odds_1x2_dict.items() if v and v > 1.0}
                total_implied = sum(implied_1x2.values())
                if total_implied > 0:
                    fair_1x2 = {k: v / total_implied for k, v in implied_1x2.items()}
                    P_MARKET = reconcile_cs_with_1x2(P_CS, fair_1x2)
                else:
                    P_MARKET = P_CS
            else:
                P_MARKET = P_CS
    # Jika P_MARKET masih None, lewati

    # 3. P_LEAGUE
    P_LEAGUE = _build_league_distribution(league_profile_dict)

    # 4. Fusion
    distributions = [P_MODEL]
    weights = [0.55]
    if P_MARKET is not None:
        distributions.append(P_MARKET)
        weights.append(0.30)
    distributions.append(P_LEAGUE)
    weights.append(0.15)

    # Normalisasi bobot sesuai jumlah distribusi yang ada
    # (Jika tidak ada P_MARKET, bobot model dan liga akan menjadi 0.55/(0.55+0.15) dan 0.15/(0.55+0.15))
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    P_STAR = fuse_score_distributions(distributions, weights)

    # 5. Turunkan probabilitas pasar
    ou_line = float(df.get('current_ou', 2.5))
    shadow_prob_over = prob_over(P_STAR, ou_line)
    shadow_prob_under = prob_under(P_STAR, ou_line)

    # 1X2
    marg_1x2 = marginalize(P_STAR, '1x2')
    shadow_prob_home = marg_1x2['home']
    shadow_prob_draw = marg_1x2['draw']
    shadow_prob_away = marg_1x2['away']

    # BTTS
    marg_btts = marginalize(P_STAR, 'btts')
    shadow_prob_btts = marg_btts['yes']

    return {
        'shadow_prob_over': shadow_prob_over,
        'shadow_prob_under': shadow_prob_under,
        'shadow_prob_home': shadow_prob_home,
        'shadow_prob_draw': shadow_prob_draw,
        'shadow_prob_away': shadow_prob_away,
        'shadow_prob_btts': shadow_prob_btts,
    }
