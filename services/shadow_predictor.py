# services/shadow_predictor.py (updated)
"""
Pipeline prediksi paralel (shadow mode) berbasis Probability Fusion.
Fase 1 – Arsitektur Baru:
  - Draw probability dari P_STAR (marginal goal difference)
  - Distribusi Goal Difference terkompresi & exact
  - Top 3 Correct Score dari P_STAR
  - Metadata versi & bobot fusion
"""
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import json
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
    dist = {}
    for h, a, p in score_probs:
        dist[(int(h), int(a))] = float(p)
    return normalize_score_distribution(dist)


def _build_league_distribution(league_profile: Dict[str, float]) -> Dict[Tuple[int, int], float]:
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


def _compute_goal_diff_distribution(P_STAR: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """Distribusi goal difference terkompresi untuk UI."""
    dist = {}
    for (h, a), prob in P_STAR.items():
        diff = h - a
        if diff <= -3:
            key = "-3"
        elif diff >= 3:
            key = "+3"
        else:
            key = f"{diff:+d}"
        dist[key] = dist.get(key, 0.0) + prob
    return dist


def _compute_goal_diff_exact(P_STAR: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """Distribusi goal difference lengkap (untuk Handicap)."""
    exact = {}
    for (h, a), prob in P_STAR.items():
        diff = h - a
        exact[str(diff)] = exact.get(str(diff), 0.0) + prob
    return exact


def _top3_correct_scores(P_STAR: Dict[Tuple[int, int], float]) -> List[Tuple[int, int, float]]:
    sorted_scores = sorted(P_STAR.items(), key=lambda x: x[1], reverse=True)
    return [(int(h), int(a), float(p)) for (h, a), p in sorted_scores[:3]]


def compute_shadow_prediction(
    r: Dict[str, Any],
    df: pd.Series,
    odds_1x2_dict: Optional[Dict[str, float]],
    odds_dict: Optional[Dict[str, float]],
    league_profile_dict: Dict[str, float],
    storage: Any,
) -> Dict[str, Any]:
    # 1. P_MODEL
    score_probs = r.get('score_probs')
    if not score_probs:
        raise ValueError("score_probs tidak tersedia di prediction result")
    P_MODEL = _build_model_distribution(score_probs)

    # 2. P_MARKET
    P_MARKET = None
    if odds_dict:
        P_CS = de_vig_correct_score(
            odds_dict,
            method='poisson_tail',
            model_score_probs=score_probs,
        )
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

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    P_STAR = fuse_score_distributions(distributions, weights)

    # 5. Turunkan probabilitas
    ou_line = float(df.get('current_ou', 2.5))
    # APPROXIMATION UNTUK DISPLAY — EV production memerlukan settlement states penuh
    shadow_prob_over = prob_over(P_STAR, ou_line)
    shadow_prob_under = prob_under(P_STAR, ou_line)

    marg_1x2 = marginalize(P_STAR, '1x2')
    shadow_prob_home = marg_1x2['home']
    shadow_prob_draw = marg_1x2['draw']
    shadow_prob_away = marg_1x2['away']

    marg_btts = marginalize(P_STAR, 'btts')
    shadow_prob_btts = marg_btts['yes']

    goal_diff_dist = _compute_goal_diff_distribution(P_STAR)
    goal_diff_exact = _compute_goal_diff_exact(P_STAR)
    top3 = _top3_correct_scores(P_STAR)

    return {
        'shadow_prob_home': shadow_prob_home,
        'shadow_prob_draw': shadow_prob_draw,
        'shadow_prob_away': shadow_prob_away,
        'shadow_prob_over': shadow_prob_over,
        'shadow_prob_under': shadow_prob_under,
        'shadow_prob_btts': shadow_prob_btts,
        'shadow_goal_diff_distribution': goal_diff_dist,  # kompatibel dengan UI
        'shadow_goal_diff_exact': goal_diff_exact,        # untuk Handicap
        'shadow_top3_scores': top3,
        # Metadata tambahan
        'shadow_prob_1x2_home': shadow_prob_home,
        'shadow_prob_1x2_draw': shadow_prob_draw,
        'shadow_prob_1x2_away': shadow_prob_away,
        'shadow_prob_btts_yes': shadow_prob_btts,
        'shadow_prob_btts_no': 1.0 - shadow_prob_btts,
        'fusion_weights': weights,
        'fusion_version': '1.0.0',
    }
