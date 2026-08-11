# services/distribution_analysis.py
"""
Distribusi gol dan analisis perbandingan tiga sumber:
  - Model (score_probs)
  - Pasar Correct Score (fair_probs)
  - Pasar 1X2 + profil liga

Semua fungsi murni, tanpa dependensi Streamlit.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.stats import poisson


def marginal_distribution(
    score_probs: List[Tuple[int, int, float]],
    team: str = 'home'
) -> Dict[int, float]:
    """Probabilitas marginal gol untuk satu tim (0-7) dari distribusi skor model.

    Args:
        score_probs: List of (home_goals, away_goals, probability)
        team: 'home' atau 'away'

    Returns:
        Dictionary {gol: probabilitas} untuk gol 0 sampai 7.
    """
    max_goals = 7
    probs = [0.0] * (max_goals + 1)
    for h, a, p in score_probs:
        if team == 'home':
            g = h
        else:
            g = a
        if 0 <= g <= max_goals:
            probs[g] += p
    total = sum(probs)
    if total > 0:
        probs = [v / total for v in probs]
    return {g: probs[g] for g in range(max_goals + 1)}


def market_marginal_distribution(
    fair_probs: Dict[str, float],
    score_probs: List[Tuple[int, int, float]]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Gabungkan probabilitas pasar Correct Score dengan model sebagai fallback.

    Args:
        fair_probs: Dictionary dari odds pasar (mis. {'1:0': 0.1, '2:1': 0.05, 'OTHER': ...})
        score_probs: List (h, a, prob) dari model.

    Returns:
        (home_dist, away_dist) masing-masing dictionary {gol: probabilitas} untuk 0-7.
    """
    max_goals = 7
    home_total = [0.0] * (max_goals + 1)
    away_total = [0.0] * (max_goals + 1)
    used_scores = set()

    for key, prob in fair_probs.items():
        if key == "OTHER":
            continue
        parts = key.split(':')
        if len(parts) == 2:
            try:
                h = int(parts[0])
                a = int(parts[1])
            except ValueError:
                continue
            if 0 <= h <= max_goals and 0 <= a <= max_goals:
                home_total[h] += prob
                away_total[a] += prob
                used_scores.add((h, a))

    # Fallback ke model untuk skor yang tidak ada di pasar
    for h, a, p in score_probs:
        if (h, a) not in used_scores and 0 <= h <= max_goals and 0 <= a <= max_goals:
            home_total[h] += p
            away_total[a] += p

    # Normalisasi masing-masing
    sum_home = sum(home_total)
    sum_away = sum(away_total)
    if sum_home > 0:
        home_total = [v / sum_home for v in home_total]
    if sum_away > 0:
        away_total = [v / sum_away for v in away_total]

    home_dist = {g: home_total[g] for g in range(max_goals + 1)}
    away_dist = {g: away_total[g] for g in range(max_goals + 1)}
    return home_dist, away_dist


def distribution_from_1x2(
    fair_1x2: Dict[str, float],
    league_profile: Dict[str, float]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Estimasi distribusi gol berdasarkan probabilitas 1X2 dan profil liga.

    Args:
        fair_1x2: {'home': prob, 'draw': prob, 'away': prob}
        league_profile: Dictionary dengan 'league_avg_goals', 'home_win_pct',
                        'away_win_pct', 'draw_pct'.

    Returns:
        (home_dist, away_dist) masing-masing 0-7 gol.
    """
    avg_goals = float(league_profile.get('league_avg_goals', 2.5))
    home_win_pct = float(league_profile.get('home_win_pct', 0.4))
    away_win_pct = float(league_profile.get('away_win_pct', 0.3))
    draw_pct = float(league_profile.get('draw_pct', 0.3))

    # Ekspektasi dasar dari profil liga
    home_exp_base = avg_goals * (home_win_pct + 0.5 * draw_pct)
    away_exp_base = avg_goals * (away_win_pct + 0.5 * draw_pct)

    # Penyesuaian berdasarkan selisih probabilitas pasar vs liga
    market_home = fair_1x2.get('home', home_win_pct)
    market_away = fair_1x2.get('away', away_win_pct)

    delta_home = market_home - home_win_pct
    delta_away = market_away - away_win_pct

    # Skalakan: perubahan 10% prob menghasilkan perubahan sebesar avg_goals/10?
    # Kita gunakan: adjustment = delta * avg_goals (1:1 mapping)
    home_exp = max(0.0, home_exp_base + delta_home * avg_goals)
    away_exp = max(0.0, away_exp_base + delta_away * avg_goals)

    max_goals = 7
    home_dist = {}
    away_dist = {}
    for g in range(max_goals + 1):
        home_dist[g] = poisson.pmf(g, home_exp)
        away_dist[g] = poisson.pmf(g, away_exp)

    # Normalisasi agar total probabilitas = 1 (karena kita potong di 7)
    sum_home = sum(home_dist.values())
    sum_away = sum(away_dist.values())
    if sum_home > 0:
        home_dist = {g: v / sum_home for g, v in home_dist.items()}
    if sum_away > 0:
        away_dist = {g: v / sum_away for g, v in away_dist.items()}

    return home_dist, away_dist


def compare_distributions(
    model_home: Dict[int, float],
    model_away: Dict[int, float],
    market_home: Dict[int, float],
    market_away: Dict[int, float],
    x12_home: Dict[int, float],
    x12_away: Dict[int, float],
) -> Dict[str, Any]:
    """Bandingkan distribusi gol dari tiga sumber.

    Returns:
        Dict dengan:
            'home': perbandingan per kategori gol
            'away': perbandingan per kategori gol
            'significant_diff': list of string
            'consensus_home': distribusi konsensus (rata-rata)
            'consensus_away': distribusi konsensus (rata-rata)
    """
    def cat_key(g):
        if g <= 4:
            return str(g)
        else:
            return "5+"

    categories = ["0", "1", "2", "3", "4", "5+"]

    def build_table(dist_model, dist_market, dist_x12):
        table = []
        for cat in categories:
            if cat == "5+":
                vals = [
                    sum(v for g, v in dist_model.items() if g >= 5),
                    sum(v for g, v in dist_market.items() if g >= 5),
                    sum(v for g, v in dist_x12.items() if g >= 5),
                ]
            else:
                g = int(cat)
                vals = [
                    dist_model.get(g, 0.0),
                    dist_market.get(g, 0.0),
                    dist_x12.get(g, 0.0),
                ]
            table.append({
                'goals': cat,
                'model': round(vals[0], 4),
                'market': round(vals[1], 4),
                '1x2': round(vals[2], 4),
                'max_diff': round(max(vals) - min(vals), 4),
                'significant': (max(vals) - min(vals)) > 0.05,
            })
        return table

    home_table = build_table(model_home, market_home, x12_home)
    away_table = build_table(model_away, market_away, x12_away)

    sig_diffs = []
    for t in home_table:
        if t['significant']:
            sig_diffs.append(f"Home {t['goals']} goals: diff {t['max_diff']:.3f}")
    for t in away_table:
        if t['significant']:
            sig_diffs.append(f"Away {t['goals']} goals: diff {t['max_diff']:.3f}")

    # Konsensus: rata-rata sederhana
    def consensus(d1, d2, d3):
        cons = {}
        for g in range(8):
            v = (d1.get(g, 0.0) + d2.get(g, 0.0) + d3.get(g, 0.0)) / 3.0
            cons[g] = v
        return cons

    cons_home = consensus(model_home, market_home, x12_home)
    cons_away = consensus(model_away, market_away, x12_away)

    return {
        'home': home_table,
        'away': away_table,
        'significant_diff': sig_diffs,
        'consensus_home': cons_home,
        'consensus_away': cons_away,
    }


def enrich_1x2_analysis(
    fair_1x2: Dict[str, float],
    league_profile: Dict[str, float]
) -> Dict[str, Any]:
    """Perkaya analisis 1X2 dengan perbandingan terhadap profil liga.

    Args:
        fair_1x2: Probabilitas implisit pasar (home, draw, away).
        league_profile: Data profil liga (home_win_pct, away_win_pct, draw_pct).

    Returns:
        Dict dengan market_probs, league_probs, selisih, dan flag bias.
    """
    market = {
        'home': fair_1x2.get('home', 0.0),
        'draw': fair_1x2.get('draw', 0.0),
        'away': fair_1x2.get('away', 0.0),
    }
    league = {
        'home': float(league_profile.get('home_win_pct', 0.4)),
        'draw': float(league_profile.get('draw_pct', 0.3)),
        'away': float(league_profile.get('away_win_pct', 0.3)),
    }

    diff = {k: round(market[k] - league[k], 4) for k in ['home', 'draw', 'away']}
    flags = []
    for k in ['home', 'draw', 'away']:
        if abs(diff[k]) > 0.10:
            flags.append(f"Strong Market Bias toward {k.upper()}")

    return {
        'market_probs': market,
        'league_probs': league,
        'diff': diff,
        'flags': flags,
    }


def enrich_ou_analysis(
    prob_over_model: float,
    fair_ou_market: Optional[float],
    league_profile: Dict[str, float]
) -> Dict[str, Any]:
    """Perkaya analisis Over/Under dengan profil liga.

    Args:
        prob_over_model: Probabilitas Over dari model.
        fair_ou_market: Probabilitas Over dari pasar (jika ada).
        league_profile: Profil liga dengan 'league_over25_pct', dll.

    Returns:
        Dict dengan perbandingan dan flag peringatan.
    """
    league_over = float(league_profile.get('league_over25_pct', 0.5))
    result = {
        'model_prob': prob_over_model,
        'league_baseline': league_over,
        'model_league_diff': round(abs(prob_over_model - league_over), 4),
        'model_league_flag': abs(prob_over_model - league_over) > 0.15,
    }
    if fair_ou_market is not None:
        result['market_prob'] = fair_ou_market
        result['market_league_diff'] = round(abs(fair_ou_market - league_over), 4)
        result['market_league_flag'] = abs(fair_ou_market - league_over) > 0.15
    return result


def ou_market_implied_prob(over_odds: float, under_odds: float) -> float:
    """Hitung probabilitas Over implisit dari odds pasar (margin dihapus)."""
    if over_odds <= 1.0 or under_odds <= 1.0:
        return np.nan
    over_implied = 1.0 / over_odds
    under_implied = 1.0 / under_odds
    total = over_implied + under_implied
    if total <= 0:
        return np.nan
    return over_implied / total
