# services/market_reconciliation.py
"""
Rekonsiliasi distribusi skor dari odds Correct Score yang tidak lengkap
dengan memperhitungkan tail events dan konsistensi marginal 1X2.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import poisson
from services.probability_fusion import normalize_score_distribution, MAX_GOALS


def de_vig_correct_score(
    cs_odds: Dict[str, float],
    method: str = 'basic',
    model_score_probs: Optional[List[Tuple[int, int, float]]] = None,
    expected_goals: Optional[Tuple[float, float]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Konversi odds Correct Score ke distribusi probabilitas lengkap (0-0 hingga 7-7).

    Args:
        cs_odds: dictionary mapping "h:a" ke odds (misal {"1:0": 8.2, ...})
        method: 'basic' atau 'poisson_tail'
        model_score_probs: list of (h,a,prob) dari model, untuk fallback tail
        expected_goals: (home_exp, away_exp) untuk Poisson tail jika model tidak tersedia

    Returns:
        Dictionary (h,a) -> probabilitas untuk semua skor 0-0 hingga 7-7
    """
    if not cs_odds:
        if model_score_probs:
            return {(int(h), int(a)): p for h, a, p in model_score_probs if h <= MAX_GOALS and a <= MAX_GOALS}
        return {}

    # Konversi odds ke implied probability
    implied = {}
    for key, odds in cs_odds.items():
        if key == "OTHER" or odds is None or odds <= 1.0:
            continue
        parts = key.split(':')
        if len(parts) == 2:
            try:
                h, a = int(parts[0]), int(parts[1])
                if 0 <= h <= MAX_GOALS and 0 <= a <= MAX_GOALS:
                    implied[(h, a)] = 1.0 / odds
            except ValueError:
                continue

    # Jika ada "OTHER", tambahkan ke implied (tapi kita tidak tahu distribusinya)
    if "OTHER" in cs_odds and cs_odds["OTHER"] and cs_odds["OTHER"] > 1.0:
        # OTHER akan didistribusikan ke tail nanti
        pass

    if not implied:
        if model_score_probs:
            return {(int(h), int(a)): p for h, a, p in model_score_probs if h <= MAX_GOALS and a <= MAX_GOALS}
        return {}

    # Normalisasi (menghilangkan overround)
    total_implied = sum(implied.values())
    if total_implied <= 0:
        return {}
    cs_dist = {k: v / total_implied for k, v in implied.items()}

    # Tail treatment
    if method == 'poisson_tail':
        cs_dist = estimate_tail_from_model(cs_dist, model_score_probs, expected_goals)

    return cs_dist


def estimate_tail_from_model(
    cs_dist: Dict[Tuple[int, int], float],
    model_score_probs: Optional[List[Tuple[int, int, float]]] = None,
    expected_goals: Optional[Tuple[float, float]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Isi skor-skor yang tidak ada di pasar (tail) menggunakan model atau Poisson.

    Args:
        cs_dist: distribusi dari pasar CS (hanya skor yang ada odds-nya)
        model_score_probs: list of (h,a,prob) dari model
        expected_goals: (home_exp, away_exp) untuk Poisson fallback
    """
    # Probabilitas total yang sudah terpakai
    used_prob = sum(cs_dist.values())
    remaining_prob = 1.0 - used_prob

    if remaining_prob <= 0:
        # Sudah penuh, normalisasi saja
        return normalize_score_distribution(cs_dist)

    # Buat distribusi tail
    tail_dist = {}
    if model_score_probs:
        # Gunakan model untuk mengisi tail
        total_model_tail = 0.0
        for h, a, p in model_score_probs:
            if h > MAX_GOALS or a > MAX_GOALS:
                continue
            if (h, a) not in cs_dist:
                tail_dist[(int(h), int(a))] = p
                total_model_tail += p
        if total_model_tail > 0:
            # Normalisasi tail
            tail_dist = {k: v / total_model_tail for k, v in tail_dist.items()}
        else:
            # Fallback ke uniform? Tidak, lebih baik ke Poisson
            tail_dist = _poisson_tail(expected_goals, cs_dist)
    elif expected_goals:
        tail_dist = _poisson_tail(expected_goals, cs_dist)
    else:
        # Tidak ada informasi tail, distribusikan secara merata ke semua skor yang belum ada
        all_scores = [(h, a) for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if (h, a) not in cs_dist]
        if all_scores:
            equal_prob = remaining_prob / len(all_scores)
            tail_dist = {score: equal_prob for score in all_scores}

    # Gabungkan
    result = {}
    for k, v in cs_dist.items():
        result[k] = v
    for k, v in tail_dist.items():
        result[k] = result.get(k, 0.0) + remaining_prob * v

    return normalize_score_distribution(result)


def _poisson_tail(
    expected_goals: Optional[Tuple[float, float]],
    exclude_scores: Optional[Dict[Tuple[int, int], float]] = None
) -> Dict[Tuple[int, int], float]:
    """Buat distribusi Poisson untuk tail, mengecualikan skor yang sudah ada."""
    if not expected_goals:
        # Default
        home_exp, away_exp = 1.2, 1.0
    else:
        home_exp, away_exp = expected_goals

    tail = {}
    exclude = set(exclude_scores.keys()) if exclude_scores else set()
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            if (h, a) not in exclude:
                tail[(h, a)] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)

    return normalize_score_distribution(tail)


def reconcile_cs_with_1x2(
    cs_dist: Dict[Tuple[int, int], float],
    fair_1x2: Dict[str, float],
    max_iter: int = 50,
    tolerance: float = 1e-6
) -> Dict[Tuple[int, int], float]:
    """
    Iterative Proportional Fitting (IPF) untuk menyelaraskan distribusi CS dengan marginal 1X2.

    Args:
        cs_dist: distribusi dari Correct Score (mungkin sudah lengkap)
        fair_1x2: dictionary {"home": prob, "draw": prob, "away": prob}
        max_iter: iterasi maksimum
        tolerance: konvergensi jika perubahan total < tolerance

    Returns:
        Distribusi yang sudah disesuaikan dengan target marginal 1X2
    """
    target = {
        'home': fair_1x2.get('home', 0.0),
        'draw': fair_1x2.get('draw', 0.0),
        'away': fair_1x2.get('away', 0.0),
    }

    # Normalisasi target jika belum
    t_total = sum(target.values())
    if t_total > 0:
        target = {k: v / t_total for k, v in target.items()}

    current = normalize_score_distribution(cs_dist)

    for _ in range(max_iter):
        # Hitung marginal 1X2 saat ini
        marg = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
        for (h, a), p in current.items():
            if h > a:
                marg['home'] += p
            elif h == a:
                marg['draw'] += p
            else:
                marg['away'] += p

        # Cek konvergensi
        max_diff = max(abs(marg[k] - target[k]) for k in ['home', 'draw', 'away'])
        if max_diff < tolerance:
            break

        # Hitung faktor koreksi
        factors = {}
        for k in ['home', 'draw', 'away']:
            if marg[k] > 0:
                factors[k] = target[k] / marg[k]
            else:
                factors[k] = 1.0  # Tidak bisa menyesuaikan jika nol, biarkan

        # Sesuaikan setiap sel
        updated = {}
        for (h, a), p in current.items():
            if h > a:
                f = factors['home']
            elif h == a:
                f = factors['draw']
            else:
                f = factors['away']
            updated[(h, a)] = p * f

        current = normalize_score_distribution(updated)

    return current
