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

    Perbaikan: quoted scores tidak dinormalisasi penuh dulu.
    Hitung quoted_mass, distribusikan tail, baru normalisasi akhir.
    """
    if not cs_odds:
        if model_score_probs:
            return {(int(h), int(a)): p for h, a, p in model_score_probs if h <= MAX_GOALS and a <= MAX_GOALS}
        return {}

    # Konversi odds ke implied probability (belum normalisasi)
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

    if not implied:
        if model_score_probs:
            return {(int(h), int(a)): p for h, a, p in model_score_probs if h <= MAX_GOALS and a <= MAX_GOALS}
        return {}

    # Hitung total implied (overround)
    total_implied = sum(implied.values())
    if total_implied <= 0:
        return {}

    # Probabilitas quoted (belum normalisasi total)
    quoted_mass = sum(implied.values()) / total_implied  # akan 1.0
    # Sebenarnya kita ingin quoted_mass = sum(implied) / total_implied = 1.0, jadi sisa 0.
    # Tetapi kita harus menghitung quoted sebagai proporsi dari total_implied.
    # Cara yang benar: quoted_prob = {k: v / total_implied for k,v in implied.items()}
    quoted_prob = {k: v / total_implied for k, v in implied.items()}
    quoted_mass = sum(quoted_prob.values())  # ≈ 1.0

    remaining_prob = 1.0 - quoted_mass

    if method == 'poisson_tail' and remaining_prob > 0:
        tail_dist = estimate_tail_from_model(
            quoted_prob,        # distribusi quoted (belum final)
            model_score_probs,
            expected_goals,
            exclude_scores=set(quoted_prob.keys())
        )
        # Gabungkan quoted + tail
        result = {}
        for k, v in quoted_prob.items():
            result[k] = v
        for k, v in tail_dist.items():
            result[k] = result.get(k, 0.0) + remaining_prob * v
        return normalize_score_distribution(result)
    else:
        # Tanpa tail, langsung normalisasi (tidak ada perubahan)
        return normalize_score_distribution(quoted_prob)


def estimate_tail_from_model(
    cs_dist: Dict[Tuple[int, int], float],
    model_score_probs: Optional[List[Tuple[int, int, float]]] = None,
    expected_goals: Optional[Tuple[float, float]] = None,
    exclude_scores: Optional[set] = None
) -> Dict[Tuple[int, int], float]:
    """
    Isi skor-skor yang tidak ada di pasar (tail) menggunakan model atau Poisson.
    """
    exclude = exclude_scores or set(cs_dist.keys())
    remaining_prob = 1.0 - sum(cs_dist.values())

    if remaining_prob <= 0:
        return normalize_score_distribution(cs_dist)

    tail_dist = {}
    if model_score_probs:
        total_model_tail = 0.0
        for h, a, p in model_score_probs:
            if h > MAX_GOALS or a > MAX_GOALS:
                continue
            if (h, a) not in exclude:
                tail_dist[(int(h), int(a))] = p
                total_model_tail += p
        if total_model_tail > 0:
            tail_dist = {k: v / total_model_tail for k, v in tail_dist.items()}
        else:
            tail_dist = _poisson_tail(expected_goals, exclude)
    elif expected_goals:
        tail_dist = _poisson_tail(expected_goals, exclude)
    else:
        # Fallback seragam
        all_scores = [(h, a) for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if (h, a) not in exclude]
        if all_scores:
            equal_prob = remaining_prob / len(all_scores)
            tail_dist = {score: equal_prob for score in all_scores}

    return normalize_score_distribution(tail_dist)


def _poisson_tail(
    expected_goals: Optional[Tuple[float, float]],
    exclude_scores: set
) -> Dict[Tuple[int, int], float]:
    if not expected_goals:
        home_exp, away_exp = 1.2, 1.0
    else:
        home_exp, away_exp = expected_goals
    tail = {}
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            if (h, a) not in exclude_scores:
                tail[(h, a)] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)
    return normalize_score_distribution(tail)


def reconcile_cs_with_1x2(
    cs_dist: Dict[Tuple[int, int], float],
    fair_1x2: Dict[str, float],
    max_iter: int = 50,
    tolerance: float = 1e-6
) -> Dict[Tuple[int, int], float]:
    """
    EXPERIMENTAL BASELINE — final design akan menggunakan soft KL projection.
    IPF untuk menyelaraskan distribusi CS dengan marginal 1X2.
    """
    target = {
        'home': fair_1x2.get('home', 0.0),
        'draw': fair_1x2.get('draw', 0.0),
        'away': fair_1x2.get('away', 0.0),
    }
    t_total = sum(target.values())
    if t_total > 0:
        target = {k: v / t_total for k, v in target.items()}

    current = normalize_score_distribution(cs_dist)

    for _ in range(max_iter):
        marg = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
        for (h, a), p in current.items():
            if h > a:
                marg['home'] += p
            elif h == a:
                marg['draw'] += p
            else:
                marg['away'] += p

        max_diff = max(abs(marg[k] - target[k]) for k in ['home', 'draw', 'away'])
        if max_diff < tolerance:
            break

        factors = {}
        for k in ['home', 'draw', 'away']:
            if marg[k] > 0:
                factors[k] = target[k] / marg[k]
            else:
                factors[k] = 1.0

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
