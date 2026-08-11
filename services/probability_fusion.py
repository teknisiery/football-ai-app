# services/probability_fusion.py
"""Penggabungan beberapa distribusi skor menjadi posterior tunggal (logarithmic opinion pool)."""
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

MAX_GOALS = 7
EPS = 1e-12


def normalize_score_distribution(dist: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """Normalisasi dictionary probabilitas skor menjadi total 1."""
    total = sum(dist.values())
    if total <= 0:
        return dist
    return {k: v / total for k, v in dist.items()}


def fuse_score_distributions(
    distributions: List[Dict[Tuple[int, int], float]],
    weights: List[float]
) -> Dict[Tuple[int, int], float]:
    """
    Logarithmic opinion pool untuk menggabungkan beberapa distribusi.
    P*(h,a) ∝ exp( Σ w_i * ln(P_i(h,a) + ε) )
    """
    if not distributions or len(distributions) != len(weights):
        raise ValueError("distributions dan weights harus memiliki panjang yang sama dan tidak kosong")

    # Normalisasi bobot
    w_total = sum(weights)
    if w_total <= 0:
        raise ValueError("Total bobot harus > 0")
    weights = [w / w_total for w in weights]

    # Kumpulkan semua pasangan skor yang muncul di setidaknya satu distribusi
    all_scores = set()
    for d in distributions:
        all_scores.update(d.keys())

    fused = {}
    for h, a in all_scores:
        log_sum = 0.0
        for w, d in zip(weights, distributions):
            p = d.get((h, a), 0.0)
            log_sum += w * math.log(p + EPS)
        fused[(h, a)] = math.exp(log_sum)

    return normalize_score_distribution(fused)


def marginalize(
    dist: Dict[Tuple[int, int], float],
    target: str,
    line: Optional[float] = None,
    max_goals: int = MAX_GOALS
) -> Dict[str, float]:
    """
    Hitung probabilitas marginal dari distribusi skor.

    target bisa: 'home', 'away', 'total', '1x2', 'btts', 'over', 'under'
    Untuk 'over'/'under' memerlukan parameter line.
    """
    if target in ('home', 'away', 'total'):
        result = {str(g): 0.0 for g in range(max_goals + 1)}
    elif target == '1x2':
        result = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    elif target == 'btts':
        result = {'yes': 0.0, 'no': 0.0}
    elif target in ('over', 'under'):
        if line is None:
            raise ValueError(f"Target '{target}' memerlukan parameter line")
        result = {'over': 0.0, 'under': 0.0, 'push': 0.0}
    else:
        raise ValueError(f"Target tidak dikenal: {target}")

    for (h, a), p in dist.items():
        if h > max_goals or a > max_goals:
            continue
        if target == 'home':
            result[str(h)] += p
        elif target == 'away':
            result[str(a)] += p
        elif target == 'total':
            total_g = h + a
            if total_g <= max_goals:
                result[str(total_g)] += p
        elif target == '1x2':
            if h > a:
                result['home'] += p
            elif h == a:
                result['draw'] += p
            else:
                result['away'] += p
        elif target == 'btts':
            if h > 0 and a > 0:
                result['yes'] += p
            else:
                result['no'] += p
        elif target == 'over':
            total_g = h + a
            if total_g > line:
                result['over'] += p
            elif total_g < line:
                result['under'] += p
            else:
                result['push'] += p

    return result


def prob_over(dist: Dict[Tuple[int, int], float], line: float) -> float:
    """
    Probabilitas Over efektif (mengikuti konvensi Asian handicap).
    Untuk line bulat: over + push/2 (push dikembalikan setengah).
    Untuk line quarter: hanya over, tanpa push (karena di model sudah dihitung dengan probabilitas tepat).
    """
    marg = marginalize(dist, 'over', line=line)
    if line % 1 == 0:
        return marg['over'] + 0.5 * marg['push']
    else:
        return marg['over']


def prob_under(dist: Dict[Tuple[int, int], float], line: float) -> float:
    """
    Probabilitas Under efektif.
    Untuk line bulat: under + push/2.
    Untuk line quarter: hanya under.
    """
    marg = marginalize(dist, 'over', line=line)
    if line % 1 == 0:
        return marg['under'] + 0.5 * marg['push']
    else:
        return marg['under']


def apply_btts_evidence(
    dist: Dict[Tuple[int, int], float],
    btts_prob: float,
    weight: float = 0.3
) -> Dict[Tuple[int, int], float]:
    """
    Perbarui distribusi berdasarkan bukti BTTS.
    Untuk setiap sel: prob_baru ∝ prob_lama * ( btts_prob^weight jika btts, else (1-btts_prob)^weight )
    """
    if not 0 < btts_prob < 1:
        raise ValueError("btts_prob harus di antara 0 dan 1 (eksklusif)")
    if not 0 < weight < 1:
        raise ValueError("weight harus di antara 0 dan 1")

    updated = {}
    for (h, a), p in dist.items():
        is_btts = (h > 0 and a > 0)
        factor = btts_prob if is_btts else (1 - btts_prob)
        updated[(h, a)] = p * (factor ** weight)

    return normalize_score_distribution(updated)
