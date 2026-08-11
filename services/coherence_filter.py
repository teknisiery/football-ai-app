# services/coherence_filter.py
"""
Market Coherence Filter – deteksi kontradiksi antar pasar.
Versi 1: aturan sederhana untuk menolak sinyal yang bertentangan.
"""

from typing import Dict, Any, Tuple, Optional, List
from services.distribution_analysis import (
    marginal_distribution,
    market_marginal_distribution,
    distribution_from_1x2,
    compare_distributions,
    enrich_1x2_analysis,
    enrich_ou_analysis,
    ou_market_implied_prob,
)


def evaluate_coherence(
    ou_pred: str,
    ou_line: float,
    recommendation_btts: Optional[str],
    prediction_1x2: Optional[str],
    prob_over_model: float,
    score_probs: List[Tuple[int, int, float]],
    fair_probs_cs: Optional[Dict[str, float]],
    fair_1x2: Optional[Dict[str, float]],
    league_profile: Dict[str, Any],
    over_odds: Optional[float] = None,
    under_odds: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Evaluasi koherensi antar sinyal pasar.

    Returns:
        (passed, reason) – passed=True jika semua pemeriksaan lolos.
    """
    reasons: List[str] = []

    # --- Aturan 1: UNDER + BTTS YES = kontradiksi (kecuali line ≤1.5) ---
    if (
        ou_pred == "UNDER"
        and recommendation_btts == "YES"
        and ou_line > 1.5
    ):
        reasons.append(
            "Sinyal pasar bertentangan: OU/UNDER vs BTTS/YES"
        )

    # --- Aturan 2: 1X2 HOME tapi prob_over_model < 0.35 → warning ---
    warning_flags: List[str] = []
    if prediction_1x2 == "HOME" and prob_over_model < 0.35:
        warning_flags.append(
            "1X2 HOME tetapi probabilitas Over sangat rendah (<0.35) – potensi kontradiksi"
        )

    # --- Aturan 3: enrichment flags ---
    # enrich_1x2_analysis
    if fair_1x2 is not None:
        enrich_1x2 = enrich_1x2_analysis(fair_1x2, league_profile)
        for flag in enrich_1x2.get("flags", []):
            reasons.append(flag)

    # enrich_ou_analysis
    market_ou = None
    if over_odds is not None and under_odds is not None:
        market_ou = ou_market_implied_prob(over_odds, under_odds)
    enrich_ou = enrich_ou_analysis(prob_over_model, market_ou, league_profile)
    if enrich_ou.get("model_league_flag"):
        reasons.append(
            "Model‑League Discrepancy pada Over/Under (selisih >15%)"
        )
    if enrich_ou.get("market_league_flag"):
        reasons.append(
            "Market‑League Discrepancy pada Over/Under (selisih >15%)"
        )

    # --- Aturan 4: perbedaan signifikan pada distribusi gol 0 atau 1 ---
    # Hitung distribusi model dan pasar (jika ada)
    model_home = marginal_distribution(score_probs, 'home')
    model_away = marginal_distribution(score_probs, 'away')

    if fair_probs_cs is not None:
        market_home, market_away = market_marginal_distribution(fair_probs_cs, score_probs)
    else:
        market_home, market_away = model_home, model_away  # fallback identik

    if fair_1x2 is not None:
        x12_home, x12_away = distribution_from_1x2(fair_1x2, league_profile)
    else:
        x12_home, x12_away = model_home, model_away

    comp = compare_distributions(model_home, model_away, market_home, market_away, x12_home, x12_away)

    # Cari signifikan pada gol 0 atau 1 (Home/Away)
    for side, table in [("Home", comp["home"]), ("Away", comp["away"])]:
        for row in table:
            if row["goals"] in ("0", "1") and row["significant"]:
                reasons.append(
                    f"Perbedaan distribusi gol {side} {row['goals']} >5% antar sumber → ketidakpastian tinggi"
                )

    # Gabungkan alasan
    if reasons:
        return False, "; ".join(reasons)
    if warning_flags:
        # Hanya warning – tidak menggagalkan, tapi bisa dicatat untuk log
        return True, "Warning: " + "; ".join(warning_flags)
    return True, "OK"
