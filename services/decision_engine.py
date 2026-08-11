# services/decision_engine.py
"""Fungsi-fungsi pengambilan keputusan untuk OU dan 1X2."""
from typing import Dict, Optional, Tuple
from services.match_pnl import apply_1x2_odds_floor


def evaluate_ou_decision(
    prob_over: float,
    ev_over: float,
    ev_under: float,
    ev_th_over: float = 0.01,
    ev_th_under: float = 0.02,
    default_stake: float = 100000.0,
) -> Dict:
    """
    Tentukan rekomendasi Over/Under berdasarkan probabilitas dan EV.

    Returns
    -------
    dict with keys:
        ou_pred (str): "OVER" atau "UNDER"
        recommendation (str): "TARUHAN OVER"/"TARUHAN UNDER"/"NO BET"
        rec_color (str): "a" untuk taruhan, "d" untuk NO BET
        stake (float): 0 jika NO BET
    """
    prob_under = 1.0 - prob_over
    if prob_over >= 0.10 and ev_over > ev_th_over:
        return {
            "ou_pred": "OVER",
            "recommendation": "TARUHAN OVER",
            "rec_color": "a",
            "stake": default_stake,
        }
    elif prob_under >= 0.10 and ev_under > ev_th_under:
        return {
            "ou_pred": "UNDER",
            "recommendation": "TARUHAN UNDER",
            "rec_color": "a",
            "stake": default_stake,
        }
    else:
        return {
            "ou_pred": "OVER" if prob_over >= 0.5 else "UNDER",
            "recommendation": "NO BET",
            "rec_color": "d",
            "stake": 0.0,
        }


def compute_1x2_hybrid_and_ev(
    prob_1x2_model: Dict[str, float],
    prob_1x2_league: Dict[str, float],
    odds_1x2_dict: Optional[Dict[str, float]],
    ev_threshold: float = 0.01,
    target_net_profit: float = 100000.0,
) -> Dict:
    """
    Hitung probabilitas hybrid 1X2, fair odds, EV, dan tentukan rekomendasi taruhan.

    Parameters
    ----------
    prob_1x2_model : dict
        Probabilitas dari model (home, draw, away).
    prob_1x2_league : dict
        Probabilitas dari profil liga (home, draw, away).
    odds_1x2_dict : dict or None
        Odds pasar saat ini {home, draw, away}. Jika None, hanya hitung hybrid tanpa EV.
    ev_threshold : float
        Threshold minimum EV untuk taruhan.
    target_net_profit : float
        Target keuntungan bersih untuk perhitungan stake.

    Returns
    -------
    dict dengan kunci:
        prob_1x2_hybrid_final, fair_odds_1x2, ev_home, ev_draw, ev_away,
        prediction_1x2, stake_1x2
    """
    # Default jika tidak ada odds pasar
    if not odds_1x2_dict:
        if prob_1x2_model and prob_1x2_league:
            hybrid = {
                'home': prob_1x2_model.get('home', 0.4) * prob_1x2_league.get('home', 0.4),
                'draw': prob_1x2_model.get('draw', 0.3) * prob_1x2_league.get('draw', 0.3),
                'away': prob_1x2_model.get('away', 0.3) * prob_1x2_league.get('away', 0.3),
            }
            total = sum(hybrid.values())
            if total > 0:
                prob_1x2_hybrid_final = {k: v / total for k, v in hybrid.items()}
            else:
                prob_1x2_hybrid_final = {'home': 0.4, 'draw': 0.3, 'away': 0.3}
        else:
            prob_1x2_hybrid_final = None

        return {
            "prob_1x2_hybrid_final": prob_1x2_hybrid_final,
            "fair_odds_1x2": None,
            "ev_home": None,
            "ev_draw": None,
            "ev_away": None,
            "prediction_1x2": None,
            "stake_1x2": 0.0,
        }

    # Hitung probabilitas pasar dari odds (implied)
    implied = {}
    for k in ['home', 'draw', 'away']:
        odds = odds_1x2_dict.get(k)
        if odds and odds > 1.0:
            implied[k] = 1.0 / odds
        else:
            implied[k] = 0.0
    total_implied = sum(implied.values())
    if total_implied > 0:
        fair_1x2 = {k: v / total_implied for k, v in implied.items()}
    else:
        fair_1x2 = {'home': 0.33, 'draw': 0.34, 'away': 0.33}

    # Ambil prob model dan liga, beri default jika tidak ada
    model = prob_1x2_model if prob_1x2_model else {'home': 0.4, 'draw': 0.3, 'away': 0.3}
    liga = prob_1x2_league if prob_1x2_league else {'home': 0.4, 'draw': 0.3, 'away': 0.3}
    pasar = fair_1x2

    # Hybrid = model * league * market
    h = model.get('home', 0.4) * liga.get('home', 0.4) * pasar.get('home', 0.33)
    d = model.get('draw', 0.3) * liga.get('draw', 0.3) * pasar.get('draw', 0.33)
    a = model.get('away', 0.3) * liga.get('away', 0.3) * pasar.get('away', 0.33)
    total = h + d + a
    if total > 0:
        prob_1x2_hybrid_final = {
            'home': h / total,
            'draw': d / total,
            'away': a / total,
        }
        fair_odds_1x2 = {
            'home': 1.0 / prob_1x2_hybrid_final['home'] if prob_1x2_hybrid_final['home'] > 0 else None,
            'draw': 1.0 / prob_1x2_hybrid_final['draw'] if prob_1x2_hybrid_final['draw'] > 0 else None,
            'away': 1.0 / prob_1x2_hybrid_final['away'] if prob_1x2_hybrid_final['away'] > 0 else None,
        }
    else:
        prob_1x2_hybrid_final = {'home': 0.4, 'draw': 0.3, 'away': 0.3}
        fair_odds_1x2 = {'home': 2.5, 'draw': 3.333, 'away': 3.333}

    # Hitung EV untuk setiap outcome
    evs = {}
    for outcome in ['home', 'draw', 'away']:
        prob = prob_1x2_hybrid_final.get(outcome, 0)
        odds = odds_1x2_dict.get(outcome)
        if prob and odds and odds > 0:
            evs[outcome] = prob * odds - 1.0
        else:
            evs[outcome] = None

    ev_home = evs['home']
    ev_draw = evs['draw']
    ev_away = evs['away']

    # Pilih outcome dengan EV tertinggi di atas threshold
    candidates = [(outcome.upper(), evs[outcome]) for outcome in ['home', 'draw', 'away']
                  if evs[outcome] is not None and evs[outcome] > ev_threshold]
    if candidates:
        prediction_1x2, best_ev = max(candidates, key=lambda x: x[1])
        outcome_key = prediction_1x2.lower()
        market_odds = odds_1x2_dict.get(outcome_key, 0)
        # Terapkan odds floor dan hitung stake
        prediction_1x2, stake_1x2 = apply_1x2_odds_floor(
            prediction_1x2, market_odds, target_net_profit
        )
    else:
        prediction_1x2 = 'NO BET'
        stake_1x2 = 0.0

    return {
        "prob_1x2_hybrid_final": prob_1x2_hybrid_final,
        "fair_odds_1x2": fair_odds_1x2,
        "ev_home": ev_home,
        "ev_draw": ev_draw,
        "ev_away": ev_away,
        "prediction_1x2": prediction_1x2,
        "stake_1x2": stake_1x2,
    }
