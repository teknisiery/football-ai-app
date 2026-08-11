# services/shadow_evaluator.py
"""
Alat evaluasi offline untuk membandingkan performa sistem Production (hybrid)
vs Shadow (Probability Fusion) menggunakan data historis dari history_ou.csv.

Tidak terintegrasi dengan app.py atau UI. Digunakan secara manual/offline.
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

EPSILON = 1e-15
PROB_SUM_TOLERANCE = 0.01  # toleransi penjumlahan probabilitas 1X2


def _clip(p: float) -> float:
    return min(max(p, EPSILON), 1.0 - EPSILON)


def _binary_brier(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(np.mean((p_pred - y_true) ** 2))


def _binary_logloss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    p_clipped = np.clip(p_pred, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(y_true * np.log(p_clipped) + (1 - y_true) * np.log(1.0 - p_clipped)))


def _multiclass_brier(y_true_home: np.ndarray, y_true_draw: np.ndarray, y_true_away: np.ndarray,
                      p_home: np.ndarray, p_draw: np.ndarray, p_away: np.ndarray) -> float:
    n = len(y_true_home)
    if n == 0:
        return np.nan
    error = (p_home - y_true_home)**2 + (p_draw - y_true_draw)**2 + (p_away - y_true_away)**2
    return float(np.mean(error))


def _multiclass_logloss(y_true: np.ndarray, p_home: np.ndarray, p_draw: np.ndarray,
                        p_away: np.ndarray) -> float:
    n = len(y_true)
    if n == 0:
        return np.nan
    p_actual = np.where(y_true == 0, p_home, np.where(y_true == 1, p_draw, p_away))
    p_actual = np.clip(p_actual, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(np.log(p_actual)))


def _accuracy(y_true: np.ndarray, p_home: np.ndarray, p_draw: np.ndarray,
              p_away: np.ndarray) -> float:
    n = len(y_true)
    if n == 0:
        return np.nan
    pred = np.argmax(np.column_stack([p_home, p_draw, p_away]), axis=1)
    return float(np.mean(pred == y_true))


def _calibration_buckets(preds: List[float], actuals: List[int]) -> Dict[str, Dict[str, Any]]:
    """Hitung calibration curve untuk probabilitas binary."""
    buckets = {f"{i}0-{i+1}0%": {'count': 0, 'predicted_sum': 0.0, 'actual_sum': 0}
               for i in range(10)}
    for p, y in zip(preds, actuals):
        idx = min(9, int(p * 10))
        key = f"{idx}0-{idx+1}0%"
        buckets[key]['count'] += 1
        buckets[key]['predicted_sum'] += p
        buckets[key]['actual_sum'] += y
    result = {}
    for k, v in buckets.items():
        if v['count'] == 0:
            result[k] = {"count": 0, "predicted": None, "actual": None}
        else:
            result[k] = {
                "count": v['count'],
                "predicted": round(v['predicted_sum'] / v['count'], 6),
                "actual": round(v['actual_sum'] / v['count'], 6)
            }
    return result


def evaluate_shadow_vs_production(history_df: pd.DataFrame) -> dict:
    """
    Evaluasi performa sistem lama (hybrid) vs sistem baru (shadow P*).
    Menggunakan common paired sample untuk perbandingan yang adil.
    """
    # --- Kumpulkan data mentah ---
    # Lists untuk perhitungan paired
    draw_prod_paired = []
    draw_shadow_paired = []
    draw_actual_paired = []

    prod_1x2_home = []
    prod_1x2_draw = []
    prod_1x2_away = []
    shadow_1x2_home = []
    shadow_1x2_draw = []
    shadow_1x2_away = []
    actual_1x2_label = []  # 0=home,1=draw,2=away

    prod_ou_prob = []
    shadow_ou_prob = []
    actual_ou = []

    prod_btts_prob = []
    shadow_btts_prob = []
    actual_btts = []

    # Goal difference calibration (shadow only)
    goal_diff_preds: Dict[str, List[float]] = {k: [] for k in ['-3','-2','-1','0','+1','+2','+3']}
    goal_diff_actual_indicators: Dict[str, List[int]] = {k: [] for k in goal_diff_preds}
    gd_valid_count = 0  # jumlah pertandingan dengan shadow_goal_diff_exact

    # Baseline draw: semua pertandingan settled (ada home_goals & away_goals)
    baseline_draw_actual = []

    # Counts
    total_settled = 0
    missing_shadow_count = 0
    missing_production_prob_count = 0
    invalid_probability_count = 0

    # Loop
    for _, row in history_df.iterrows():
        if pd.isna(row.get('home_goals')) or pd.isna(row.get('away_goals')):
            continue
        home_goals = int(row.get('home_goals', 0) or 0)
        away_goals = int(row.get('away_goals', 0) or 0)
        total_settled += 1

        actual_draw = 1 if home_goals == away_goals else 0
        actual_home = 1 if home_goals > away_goals else 0
        actual_away = 1 if home_goals < away_goals else 0

        # Actual over/under (non-push binary)
        if 'actual_over' in row and not pd.isna(row['actual_over']):
            actual_over_val = int(row['actual_over'])
        else:
            current_ou = row.get('current_ou', None)
            if current_ou is not None and not pd.isna(current_ou):
                total_goals = home_goals + away_goals
                if total_goals > current_ou:
                    actual_over_val = 1
                elif total_goals < current_ou:
                    actual_over_val = 0
                else:
                    actual_over_val = None  # push, diabaikan
            else:
                actual_over_val = None

        actual_btts_val = 1 if (home_goals > 0 and away_goals > 0) else 0

        # Baseline draw: semua settled
        baseline_draw_actual.append(actual_draw)

        # Probabilitas produksi
        prod_home = row.get('prob_1x2_hybrid_home')
        prod_draw = row.get('prob_1x2_hybrid_draw')
        prod_away = row.get('prob_1x2_hybrid_away')
        prod_ou = row.get('prob_over')
        prod_btts = row.get('prob_btts')

        # Shadow probabilitas
        shadow_home = row.get('shadow_prob_home')
        shadow_draw = row.get('shadow_prob_draw')
        shadow_away = row.get('shadow_prob_away')
        shadow_ou = row.get('shadow_prob_over')
        shadow_btts = row.get('shadow_prob_btts')
        if shadow_btts is None or pd.isna(shadow_btts):
            shadow_btts = row.get('shadow_prob_btts_yes')

        has_prod_1x2 = all(not pd.isna(x) for x in [prod_home, prod_draw, prod_away])
        has_shadow_1x2 = all(not pd.isna(x) for x in [shadow_home, shadow_draw, shadow_away])
        has_shadow = has_shadow_1x2  # minimal

        if not has_shadow:
            missing_shadow_count += 1
        if not has_prod_1x2:
            missing_production_prob_count += 1

        # Common paired sample untuk Draw
        if (not pd.isna(prod_draw)) and (not pd.isna(shadow_draw)):
            draw_prod_paired.append(prod_draw)
            draw_shadow_paired.append(shadow_draw)
            draw_actual_paired.append(actual_draw)

        # Common paired untuk 1X2
        if has_prod_1x2 and has_shadow_1x2:
            # Validasi penjumlahan ~1
            prod_sum = prod_home + prod_draw + prod_away
            shadow_sum = shadow_home + shadow_draw + shadow_away
            if abs(prod_sum - 1.0) > PROB_SUM_TOLERANCE or abs(shadow_sum - 1.0) > PROB_SUM_TOLERANCE:
                invalid_probability_count += 1
            else:
                prod_1x2_home.append(prod_home)
                prod_1x2_draw.append(prod_draw)
                prod_1x2_away.append(prod_away)
                shadow_1x2_home.append(shadow_home)
                shadow_1x2_draw.append(shadow_draw)
                shadow_1x2_away.append(shadow_away)
                if actual_home == 1:
                    actual_1x2_label.append(0)
                elif actual_draw == 1:
                    actual_1x2_label.append(1)
                else:
                    actual_1x2_label.append(2)

        # Common paired untuk OU
        if (not pd.isna(prod_ou)) and (not pd.isna(shadow_ou)) and actual_over_val is not None:
            prod_ou_prob.append(prod_ou)
            shadow_ou_prob.append(shadow_ou)
            actual_ou.append(actual_over_val)

        # Common paired untuk BTTS
        if (not pd.isna(prod_btts)) and (not pd.isna(shadow_btts)):
            prod_btts_prob.append(prod_btts)
            shadow_btts_prob.append(shadow_btts)
            actual_btts.append(actual_btts_val)

        # Goal Difference calibration (shadow only)
        if has_shadow_1x2 and 'shadow_goal_diff_exact' in row and not pd.isna(row['shadow_goal_diff_exact']):
            try:
                gd_exact = json.loads(row['shadow_goal_diff_exact'])
                # gd_exact adalah dict str->prob, misal {"-3": 0.05, ...}
                gd_dict = {int(k): v for k, v in gd_exact.items() if v is not None}
            except:
                gd_dict = {}
            actual_gd = home_goals - away_goals
            gd_valid_count += 1

            # Hitung probabilitas untuk setiap bucket
            # -3: P(GD <= -3)
            prob_minus3 = sum(v for k, v in gd_dict.items() if k <= -3)
            # +3: P(GD >= 3)
            prob_plus3 = sum(v for k, v in gd_dict.items() if k >= 3)
            # lainnya: langsung
            prob_minus2 = gd_dict.get(-2, 0.0)
            prob_minus1 = gd_dict.get(-1, 0.0)
            prob_0 = gd_dict.get(0, 0.0)
            prob_plus1 = gd_dict.get(1, 0.0)
            prob_plus2 = gd_dict.get(2, 0.0)

            goal_diff_preds['-3'].append(prob_minus3)
            goal_diff_preds['-2'].append(prob_minus2)
            goal_diff_preds['-1'].append(prob_minus1)
            goal_diff_preds['0'].append(prob_0)
            goal_diff_preds['+1'].append(prob_plus1)
            goal_diff_preds['+2'].append(prob_plus2)
            goal_diff_preds['+3'].append(prob_plus3)

            # Indicator aktual untuk setiap bucket
            ind_minus3 = 1 if actual_gd <= -3 else 0
            ind_minus2 = 1 if actual_gd == -2 else 0
            ind_minus1 = 1 if actual_gd == -1 else 0
            ind_0 = 1 if actual_gd == 0 else 0
            ind_plus1 = 1 if actual_gd == 1 else 0
            ind_plus2 = 1 if actual_gd == 2 else 0
            ind_plus3 = 1 if actual_gd >= 3 else 0

            goal_diff_actual_indicators['-3'].append(ind_minus3)
            goal_diff_actual_indicators['-2'].append(ind_minus2)
            goal_diff_actual_indicators['-1'].append(ind_minus1)
            goal_diff_actual_indicators['0'].append(ind_0)
            goal_diff_actual_indicators['+1'].append(ind_plus1)
            goal_diff_actual_indicators['+2'].append(ind_plus2)
            goal_diff_actual_indicators['+3'].append(ind_plus3)

    # --- Hitung metrik global ---
    # Baseline draw
    baseline_draw_prob = float(np.mean(baseline_draw_actual)) if baseline_draw_actual else 0.0
    baseline_draw_brier = _binary_brier(np.array(baseline_draw_actual),
                                        np.full(len(baseline_draw_actual), baseline_draw_prob))
    baseline_draw_logloss = _binary_logloss(np.array(baseline_draw_actual),
                                            np.full(len(baseline_draw_actual), baseline_draw_prob))

    # Draw
    prod_draw_arr = np.array(draw_prod_paired)
    shadow_draw_arr = np.array(draw_shadow_paired)
    actual_draw_arr = np.array(draw_actual_paired)
    prod_draw_brier = _binary_brier(actual_draw_arr, prod_draw_arr)
    prod_draw_logloss = _binary_logloss(actual_draw_arr, prod_draw_arr)
    shadow_draw_brier = _binary_brier(actual_draw_arr, shadow_draw_arr)
    shadow_draw_logloss = _binary_logloss(actual_draw_arr, shadow_draw_arr)

    # 1X2
    prod_home_arr = np.array(prod_1x2_home)
    prod_draw_arr1x2 = np.array(prod_1x2_draw)
    prod_away_arr = np.array(prod_1x2_away)
    shadow_home_arr = np.array(shadow_1x2_home)
    shadow_draw_arr1x2 = np.array(shadow_1x2_draw)
    shadow_away_arr = np.array(shadow_1x2_away)
    actual_1x2_arr = np.array(actual_1x2_label)

    prod_1x2_logloss = _multiclass_logloss(actual_1x2_arr, prod_home_arr, prod_draw_arr1x2, prod_away_arr)
    shadow_1x2_logloss = _multiclass_logloss(actual_1x2_arr, shadow_home_arr, shadow_draw_arr1x2, shadow_away_arr)

    # Brier 1X2
    prod_1x2_true_home = np.array([1 if a==0 else 0 for a in actual_1x2_arr])
    prod_1x2_true_draw = np.array([1 if a==1 else 0 for a in actual_1x2_arr])
    prod_1x2_true_away = np.array([1 if a==2 else 0 for a in actual_1x2_arr])
    prod_1x2_brier = _multiclass_brier(prod_1x2_true_home, prod_1x2_true_draw, prod_1x2_true_away,
                                       prod_home_arr, prod_draw_arr1x2, prod_away_arr)
    shadow_1x2_true_home = np.array([1 if a==0 else 0 for a in actual_1x2_arr])
    shadow_1x2_true_draw = np.array([1 if a==1 else 0 for a in actual_1x2_arr])
    shadow_1x2_true_away = np.array([1 if a==2 else 0 for a in actual_1x2_arr])
    shadow_1x2_brier = _multiclass_brier(shadow_1x2_true_home, shadow_1x2_true_draw, shadow_1x2_true_away,
                                         shadow_home_arr, shadow_draw_arr1x2, shadow_away_arr)

    prod_1x2_accuracy = _accuracy(actual_1x2_arr, prod_home_arr, prod_draw_arr1x2, prod_away_arr)
    shadow_1x2_accuracy = _accuracy(actual_1x2_arr, shadow_home_arr, shadow_draw_arr1x2, shadow_away_arr)

    # OU
    prod_ou_brier = _binary_brier(np.array(actual_ou), np.array(prod_ou_prob))
    shadow_ou_brier = _binary_brier(np.array(actual_ou), np.array(shadow_ou_prob))

    # BTTS
    prod_btts_brier = _binary_brier(np.array(actual_btts), np.array(prod_btts_prob))
    shadow_btts_brier = _binary_brier(np.array(actual_btts), np.array(shadow_btts_prob))

    # Calibration buckets untuk Draw (menggunakan data paired)
    prod_draw_calibration = _calibration_buckets(draw_prod_paired, draw_actual_paired)
    shadow_draw_calibration = _calibration_buckets(draw_shadow_paired, draw_actual_paired)

    # Goal Difference calibration
    goal_diff_calibration = {}
    if gd_valid_count > 0:
        for bucket in ['-3', '-2', '-1', '0', '+1', '+2', '+3']:
            preds = goal_diff_preds[bucket]
            actuals = goal_diff_actual_indicators[bucket]
            # count = gd_valid_count (sama untuk semua bucket karena setiap pertandingan menyumbang satu prediksi ke setiap bucket)
            avg_pred = float(np.mean(preds)) if preds else None
            avg_actual = float(np.mean(actuals)) if actuals else 0.0
            goal_diff_calibration[bucket] = {
                "predicted": round(avg_pred, 6) if avg_pred is not None else None,
                "actual": round(avg_actual, 6),
                "count": gd_valid_count
            }
    else:
        for bucket in ['-3', '-2', '-1', '0', '+1', '+2', '+3']:
            goal_diff_calibration[bucket] = {"predicted": None, "actual": None, "count": 0}

    # Common paired counts
    valid_draw_matches = len(draw_prod_paired)
    valid_1x2_matches = len(actual_1x2_label)
    valid_ou_matches = len(actual_ou)
    valid_btts_matches = len(actual_btts)

    return {
        "total_matches": total_settled,
        "valid_draw_matches": valid_draw_matches,
        "valid_1x2_matches": valid_1x2_matches,
        "valid_ou_matches": valid_ou_matches,
        "valid_btts_matches": valid_btts_matches,
        "missing_shadow_count": missing_shadow_count,
        "missing_production_prob_count": missing_production_prob_count,
        "invalid_probability_count": invalid_probability_count,
        "baseline_draw_prob": baseline_draw_prob,
        "baseline_draw_brier": baseline_draw_brier,
        "baseline_draw_logloss": baseline_draw_logloss,
        "prod_draw_brier": prod_draw_brier,
        "prod_draw_logloss": prod_draw_logloss,
        "prod_1x2_logloss": prod_1x2_logloss,
        "prod_1x2_brier": prod_1x2_brier,
        "prod_1x2_accuracy": prod_1x2_accuracy,
        "prod_ou_brier": prod_ou_brier,
        "prod_btts_brier": prod_btts_brier,
        "shadow_draw_brier": shadow_draw_brier,
        "shadow_draw_logloss": shadow_draw_logloss,
        "shadow_1x2_logloss": shadow_1x2_logloss,
        "shadow_1x2_brier": shadow_1x2_brier,
        "shadow_1x2_accuracy": shadow_1x2_accuracy,
        "shadow_ou_brier": shadow_ou_brier,
        "shadow_btts_brier": shadow_btts_brier,
        "prod_draw_calibration": prod_draw_calibration,
        "shadow_draw_calibration": shadow_draw_calibration,
        "shadow_goal_diff_calibration": goal_diff_calibration,
    }
