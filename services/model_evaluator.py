"""
Evaluasi performa model Football AI V2.
Menghitung metrik regresi, klasifikasi, kalibrasi, dan finansial.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error
)
from services.settlement import SettlementEngine
from services.profit_calculator import ProfitCalculator        # <-- import baru


# ----------------------------------------------------------------------
# Duplikasi minimal ResourceRegistry (sementara)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Resource:
    id: str
    type: str
    serializer: str
    default_filename: str
    version: int = 1
    description: str = ""

class ResourceRegistry:
    HISTORY = Resource("history", "dataframe", "csv", "history_ou.csv")
# ----------------------------------------------------------------------


def evaluate_model(storage) -> dict:
    result = {}
    try:
        df = storage.load_dataframe(ResourceRegistry.HISTORY)
    except:
        result = {"error": "Gagal membaca history."}
        return result

    if df.empty:
        result = {"error": "History kosong."}
        return result

    if 'totalgol_ft' not in df.columns:
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            df['totalgol_ft'] = df['home_goals'] + df['away_goals']
        else:
            result = {"error": "Data tidak memiliki kolom skor."}
            return result

    if 'expected_goal' not in df.columns:
        result = {"error": "Data tidak memiliki expected_goal."}
        return result

    reg_df = df.dropna(subset=['expected_goal', 'totalgol_ft'])
    if len(reg_df) < 2:
        result = {"error": "Belum cukup data untuk evaluasi regresi."}
        return result

    y_true_reg = reg_df['totalgol_ft']
    y_pred_reg = reg_df['expected_goal']
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))

    df['total_goals'] = df['home_goals'] + df['away_goals']
    df['actual_over'] = np.where(
        df['total_goals'] > df['current_ou'], 1,
        np.where(df['total_goals'] < df['current_ou'], 0, np.nan)
    )
    class_df = df.dropna(subset=['actual_over', 'prediction_ou'])
    if len(class_df) < 2:
        accuracy = precision = recall = f1 = cm = brier = logloss = None
    else:
        y_true_cls = class_df['actual_over'].astype(int)
        y_pred_cls = class_df['prediction_ou'].astype(int)
        accuracy = accuracy_score(y_true_cls, y_pred_cls)
        precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
        cm = confusion_matrix(y_true_cls, y_pred_cls)
        if 'prob_over' in class_df.columns:
            brier = brier_score_loss(y_true_cls, class_df['prob_over'])
            logloss = log_loss(y_true_cls, class_df['prob_over'])
        else:
            brier = logloss = None

    pc = ProfitCalculator()
    fin_df = df.copy()
    profits_statuses = [pc.calculate(row.to_dict(), 100000) for _, row in fin_df.iterrows()]
    fin_df['profit'], fin_df['status'] = zip(*profits_statuses)
    bet_df = fin_df[fin_df['status'] != 'NO BET']
    if bet_df.empty:
        win_rate = roi = yield_pct = total_profit = total_bets = None
    else:
        total_bets = len(bet_df)
        total_profit = bet_df['profit'].sum()
        roi = total_profit / (total_bets * 100000) * 100
        yield_pct = total_profit / (total_bets * 100000) * 100
        win_rate = ((bet_df['status'] == 'FULL WIN').sum() + (bet_df['status'] == 'HALF WIN').sum()) / total_bets * 100

    brier_btts = logloss_btts = None
    if 'prob_btts' in df.columns and 'home_goals' in df.columns:
        df['btts_actual'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        btts_valid = df.dropna(subset=['prob_btts', 'btts_actual'])
        if len(btts_valid) > 1:
            brier_btts = brier_score_loss(btts_valid['btts_actual'], btts_valid['prob_btts'])
            logloss_btts = log_loss(btts_valid['btts_actual'], btts_valid['prob_btts'])

    result = {
        "mae": mae, "rmse": rmse,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "cm": cm, "brier": brier, "logloss": logloss,
        "win_rate": win_rate, "roi": roi, "yield_pct": yield_pct,
        "total_bets_fin": total_bets, "total_profit_fin": total_profit,
        "brier_btts": brier_btts, "logloss_btts": logloss_btts,
        "error": None
    }
    return result
