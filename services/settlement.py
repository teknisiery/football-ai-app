# services/settlement.py
import numpy as np
from typing import Dict
from utils import settle_basic, split_quarter_line

class SettlementEngine:
    @staticmethod
    def evaluate(row: dict, home_goals: int, away_goals: int) -> Dict:
        """
        Menghitung seluruh hasil settlement untuk satu pertandingan.
        row: dictionary berisi data pertandingan (seperti yang disimpan di history).
        home_goals, away_goals: skor akhir.
        Return: dict dengan profit_ou, result_ou, actual_over, actual_btts, profit_cs.
        """
        total_goals = home_goals + away_goals
        ou_line = float(row.get('current_ou', 2.5))
        rec = str(row.get('recommendation', '')).strip().upper()
        stake_val = row.get('stake', 100000)

        # Over/Under settlement
        if rec == 'NO BET' or (stake_val is not None and float(stake_val) == 0):
            profit_ou = 0.0
            result_ou = "NO BET"
        else:
            effective_stake = float(stake_val) if (stake_val is not None and float(stake_val) > 0) else 100000.0
            pred_str = str(row.get('prediction', '')).strip().upper()
            if pred_str.startswith('OVER'):
                bet_type = "OVER"
                odds = float(row.get('current_over_odds', 1.0))
            elif pred_str.startswith('UNDER'):
                bet_type = "UNDER"
                odds = float(row.get('current_under_odds', 1.0))
            else:
                # Tidak bisa menentukan arah taruhan, anggap NO BET
                return {
                    'profit': 0.0,
                    'result': 'UNKNOWN',
                    'actual_over': np.nan,
                    'actual_btts': 0,
                    'cs_profit': 0.0
                }

            splits = split_quarter_line(ou_line)
            total_profit = 0.0
            results = []
            for line, weight in splits:
                result = settle_basic(total_goals, line, bet_type)
                results.append(result)
                if result == "WIN":
                    total_profit += effective_stake * weight * (odds - 1)
                elif result == "LOSE":
                    total_profit -= effective_stake * weight

            if len(results) == 1:
                if results[0] == "WIN": result_ou = "FULL WIN"
                elif results[0] == "LOSE": result_ou = "FULL LOSE"
                else: result_ou = "PUSH"
            else:
                r1, r2 = results
                if r1 == "WIN" and r2 == "WIN": result_ou = "FULL WIN"
                elif r1 == "LOSE" and r2 == "LOSE": result_ou = "FULL LOSE"
                elif r1 == "PUSH" and r2 == "PUSH": result_ou = "PUSH"
                elif (r1 == "WIN" and r2 == "PUSH") or (r1 == "PUSH" and r2 == "WIN"): result_ou = "HALF WIN"
                elif (r1 == "LOSE" and r2 == "PUSH") or (r1 == "PUSH" and r2 == "LOSE"): result_ou = "HALF LOSE"
                # Blok WIN + LOSE telah dihapus karena tidak mungkin terjadi secara matematis
                else: result_ou = "UNKNOWN"
            profit_ou = total_profit

        # Actual Over/BTTS
        if total_goals > ou_line:
            actual_over = 1
        elif total_goals < ou_line:
            actual_over = 0
        else:
            actual_over = np.nan
        actual_btts = 1 if (home_goals > 0 and away_goals > 0) else 0

        # Correct Score settlement
        cs_profit = 0.0
        for i in range(1, 4):
            score_str = row.get(f'cs_score_{i}')
            odds = row.get(f'cs_odds_{i}')
            stake_cs = row.get(f'cs_stake_{i}')
            if score_str and odds is not None and stake_cs is not None:
                try:
                    h, a = map(int, score_str.split(':'))
                    if h == home_goals and a == away_goals:
                        cs_profit += stake_cs * (odds - 1)
                    else:
                        cs_profit -= stake_cs
                except:
                    pass

        return {
            'profit': profit_ou,
            'result': result_ou,
            'actual_over': actual_over,
            'actual_btts': actual_btts,
            'cs_profit': cs_profit
        }
