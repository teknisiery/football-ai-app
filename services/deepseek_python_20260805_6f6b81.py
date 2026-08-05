"""
Kalkulasi profit untuk Football AI V2.
Wrapper sederhana di atas SettlementEngine untuk menghitung profit dan status hasil taruhan.
"""
from typing import Tuple
from services.settlement import SettlementEngine

class ProfitCalculator:
    @staticmethod
    def calculate(row: dict, stake: float = 100000.0) -> Tuple[float, str]:
        home_goals = int(row.get('home_goals', 0) or 0)
        away_goals = int(row.get('away_goals', 0) or 0)
        res = SettlementEngine.evaluate(row, home_goals, away_goals)
        return res['profit'], res['result']