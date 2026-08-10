"""Per-match P&L calculations for Football AI V2.

This module contains only deterministic calculations. Streamlit rendering stays
in app.py, while history_ou.csv remains the source of truth for settled data.
"""

from typing import Any, Dict, List, Tuple
from html import escape

from services.settlement import SettlementEngine, normalize_correct_score_stake

ONE_X_TWO_MIN_ODDS = 1.56


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _leg(market: str, stake: float, odds: float, result: str, pnl: float) -> Dict[str, Any]:
    stake = max(0.0, float(stake))
    pnl = float(pnl)
    return {
        "market": market,
        "stake": stake,
        "odds": float(odds),
        "result": result,
        "return": stake + pnl,
        "pnl": pnl,
    }


def apply_1x2_odds_floor(
    prediction: str,
    odds: float,
    target_net_profit: float = 100000.0,
) -> Tuple[str, float]:
    """Apply the hard minimum current 1X2 odds rule.

    Odds below 1.56 are always NO BET. Odds at or above 1.56 retain the
    existing fixed-net-profit stake formula used by the application.
    """
    prediction = str(prediction or "").strip().upper()
    odds = _float(odds)
    target_net_profit = _float(target_net_profit)

    if prediction not in {"HOME", "DRAW", "AWAY"}:
        return "NO BET", 0.0
    if odds < ONE_X_TWO_MIN_ODDS or odds <= 1.0 or target_net_profit <= 0.0:
        return "NO BET", 0.0

    return prediction, target_net_profit / (odds - 1.0)


def _add_ou_leg(row: Dict[str, Any], home_goals: int, away_goals: int, legs: List[Dict[str, Any]]) -> None:
    prediction = str(row.get("prediction", "")).strip().upper()
    stake = _float(row.get("stake", 0.0))
    if not prediction or prediction == "NO BET" or stake <= 0.0:
        return

    try:
        settlement = SettlementEngine.evaluate(row, home_goals, away_goals)
    except Exception:
        return

    result = str(settlement.get("result", "UNKNOWN"))
    pnl = _float(settlement.get("profit", 0.0))
    odds = _float(
        row.get("current_over_odds") if prediction.startswith("OVER") else row.get("current_under_odds")
    )
    if odds <= 1.0:
        return

    line = _float(row.get("current_ou", 2.5), 2.5)
    line_text = f"{line:.1f}"
    side = "OVER" if prediction.startswith("OVER") else "UNDER"
    legs.append(_leg(f"{side} {line_text}", stake, odds, result, pnl))


def _add_1x2_leg(row: Dict[str, Any], home_goals: int, away_goals: int, legs: List[Dict[str, Any]]) -> None:
    prediction = str(row.get("prediction_1x2", "")).strip().upper()
    stake = _float(row.get("stake_1x2", 0.0))
    if prediction not in {"HOME", "DRAW", "AWAY"} or stake <= 0.0:
        return

    actual = "HOME" if home_goals > away_goals else "DRAW" if home_goals == away_goals else "AWAY"
    odds = _float(row.get(f"market_odds_1x2_{prediction.lower()}"))
    if odds <= 1.0:
        return

    won = prediction == actual
    result = "WIN" if won else "LOSE"
    pnl = stake * (odds - 1.0) if won else -stake
    legs.append(_leg("1X2", stake, odds, result, pnl))


def _add_btts_leg(row: Dict[str, Any], home_goals: int, away_goals: int, legs: List[Dict[str, Any]]) -> None:
    prediction = str(row.get("recommendation_btts", "")).strip().upper()
    stake = _float(row.get("stake_btts", 0.0))
    if prediction not in {"YES", "NO"} or stake <= 0.0:
        return

    actual = "YES" if home_goals > 0 and away_goals > 0 else "NO"
    odds_col = "market_odds_btts_yes" if prediction == "YES" else "market_odds_btts_no"
    odds = _float(row.get(odds_col))
    if odds <= 1.0:
        return

    won = prediction == actual
    result = "WIN" if won else "LOSE"
    pnl = stake * (odds - 1.0) if won else -stake
    legs.append(_leg(f"BTTS {prediction}", stake, odds, result, pnl))


def _add_cs_legs(row: Dict[str, Any], home_goals: int, away_goals: int, legs: List[Dict[str, Any]]) -> None:
    for i in range(1, 4):
        score_raw = row.get(f"cs_score_{i}")
        odds = _float(row.get(f"cs_odds_{i}"))
        if score_raw is None or odds <= 1.0:
            continue

        try:
            h, a = map(int, str(score_raw).strip().split(":", 1))
        except (TypeError, ValueError):
            continue

        stake_raw = row.get(f"cs_stake_{i}")
        stake = _float(stake_raw)
        if stake <= 0.0:
            stake = normalize_correct_score_stake(odds)
        if stake <= 0.0:
            continue

        won = h == home_goals and a == away_goals
        result = "WIN" if won else "LOSE"
        pnl = stake * (odds - 1.0) if won else -stake
        legs.append(_leg(f"CS {h}:{a}", stake, odds, result, pnl))



def format_pnl_table_html(legs: List[Dict[str, Any]]) -> str:
    """Render compact, full-row-color P&L detail as one HTML table."""
    result_styles = {
        "WIN": ("#16a34a", "white"),
        "HALF WIN": ("#65a30d", "white"),
        "PUSH": ("#475569", "white"),
        "HALF LOSE": ("#f59e0b", "black"),
        "LOSE": ("#ef4444", "white"),
        "NO BET": ("#475569", "white"),
    }

    rows = [
        "<table style='width:100%; border-collapse:collapse; font-size:0.82rem;'>",
        "<thead><tr style='background:#1f2937; color:white;'>"
        "<th>Jenis Taruhan</th><th>Modal</th><th>Odds</th><th>Hasil</th><th>P/L</th>"
        "</tr></thead><tbody>",
    ]
    for leg in legs:
        result = str(leg.get("result", "UNKNOWN")).upper()
        bg, fg = result_styles.get(result, ("#334155", "white"))
        pnl = float(leg.get("pnl", 0.0) or 0.0)
        sign = "+" if pnl > 0 else ""
        rows.append(
            f"<tr style='background:{bg}; color:{fg};'>"
            f"<td style='padding:8px; font-weight:700;'>{escape(str(leg.get('market', '')))}</td>"
            f"<td style='padding:8px; text-align:right;'>Rp{float(leg.get('stake', 0.0)):,.0f}</td>"
            f"<td style='padding:8px; text-align:right;'>{float(leg.get('odds', 0.0)):.2f}</td>"
            f"<td style='padding:8px; font-weight:700;'>{escape(result)}</td>"
            f"<td style='padding:8px; text-align:right; font-weight:700;'>Rp{sign}{pnl:,.0f}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "".join(rows)

def build_match_pnl(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build the complete P&L ledger for one settled match history row."""
    home_goals = _int(row.get("home_goals"))
    away_goals = _int(row.get("away_goals"))
    legs: List[Dict[str, Any]] = []

    _add_ou_leg(row, home_goals, away_goals, legs)
    _add_1x2_leg(row, home_goals, away_goals, legs)
    _add_btts_leg(row, home_goals, away_goals, legs)
    _add_cs_legs(row, home_goals, away_goals, legs)

    total_stake = sum(item["stake"] for item in legs)
    total_return = sum(item["return"] for item in legs)
    net_pnl = total_return - total_stake

    return {
        "match_uid": row.get("match_uid"),
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "legs": legs,
        "total_stake": total_stake,
        "total_return": total_return,
        "net_pnl": net_pnl,
    }
