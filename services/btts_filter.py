"""Deterministic BTTS market-protection filter.

The filter separates raw model/value evidence from the final betting decision.
All evidence remains available to callers so future experiments can compare
filtered versus unfiltered decisions.
"""

from typing import Any, Dict, Optional

BTTS_FILTER_VERSION = "BTTS_LOW_ODDS_V2_135_OPPOSITE_EV30"
BTTS_LOW_ODDS = 1.35
BTTS_OPPOSITE_EV_MIN = 0.30


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if value != value:
            return None
        return value
    except (TypeError, ValueError):
        return None


def evaluate_btts_filter(
    prob_btts: Any,
    market_yes: Any,
    market_no: Any,
    ev_yes: Any,
    ev_no: Any,
) -> Dict[str, Any]:
    """Return the final BTTS decision while retaining raw market evidence.

    If either market side is priced below 1.35, that side is protected from
    betting. The opposite side is eligible only when its EV is strictly above
    +0.30. If neither side is below 1.35, the highest positive EV wins.
    """
    prob = _num(prob_btts)
    yes_odds = _num(market_yes)
    no_odds = _num(market_no)
    yes_ev = _num(ev_yes)
    no_ev = _num(ev_no)

    result: Dict[str, Any] = {
        "version": BTTS_FILTER_VERSION,
        "filtered": False,
        "recommendation": "NO BET",
        "stake_side": None,
        "reason": "NO_POSITIVE_VALUE",
        "prob_btts": prob,
        "market_yes": yes_odds,
        "market_no": no_odds,
        "ev_yes": yes_ev,
        "ev_no": no_ev,
    }

    low_yes = yes_odds is not None and yes_odds < BTTS_LOW_ODDS
    low_no = no_odds is not None and no_odds < BTTS_LOW_ODDS

    if low_yes or low_no:
        result["filtered"] = True
        if low_yes and low_no:
            result["reason"] = "BOTH_SIDES_LOW_ODDS"
            return result

        if low_yes:
            if no_ev is not None and no_ev > BTTS_OPPOSITE_EV_MIN:
                result.update(
                    recommendation="NO",
                    stake_side="NO",
                    reason="LOW_ODDS_OPPOSITE_VALUE",
                )
            else:
                result["reason"] = "LOW_ODDS_OPPOSITE_EV_TOO_LOW"
            return result

        if yes_ev is not None and yes_ev > BTTS_OPPOSITE_EV_MIN:
            result.update(
                recommendation="YES",
                stake_side="YES",
                reason="LOW_ODDS_OPPOSITE_VALUE",
            )
        else:
            result["reason"] = "LOW_ODDS_OPPOSITE_EV_TOO_LOW"
        return result

    positive = []
    if yes_ev is not None and yes_ev > 0:
        positive.append(("YES", yes_ev))
    if no_ev is not None and no_ev > 0:
        positive.append(("NO", no_ev))

    if positive:
        side, _ = max(positive, key=lambda item: item[1])
        result.update(
            recommendation=side,
            stake_side=side,
            reason="NORMAL_VALUE",
        )

    return result


def normalize_btts_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Re-evaluate stored BTTS evidence without deleting the original decision."""
    normalized = dict(row)
    recommendation = str(normalized.get("recommendation_btts", "")).strip().upper()

    if "recommendation_btts_raw" not in normalized:
        normalized["recommendation_btts_raw"] = recommendation or "NO BET"
    if "stake_btts_raw" not in normalized:
        normalized["stake_btts_raw"] = normalized.get("stake_btts", 0.0) or 0.0

    decision = evaluate_btts_filter(
        normalized.get("prob_btts"),
        normalized.get("market_odds_btts_yes"),
        normalized.get("market_odds_btts_no"),
        normalized.get("ev_btts_yes"),
        normalized.get("ev_btts_no"),
    )

    has_evidence = any(
        decision[key] is not None
        for key in ("market_yes", "market_no", "ev_yes", "ev_no")
    )
    if has_evidence:
        final_rec = decision["recommendation"]
        previous_rec = recommendation
        normalized["recommendation_btts"] = final_rec
        normalized["btts_filter_version"] = decision["version"]
        normalized["btts_filter_reason"] = decision["reason"]
        normalized["btts_filtered"] = bool(decision["filtered"])

        if final_rec == previous_rec and final_rec in {"YES", "NO"}:
            try:
                normalized["stake_btts"] = max(0.0, float(normalized.get("stake_btts", 0.0) or 0.0))
            except (TypeError, ValueError):
                normalized["stake_btts"] = 0.0
        elif final_rec in {"YES", "NO"}:
            odds_key = "market_odds_btts_yes" if final_rec == "YES" else "market_odds_btts_no"
            try:
                odds = float(normalized.get(odds_key))
                normalized["stake_btts"] = 100000.0 / (odds - 1.0) if odds > 1.0 else 0.0
            except (TypeError, ValueError):
                normalized["stake_btts"] = 0.0
        else:
            normalized["stake_btts"] = 0.0
    elif recommendation not in {"YES", "NO"}:
        normalized["recommendation_btts"] = "NO BET"
        normalized["stake_btts"] = 0.0

    return normalized
