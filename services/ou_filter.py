"""Experimental OU decision filter backed by historical testing.

This filter is intentionally isolated so its thresholds can be audited and
changed without rewriting the prediction engine.
"""

from typing import Any, Dict, Optional

OU_FILTER_VERSION = "OU_FILTER_V1_50_70_EV5_ODDS185_LINE275_MOVE-001"
OU_FILTER_MIN_CONFIDENCE = 0.50
OU_FILTER_MAX_CONFIDENCE = 0.70
OU_FILTER_MIN_EV = 0.05
OU_FILTER_MIN_ODDS = 1.85
OU_FILTER_MIN_LINE = 2.75
OU_FILTER_MAX_SELECTED_MOVEMENT = -0.01


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        value = float(value)
        if value != value:  # NaN
            return default
        return value
    except (TypeError, ValueError):
        return default


def evaluate_ou_filter(
    confidence: Any,
    ou_line: Any,
    prediction: str,
    over_odds: Any,
    under_odds: Any,
    ev_over: Any,
    ev_under: Any,
    over_move: Any,
    under_move: Any,
) -> Dict[str, Any]:
    """Evaluate the experimental 50-70% OU filter.

    Outside the 50-70% confidence band the filter is not applied and the
    existing OU decision remains untouched. Inside the band, all conditions
    must pass:
      - EV >= 5%
      - selected odds >= 1.85
      - OU line >= 2.75
      - selected-side odds movement <= -0.01
    """
    conf = _num(confidence)
    line = _num(ou_line)
    pred = str(prediction or "").strip().upper()

    result = {
        "version": OU_FILTER_VERSION,
        "applied": False,
        "passed": True,
        "reason": "OUTSIDE_CONFIDENCE_BAND",
        "selected_ev": None,
        "selected_odds": None,
        "selected_movement": None,
    }

    if conf is None or not (OU_FILTER_MIN_CONFIDENCE <= conf <= OU_FILTER_MAX_CONFIDENCE):
        return result

    result["applied"] = True

    if pred not in ("OVER", "UNDER"):
        result.update(passed=False, reason="NO_VALID_OU_SIDE")
        return result

    if pred == "OVER":
        selected_ev = _num(ev_over)
        selected_odds = _num(over_odds)
        selected_movement = _num(over_move)
    else:
        selected_ev = _num(ev_under)
        selected_odds = _num(under_odds)
        selected_movement = _num(under_move)

    result["selected_ev"] = selected_ev
    result["selected_odds"] = selected_odds
    result["selected_movement"] = selected_movement

    failures = []
    if selected_ev is None or selected_ev < OU_FILTER_MIN_EV:
        failures.append("EV<5%")
    if selected_odds is None or selected_odds < OU_FILTER_MIN_ODDS:
        failures.append("ODDS<1.85")
    if line is None or line < OU_FILTER_MIN_LINE:
        failures.append("LINE<2.75")
    if selected_movement is None or selected_movement > OU_FILTER_MAX_SELECTED_MOVEMENT:
        failures.append("MOVEMENT_NOT_SHORTENING")

    if failures:
        result.update(passed=False, reason=";".join(failures))
    else:
        result.update(passed=True, reason="ALL_FILTERS_PASS")

    return result
