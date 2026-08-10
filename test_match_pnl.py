from datetime import date, timedelta

from services.match_pnl import visible_match_dates


def test_visible_match_dates_contains_today_and_previous_six_days():
    today = date(2026, 8, 10)
    dates = visible_match_dates(today)
    assert len(dates) == 7
    assert dates[0] == today
    assert dates[-1] == today - timedelta(days=6)
    assert dates == sorted(dates, reverse=True)
