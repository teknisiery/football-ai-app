# ui/components.py
"""Komponen UI yang dapat digunakan kembali untuk Football AI V2."""
import streamlit as st
from utils import safe_html


def _render_horizontal_metric_row(cards, allow_html=False):
    """Render baris metrik horizontal di dalam prediction card."""
    divs = []
    for icon, label, value, bg in cards:
        val = value if allow_html else safe_html(str(value))
        divs.append(
            f'<div class="brain-card" style="background:{bg};">'
            f'<div class="icon">{icon}</div>'
            f'<div class="label">{safe_html(label)}</div>'
            f'<div class="badge-value">{val}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="brain-row">{"".join(divs)}</div>', unsafe_allow_html=True)


def render_prediction_card(summary: dict):
    """Tampilkan kartu prediksi lengkap untuk satu pertandingan."""
    if not summary:
        return

    home = safe_html(summary['home'])
    away = safe_html(summary['away'])
    league = safe_html(summary['league'])
    ou_pred = safe_html(summary.get('ou_pred', ''))
    ou_line = summary.get('ou_line', '')
    rec = safe_html(summary.get('recommendation', ''))
    rec_color = summary.get('rec_color', 'd')
    over_odds = summary.get('over_odds', 0)
    under_odds = summary.get('under_odds', 0)

    # Header pertandingan
    st.markdown(
        f"""<div class="prediction-card"><div style="text-align:center;">
        <h3>⚽ {home} vs {away}</h3>
        <p style="color:#a0a0b0; margin:0;">{league}</p>
        <p style="font-size:1.2rem; font-weight:700; margin:4px 0 8px 0;">
            <span class="badge badge-{rec_color}">{ou_pred} {ou_line} — {rec}</span>
        </p>
    </div>""",
        unsafe_allow_html=True
    )

    # Fungsi bantu warna
    def eg_c(v):
        return "#16a34a" if v >= 2.8 else "#eab308" if v >= 2.0 else "#ef4444"

    def ev_c(v):
        return "#16a34a" if v > 0.02 else "#eab308" if v > 0 else "#ef4444"

    def conf_color(pct):
        if pct >= 80: return "#16a34a"
        elif pct >= 70: return "#65a30d"
        elif pct >= 60: return "#eab308"
        else: return "#334155"

    def btts_c(p):
        return "#16a34a" if p > 0.6 else "#eab308" if p > 0.4 else "#ef4444"

    def btts_card_color(rec):
        if rec == 'YES': return "#16a34a"
        elif rec == 'NO': return "#eab308"
        elif rec == 'NO BET': return "#ef4444"
        else: return btts_c(summary.get('confidence_btts', 0) if summary.get('confidence_btts') is not None else 0.5)

    expected_goal = summary.get('expected_goal', 0)
    confidence_ou = summary.get('confidence_ou', 0)
    ev_over = summary.get('ev_over', 0)
    ev_under = summary.get('ev_under', 0)

    # Baris pertama metrik
    _render_horizontal_metric_row([
        ("⚽", "Expected Goal", f"{expected_goal:.2f}", eg_c(expected_goal)),
        ("📈", "Confidence", f"{confidence_ou:.0%}", conf_color(confidence_ou * 100)),
        ("💰", "EV Over", f"{ev_over:+.3f}<br>Odds: {over_odds:.2f}", ev_c(ev_over)),
        ("💰", "EV Under", f"{ev_under:+.3f}<br>Odds: {under_odds:.2f}", ev_c(ev_under)),
    ], allow_html=True)

    # Baris kedua: 1X2, Stake 1X2, BTTS, Stake OU
    stake_val = summary.get('stake', 0)
    rec_1x2 = summary.get('prediction_1x2')
    stake_1x2 = summary.get('stake_1x2', 0)
    rec_btts = summary.get('recommendation_btts')
    stake_btts = summary.get('stake_btts', 0)

    if rec_btts:
        btts_value = f"{rec_btts}"
        if rec_btts != 'NO BET' and stake_btts > 0:
            btts_value += f"<br><small>Stake: Rp{stake_btts:,.0f}</small>"
    else:
        btts_value = safe_html(str(summary.get('btts_pred', 'N/A')))
        if summary.get('confidence_btts') is not None:
            btts_value += f" ({summary.get('confidence_btts', 0):.0%})"

    cards_row2 = [
        ("🎯", "1X2",
         rec_1x2 if rec_1x2 else "N/A",
         "#16a34a" if rec_1x2 and rec_1x2 != 'NO BET' else ("#ef4444" if rec_1x2 == 'NO BET' else "#334155")),
        ("💵", "Stake 1X2",
         f"Rp{stake_1x2:,.0f}" if (rec_1x2 and rec_1x2 != 'NO BET') else "Rp0",
         "#16a34a" if stake_1x2 > 0 else "#6b7280"),
        ("🤝", "BTTS",
         btts_value,
         btts_card_color(rec_btts) if rec_btts else btts_c(
             summary.get('confidence_btts', 0) if summary.get('confidence_btts') is not None else 0.5)),
        ("💲", "Stake",
         f"Rp{stake_val:,.0f}" if stake_val > 0 else "Rp0",
         "#16a34a" if stake_val > 0 else "#6b7280"),
    ]
    _render_horizontal_metric_row(cards_row2, allow_html=True)

    # Baris ketiga: Correct Score top 3
    if summary.get('top3_scores'):
        top3 = summary.get('hybrid_top3') or summary['top3_scores']
        cs_recs = summary.get('cs_recommendations')
        colors = ["#16a34a", "#2563eb", "#eab308"]
        cards_cs = []
        for i, (h, a, prob) in enumerate(top3):
            label = f"{h}-{a}"
            value = f"{prob*100:.1f}%"
            odds_str = ""
            if cs_recs and i < len(cs_recs):
                rec = cs_recs[i]
                if rec[0] == h and rec[1] == a:
                    odds = rec[2]
                    stake_cs = 200000.0 / (odds - 1) if odds > 1 else 0
                    odds_str = f"Odds: {odds:.2f} | Stake: Rp{stake_cs:,.0f}"
            full_label = f"{label}  {odds_str}" if odds_str else label
            color = colors[i] if i < len(colors) else "#1e293b"
            cards_cs.append(("⚽", full_label, value, color))
        _render_horizontal_metric_row(cards_cs)

    # Movement 1X2
    movement = summary.get('movement')
    if movement:
        parts = []
        for outcome, label in [('home', 'Home'), ('draw', 'Draw'), ('away', 'Away')]:
            val = movement.get(outcome, 0)
            if val < -0.05:
                icon = "↓"; color = "#16a34a"
            elif val > 0.05:
                icon = "↑"; color = "#ef4444"
            else:
                icon = "→"; color = "#a0a0b0"
            parts.append(f"{label} {icon} {val:+.2f}")
        st.markdown(
            f"<p style='text-align:center; font-size:0.7rem; color:#a0a0b0;'>📊 Movement: {' | '.join(parts)}</p>",
            unsafe_allow_html=True
        )

    # Fair odds BTTS dan input market odds
    if summary.get('prob_btts') is not None:
        prob = summary['prob_btts']
        if 0 < prob < 1:
            fair_yes = 1 / prob
            fair_no = 1 / (1 - prob)
            st.markdown(
                f"<p style='text-align:center; font-size:0.7rem; color:#a0a0b0;'>"
                f"Fair Odds YES: {fair_yes:.2f} | Fair Odds NO: {fair_no:.2f}"
                f"</p>",
                unsafe_allow_html=True
            )

            col_yes, col_no = st.columns(2)
            with col_yes:
                market_yes = st.number_input(
                    "Market Odds YES", min_value=1.01, value=1.80, step=0.01,
                    key="market_btss_yes"
                )
            with col_no:
                market_no = st.number_input(
                    "Market Odds NO", min_value=1.01, value=1.80, step=0.01,
                    key="market_btss_no"
                )

            st.session_state['_market_odds_btts_'] = {'yes': market_yes, 'no': market_no}

            value_yes = market_yes - fair_yes
            value_no = market_no - fair_no

            st.markdown("**📊 Value Analysis**")
            if value_yes > 0.05:
                st.success(f"🟢 BTTS YES: Market {market_yes:.2f} > Fair {fair_yes:.2f} (Value +{value_yes:.2f})")
            elif value_yes < -0.05:
                st.error(f"🔴 BTTS YES: Market {market_yes:.2f} < Fair {fair_yes:.2f} (No Value)")
            else:
                st.info(f"⚪ BTTS YES: Market {market_yes:.2f} ≈ Fair {fair_yes:.2f} (Netral)")

            if value_no > 0.05:
                st.success(f"🟢 BTTS NO: Market {market_no:.2f} > Fair {fair_no:.2f} (Value +{value_no:.2f})")
            elif value_no < -0.05:
                st.error(f"🔴 BTTS NO: Market {market_no:.2f} < Fair {fair_no:.2f} (No Value)")
            else:
                st.info(f"⚪ BTTS NO: Market {market_no:.2f} ≈ Fair {fair_no:.2f} (Netral)")

    st.markdown("</div>", unsafe_allow_html=True)
