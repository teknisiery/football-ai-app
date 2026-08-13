# ui/league_editor.py
"""Editor statistik liga untuk sidebar."""
import streamlit as st
import pandas as pd
import json
import re

from services.resource_registry import ResourceRegistry


def parse_score_distribution_text(raw_text: str, total_matches: int = 0) -> dict:
    """Parse teks distribusi skor dari sumber eksternal.

    Mengabaikan persentase, hanya mengambil angka terakhir di baris sebagai jumlah.
    Skor dengan gol individu ≥5 otomatis masuk kategori "Other".
    """
    categories = [
        "0:0", "1:0", "1:1", "2:0", "2:1", "2:2",
        "3:0", "3:1", "3:2", "3:3",
        "4:0", "4:1", "4:2", "4:3", "4:4",
    ]
    counts = {cat: 0 for cat in categories}
    counts["Other"] = 0

    text = raw_text.replace(',', ' ')
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for line in lines:
        m_score = re.search(r'(\d{1,2})-(\d{1,2})', line)
        if not m_score:
            continue

        h = int(m_score.group(1))
        a = int(m_score.group(2))

        numbers = re.findall(r'\d+', line)
        if len(numbers) < 2:
            continue
        count = int(numbers[-1])

        if h >= 5 or a >= 5:
            key = "Other"
        elif h > a:
            key = f"{h}:{a}"
        elif h < a:
            key = f"{a}:{h}"
        else:
            key = f"{h}:{a}"

        if key in counts:
            counts[key] += count
        else:
            counts["Other"] += count

    if sum(counts.values()) == 0:
        return {}

    return counts


def render_league_editor(db_storage, session):
    """Render seluruh UI edit statistik liga di sidebar."""
    with st.expander("✏️ Edit Statistik Liga"):
        try:
            profil_league = db_storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
        except:
            profil_league = pd.DataFrame()

        # Pastikan kolom default
        for col, default in [('home_win_pct', 0.40), ('away_win_pct', 0.30), ('draw_pct', 0.30)]:
            if col not in profil_league.columns:
                profil_league[col] = default

        if profil_league.empty:
            st.info("File profil liga tidak tersedia.")
            return

        profil_league = profil_league.sort_values('league_name', ascending=True)
        league_options = [f"[{int(row['league_code'])}] {row['league_name']}" for _, row in profil_league.iterrows()]
        selected_league_str = st.selectbox("Pilih Liga", league_options, key="edit_league_select")

        if st.button("📂 Load Data Liga"):
            selected_code = int(selected_league_str.split(']')[0].replace('[', ''))
            st.session_state['edit_league_code'] = selected_code
            st.rerun()

        if 'edit_league_code' not in st.session_state or st.session_state['edit_league_code'] is None:
            st.info("Pilih liga dari dropdown, lalu klik 'Load Data Liga' untuk mengedit.")
            return

        selected_code = st.session_state['edit_league_code']
        mask = profil_league['league_code'] == selected_code
        if not mask.any():
            st.error("Kode liga tidak ditemukan di profil.")
            st.session_state.pop('edit_league_code', None)
            return

        liga_row = profil_league[mask].iloc[0]

        # --- Statistik Dasar ---
        col_a, col_b = st.columns(2)
        with col_a:
            new_avg = st.number_input("Avg Goals", min_value=0.0, value=float(liga_row.get('league_avg_goals', 2.5)), step=0.1, key="edit_avg")
        with col_b:
            new_over25 = st.number_input("Over 2.5 %", min_value=0.0, max_value=1.0, value=float(liga_row.get('league_over25_pct', 0.5)), step=0.01, key="edit_over25")

        col_c, col_d = st.columns(2)
        with col_c:
            new_btts_pct = st.number_input("BTTS %", min_value=0.0, max_value=1.0, value=float(liga_row.get('league_btts_pct', 0.5)), step=0.01, key="edit_btts")
        with col_d:
            new_under35 = st.number_input("Under 3.5 %", min_value=0.0, max_value=1.0, value=float(liga_row.get('league_under35_pct', 0.7)), step=0.01, key="edit_under35")

        col_e, col_f = st.columns(2)
        with col_e:
            new_home_win = st.number_input("Home Win %", min_value=0.0, max_value=1.0, value=float(liga_row.get('home_win_pct', 0.40)), step=0.01, key="edit_home_win")
        with col_f:
            new_draw = st.number_input("Draw %", min_value=0.0, max_value=1.0, value=float(liga_row.get('draw_pct', 0.30)), step=0.01, key="edit_draw")

        new_away_win = st.number_input("Away Win %", min_value=0.0, max_value=1.0, value=float(liga_row.get('away_win_pct', 0.30)), step=0.01, key="edit_away_win")

        # --- Home/Away Avg Goals ---
        new_home_avg_goals = st.number_input(
            "Home Avg Goals",
            min_value=0.0,
            value=float(liga_row.get('home_avg_goals', 1.20) or 1.20),
            step=0.01,
            key="edit_home_avg_goals"
        )

        new_away_avg_goals = st.number_input(
            "Away Avg Goals",
            min_value=0.0,
            value=float(liga_row.get('away_avg_goals', 0.90) or 0.90),
            step=0.01,
            key="edit_away_avg_goals"
        )

        # --- Total Matches ---
        new_total_matches = st.number_input(
            "Total Matches",
            min_value=0,
            value=int(liga_row.get('total_matches', 0) or 0),
            step=1,
            key="edit_total_matches"
        )

        # --- Distribusi Skor Kombinasi ---
        with st.expander("⚽ Distribusi Skor Kombinasi"):
            st.caption("Isi jumlah kejadian untuk setiap kategori. Sisa otomatis menjadi 'Other' (termasuk skor 5+).")

            # Inisialisasi session state untuk hasil parse
            if 'parsed_score_counts' not in st.session_state:
                st.session_state['parsed_score_counts'] = {}

            # Blok parse diletakkan SEBELUM input number agar nilai hasil parse terbaca
            st.markdown("**📥 Paste Distribusi Skor (hanya untuk mengisi input di atas)**")
            raw_paste = st.text_area(
                "Tempel teks distribusi skor",
                height=150,
                key="paste_score_dist"
            )
            if st.button("📥 Parse Teks", key="parse_score_dist_btn"):
                parsed = parse_score_distribution_text(raw_paste, int(new_total_matches))
                if parsed:
                    st.session_state['parsed_score_counts'] = parsed
                    st.success("Teks berhasil diparse. Silakan periksa input di atas.")
                else:
                    st.warning("Tidak dapat memparse teks. Pastikan format mengandung skor dan jumlah.")

            st.markdown("---")
            raw_dist = liga_row.get('score_combination_distribution', '{}')
            try:
                current_dist = json.loads(raw_dist) if raw_dist else {}
            except:
                current_dist = {}

            current_total = int(liga_row.get('total_matches', 0) or 0)
            score_counts = {}
            categories = [
                "0:0", "1:0", "1:1", "2:0", "2:1", "2:2",
                "3:0", "3:1", "3:2", "3:3",
                "4:0", "4:1", "4:2", "4:3", "4:4"
            ]

            parsed_counts = st.session_state.get('parsed_score_counts', {})
            for cat in categories:
                default_count = parsed_counts.get(
                    cat,
                    int(round(current_dist.get(cat, 0.0) * current_total)) if current_total else 0
                )
                score_counts[cat] = st.number_input(
                    f"Skor {cat}",
                    min_value=0,
                    value=int(default_count),
                    step=1,
                    key=f"edit_comb_{cat.replace(':', '_')}"
                )

            sum_input = sum(score_counts.values())
            remaining = new_total_matches - sum_input
            if remaining < 0:
                st.error(f"Jumlah seluruh skor melebihi Total Matches. Selisih: {remaining}")
            else:
                st.info(f"Sisa {remaining} otomatis masuk kategori 'Other' (termasuk skor 5+).")

        # --- Simpan ---
        if st.button("💾 Simpan Statistik"):
            profil_league.loc[mask, 'league_avg_goals'] = new_avg
            profil_league.loc[mask, 'league_over25_pct'] = new_over25
            profil_league.loc[mask, 'league_btts_pct'] = new_btts_pct
            profil_league.loc[mask, 'league_under35_pct'] = new_under35
            profil_league.loc[mask, 'home_win_pct'] = new_home_win
            profil_league.loc[mask, 'away_win_pct'] = new_away_win
            profil_league.loc[mask, 'draw_pct'] = new_draw

            profil_league.loc[mask, 'home_avg_goals'] = new_home_avg_goals
            profil_league.loc[mask, 'away_avg_goals'] = new_away_avg_goals

            profil_league.loc[mask, 'total_matches'] = new_total_matches

            final_dist = {}
            if new_total_matches > 0:
                for cat, count in score_counts.items():
                    if count > 0:
                        final_dist[cat] = count / new_total_matches
                if remaining > 0:
                    final_dist["Other"] = remaining / new_total_matches

            profil_league.loc[mask, 'score_combination_distribution'] = json.dumps(final_dist)

            db_storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profil_league)
            st.session_state.pop('cached_profil', None)
            st.session_state.pop('parsed_score_counts', None)
            session.invalidate_league_profile_cache()
            st.cache_resource.clear()
            st.success("Statistik liga berhasil diperbarui.")
            st.rerun()
