# utils.py
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from typing import List, Tuple, Optional

def safe_html(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#x27;"))

def calc_kelly(prob: float, odds: float) -> float:
    if odds <= 1.0 or prob <= 0:
        return 0.0
    k = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    return max(0.0, min(0.25, k))

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = ['league_code', 'kickoff_time', 'home_team', 'away_team']
    existing_front = [c for c in front if c in df.columns]
    other = [c for c in df.columns if c not in front]
    return df[existing_front + other]

def get_valid_time(row, primary_col='kickoff_time', secondary_col='settlement_time'):
    for col in [primary_col, secondary_col]:
        if col in row and pd.notna(row[col]):
            try:
                ts = pd.to_datetime(row[col])
                if ts is not pd.NaT:
                    return ts
            except:
                pass
    return None

def normalize_kickoff(dt_str):
    if pd.isna(dt_str) or dt_str is None or str(dt_str).strip() == '':
        return None
    try:
        ts = pd.to_datetime(str(dt_str).strip(), errors='coerce')
        if ts is pd.NaT:
            return None
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return None

def parse_odds_csv(file_content: bytes) -> dict:
    df = pd.read_csv(BytesIO(file_content))
    odds_dict = {}
    score_col = None
    odds_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'score' in col_lower:
            score_col = col
        elif 'odds' in col_lower:
            odds_col = col
    if score_col is None or odds_col is None:
        if len(df.columns) >= 3:
            score_col = df.columns[1]
            odds_col = df.columns[2]
        else:
            return {}
    for _, row in df.iterrows():
        try:
            score_str = str(row[score_col]).strip()
            odds_val = float(row[odds_col])
            type_str = str(row.iloc[0]).strip().lower()
            if score_str.lower() == 'other':
                odds_dict["OTHER"] = odds_val
                continue
            if 'home' in type_str:
                odds_dict[score_str] = odds_val
            elif 'away' in type_str:
                a_goals, h_goals = map(int, score_str.split(':'))
                odds_dict[f"{h_goals}:{a_goals}"] = odds_val
            elif 'draw' in type_str:
                odds_dict[score_str] = odds_val
        except:
            pass
    return odds_dict

def get_cs_recommendations(top3_scores, odds_dict):
    recs = []
    for h, a, prob in top3_scores:
        key = f"{h}:{a}"
        odds = odds_dict.get(key)
        if odds is None:
            odds = odds_dict.get("OTHER")
        if odds is not None:
            recs.append((h, a, odds))
    return recs[:3]

def convert_odds_to_wide(match_uid, file_content):
    df = pd.read_csv(BytesIO(file_content))
    row = {'match_uid': match_uid, 'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    type_col = df.columns[0]
    score_col = df.columns[1]
    odds_col = df.columns[2]
    for _, r in df.iterrows():
        try:
            type_str = str(r[type_col]).strip().lower()
            score_str = str(r[score_col]).strip()
            odds_val = float(r[odds_col])
            if score_str.lower() == 'other':
                row['odds_other_all'] = odds_val
                continue
            if 'home' in type_str:
                col = f"odds_{score_str.replace(':', '_')}_home"
            elif 'away' in type_str:
                col = f"odds_{score_str.replace(':', '_')}_away"
            elif 'draw' in type_str:
                col = f"odds_{score_str.replace(':', '_')}_draw"
            else:
                continue
            row[col] = odds_val
        except:
            pass
    return row

def settle_basic(total_goals: int, line: float, bet_type: str) -> str:
    if bet_type == "OVER":
        if total_goals > line: return "WIN"
        elif total_goals == line: return "PUSH"
        else: return "LOSE"
    else:
        if total_goals < line: return "WIN"
        elif total_goals == line: return "PUSH"
        else: return "LOSE"

def split_quarter_line(ou_line: float) -> List[Tuple[float, float]]:
    remainder = round(ou_line % 1, 2)
    if remainder == 0.0 or remainder == 0.5:
        return [(ou_line, 1.0)]
    elif remainder == 0.25:
        return [(float(int(ou_line)), 0.5), (int(ou_line) + 0.5, 0.5)]
    elif remainder == 0.75:
        return [(int(ou_line) + 0.5, 0.5), (int(ou_line) + 1.0, 0.5)]
    else:
        return [(ou_line, 1.0)]

# ============================================================
# FUNGSI BARU UNTUK TASK 16a – HYBRID TOP 3 SCORE
# ============================================================

def calculate_fair_probs(odds_dict):
    """
    Menghitung fair probability (tanpa margin) dari dictionary odds.
    Mengembalikan dict: {skor: fair_prob, "OTHER": fair_prob_other}
    """
    if not odds_dict:
        return {}
    
    # Hitung 1/odds untuk semua skor
    implied = {}
    other_odds = None
    for key, odds in odds_dict.items():
        if key == "OTHER":
            other_odds = odds
            continue
        if odds and odds > 1.0:
            implied[key] = 1.0 / odds
    
    if other_odds and other_odds > 1.0:
        implied["OTHER"] = 1.0 / other_odds
    
    total_implied = sum(implied.values())
    if total_implied <= 0:
        return {}
    
    # Normalisasi ke fair probability
    fair_probs = {k: v / total_implied for k, v in implied.items()}
    return fair_probs


def get_hybrid_top3(score_probs, fair_probs):
    """
    Gabungkan probabilitas model dengan fair probability pasar.
    score_probs: list of (h, a, prob)
    fair_probs: dict dari calculate_fair_probs
    """
    if not fair_probs:
        # Fallback ke Top 3 model biasa
        top3 = sorted(score_probs, key=lambda x: x[2], reverse=True)[:3]
        return [(int(h), int(a), float(p)) for h, a, p in top3]
    
    hybrid = []
    for h, a, prob in score_probs:
        key = f"{h}:{a}"
        # Cari fair probability pasar
        market_prob = fair_probs.get(key)
        if market_prob is None:
            # Coba fallback ke "OTHER" jika skor tidak ada di daftar spesifik
            market_prob = fair_probs.get("OTHER", 1.0)  # 1.0 = tidak mempengaruhi
        # Hybrid = model * pasar
        hybrid.append((h, a, prob * market_prob))
    
    # Ambil 3 teratas
    top3 = sorted(hybrid, key=lambda x: x[2], reverse=True)[:3]
    return [(int(h), int(a), float(p)) for h, a, p in top3]


# ============================================================
# FUNGSI BARU UNTUK 1X2
# ============================================================

def parse_odds_1x2_csv(file_content: bytes) -> dict:
    """
    Parse CSV odds 1X2.
    Format yang diterima: kolom Home, Draw, Away (dengan header) atau tiga kolom pertama.
    Mengembalikan dict: {'home': odds, 'draw': odds, 'away': odds}
    """
    df = pd.read_csv(BytesIO(file_content))
    # Coba deteksi kolom berdasarkan nama (case-insensitive)
    home_col = draw_col = away_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['home', '1']:
            home_col = col
        elif col_lower in ['draw', 'x', '0']:
            draw_col = col
        elif col_lower in ['away', '2']:
            away_col = col
    if home_col is None or draw_col is None or away_col is None:
        # Fallback: gunakan tiga kolom pertama
        if len(df.columns) >= 3:
            home_col = df.columns[0]
            draw_col = df.columns[1]
            away_col = df.columns[2]
        else:
            return {}
    # Ambil baris pertama
    row = df.iloc[0]
    try:
        return {
            'home': float(row[home_col]),
            'draw': float(row[draw_col]),
            'away': float(row[away_col])
        }
    except:
        return {}
