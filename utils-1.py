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
# FUNGSI HYBRID TOP 3 SCORE
# ============================================================

def calculate_fair_probs(odds_dict):
    if not odds_dict:
        return {}
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
    fair_probs = {k: v / total_implied for k, v in implied.items()}
    return fair_probs

def get_hybrid_top3(score_probs, fair_probs):
    if not fair_probs:
        top3 = sorted(score_probs, key=lambda x: x[2], reverse=True)[:3]
        return [(int(h), int(a), float(p)) for h, a, p in top3]
    hybrid = []
    for h, a, prob in score_probs:
        key = f"{h}:{a}"
        market_prob = fair_probs.get(key)
        if market_prob is None:
            market_prob = fair_probs.get("OTHER", 1.0)
        hybrid.append((h, a, prob * market_prob))
    top3 = sorted(hybrid, key=lambda x: x[2], reverse=True)[:3]
    return [(int(h), int(a), float(p)) for h, a, p in top3]

# ============================================================
# FUNGSI 1X2
# ============================================================

def parse_odds_1x2_csv(file_content: bytes) -> dict:
    df = pd.read_csv(BytesIO(file_content))
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
        if len(df.columns) >= 3:
            home_col = df.columns[0]
            draw_col = df.columns[1]
            away_col = df.columns[2]
        else:
            return {}
    row = df.iloc[0]
    try:
        return {
            'home': float(row[home_col]),
            'draw': float(row[draw_col]),
            'away': float(row[away_col])
        }
    except:
        return {}

def parse_combined_odds_csv(file_content: bytes) -> dict:
    """Robust parser for the combined 1X2 + Correct Score CSV format.

    Accepts the current production format used by the app, including a
    1X2 header/data block followed by a Type,Score,Odds block.  Parsing is
    intentionally tolerant of BOM, whitespace, capitalization and blank lines.
    Returns a diagnostics-rich dictionary so the UI can explain failures
    instead of silently doing nothing.
    """
    import csv

    result = {
        '1x2': None,
        'cs': None,
        'open_1x2': None,
        'errors': [],
        'warnings': [],
        'format': None,
    }

    if not file_content:
        result['errors'].append('File kosong.')
        return result

    try:
        text = file_content.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:
            text = file_content.decode('utf-8')
        except Exception as e:
            result['errors'].append(f'Encoding CSV tidak dapat dibaca: {e}')
            return result

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        result['errors'].append('CSV tidak memiliki baris data.')
        return result

    def norm(x):
        return str(x).strip().lower().replace('\ufeff', '')

    # -------------------------
    # 1X2 block
    # -------------------------
    for i, line in enumerate(lines):
        try:
            row = next(csv.reader([line]))
        except Exception:
            continue
        headers = [norm(x) for x in row]
        if {'open_1x2_home', 'open_1x2_draw', 'open_1x2_away',
            'current_1x2_home', 'current_1x2_draw', 'current_1x2_away'}.issubset(set(headers)):
            if i + 1 < len(lines):
                try:
                    vals = next(csv.reader([lines[i + 1]]))
                    mapping = {headers[j]: vals[j].strip() for j in range(min(len(headers), len(vals)))}
                    result['open_1x2'] = {
                        'home': float(mapping['open_1x2_home']),
                        'draw': float(mapping['open_1x2_draw']),
                        'away': float(mapping['open_1x2_away']),
                    }
                    result['1x2'] = {
                        'home': float(mapping['current_1x2_home']),
                        'draw': float(mapping['current_1x2_draw']),
                        'away': float(mapping['current_1x2_away']),
                    }
                    result['format'] = 'combined_open_current_1x2'
                except Exception as e:
                    result['errors'].append(f'1X2 open/current tidak valid: {e}')
            else:
                result['errors'].append('Baris data 1X2 tidak ditemukan.')
            break

    # Legacy/current simple 1X2 block, if the structured block was absent.
    if result['1x2'] is None:
        for i, line in enumerate(lines):
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            headers = [norm(x) for x in row]
            if {'home', 'draw', 'away'}.issubset(set(headers)) or {'1', 'x', '2'}.issubset(set(headers)):
                if i + 1 < len(lines):
                    try:
                        vals = next(csv.reader([lines[i + 1]]))
                        mapping = {headers[j]: vals[j].strip() for j in range(min(len(headers), len(vals)))}
                        hk, dk, ak = ('home', 'draw', 'away') if 'home' in headers else ('1', 'x', '2')
                        result['1x2'] = {
                            'home': float(mapping[hk]), 'draw': float(mapping[dk]), 'away': float(mapping[ak])
                        }
                        result['format'] = result['format'] or 'simple_1x2'
                    except Exception as e:
                        result['errors'].append(f'1X2 sederhana tidak valid: {e}')
                break

    # -------------------------
    # Correct Score block
    # -------------------------
    cs_start = None
    for i, line in enumerate(lines):
        try:
            row = [norm(x) for x in next(csv.reader([line]))]
        except Exception:
            continue
        if len(row) >= 3 and 'type' in row and 'score' in row and 'odds' in row:
            cs_start = i
            break

    if cs_start is not None:
        try:
            reader = csv.reader(lines[cs_start:])
            header = [norm(x) for x in next(reader)]
            idx_type, idx_score, idx_odds = header.index('type'), header.index('score'), header.index('odds')
            cs = {}
            for row in reader:
                if len(row) <= max(idx_type, idx_score, idx_odds):
                    continue
                score = str(row[idx_score]).strip().upper()
                bet_type = str(row[idx_type]).strip().lower()
                if not score:
                    continue
                try:
                    odd = float(str(row[idx_odds]).strip())
                except (TypeError, ValueError):
                    continue
                if odd <= 1.0:
                    continue
                # Preserve the application's existing parse_odds_csv semantics:
                # the source file expresses Away scores from the away side's
                # perspective (1:0 means 0:1 in Home:Away notation).
                if norm(score) == 'other':
                    key = 'OTHER'
                else:
                    parts = score.replace(' ', '').split(':')
                    if len(parts) == 2 and bet_type == 'away':
                        key = f"{parts[1]}:{parts[0]}"
                    else:
                        key = score.replace(' ', '')
                cs[key] = odd
            if cs:
                result['cs'] = cs
                result['format'] = result['format'] or 'correct_score'
            else:
                result['errors'].append('Bagian Correct Score ditemukan tetapi tidak ada odds valid.')
        except Exception as e:
            result['errors'].append(f'Correct Score tidak valid: {e}')

    if result['1x2'] is None:
        result['errors'].append('Odds 1X2 tidak ditemukan.')
    if result['cs'] is None:
        result['errors'].append('Odds Correct Score tidak ditemukan.')

    if result['1x2'] is not None and result['cs'] is not None:
        result['warnings'].append(f"Berhasil membaca 1X2 + {len(result['cs'])} Correct Score odds.")

    return result

