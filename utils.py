# utils.py
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

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


def _parse_ah_line(val: str) -> float:
    """Convert AH line from string like '0.5/1' to decimal."""
    val = val.strip()
    if '/' in val:
        parts = val.split('/')
        try:
            a = float(parts[0])
            b = float(parts[1])
            return (a + b) / 2.0
        except:
            return 0.0
    else:
        try:
            return float(val)
        except:
            return 0.0


def _convert_hk_odds(odds: List[float]) -> List[float]:
    """Convert Hong Kong odds to decimal (+1) if all < 2.0, else keep."""
    if all(o < 2.0 for o in odds):
        return [o + 1.0 for o in odds]
    return odds


def parse_combined_odds_csv(file_content: bytes) -> Dict[str, Any]:
    """Robust parser untuk format gabungan 1X2 + Correct Score + Asian Handicap + BTTS."""
    import csv

    result = {
        '1x2': None,
        'cs': None,
        'open_1x2': None,
        'ah': None,
        'btts': None,
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

    # Legacy simple 1X2
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
            idx_type = header.index('type')
            idx_score = header.index('score')
            idx_odds = header.index('odds')
            cs = {}
            for row in reader:
                # Baris kosong atau tidak cukup kolom dilewati
                if len(row) < 3:
                    continue
                # Hentikan jika bertemu header blok berikutnya (jumlah kolom bukan 3 atau mengandung kata kunci AH/BTTS)
                if len(row) != 3 or any(kw in norm(row[0]) for kw in ['ah_line', 'ah_home', 'btts_yes', 'btts_no']):
                    break
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

    # -------------------------
    # Asian Handicap block (setelah CS)
    # -------------------------
    ah_start = None
    for i in range(cs_start + 1 if cs_start else 0, len(lines)):
        try:
            row = [norm(x) for x in next(csv.reader([lines[i]]))]
        except Exception:
            continue
        if any('ah_line' in x or 'ah_home' in x or 'ah_away' in x for x in row):
            ah_start = i
            break

    if ah_start is not None:
        try:
            reader = csv.reader(lines[ah_start:])
            header = [norm(x) for x in next(reader)]
            def find_col(*names):
                for name in names:
                    if name in header:
                        return header.index(name)
                return None
            open_line_idx = find_col('open_ah_line')
            open_home_idx = find_col('open_ah_home')
            open_away_idx = find_col('open_ah_away')
            cur_line_idx = find_col('current_ah_line')
            cur_home_idx = find_col('current_ah_home')
            cur_away_idx = find_col('current_ah_away')
            if None in (open_line_idx, open_home_idx, open_away_idx, cur_line_idx, cur_home_idx, cur_away_idx):
                result['errors'].append('Kolom Asian Handicap tidak lengkap.')
            else:
                data_row = next(reader)
                open_line = _parse_ah_line(data_row[open_line_idx])
                cur_line = _parse_ah_line(data_row[cur_line_idx])
                raw_odds = [
                    float(data_row[open_home_idx]),
                    float(data_row[open_away_idx]),
                    float(data_row[cur_home_idx]),
                    float(data_row[cur_away_idx])
                ]
                odds_dec = _convert_hk_odds(raw_odds)
                result['ah'] = {
                    'open_line': open_line,
                    'open_home': odds_dec[0],
                    'open_away': odds_dec[1],
                    'current_line': cur_line,
                    'current_home': odds_dec[2],
                    'current_away': odds_dec[3]
                }
        except Exception as e:
            result['errors'].append(f'Asian Handicap tidak valid: {e}')

    # -------------------------
    # BTTS block (setelah AH atau setelah CS)
    # -------------------------
    btts_start = None
    search_start = (ah_start + 1) if ah_start else (cs_start + 1 if cs_start else 0)
    for i in range(search_start, len(lines)):
        try:
            row = [norm(x) for x in next(csv.reader([lines[i]]))]
        except Exception:
            continue
        if 'open_btts_yes' in row or 'current_btts_yes' in row:
            btts_start = i
            break

    if btts_start is not None:
        try:
            reader = csv.reader(lines[btts_start:])
            header = [norm(x) for x in next(reader)]
            def find_col(name):
                if name in header: return header.index(name)
                return None
            open_yes_idx = find_col('open_btts_yes')
            open_no_idx = find_col('open_btts_no')
            cur_yes_idx = find_col('current_btts_yes')
            cur_no_idx = find_col('current_btts_no')
            if None in (open_yes_idx, open_no_idx, cur_yes_idx, cur_no_idx):
                result['errors'].append('Kolom BTTS tidak lengkap.')
            else:
                data_row = next(reader)
                result['btts'] = {
                    'open_yes': float(data_row[open_yes_idx]),
                    'open_no': float(data_row[open_no_idx]),
                    'current_yes': float(data_row[cur_yes_idx]),
                    'current_no': float(data_row[cur_no_idx])
                }
        except Exception as e:
            result['errors'].append(f'BTTS tidak valid: {e}')

    if result['1x2'] is None:
        result['errors'].append('Odds 1X2 tidak ditemukan.')
    if result['cs'] is None:
        result['errors'].append('Odds Correct Score tidak ditemukan.')

    if result['1x2'] is not None and result['cs'] is not None:
        result['warnings'].append(f"Berhasil membaca 1X2 + {len(result['cs'])} Correct Score odds.")

    return result
