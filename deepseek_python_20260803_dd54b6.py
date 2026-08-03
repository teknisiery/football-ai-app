# ============================================================
# FOOTBALL AI V2 – PRODUCTION (DUAL STORAGE)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import base64
import requests
from io import BytesIO
from datetime import datetime, timedelta
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import poisson
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error
)
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import plotly.express as px

# ============================================================
# CONFIGURATION
# ============================================================
APP_TITLE = "Football AI V2"
APP_VERSION = "2.3.0"
BASE_DIR = Path(__file__).resolve().parent
EV_THRESHOLD_FILE = BASE_DIR / "ev_threshold.json"
PROFIT_SUMMARY_FILE = BASE_DIR / "profit_summary.json"

def load_league_round_config():
    config_file = BASE_DIR / "league_round_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
        except:
            pass
    return {
        390: {'teams': 20, 'matches_per_round': 10},
        40: {'teams': 16, 'matches_per_round': 8},
        22: {'teams': 16, 'matches_per_round': 8},
        11653: {'teams': 15, 'matches_per_round': 7},
        682: {'teams': 15, 'matches_per_round': 7},
        649: {'teams': 14, 'matches_per_round': 7},
        782: {'teams': 13, 'matches_per_round': 6},
        188: {'teams': 12, 'matches_per_round': 6},
        41: {'teams': 12, 'matches_per_round': 6},
        55: {'teams': 10, 'matches_per_round': 5},
        178: {'teams': 9, 'matches_per_round': 4},
        198: {'teams': 9, 'matches_per_round': 4},
        197: {'teams': 9, 'matches_per_round': 4},
        192: {'teams': 10, 'matches_per_round': 5},
        1275: {'teams': 8, 'matches_per_round': 3},
        20: {'teams': 16, 'matches_per_round': 8},
        46: {'teams': 16, 'matches_per_round': 8},
        16736: {'teams': 13, 'matches_per_round': 6},
        278: {'teams': 16, 'matches_per_round': 8},
        240: {'teams': 15, 'matches_per_round': 7},
        242: {'teams': 28, 'matches_per_round': 14},
        1240: {'teams': 16, 'matches_per_round': 8},
        11539: {'teams': 20, 'matches_per_round': 10},
        352: {'teams': 17, 'matches_per_round': 8},
        1260: {'teams': 9, 'matches_per_round': 4},
        1274: {'teams': 11, 'matches_per_round': 4},
    }

LEAGUE_ROUND_CONFIG = load_league_round_config()

EXPECTED_FEATURES = [
    'league_code', 'home_xg', 'away_xg', 'home_xga', 'away_xga',
    'last5_home_xg', 'last5_away_xg', 'last5_home_xga', 'last5_away_xga',
    'last5_home_avg_goals', 'last5_away_avg_goals',
    'last5_home_conceded', 'last5_away_conceded',
    'last5_home_over25', 'last5_away_over25',
    'last5_home_btts', 'last5_away_btts',
    'open_over_odds', 'open_under_odds',
    'current_over_odds', 'current_under_odds',
    'open_ou', 'current_ou', 'over_move', 'under_move',
    'xg_ratio_home', 'xg_ratio_away',
    'goal_diff_home', 'goal_diff_away',
    'xg_diff_home', 'xg_diff_away',
    'btts_potential', 'over25_potential',
    'odds_ratio', 'momentum_home', 'momentum_away',
    'xg_interact', 'odds_momentum',
    'league_avg_goals', 'league_over25_pct', 'league_btts_pct', 'league_under35_pct',
    'xghome_x_leagueavg', 'xgaway_x_leagueavg',
    'xgahome_x_leagueavg', 'xgaaway_x_leagueavg',
    'last5_home_xg_x_leagueavg', 'last5_away_xg_x_leagueavg',
    'last5_home_xga_x_leagueavg', 'last5_away_xga_x_leagueavg',
    'ou_line_x_leagueavg', 'over25_x_leaguepct', 'btts_x_leaguepct'
]

# ============================================================
# CUSTOM CSS
# ============================================================
def load_css():
    st.markdown("""
    <style>
    body { background: #0f1117; color: #ffffff; }
    .block-container { padding-top: 2rem; }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 24px;
        padding: 18px 20px 14px 20px;
        margin: 12px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 2px solid #2563eb;
    }
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 40px;
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-s { background: #0d6e0d; color: #ffffff; }
    .badge-a { background: #16a34a; color: #ffffff; }
    .badge-b { background: #2563eb; color: #ffffff; }
    .badge-c { background: #f97316; color: #ffffff; }
    .badge-d { background: #ef4444; color: #ffffff; }
    .brain-row {
        display: flex; flex-direction: row; justify-content: space-between;
        align-items: stretch; gap: 10px; flex-wrap: nowrap; width: 100%; margin: 10px 0;
    }
    .brain-card {
        flex: 1; border-radius: 16px; padding: 12px 6px; text-align: center;
        color: white; box-shadow: 0 6px 16px rgba(0,0,0,0.25);
        display: flex; flex-direction: column; justify-content: center; min-width: 0;
    }
    .brain-card .icon { font-size: 1.3rem; margin-bottom: 3px; }
    .brain-card .label { font-size: 0.6rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.3px; }
    .brain-card .badge-value {
        display: inline-block; padding: 4px 10px; border-radius: 12px;
        font-size: 1.2rem; font-weight: 800; margin-top: 4px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white; border-radius: 12px; border: none; padding: 12px 24px;
        font-weight: 700; font-size: 1rem; transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(37,99,235,0.6); }
    .stButton > button:disabled { background: #4b5563; box-shadow: none; transform: none; }
    .stFileUploader > div {
        border-radius: 16px; border: 2px dashed #4b5563; background: #1c1f26; padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# UTILS
# ============================================================
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
            st.error("Format CSV odds tidak dikenal. Harus memiliki kolom Score dan Odds.")
            return {}
    for _, row in df.iterrows():
        try:
            score_str = str(row[score_col]).strip()
            odds_val = float(row[odds_col])
            odds_dict[score_str] = odds_val
        except:
            pass
    return odds_dict

def get_cs_recommendations(top3_scores, odds_dict):
    """Ambil odds untuk Top 3 Skor, kembalikan list (h, a, odds)."""
    recs = []
    for h, a, prob in top3_scores:
        key = f"{h}:{a}"
        odds = odds_dict.get(key)
        if odds is not None:
            recs.append((h, a, odds))
    return recs[:3]

# ============================================================
# THRESHOLD SERVICE
# ============================================================
class ThresholdService:
    @staticmethod
    def get_thresholds(storage: Optional[Any] = None) -> Tuple[float, float]:
        if storage and storage.exists(ResourceRegistry.THRESHOLD):
            data = storage.load_json(ResourceRegistry.THRESHOLD)
            return data.get('ev_over', 0.01), data.get('ev_under', 0.02)
        if os.path.exists(EV_THRESHOLD_FILE):
            try:
                with open(EV_THRESHOLD_FILE) as f:
                    data = json.load(f)
                    return data.get('ev_over', 0.01), data.get('ev_under', 0.02)
            except:
                pass
        return 0.01, 0.02

    @staticmethod
    def get_btts_threshold(storage, league_code):
        if storage and storage.exists(ResourceRegistry.LEAGUE_PROFILE):
            df = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
            if 'btts_threshold' in df.columns:
                row = df[df['league_code'] == league_code]
                if not row.empty:
                    return row['btts_threshold'].values[0]
        return 0.22

# ============================================================
# PROFIT CALCULATOR
# ============================================================
class ProfitCalculator:
    @staticmethod
    def calculate(row: dict, stake: float = 100000.0) -> Tuple[float, str]:
        rec = str(row.get('recommendation', '')).strip().upper()
        stake_val = row.get('stake')
        if rec == 'NO BET' or (stake_val is not None and float(stake_val) == 0):
            return 0.0, "NO BET"
        if stake_val is not None and float(stake_val) > 0:
            effective_stake = float(stake_val)
        else:
            effective_stake = stake
        total_goals = int(row.get('home_goals', 0) or 0) + int(row.get('away_goals', 0) or 0)
        ou_line = float(row.get('current_ou', 2.5))
        pred_str = str(row.get('prediction', '')).strip().upper()
        if pred_str.startswith('OVER'):
            is_over = True
        elif pred_str.startswith('UNDER'):
            is_over = False
        else:
            return 0.0, "UNKNOWN"
        bet_type = "OVER" if is_over else "UNDER"
        odds = float(row.get('current_over_odds' if is_over else 'current_under_odds', 1.0))
        splits = split_quarter_line(ou_line)
        total_profit = 0.0
        results = []
        for line, weight in splits:
            result = settle_basic(total_goals, line, bet_type)
            results.append(result)
            if result == "WIN":
                total_profit += effective_stake * weight * (odds - 1)
            elif result == "LOSE":
                total_profit -= effective_stake * weight
        if len(results) == 1:
            if results[0] == "WIN": status = "FULL WIN"
            elif results[0] == "LOSE": status = "FULL LOSE"
            else: status = "PUSH"
        else:
            r1, r2 = results
            if r1 == "WIN" and r2 == "WIN": status = "FULL WIN"
            elif r1 == "LOSE" and r2 == "LOSE": status = "FULL LOSE"
            elif r1 == "PUSH" and r2 == "PUSH": status = "PUSH"
            elif (r1 == "WIN" and r2 == "PUSH") or (r1 == "PUSH" and r2 == "WIN"): status = "HALF WIN"
            elif (r1 == "LOSE" and r2 == "PUSH") or (r1 == "PUSH" and r2 == "LOSE"): status = "HALF LOSE"
            elif (r1 == "WIN" and r2 == "LOSE") or (r1 == "LOSE" and r2 == "WIN"):
                status = "PUSH"
                total_profit = 0.0
            else: status = "UNKNOWN"
        return total_profit, status

# ============================================================
# SPLIT HANDICAP SETTLEMENT (QUARTER LINE SUPPORT)
# ============================================================
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
# COMPUTE PROFITS
# ============================================================
@st.cache_data(ttl=300)
def compute_detailed_profits(history_df: pd.DataFrame):
    if history_df.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame(), {
            "total_bets": 0, "full_win": 0, "half_win": 0, "push": 0,
            "half_lose": 0, "full_lose": 0
        }, {}

    df = history_df.copy()
    pc = ProfitCalculator()
    profits = []
    statuses = []
    for _, row in df.iterrows():
        profit, status = pc.calculate(row.to_dict(), 100000)
        profits.append(profit)
        statuses.append(status)
    df['profit'] = profits
    df['result'] = statuses

    bet_df = df[df['result'] != 'NO BET'].copy()
    total_profit = bet_df['profit'].sum() if not bet_df.empty else 0.0

    if not bet_df.empty:
        status_counts = bet_df['result'].value_counts()
        summary = {
            'total_bets': len(bet_df),
            'full_win': status_counts.get('FULL WIN', 0),
            'half_win': status_counts.get('HALF WIN', 0),
            'push': status_counts.get('PUSH', 0),
            'half_lose': status_counts.get('HALF LOSE', 0),
            'full_lose': status_counts.get('FULL LOSE', 0)
        }
        profit_by_status = bet_df.groupby('result')['profit'].sum().to_dict()
    else:
        summary = {
            'total_bets': 0, 'full_win': 0, 'half_win': 0,
            'push': 0, 'half_lose': 0, 'full_lose': 0
        }
        profit_by_status = {}

    bet_df['_valid_time'] = bet_df.apply(lambda row: get_valid_time(row), axis=1)
    invalid = bet_df['_valid_time'].isna().sum()
    if invalid > 0:
        st.warning(f"⚠️ {invalid} baris memiliki timestamp tidak valid dan diabaikan dalam grafik bulanan.")
    bet_df = bet_df.dropna(subset=['_valid_time'])

    if not bet_df.empty:
        df_bulan = bet_df.copy()
        df_bulan['bulan'] = df_bulan['_valid_time'].dt.strftime('%Y-%m')
        df_bulan['bulan'] = df_bulan['bulan'].fillna('Tanpa Tanggal')
        monthly_groups = df_bulan.groupby('bulan')
        monthly_data = {}
        for bulan, group in monthly_groups:
            monthly_profit = group['profit'].sum()
            monthly_summary = {
                'total_bets': len(group),
                'full_win': group['result'].tolist().count('FULL WIN'),
                'half_win': group['result'].tolist().count('HALF WIN'),
                'push': group['result'].tolist().count('PUSH'),
                'half_lose': group['result'].tolist().count('HALF LOSE'),
                'full_lose': group['result'].tolist().count('FULL LOSE')
            }
            monthly_data[bulan] = {
                'profit': monthly_profit,
                'summary': monthly_summary,
                'df': group,
                'time_col': '_valid_time'
            }
    else:
        monthly_data = {}

    return total_profit, bet_df, summary, monthly_data, profit_by_status

def save_profit_summary(league_data: dict):
    slim = {}
    for league, data in league_data.items():
        slim[league] = {
            'profit': data['profit'],
            'summary': data['summary']
        }
    with open(PROFIT_SUMMARY_FILE, 'w') as f:
        json.dump(slim, f, indent=2)

def load_profit_summary():
    if PROFIT_SUMMARY_FILE.exists():
        try:
            with open(PROFIT_SUMMARY_FILE) as f:
                return json.load(f)
        except:
            pass
    return None

def compute_profits_by_league(history_df: pd.DataFrame):
    if history_df.empty:
        return {}

    df = history_df.copy()
    pc = ProfitCalculator()
    profits = []
    statuses = []
    for _, row in df.iterrows():
        profit, status = pc.calculate(row.to_dict(), 100000)
        profits.append(profit)
        statuses.append(status)
    df['profit'] = profits
    df['result'] = statuses

    bet_df = df[df['result'] != 'NO BET'].copy()
    if bet_df.empty:
        return {}

    if 'league_name' not in bet_df.columns:
        bet_df['league_name'] = 'Unknown'
    else:
        bet_df['league_name'] = bet_df['league_name'].str.title().str.strip()

    bet_df['_valid_time'] = bet_df.apply(lambda row: get_valid_time(row), axis=1)
    bet_df = bet_df.dropna(subset=['_valid_time'])

    league_groups = bet_df.groupby('league_name')
    league_data = {}
    for league, group in league_groups:
        profit = group['profit'].sum()
        summary = {
            'total_bets': len(group),
            'full_win': group['result'].tolist().count('FULL WIN'),
            'half_win': group['result'].tolist().count('HALF WIN'),
            'push': group['result'].tolist().count('PUSH'),
            'half_lose': group['result'].tolist().count('HALF LOSE'),
            'full_lose': group['result'].tolist().count('FULL LOSE')
        }
        league_data[league] = {
            'profit': profit,
            'summary': summary,
            'df': group,
            'time_col': '_valid_time'
        }

    save_profit_summary(league_data)
    return league_data

# ============================================================
# RESOURCE REGISTRY
# ============================================================
@dataclass(frozen=True)
class Resource:
    id: str
    type: str
    serializer: str
    default_filename: str
    version: int = 1
    description: str = ""

class ResourceRegistry:
    HISTORY = Resource("history", "dataframe", "csv", "history_ou.csv")
    DATASET = Resource("dataset", "dataframe", "csv", "dataset_ou.csv")
    DATASET_WITH_GOAL = Resource("dataset_with_goal", "dataframe", "csv", "dataset_withgoal.csv")
    PENDING = Resource("pending", "dataframe", "csv", "pending_predictions.csv")
    LEAGUE_STATS = Resource("league_statistics", "dataframe", "csv", "league_statistics.csv")
    LEAGUE_PROFILE = Resource("league_profile", "dataframe", "csv", "profil_league.csv")
    LEAGUE_PROFILE_HISTORY = Resource("league_profile_history", "dataframe", "csv", "league_profile_history.csv")
    THRESHOLD = Resource("threshold", "dict", "json", "ev_threshold.json")
    MODEL = Resource("model", "pickle", "pickle", "model.pkl")
    FEATURE_COLUMNS = Resource("feature_columns", "dict", "json", "feature_columns.json")
    LEAGUE_THRESHOLD = Resource("league_threshold", "dataframe", "csv", "league_threshold.csv")

OPTIONAL_RESOURCES = {
    ResourceRegistry.HISTORY.id, ResourceRegistry.DATASET.id,
    ResourceRegistry.DATASET_WITH_GOAL.id, ResourceRegistry.PENDING.id,
    ResourceRegistry.LEAGUE_STATS.id, ResourceRegistry.THRESHOLD.id,
    ResourceRegistry.FEATURE_COLUMNS.id, ResourceRegistry.LEAGUE_THRESHOLD.id,
    ResourceRegistry.LEAGUE_PROFILE.id, ResourceRegistry.LEAGUE_PROFILE_HISTORY.id,
}

# ============================================================
# STORAGE PROVIDERS
# ============================================================
class StorageProvider(ABC):
    @abstractmethod
    def load_dataframe(self, resource: Resource) -> pd.DataFrame: ...
    @abstractmethod
    def save_dataframe(self, resource: Resource, df: pd.DataFrame): ...
    @abstractmethod
    def load_json(self, resource: Resource) -> dict: ...
    @abstractmethod
    def save_json(self, resource: Resource, data: dict): ...
    @abstractmethod
    def load_pickle(self, resource: Resource) -> Any: ...
    @abstractmethod
    def save_pickle(self, resource: Resource, obj: Any): ...
    @abstractmethod
    def exists(self, resource: Resource) -> bool: ...
    @abstractmethod
    def delete(self, resource: Resource): ...

class LocalStorageProvider(StorageProvider):
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir
    def _path(self, r): return self.base_dir / r.default_filename
    def load_dataframe(self, r):
        p = self._path(r)
        if not p.exists():
            if r.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {r.id} not found at {p}")
        return pd.read_csv(p)
    def save_dataframe(self, r, df): df.to_csv(self._path(r), index=False)
    def load_json(self, r):
        p = self._path(r)
        if not p.exists():
            if r.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {r.id} not found at {p}")
        with open(p) as f: return json.load(f)
    def save_json(self, r, d):
        with open(self._path(r), 'w') as f: json.dump(d, f, indent=2)
    def load_pickle(self, r): return joblib.load(self._path(r))
    def save_pickle(self, r, o): joblib.dump(o, self._path(r))
    def exists(self, r): return self._path(r).exists()
    def delete(self, r): self._path(r).unlink(missing_ok=True)

class GitHubStorageProvider(StorageProvider):
    def __init__(self, owner, repo, branch, token):
        self.api = f"https://api.github.com/repos/{owner}/{repo}/contents"
        self.branch = branch
        self.token = token
    def _headers(self): return {"Authorization": f"token {self.token}", "Accept": "application/vnd.github.v3+json"}
    def _get_sha(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        return resp.json().get("sha") if resp.status_code == 200 else None
    def _crud(self, method, r, data=None):
        url = f"{self.api}/{r.default_filename}"
        sha = self._get_sha(r)
        payload = {"message": f"Update {r.id}", "branch": self.branch}
        if sha: payload["sha"] = sha
        if method == "put" and data: payload["content"] = base64.b64encode(data).decode()
        resp = requests.request(method, url, headers=self._headers(), json=payload)
        if resp.status_code == 409:
            sha = self._get_sha(r)
            if sha:
                payload["sha"] = sha
                resp = requests.request(method, url, headers=self._headers(), json=payload)
                resp.raise_for_status()
            else:
                raise RuntimeError("Conflict: file tidak ditemukan setelah konflik")
        else:
            resp.raise_for_status()
    def load_dataframe(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404:
            if r.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {r.id} not found in GitHub")
        resp.raise_for_status()
        content = base64.b64decode(resp.json()["content"])
        return pd.read_csv(BytesIO(content)) if content.strip() else pd.DataFrame()
    def save_dataframe(self, r, df): self._crud("put", r, df.to_csv(index=False).encode())
    def load_json(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404:
            if r.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {r.id} not found in GitHub")
        resp.raise_for_status()
        return json.loads(base64.b64decode(resp.json()["content"]))
    def save_json(self, r, d): self._crud("put", r, json.dumps(d, indent=2).encode())
    def load_pickle(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404: raise FileNotFoundError
        resp.raise_for_status()
        return joblib.load(BytesIO(base64.b64decode(resp.json()["content"])))
    def save_pickle(self, r, o):
        buf = BytesIO()
        joblib.dump(o, buf)
        self._crud("put", r, buf.getvalue())
    def exists(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        return requests.get(url, headers=self._headers()).status_code == 200
    def delete(self, r):
        sha = self._get_sha(r)
        if sha: requests.delete(f"{self.api}/{r.default_filename}", headers=self._headers(), json={"message":"delete","sha":sha,"branch":self.branch})

# ============================================================
# DATABASE MANAGER
# ============================================================
class DatabaseManager:
    def __init__(self, storage):
        self.storage = storage
    def load_history(self): return self.storage.load_dataframe(ResourceRegistry.HISTORY)
    def save_history(self, df): self.storage.save_dataframe(ResourceRegistry.HISTORY, df)
    def load_dataset(self): return self.storage.load_dataframe(ResourceRegistry.DATASET)
    def save_dataset(self, df): self.storage.save_dataframe(ResourceRegistry.DATASET, df)
    def load_dataset_with_goal(self): return self.storage.load_dataframe(ResourceRegistry.DATASET_WITH_GOAL)
    def save_dataset_with_goal(self, df): self.storage.save_dataframe(ResourceRegistry.DATASET_WITH_GOAL, df)
    def load_pending(self): return self.storage.load_dataframe(ResourceRegistry.PENDING)
    def save_pending(self, df): self.storage.save_dataframe(ResourceRegistry.PENDING, df)
    def load_model(self): return self.storage.load_pickle(ResourceRegistry.MODEL)
    def save_model(self, b): self.storage.save_pickle(ResourceRegistry.MODEL, b)
    def load_threshold(self): return self.storage.load_json(ResourceRegistry.THRESHOLD) if self.storage.exists(ResourceRegistry.THRESHOLD) else {}
    def save_threshold(self, d): self.storage.save_json(ResourceRegistry.THRESHOLD, d)
    def load_league_profile(self): return self.storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE) if self.storage.exists(ResourceRegistry.LEAGUE_PROFILE) else pd.DataFrame()
    def save_league_profile(self, df): self.storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, df)
    def is_model_ready(self): return self.storage.exists(ResourceRegistry.MODEL)

# ============================================================
# CONTRACTS
# ============================================================
class PendingContract:
    CORE_COLUMNS = [
        "match_uid","home_team","away_team","league_name","kickoff_time",
        "prediction","grade","confidence","prediction_id","prediction_time",
        "prediction_status","prediction_version","prediction_source","prediction_notes",
        "home_ht_goals","away_ht_goals","home_goals","away_goals",
        "actual_over","actual_btts","settlement_time",
        "recommendation","stake",
        "cs_score_1", "cs_odds_1", "cs_stake_1",
        "cs_score_2", "cs_odds_2", "cs_stake_2",
        "cs_score_3", "cs_odds_3", "cs_stake_3",
        "cs_profit"
    ]
    @classmethod
    def normalize(cls, df):
        for c in cls.CORE_COLUMNS:
            if c not in df.columns: df[c] = None
        return df

class ScoreValidator:
    @staticmethod
    def validate(row, ht_home, ht_away, ft_home, ft_away):
        errors = []
        if not row.get('match_uid'): errors.append("Match UID kosong")
        if not row.get('prediction'): errors.append("Prediction kosong")
        if ht_home is None or ht_away is None: errors.append("HT Score kosong")
        if ft_home is None or ft_away is None: errors.append("FT Score kosong")
        if (ht_home or 0)+(ht_away or 0) > (ft_home or 0)+(ft_away or 0): errors.append("HT > FT")
        return len(errors)==0, errors

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    if 'open_over_odds' in cols and 'current_over_odds' in cols:
        df['over_move'] = (df['current_over_odds'] - df['open_over_odds']).round(2)
    if 'open_under_odds' in cols and 'current_under_odds' in cols:
        df['under_move'] = (df['current_under_odds'] - df['open_under_odds']).round(2)
    if 'open_ou' in cols and 'current_ou' in cols:
        df['ou_movement'] = (df['current_ou'] - df['open_ou']).round(2)

    df['xg_ratio_home'] = (df.get('home_xg',0) / df.get('home_xga',1).replace(0, np.nan)).fillna(0)
    df['xg_ratio_away'] = (df.get('away_xg',0) / df.get('away_xga',1).replace(0, np.nan)).fillna(0)

    home_avg = df.get('last5_home_avg_goals', pd.Series(0, index=df.index))
    away_avg = df.get('last5_away_avg_goals', pd.Series(0, index=df.index))
    home_con = df.get('last5_home_conceded', pd.Series(0, index=df.index))
    away_con = df.get('last5_away_conceded', pd.Series(0, index=df.index))
    df['goal_diff_home'] = home_avg - home_con
    df['goal_diff_away'] = away_avg - away_con

    df['xg_diff_home'] = df['home_xg'] - df['home_xga']
    df['xg_diff_away'] = df['away_xg'] - df['away_xga']

    home_btts = df.get('last5_home_btts', pd.Series(0, index=df.index))
    away_btts = df.get('last5_away_btts', pd.Series(0, index=df.index))
    df['btts_potential'] = (home_btts + away_btts) / 2
    home_over = df.get('last5_home_over25', pd.Series(0, index=df.index))
    away_over = df.get('last5_away_over25', pd.Series(0, index=df.index))
    df['over25_potential'] = (home_over + away_over) / 2

    df['odds_ratio'] = (df['current_over_odds'] / df['current_under_odds'].replace(0, np.nan)).fillna(0)
    df['momentum_home'] = home_avg.fillna(0)
    df['momentum_away'] = away_avg.fillna(0)
    df['xg_interact'] = df['home_xg'] * df['away_xg']
    df['odds_momentum'] = df['over_move'] * df['momentum_home'] if 'over_move' in cols else 0

    interactions = [
        ('home_xg','league_avg_goals','xghome_x_leagueavg'),
        ('away_xg','league_avg_goals','xgaway_x_leagueavg'),
        ('home_xga','league_avg_goals','xgahome_x_leagueavg'),
        ('away_xga','league_avg_goals','xgaaway_x_leagueavg'),
        ('last5_home_xg','league_avg_goals','last5_home_xg_x_leagueavg'),
        ('last5_away_xg','league_avg_goals','last5_away_xg_x_leagueavg'),
        ('last5_home_xga','league_avg_goals','last5_home_xga_x_leagueavg'),
        ('last5_away_xga','league_avg_goals','last5_away_xga_x_leagueavg'),
        ('current_ou','league_avg_goals','ou_line_x_leagueavg'),
        ('last5_home_over25','league_over25_pct','over25_x_leaguepct'),
        ('last5_home_btts','league_btts_pct','btts_x_leaguepct')
    ]
    for c1, c2, name in interactions:
        if c1 in cols and c2 in cols: df[name] = df[c1] * df[c2]
        else: df[name] = 0.0

    df[df.select_dtypes(include=np.number).columns] = df.select_dtypes(include=np.number).fillna(0)
    return df

# ============================================================
# LEAGUE PROFILE (cache lewat AppState)
# ============================================================
def get_league_profile(storage: StorageProvider, session: 'SessionManager') -> pd.DataFrame:
    state = session.state
    if state.league_profile_cache is not None:
        return state.league_profile_cache
    if storage.exists(ResourceRegistry.LEAGUE_PROFILE):
        state.league_profile_cache = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
    else:
        state.league_profile_cache = pd.DataFrame()
    return state.league_profile_cache

def attach_league_profile(storage: StorageProvider, df: pd.DataFrame, session: 'SessionManager') -> pd.DataFrame:
    profile = get_league_profile(storage, session)
    if not profile.empty and 'league_code' in profile.columns and 'league_code' in df.columns:
        df = df.merge(profile, on='league_code', how='left', suffixes=('', '_profile'))
    defaults = {'league_avg_goals':2.5,'league_over25_pct':0.5,'league_btts_pct':0.5,'league_under35_pct':0.7,'league_name':'Unknown League'}
    for col, val in defaults.items():
        if col not in df.columns: df[col] = val
        else: df[col] = df[col].fillna(val)
    return df

def update_league_profile(storage: StorageProvider, league_code: int, session: 'SessionManager' = None):
    if not storage.exists(ResourceRegistry.DATASET_WITH_GOAL): return
    df = storage.load_dataframe(ResourceRegistry.DATASET_WITH_GOAL)
    if 'league_name' in df.columns:
        df['league_name'] = df['league_name'].str.title().str.strip()

    if 'totalgol_ft' not in df.columns:
        if 'home_goals' in df.columns: df['totalgol_ft'] = df['home_goals'] + df['away_goals']
        else: return
    if 'totalgol_ht' not in df.columns:
        df['totalgol_ht'] = df['home_ht_goals'] + df['away_ht_goals'] if 'home_ht_goals' in df.columns else 0
    df = df.dropna(subset=['totalgol_ft'])
    df_league = df[df['league_code'] == league_code]
    if df_league.empty: return
    config = LEAGUE_ROUND_CONFIG.get(league_code)
    if config and len(df_league) % config['matches_per_round'] != 0: return

    df_league['btts'] = ((df_league['home_goals']>0)&(df_league['away_goals']>0)).astype(int)
    df_league['ht0'] = (df_league['totalgol_ht']==0).astype(int)
    total = len(df_league)

    profile_df = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE) if storage.exists(ResourceRegistry.LEAGUE_PROFILE) else pd.DataFrame()

    existing_name = None
    if not profile_df.empty and 'league_code' in profile_df.columns and league_code in profile_df['league_code'].values:
        existing_name = profile_df[profile_df['league_code'] == league_code]['league_name'].iloc[0]

    if existing_name:
        league_name_final = existing_name
    elif 'league_name' in df_league.columns and not str(df_league['league_name'].iloc[0]).startswith('League '):
        league_name_final = df_league['league_name'].iloc[0]
    else:
        league_name_final = f'League {league_code}'

    new_row = {
        'league_code': league_code,
        'league_name': league_name_final,
        'league_avg_goals': df_league['totalgol_ft'].mean(),
        'league_over25_pct': (df_league['totalgol_ft']>2.5).mean(),
        'league_btts_pct': df_league['btts'].mean(),
        'league_under35_pct': (df_league['totalgol_ft']<3.5).mean(),
        'eg_p25': df_league['totalgol_ft'].quantile(0.25),
        'eg_p75': df_league['totalgol_ft'].quantile(0.75),
        'btts_p25':0.0,'btts_p75':0.0,'ht0_p25':0.0,'ht0_p75':0.0,
        'ev_over_threshold':0.01,'ev_under_threshold':0.02,'total_matches':total,
        'btts_threshold': 0.22
    }

    if league_code not in profile_df['league_code'].values:
        profile_df = pd.concat([profile_df, pd.DataFrame([new_row])])
    else:
        idx = profile_df[profile_df['league_code']==league_code].index[0]
        for k,v in new_row.items():
            if k != 'btts_threshold' or 'btts_threshold' not in profile_df.columns:
                profile_df.at[idx,k] = v

    storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profile_df)
    if session:
        session.invalidate_league_profile_cache()

    history_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'league_code': league_code,
        'league_name': league_name_final,
        'league_avg_goals': new_row['league_avg_goals'],
        'league_over25_pct': new_row['league_over25_pct'],
        'league_btts_pct': new_row['league_btts_pct'],
        'league_under35_pct': new_row['league_under35_pct'],
        'total_matches': total
    }
    history_df = pd.DataFrame([history_row])
    if storage.exists(ResourceRegistry.LEAGUE_PROFILE_HISTORY):
        existing_hist = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE_HISTORY)
        history_df = pd.concat([existing_hist, history_df], ignore_index=True)
    storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE_HISTORY, history_df)

def add_new_league(league_code, league_name, avg_goals, over25_pct, btts_pct, under35_pct,
                   teams, matches_per_round, db_storage, app_storage):
    try:
        profil = db_storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
    except:
        profil = pd.DataFrame()

    if league_code in profil['league_code'].values:
        return False, "Kode liga sudah ada di profil."

    new_row = {
        'league_code': league_code,
        'league_name': league_name,
        'league_avg_goals': avg_goals,
        'league_over25_pct': over25_pct,
        'league_btts_pct': btts_pct,
        'league_under35_pct': under35_pct,
        'eg_p25': 0.0, 'eg_p75': 0.0,
        'btts_p25': 0.0, 'btts_p75': 0.0,
        'ht0_p25': 0.0, 'ht0_p75': 0.0,
        'ev_over_threshold': 0.01, 'ev_under_threshold': 0.02,
        'total_matches': 0,
        'btts_threshold': 0.22
    }
    profil = pd.concat([profil, pd.DataFrame([new_row])], ignore_index=True)
    db_storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profil)

    config_file = BASE_DIR / "league_round_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}

    config[str(league_code)] = {'teams': teams, 'matches_per_round': matches_per_round}
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    global LEAGUE_ROUND_CONFIG
    LEAGUE_ROUND_CONFIG = {int(k): v for k, v in config.items()}

    return True, "Liga baru berhasil ditambahkan."

# ============================================================
# PREDICTION ENGINE (dengan Task 10 & score_probs)
# ============================================================
@dataclass
class PredictionResult:
    expected_goal: float
    prob_over: float
    prob_under: float
    prob_over_raw: float
    ev_over: float
    ev_under: float
    kelly_over: float
    kelly_under: float
    prob_btts: Optional[float]
    prob_ht0: Optional[float]
    confidence_ou: float
    confidence_btts: Optional[float]
    confidence_ht0: Optional[float]
    prediction_ou: int
    prediction_btts: int
    prediction_ht0: int
    top3_scores: Optional[List[Tuple[int, int, float]]] = None
    score_probs: Optional[List[Tuple[int, int, float]]] = None

class Predictor:
    def __init__(self, bundle: dict, league_profile: pd.DataFrame = None):
        self.model_ou = bundle.get('model') or bundle.get('model_ou')
        self.model_btts = bundle.get('model_btts')
        self.btts_calibrator = bundle.get('btts_calibrator')
        self.feature_cols = bundle['feature_cols']
        self.calibrator = bundle.get('calibrator')
        self.league_profile = league_profile
        self.btts_available = self.model_btts is not None

    def _get_btts_threshold(self, league_code):
        if self.league_profile is not None and 'btts_threshold' in self.league_profile.columns:
            row = self.league_profile[self.league_profile['league_code'] == league_code]
            if not row.empty:
                return row['btts_threshold'].values[0]
        return 0.22

    def predict(self, features_df: pd.DataFrame) -> PredictionResult:
        X = features_df[self.feature_cols].fillna(0)
        lam = max(0.0, self.model_ou.predict(X)[0])
        ou_line = float(features_df['current_ou'].iloc[0])
        over_odds = float(features_df['current_over_odds'].iloc[0])
        under_odds = float(features_df['current_under_odds'].iloc[0])

        try:
            home_xg = float(features_df.get('home_xg', pd.Series([lam/2])).iloc[0])
            away_xg = float(features_df.get('away_xg', pd.Series([lam/2])).iloc[0])
        except:
            home_xg = lam / 2
            away_xg = lam / 2

        total_xg_input = home_xg + away_xg
        if total_xg_input > 0:
            lam_home = lam * (home_xg / total_xg_input)
            lam_away = lam * (away_xg / total_xg_input)
        else:
            lam_home = lam / 2
            lam_away = lam / 2

        rho = -0.1
        max_goals = 7

        prob_over_win = 0.0
        prob_under_win = 0.0
        prob_push = 0.0
        scores = []

        for h in range(0, max_goals+1):
            for a in range(0, max_goals+1):
                prob = poisson.pmf(h, lam_home) * poisson.pmf(a, lam_away)

                if h == 0 and a == 0:
                    prob *= (1 - lam_home * lam_away * rho)
                elif h == 1 and a == 0:
                    prob *= (1 + lam_away * rho)
                elif h == 0 and a == 1:
                    prob *= (1 + lam_home * rho)
                elif h == 1 and a == 1:
                    prob *= (1 - rho)

                scores.append((h, a, prob))

                total_goals = h + a
                if total_goals > ou_line:
                    prob_over_win += prob
                elif total_goals < ou_line:
                    prob_under_win += prob
                else:
                    prob_push += prob

        all_score_probs = [(int(h), int(a), float(p)) for h, a, p in scores]

        prob_btts = None
        if self.btts_available:
            if self.btts_calibrator:
                prob_btts = self.btts_calibrator.predict_proba(X)[0, 1]
            else:
                prob_btts = self.model_btts.predict_proba(X)[0, 1]
            conf_btts = max(prob_btts, 1 - prob_btts)
            league_code = features_df['league_code'].iloc[0]
            btts_threshold = self._get_btts_threshold(league_code)
            pred_btts = int(prob_btts >= btts_threshold)
        else:
            prob_btts = None
            conf_btts = None
            pred_btts = -1

        if self.btts_available and prob_btts is not None:
            weighted_scores = []
            for h, a, p in scores:
                is_btts = (h > 0 and a > 0)
                weight = prob_btts if is_btts else (1 - prob_btts)
                weighted_scores.append((h, a, p * weight))
            top3 = sorted(weighted_scores, key=lambda x: x[2], reverse=True)[:3]
        else:
            top3 = sorted(scores, key=lambda x: x[2], reverse=True)[:3]

        prob_raw = prob_over_win
        prob_over = prob_raw
        if self.calibrator:
            prob_over = np.clip(self.calibrator.predict([[prob_raw]])[0], 0, 1)

        if ou_line % 1 == 0:
            ev_over = (prob_over_win * over_odds) + (prob_push * 1) - 1
            ev_under = (prob_under_win * under_odds) + (prob_push * 1) - 1
        else:
            ev_over = (prob_over_win * over_odds) - 1
            ev_under = (prob_under_win * under_odds) - 1

        k_over = calc_kelly(prob_over, over_odds)
        k_under = calc_kelly(1 - prob_over, under_odds)

        pred_ou = int(prob_over >= 0.10)

        return PredictionResult(
            expected_goal=lam,
            prob_over=prob_over,
            prob_under=1 - prob_over,
            prob_over_raw=prob_raw,
            ev_over=ev_over, ev_under=ev_under,
            kelly_over=k_over, kelly_under=k_under,
            prob_btts=prob_btts, prob_ht0=None,
            confidence_ou=max(prob_over, 1 - prob_over),
            confidence_btts=conf_btts, confidence_ht0=None,
            prediction_ou=pred_ou,
            prediction_btts=pred_btts, prediction_ht0=-1,
            top3_scores=[(int(h), int(a), float(p)) for h, a, p in top3],
            score_probs=all_score_probs
        )

# ============================================================
# LIVE BET RECOMMENDATION
# ============================================================
def calculate_live_recommendation(
    lam_total: float, home_xg: float, away_xg: float,
    menit_berjalan: float, home_goals_live: int, away_goals_live: int,
    current_ou: float, current_over_odds: float, current_under_odds: float,
    storage: StorageProvider, max_goals: int = 7
) -> Dict[str, Any]:
    total_xg = home_xg + away_xg
    if total_xg > 0:
        lam_home = lam_total * (home_xg / total_xg)
        lam_away = lam_total * (away_xg / total_xg)
    else:
        lam_home = lam_total / 2
        lam_away = lam_total / 2

    menit_tersisa = max(0.0, 90.0 - menit_berjalan)
    lam_home_sisa = lam_home * (menit_tersisa / 90.0)
    lam_away_sisa = lam_away * (menit_tersisa / 90.0)

    faktor_home = 1.0
    faktor_away = 1.0
    if home_goals_live < away_goals_live:
        if home_xg > away_xg:
            faktor_home = 1.15; faktor_away = 0.95
        else:
            faktor_home = 0.95; faktor_away = 1.10
    elif home_goals_live > away_goals_live:
        if home_xg < away_xg:
            faktor_home = 0.90; faktor_away = 1.10
        else:
            faktor_home = 1.0; faktor_away = 0.95
    lam_home_adj = lam_home_sisa * faktor_home
    lam_away_adj = lam_away_sisa * faktor_away

    rho = -0.1
    prob_over_win = 0.0
    prob_under_win = 0.0
    prob_push = 0.0
    for h in range(0, max_goals+1):
        for a in range(0, max_goals+1):
            prob = poisson.pmf(h, lam_home_adj) * poisson.pmf(a, lam_away_adj)
            if h == 0 and a == 0:
                prob *= (1 - lam_home_adj * lam_away_adj * rho)
            elif h == 1 and a == 0:
                prob *= (1 + lam_away_adj * rho)
            elif h == 0 and a == 1:
                prob *= (1 + lam_home_adj * rho)
            elif h == 1 and a == 1:
                prob *= (1 - rho)
            total_gol = h + a
            if total_gol > current_ou: prob_over_win += prob
            elif total_gol < current_ou: prob_under_win += prob
            else: prob_push += prob

    prob_over = prob_over_win
    if current_ou % 1 == 0:
        ev_over = (prob_over_win * current_over_odds) + (prob_push * 1) - 1
        ev_under = (prob_under_win * current_under_odds) + (prob_push * 1) - 1
    else:
        ev_over = (prob_over_win * current_over_odds) - 1
        ev_under = (prob_under_win * current_under_odds) - 1

    kelly_over = calc_kelly(prob_over, current_over_odds)
    kelly_under = calc_kelly(1 - prob_over, current_under_odds)

    ev_th_over, ev_th_under = ThresholdService.get_thresholds(storage)
    if prob_over >= 0.10 and ev_over > ev_th_over:
        rec = "OVER"
    elif (1 - prob_over) >= 0.10 and ev_under > ev_th_under:
        rec = "UNDER"
    else:
        rec = "NO BET"

    prob_home_score = 1 - poisson.pmf(0, lam_home_adj)
    prob_away_score = 1 - poisson.pmf(0, lam_away_adj)
    prob_btts_yes = prob_home_score * prob_away_score
    confidence_btts = max(prob_btts_yes, 1 - prob_btts_yes)

    return {
        "prob_over": prob_over,
        "ev_over": ev_over,
        "ev_under": ev_under,
        "kelly_over": kelly_over,
        "kelly_under": kelly_under,
        "recommendation": rec,
        "prob_btts_yes": prob_btts_yes,
        "btts_pred": "YES" if prob_btts_yes >= 0.5 else "NO",
        "confidence_btts": confidence_btts
    }

# ============================================================
# SESSION MANAGER (AppState terpusat)
# ============================================================
@dataclass
class UploadState:
    uploaded: bool = False
    filename: str = ""
    rows: int = 0
    columns: int = 0
    upload_time: str = ""
    uploaded_data: Optional[pd.DataFrame] = None

@dataclass
class PredictionState:
    processed: bool = False
    status: str = "NOT_PROCESSED"
    prediction_count: int = 0
    prediction_dataframe: Optional[pd.DataFrame] = None
    prediction_result: Optional[dict] = None

@dataclass
class AppState:
    upload: UploadState = field(default_factory=UploadState)
    prediction: PredictionState = field(default_factory=PredictionState)
    live_results: Dict[str, dict] = field(default_factory=dict)
    analysis_needed: bool = False
    debug_trace: List[str] = field(default_factory=list)
    league_profile_cache: Optional[pd.DataFrame] = None
    uploaded_file: Optional[object] = None
    uploaded_df: Optional[pd.DataFrame] = None
    uploaded_odds: Optional[dict] = None

class SessionManager:
    STATE_KEY = "app_state"
    def __init__(self):
        if self.STATE_KEY not in st.session_state:
            st.session_state[self.STATE_KEY] = AppState()
    @property
    def state(self) -> AppState:
        return st.session_state[self.STATE_KEY]
    def get_upload_state(self): return self.state.upload
    def set_upload_state(self, s): self.state.upload = s
    def get_prediction_state(self): return self.state.prediction
    def set_prediction_state(self, s): self.state.prediction = s
    def get_live_result(self, match_uid): return self.state.live_results.get(match_uid)
    def set_live_result(self, match_uid, result): self.state.live_results[match_uid] = result
    @property
    def analysis_needed(self): return self.state.analysis_needed
    @analysis_needed.setter
    def analysis_needed(self, val): self.state.analysis_needed = val
    @property
    def uploaded_file(self): return self.state.uploaded_file
    @uploaded_file.setter
    def uploaded_file(self, file): self.state.uploaded_file = file
    @property
    def uploaded_df(self): return self.state.uploaded_df
    @uploaded_df.setter
    def uploaded_df(self, df): self.state.uploaded_df = df
    @property
    def uploaded_odds(self) -> Optional[dict]:
        return self.state.uploaded_odds
    @uploaded_odds.setter
    def uploaded_odds(self, odds: Optional[dict]):
        self.state.uploaded_odds = odds
    def add_debug(self, msg):
        self.state.debug_trace.append(msg)
        if len(self.state.debug_trace) > 500:
            self.state.debug_trace = self.state.debug_trace[-500:]
    def get_debug_trace(self): return self.state.debug_trace
    def invalidate_league_profile_cache(self):
        self.state.league_profile_cache = None
    def reset_upload_and_prediction(self):
        self.set_upload_state(UploadState())
        self.set_prediction_state(PredictionState())
        self.analysis_needed = False
        self.uploaded_file = None
        self.uploaded_df = None
        self.uploaded_odds = None

# ============================================================
# UI COMPONENTS
# ============================================================
def render_horizontal_metric_row(cards):
    divs = []
    for icon, label, value, bg in cards:
        divs.append(f'<div class="brain-card" style="background:{bg};"><div class="icon">{icon}</div><div class="label">{safe_html(label)}</div><div class="badge-value">{safe_html(str(value))}</div></div>')
    st.markdown(f'<div class="brain-row">{"".join(divs)}</div>', unsafe_allow_html=True)

def render_prediction_card(summary: dict):
    if not summary: return
    home = safe_html(summary['home'])
    away = safe_html(summary['away'])
    league = safe_html(summary['league'])
    ou_pred = safe_html(summary.get('ou_pred',''))
    ou_line = summary.get('ou_line','')
    rec = safe_html(summary.get('recommendation',''))
    rec_color = summary.get('rec_color','d')
    over_odds = summary.get('over_odds',0)
    under_odds = summary.get('under_odds',0)

    st.markdown(f"""<div class="prediction-card"><div style="text-align:center;">
        <h3>⚽ {home} vs {away}</h3><p style="color:#a0a0b0;">{league}</p></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: st.markdown(f"<div style='text-align:center;font-size:2.5rem;font-weight:900;'>{ou_pred} {ou_line}</div><span class='badge badge-{rec_color}'>{rec}</span>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div style='text-align:center;padding-top:10px;'><p style='color:#a0a0b0;'>Odds</p><p>Over: {over_odds:.2f}</p><p>Under: {under_odds:.2f}</p></div>", unsafe_allow_html=True)

    def eg_c(v): return "#16a34a" if v>=2.8 else "#eab308" if v>=2.0 else "#ef4444"
    def ev_c(v): return "#16a34a" if v>0.02 else "#eab308" if v>0 else "#ef4444"
    def k_c(v): return "#16a34a" if v>0.1 else "#eab308" if v>0.05 else "#ef4444"
    def btts_c(p): return "#16a34a" if p>0.6 else "#eab308" if p>0.4 else "#ef4444"

    render_horizontal_metric_row([
        ("⚽","Expected Goal",f"{summary.get('expected_goal',0):.2f}",eg_c(summary.get('expected_goal',0))),
        ("📈","Confidence",f"{summary.get('confidence_ou',0):.0%}","#16a34a"),
        ("💰","EV Over",f"{summary.get('ev_over',0):+.3f}",ev_c(summary.get('ev_over',0))),
        ("💰","EV Under",f"{summary.get('ev_under',0):+.3f}",ev_c(summary.get('ev_under',0))),
    ])
    stake_val = summary.get('stake',0)
    render_horizontal_metric_row([
        ("📊","Kelly Over",f"{summary.get('kelly_over',0):.1%}",k_c(summary.get('kelly_over',0))),
        ("📊","Kelly Under",f"{summary.get('kelly_under',0):.1%}",k_c(summary.get('kelly_under',0))),
        ("🤝","BTTS",f"{safe_html(str(summary.get('btts_pred','N/A')))} ({summary.get('confidence_btts',0):.0%})" if summary.get('confidence_btts') is not None else "N/A", btts_c(summary.get('confidence_btts',0) if summary.get('confidence_btts') is not None else 0.5)),
        ("💲","Stake",f"Rp{stake_val:,.0f}" if stake_val>0 else "Rp0","#16a34a" if stake_val>0 else "#6b7280"),
    ])

    if summary.get('top3_scores'):
        st.markdown("**🔮 Top 3 Skor:**")
        cols_top = st.columns(3)
        for i, (h, a, prob) in enumerate(summary['top3_scores']):
            with cols_top[i]:
                st.markdown(f"**{h}‑{a}**")
                st.caption(f"{prob*100:.1f}%")

    # Rekomendasi Correct Score (menggunakan Top 3 langsung)
    if summary.get('cs_recommendations'):
        st.markdown("---")
        st.markdown("**🎯 Rekomendasi Correct Score**")
        recs = summary['cs_recommendations']
        stakes = [30000, 40000, 50000]
        for i, (h, a, odds) in enumerate(recs):
            stake = stakes[i]
            st.markdown(f"**{h}-{a}** | Odds: {odds:.2f} | Stake: Rp{stake:,}")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGES
# ============================================================
def serialize_prediction(result: PredictionResult) -> dict:
    return {
        'expected_goal': result.expected_goal,
        'prob_over': result.prob_over,
        'ev_over': result.ev_over,
        'ev_under': result.ev_under,
        'kelly_over': result.kelly_over,
        'kelly_under': result.kelly_under,
        'prob_btts': result.prob_btts,
        'confidence_ou': result.confidence_ou,
        'confidence_btts': result.confidence_btts,
        'prediction_ou': result.prediction_ou,
        'prediction_btts': result.prediction_btts,
        'top3_scores_json': json.dumps(result.top3_scores) if result.top3_scores else None,
        'score_probs': result.score_probs
    }

def render_upload_section(session: SessionManager):
    with st.expander("📤 Upload File CSV", expanded=True):
        f = st.file_uploader("Pilih CSV", type=["csv"])
        if f:
            df = pd.read_csv(f)
            required = ['home_xg', 'away_xg', 'current_over_odds', 'current_under_odds', 'current_ou']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}. File tidak valid.")
                return
            session.uploaded_file = f
            session.uploaded_df = df
            session.set_prediction_state(PredictionState())
            session.analysis_needed = True
            st.rerun()

def run_analysis(session: SessionManager, storage: StorageProvider, predictor: Predictor):
    df = session.uploaded_df.copy()
    home, away = session.uploaded_file.name.replace('.csv','').split(' vs ')
    df = attach_league_profile(storage, df, session)
    df = add_features(df)
    result = predictor.predict(df)
    session.set_prediction_state(PredictionState(processed=True, prediction_dataframe=df, prediction_result=result.__dict__))

    if 'match_uid' not in df.columns:
        kickoff = df['kickoff_time'].iloc[0] if 'kickoff_time' in df.columns else datetime.now().strftime("%Y-%m-%d %H:%M")
        df['match_uid'] = f"{home}|{away}|{kickoff}"
    if 'home_team' not in df.columns:
        df['home_team'] = home
    if 'away_team' not in df.columns:
        df['away_team'] = away

    df_clean = df.copy()
    if 'league_name' in df_clean.columns:
        df_clean = df_clean.drop(columns=['league_name'])
    df_clean = reorder_columns(df_clean)

    if storage.exists(ResourceRegistry.DATASET):
        existing = storage.load_dataframe(ResourceRegistry.DATASET)
        if not existing.empty and 'match_uid' in existing.columns and 'match_uid' in df_clean.columns:
            new_uid = df_clean['match_uid'].iloc[0]
            if new_uid not in existing['match_uid'].values:
                combined = pd.concat([existing, df_clean], ignore_index=True)
                storage.save_dataframe(ResourceRegistry.DATASET, combined)
        else:
            combined = pd.concat([existing, df_clean], ignore_index=True)
            storage.save_dataframe(ResourceRegistry.DATASET, combined)
    else:
        storage.save_dataframe(ResourceRegistry.DATASET, df_clean)

def get_match_prediction_summary(session: SessionManager, storage: StorageProvider):
    ps = session.get_prediction_state()
    if not ps.processed: return None
    r = ps.prediction_result
    df = ps.prediction_dataframe.iloc[0]
    home, away = session.uploaded_file.name.replace('.csv','').split(' vs ')
    ou_line = df['current_ou']
    over_odds = df['current_over_odds']
    under_odds = df['current_under_odds']

    prob_over = r['prob_over']
    ev_over = r['ev_over']
    ev_under = r['ev_under']

    ev_th_over, ev_th_under = ThresholdService.get_thresholds(storage)

    if prob_over >= 0.10 and ev_over > ev_th_over:
        ou_pred, rec, rec_color, stake = "OVER", "TARUHAN OVER", "a", 100000
    elif (1 - prob_over) >= 0.10 and ev_under > ev_th_under:
        ou_pred, rec, rec_color, stake = "UNDER", "TARUHAN UNDER", "a", 100000
    else:
        ou_pred, rec, rec_color, stake = ("OVER" if prob_over >= 0.5 else "UNDER"), "NO BET", "d", 0

    # Rekomendasi Correct Score: gunakan Top 3 Skor yang sudah dihitung
    cs_recs = None
    odds_dict = session.uploaded_odds
    if odds_dict and r.get('top3_scores'):
        cs_recs = get_cs_recommendations(r['top3_scores'], odds_dict)

    return {
        "home": home, "away": away, "league": safe_html(str(df.get('league_name','Unknown'))),
        "ou_pred": ou_pred, "ou_line": ou_line, "over_odds": over_odds, "under_odds": under_odds,
        "expected_goal": r['expected_goal'], "confidence_ou": r['confidence_ou'],
        "ev_over": ev_over, "ev_under": ev_under,
        "kelly_over": r['kelly_over'], "kelly_under": r['kelly_under'],
        "recommendation": rec, "rec_color": rec_color, "stake": stake,
        "btts_pred": "YES" if r.get('prediction_btts', -1) == 1 else ("NO" if r.get('prediction_btts', -1) == 0 else "N/A"),
        "confidence_btts": r.get('confidence_btts'),
        "ht0_pred": "N/A", "insights": [],
        "top3_scores": r.get('top3_scores', []),
        "cs_recommendations": cs_recs
    }

def render_pending(session: SessionManager, storage: StorageProvider):
    st.subheader("📋 Pending Manager")
    db = DatabaseManager(storage)
    df = db.load_pending()
    if df.empty: st.info("Belum ada data."); return

    with st.expander("Status", expanded=False):
        status = st.selectbox("Status", ["ALL","PENDING","SCORE_ENTERED","VALIDATED"])
    tab_l, tab_c = st.tabs(["League","Confidence"])
    with tab_l: league = st.selectbox("League", ["ALL"]+sorted(df['league_name'].dropna().unique()))
    with tab_c: conf = st.radio("Confidence", ["ALL",">70%",">80%",">90%"], horizontal=True)
    if status!="ALL": df = df[df['prediction_status']==status]
    if league!="ALL": df = df[df['league_name']==league]
    if conf==">70%": df = df[df['confidence_ou']>0.7]
    elif conf==">80%": df = df[df['confidence_ou']>0.8]
    elif conf==">90%": df = df[df['confidence_ou']>0.9]

    for idx, row in df.iterrows():
        home_safe = safe_html(row['home_team'])
        away_safe = safe_html(row['away_team'])
        match_uid = row['match_uid']

        pred_str = str(row.get('prediction', '')).strip()
        rec_str = str(row.get('recommendation', '')).strip()
        stake_val = row.get('stake')
        info = f"{pred_str} ({rec_str} - Rp{int(float(stake_val)):,})" if (stake_val is not None and float(stake_val) > 0) else f"{pred_str} (NO BET)"

        with st.expander(f"▶ {home_safe} vs {away_safe} ({info})"):
            def _get_initial(val):
                if pd.isna(val) or val is None: return None
                try: return int(float(val))
                except: return None

            init_ht_h = _get_initial(row.get('home_ht_goals'))
            init_ht_a = _get_initial(row.get('away_ht_goals'))
            init_ft_h = _get_initial(row.get('home_goals'))
            init_ft_a = _get_initial(row.get('away_goals'))

            c1,c2 = st.columns(2)
            with c1: ht_h = st.number_input("HT Home", value=init_ht_h, key=f"hth_{match_uid}", min_value=0, step=1)
            with c2: ht_a = st.number_input("HT Away", value=init_ht_a, key=f"hta_{match_uid}", min_value=0, step=1)
            c3,c4 = st.columns(2)
            with c3: ft_h = st.number_input("FT Home", value=init_ft_h, key=f"fth_{match_uid}", min_value=0, step=1)
            with c4: ft_a = st.number_input("FT Away", value=init_ft_a, key=f"fta_{match_uid}", min_value=0, step=1)

            processing_key = f"processing_{match_uid}"
            if processing_key not in st.session_state:
                st.session_state[processing_key] = False

            def _save_callback(match_uid=match_uid):
                st.session_state[f"processing_{match_uid}"] = True

            if st.button("💾 Save Score", key=f"save_{match_uid}", disabled=st.session_state.get(f"processing_{match_uid}", False), on_click=_save_callback):
                pass

            if st.session_state.get(f"processing_{match_uid}", False):
                ht_h_val = int(ht_h or 0); ht_a_val = int(ht_a or 0)
                ft_h_val = int(ft_h or 0); ft_a_val = int(ft_a or 0)

                valid, _ = ScoreValidator.validate(row.to_dict(), ht_h_val, ht_a_val, ft_h_val, ft_a_val)
                df.at[idx,'home_ht_goals'] = ht_h_val
                df.at[idx,'away_ht_goals'] = ht_a_val
                df.at[idx,'home_goals'] = ft_h_val
                df.at[idx,'away_goals'] = ft_a_val
                if valid:
                    df.at[idx,'prediction_status'] = 'VALIDATED'
                    full_record = df.loc[idx].to_dict()
                    full_record['home_ht_goals'] = ht_h_val
                    full_record['away_ht_goals'] = ht_a_val
                    full_record['home_goals'] = ft_h_val
                    full_record['away_goals'] = ft_a_val
                    full_record['totalgol_ft'] = ft_h_val + ft_a_val
                    full_record['totalgol_ht'] = ht_h_val + ht_a_val
                    full_record['settlement_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Perhitungan profit Correct Score
                    cs_profit = 0
                    for i in range(1, 4):
                        score_str = full_record.get(f'cs_score_{i}')
                        odds = full_record.get(f'cs_odds_{i}')
                        stake_cs = full_record.get(f'cs_stake_{i}')
                        if score_str and odds is not None and stake_cs is not None:
                            try:
                                h, a = map(int, score_str.split(':'))
                                if h == ft_h_val and a == ft_a_val:
                                    cs_profit += stake_cs * (odds - 1)
                                else:
                                    cs_profit -= stake_cs
                            except:
                                pass
                    full_record['cs_profit'] = cs_profit

                    existing_hist = db.load_history()
                    if not existing_hist.empty and 'match_uid' in existing_hist.columns:
                        if full_record['match_uid'] in existing_hist['match_uid'].values:
                            st.warning("Pertandingan sudah ada di History.")
                            st.session_state[processing_key] = False
                            st.stop()

                    hist_record_df = pd.DataFrame([full_record])
                    hist_record_df = reorder_columns(hist_record_df)
                    dataset_record_df = hist_record_df.copy()
                    if 'league_name' in dataset_record_df.columns:
                        dataset_record_df = dataset_record_df.drop(columns=['league_name'])

                    hist = db.load_history()
                    hist = pd.concat([hist, hist_record_df], ignore_index=True)
                    db.save_history(hist)

                    dataset_wg = db.load_dataset_with_goal()
                    if not dataset_wg.empty and 'match_uid' in dataset_wg.columns:
                        if full_record['match_uid'] not in dataset_wg['match_uid'].values:
                            dataset_wg = pd.concat([dataset_wg, dataset_record_df], ignore_index=True)
                            db.save_dataset_with_goal(dataset_wg)
                    else:
                        db.save_dataset_with_goal(dataset_record_df)

                    df.drop(idx, inplace=True)
                    db.save_pending(df)
                    update_league_profile(storage, int(row.get('league_code',0)), session)
                    st.success("Skor disimpan dan dipindahkan ke History.")
                else:
                    db.save_pending(df)
                    st.warning("Skor disimpan dengan catatan.")
                st.session_state[processing_key] = False
                st.rerun()

            with st.expander("🔴 Live Bet (klik untuk buka)"):
                st.markdown("---")
                st.subheader("🔴 Live Bet (hanya sesi ini)")
                lam_total = row.get('expected_goal', 2.5)
                home_xg = row.get('home_xg', lam_total/2)
                away_xg = row.get('away_xg', lam_total/2)
                default_ou = row.get('current_ou', 2.5)
                default_over_odds = row.get('current_over_odds', 1.8)
                default_under_odds = row.get('current_under_odds', 1.8)

                live_menit = st.number_input("Menit berjalan", 0, 120, value=0, key=f"live_menit_{match_uid}")
                live_home_goals = st.number_input("Skor Home Live", 0, 20, value=0, key=f"live_hg_{match_uid}")
                live_away_goals = st.number_input("Skor Away Live", 0, 20, value=0, key=f"live_ag_{match_uid}")
                live_ou = st.number_input("Line Gol Live", step=0.25, value=float(default_ou), key=f"live_ou_{match_uid}")
                live_over_odds = st.number_input("Odds Over Live", step=0.01, value=float(default_over_odds), key=f"live_o_{match_uid}")
                live_under_odds = st.number_input("Odds Under Live", step=0.01, value=float(default_under_odds), key=f"live_u_{match_uid}")

                if st.button("Hitung Rekomendasi Live", key=f"live_btn_{match_uid}"):
                    with st.spinner("Menghitung..."):
                        result_live = calculate_live_recommendation(
                            lam_total=float(lam_total), home_xg=float(home_xg), away_xg=float(away_xg),
                            menit_berjalan=float(live_menit), home_goals_live=int(live_home_goals),
                            away_goals_live=int(live_away_goals), current_ou=float(live_ou),
                            current_over_odds=float(live_over_odds), current_under_odds=float(live_under_odds),
                            storage=storage
                        )
                        session.set_live_result(match_uid, result_live)

                if session.get_live_result(match_uid):
                    res = session.get_live_result(match_uid)
                    st.markdown(f"**Rekomendasi:** :{'green' if res['recommendation'] != 'NO BET' else 'red'}[{res['recommendation']}]")
                    st.markdown(f"**EV Over:** {res['ev_over']:+.3f} | **EV Under:** {res['ev_under']:+.3f}")
                    st.markdown(f"**Kelly Over:** {res['kelly_over']:.1%} | **Kelly Under:** {res['kelly_under']:.1%}")
                    st.markdown(f"**BTTS:** {res['btts_pred']} ({res['prob_btts_yes']:.1%} confidence {res['confidence_btts']:.1%})")

def render_settlement(session: SessionManager, storage: StorageProvider):
    st.subheader("📝 Settlement Audit")
    db = DatabaseManager(storage)
    try: raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except: raw = pd.DataFrame()
    if raw.empty: st.info("Belum ada data."); return
    if 'settlement_time' not in raw.columns: st.warning("Data history tidak memiliki kolom settlement_time."); return

    raw['settlement_dt'] = pd.to_datetime(raw['settlement_time'], errors='coerce')
    raw = raw.dropna(subset=['settlement_dt'])
    if raw.empty: st.info("Tidak ada data dengan settlement_time valid."); return

    raw['tanggal'] = raw['settlement_dt'].dt.date
    hari_ini = datetime.now().date()
    start_date = hari_ini - timedelta(days=6)
    mask = (raw['tanggal'] >= start_date) & (raw['tanggal'] <= hari_ini)
    df_recent = raw[mask].copy()
    if df_recent.empty: st.info("Tidak ada data settlement dalam 7 hari terakhir."); return

    pc = ProfitCalculator()
    profits = [pc.calculate(row.to_dict(), 100000)[0] for _, row in df_recent.iterrows()]
    statuses = [pc.calculate(row.to_dict(), 100000)[1] for _, row in df_recent.iterrows()]
    df_recent['profit'] = profits
    df_recent['status'] = statuses

    today_mask = df_recent['tanggal'] == hari_ini
    total_profit_hari_ini = df_recent.loc[today_mask, 'profit'].sum() if today_mask.any() else 0.0
    st.metric("Total Profit Hari Ini", f"Rp {total_profit_hari_ini:+,.0f}")

    grouped = df_recent.groupby('tanggal')
    tanggal_list = sorted(grouped.groups.keys(), reverse=True)
    hari_map = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    bulan_map = {1:'Januari',2:'Februari',3:'Maret',4:'April',5:'Mei',6:'Juni',
                 7:'Juli',8:'Agustus',9:'September',10:'Oktober',11:'November',12:'Desember'}

    for tgl in tanggal_list:
        group = grouped.get_group(tgl)
        total_profit_tgl = group['profit'].sum()
        if tgl == hari_ini: label = "Hari Ini"
        elif tgl == hari_ini - timedelta(days=1): label = "Kemarin"
        else:
            nama_hari = hari_map[tgl.weekday()]
            label = f"{nama_hari}, {tgl.day} {bulan_map[tgl.month]} {tgl.year}"

        with st.expander(f"{label} - Rp {total_profit_tgl:+,.0f}"):
            group_sorted = group.sort_values('settlement_dt', ascending=False)
            for _, row in group_sorted.iterrows():
                home = safe_html(str(row['home_team']))
                away = safe_html(str(row['away_team']))
                league = safe_html(str(row.get('league_name', '')))
                prediction = safe_html(str(row.get('prediction', '')))
                home_goals = int(row.get('home_goals', 0) or 0)
                away_goals = int(row.get('away_goals', 0) or 0)
                score = f"{home_goals}-{away_goals}"
                profit = row['profit']
                status = row['status']
                if status == "FULL WIN": bg = "#16a34a"
                elif status == "HALF WIN": bg = "#65a30d"
                elif status == "PUSH": bg = "#334155"
                elif status == "HALF LOSE": bg = "#3b82f6"
                elif status == "FULL LOSE": bg = "#ef4444"
                else: bg = "#334155"
                profit_str = f"Rp {profit:+,.0f}"
                st.markdown(
                    f"<div style='background-color:{bg}; padding:8px; border-radius:8px; "
                    f"margin:4px 0; color:white;'>"
                    f"<strong>{home} vs {away}</strong> | {league} | Pred: {prediction} | "
                    f"Skor: {score} | {status} | {profit_str}"
                    f"</div>", unsafe_allow_html=True
                )

def render_history(session: SessionManager, storage: StorageProvider):
    st.subheader("📜 History Manager")
    try: raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except: raw = pd.DataFrame()
    if raw.empty: st.warning("History kosong."); return
    if 'settlement_time' in raw.columns: raw = raw.sort_values('settlement_time', ascending=False)
    st.dataframe(raw.head(20)[['settlement_time','home_team','away_team','prediction','home_goals','away_goals']])

def render_cs_history(session: SessionManager, storage: StorageProvider):
    st.subheader("🎯 Correct Score History")
    try:
        raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except:
        raw = pd.DataFrame()
    if raw.empty:
        st.info("Belum ada data.")
        return

    cols = ['settlement_time','home_team','away_team','home_goals','away_goals',
            'cs_score_1','cs_odds_1','cs_stake_1',
            'cs_score_2','cs_odds_2','cs_stake_2',
            'cs_score_3','cs_odds_3','cs_stake_3','cs_profit']
    available_cols = [c for c in cols if c in raw.columns]
    if 'cs_profit' not in available_cols:
        st.info("Data correct score belum tersedia di history.")
        return

    df = raw[available_cols].copy()
    df['settlement_time'] = pd.to_datetime(df['settlement_time'], errors='coerce')
    df = df.dropna(subset=['settlement_time']).sort_values('settlement_time', ascending=False)

    total_cs_profit = df['cs_profit'].sum() if 'cs_profit' in df.columns else 0
    st.metric("Total Profit Correct Score", f"Rp {total_cs_profit:+,.0f}")

    wins = 0
    total_bets = len(df)
    for _, row in df.iterrows():
        home_goals = int(row.get('home_goals', 0) or 0)
        away_goals = int(row.get('away_goals', 0) or 0)
        for i in range(1, 4):
            score_str = row.get(f'cs_score_{i}')
            if score_str:
                try:
                    h, a = map(int, score_str.split(':'))
                    if h == home_goals and a == away_goals:
                        wins += 1
                        break
                except:
                    pass
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    st.metric("Win Rate (salah satu dari 3 tepat)", f"{win_rate:.1f}%")

    st.dataframe(df[['settlement_time','home_team','away_team','home_goals','away_goals',
                     'cs_score_1','cs_score_2','cs_score_3','cs_profit']])

def evaluate_model(storage: StorageProvider) -> dict:
    cache_key = "evaluate_model_cache"
    if cache_key in st.session_state: return st.session_state[cache_key]
    result = {}
    try: df = storage.load_dataframe(ResourceRegistry.HISTORY)
    except: result = {"error": "Gagal membaca history."}; st.session_state[cache_key] = result; return result
    if df.empty: result = {"error": "History kosong."}; st.session_state[cache_key] = result; return result
    if 'totalgol_ft' not in df.columns:
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            df['totalgol_ft'] = df['home_goals'] + df['away_goals']
        else: result = {"error": "Data tidak memiliki kolom skor."}; st.session_state[cache_key] = result; return result
    if 'expected_goal' not in df.columns:
        result = {"error": "Data tidak memiliki expected_goal."}; st.session_state[cache_key] = result; return result

    reg_df = df.dropna(subset=['expected_goal', 'totalgol_ft'])
    if len(reg_df) < 2: result = {"error": "Belum cukup data untuk evaluasi regresi."}; st.session_state[cache_key] = result; return result
    y_true_reg = reg_df['totalgol_ft']
    y_pred_reg = reg_df['expected_goal']
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))

    df['total_goals'] = df['home_goals'] + df['away_goals']
    df['actual_over'] = np.where(df['total_goals'] > df['current_ou'], 1,
                                 np.where(df['total_goals'] < df['current_ou'], 0, np.nan))
    class_df = df.dropna(subset=['actual_over', 'prediction_ou'])
    if len(class_df) < 2: accuracy = precision = recall = f1 = cm = brier = logloss = None
    else:
        y_true_cls = class_df['actual_over'].astype(int)
        y_pred_cls = class_df['prediction_ou'].astype(int)
        accuracy = accuracy_score(y_true_cls, y_pred_cls)
        precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
        cm = confusion_matrix(y_true_cls, y_pred_cls)
        if 'prob_over' in class_df.columns:
            brier = brier_score_loss(y_true_cls, class_df['prob_over'])
            logloss = log_loss(y_true_cls, class_df['prob_over'])
        else: brier = logloss = None

    pc = ProfitCalculator()
    fin_df = df.copy()
    profits_statuses = [pc.calculate(row.to_dict(), 100000) for _, row in fin_df.iterrows()]
    fin_df['profit'], fin_df['status'] = zip(*profits_statuses)
    bet_df = fin_df[fin_df['status'] != 'NO BET']
    if bet_df.empty: win_rate = roi = yield_pct = total_profit = total_bets = None
    else:
        total_bets = len(bet_df)
        total_profit = bet_df['profit'].sum()
        roi = total_profit / (total_bets * 100000) * 100
        yield_pct = total_profit / (total_bets * 100000) * 100
        win_rate = ((bet_df['status'] == 'FULL WIN').sum() + (bet_df['status'] == 'HALF WIN').sum()) / total_bets * 100

    brier_btts = logloss_btts = None
    if 'prob_btts' in df.columns and 'home_goals' in df.columns:
        df['btts_actual'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        btts_valid = df.dropna(subset=['prob_btts', 'btts_actual'])
        if len(btts_valid) > 1:
            brier_btts = brier_score_loss(btts_valid['btts_actual'], btts_valid['prob_btts'])
            logloss_btts = log_loss(btts_valid['btts_actual'], btts_valid['prob_btts'])

    result = {
        "mae": mae, "rmse": rmse, "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "cm": cm, "brier": brier, "logloss": logloss,
        "win_rate": win_rate, "roi": roi, "yield_pct": yield_pct,
        "total_bets_fin": total_bets, "total_profit_fin": total_profit,
        "brier_btts": brier_btts, "logloss_btts": logloss_btts,
        "error": None
    }
    st.session_state[cache_key] = result
    return result

def render_learning(session: SessionManager, storage: StorageProvider):
    st.subheader("🧠 Learning Center")
    if not storage.exists(ResourceRegistry.HISTORY): st.warning("History kosong."); return

    if st.button("🚀 Latih Ulang Model"):
        with st.spinner("Melatih..."):
            from xgboost import XGBRegressor, XGBClassifier
            from sklearn.calibration import CalibratedClassifierCV
            hist = storage.load_dataframe(ResourceRegistry.HISTORY)
            if 'totalgol_ft' not in hist.columns:
                hist['totalgol_ft'] = hist['home_goals'] + hist['away_goals']
            feats = [c for c in EXPECTED_FEATURES if c in hist.columns]
            X = hist[feats].fillna(0)
            y = hist['totalgol_ft']
            model_ou = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, objective='count:poisson', random_state=42)
            model_ou.fit(X, y)

            model_btts = None
            calibrator = None
            if 'home_goals' in hist.columns and 'away_goals' in hist.columns:
                y_btts = ((hist['home_goals'] > 0) & (hist['away_goals'] > 0)).astype(int)
                xgb_btts = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
                calibrated_btts = CalibratedClassifierCV(estimator=xgb_btts, method='isotonic', cv=5)
                calibrated_btts.fit(X, y_btts)
                model_btts = calibrated_btts
                calibrator = calibrated_btts

            bundle = {
                'model_ou': model_ou, 'feature_cols': feats,
                'model_btts': model_btts, 'btts_calibrator': calibrator,
                'calibrator': None
            }
            app_storage = LocalStorageProvider()
            app_storage.save_pickle(ResourceRegistry.MODEL, bundle)
            st.cache_resource.clear()
            st.cache_data.clear()
            if "evaluate_model_cache" in st.session_state: del st.session_state["evaluate_model_cache"]
            st.success(f"Model dilatih dari {len(hist)} data! (OU + BTTS)")

    st.markdown("---")
    st.subheader("📊 Evaluasi Model Saat Ini")
    eval_result = evaluate_model(storage)
    if eval_result.get("error"): st.info(eval_result["error"]); return

    st.markdown("**Regresi (Prediksi Gol)**")
    c1, c2 = st.columns(2)
    c1.metric("📏 MAE", f"{eval_result['mae']:.2f}")
    c2.metric("📐 RMSE", f"{eval_result['rmse']:.2f}")

    st.markdown("**Klasifikasi Over/Under**")
    if eval_result['accuracy'] is not None:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Akurasi", f"{eval_result['accuracy']:.2%}")
        c2.metric("Precision", f"{eval_result['precision']:.2%}")
        c3.metric("Recall", f"{eval_result['recall']:.2%}")
        c4.metric("F1‑Score", f"{eval_result['f1']:.2%}")
        if eval_result['cm'] is not None:
            cm_df = pd.DataFrame(eval_result['cm'], index=['Actual Under','Actual Over'], columns=['Pred Under','Pred Over'])
            st.table(cm_df)
    else: st.info("Data klasifikasi tidak mencukupi.")

    st.markdown("**Kalibrasi Probabilitas Over/Under**")
    if eval_result['brier'] is not None:
        c1, c2 = st.columns(2)
        c1.metric("Brier Score OU", f"{eval_result['brier']:.4f}")
        c2.metric("Log Loss OU", f"{eval_result['logloss']:.4f}")
    else: st.info("Data kalibrasi OU tidak tersedia.")

    st.markdown("**Evaluasi BTTS**")
    if eval_result.get('brier_btts') is not None:
        c1, c2 = st.columns(2)
        c1.metric("Brier Score BTTS", f"{eval_result['brier_btts']:.4f}")
        c2.metric("Log Loss BTTS", f"{eval_result['logloss_btts']:.4f}")
    else: st.info("Data evaluasi BTTS belum cukup.")

    st.markdown("**Finansial (hanya taruhan yang direkomendasikan)**")
    if eval_result['win_rate'] is not None:
        c1,c2,c3 = st.columns(3)
        c1.metric("Win Rate", f"{eval_result['win_rate']:.2f}%")
        c2.metric("ROI", f"{eval_result['roi']:.2f}%")
        c3.metric("Yield", f"{eval_result['yield_pct']:.2f}%")
        st.caption(f"Berdasarkan {eval_result['total_bets_fin']} taruhan, total profit Rp {eval_result['total_profit_fin']:+,.0f}")
    else: st.info("Belum ada taruhan yang dipasang.")

def render_database(session: SessionManager, storage: StorageProvider):
    st.subheader("🗄️ Database Monitor")
    resources = [ResourceRegistry.PENDING, ResourceRegistry.HISTORY, ResourceRegistry.DATASET, ResourceRegistry.DATASET_WITH_GOAL]
    data = []
    for r in resources:
        try: data.append([r.id, len(storage.load_dataframe(r)), "Active"])
        except: data.append([r.id, "Error", "Error"])
    st.table(pd.DataFrame(data, columns=["Resource","Rows","Status"]))

def render_debug(session: SessionManager):
    st.subheader("🐞 Debug Center")
    trace = session.get_debug_trace()
    if not trace: st.info("No trace")
    else:
        for line in trace[-100:]: st.text(line)

def render_performance(session: SessionManager, app_storage, db_storage):
    st.subheader("📊 Performance Center")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Model","✅" if app_storage.exists(ResourceRegistry.MODEL) else "❌")
    c2.metric("Dataset","✅" if db_storage.exists(ResourceRegistry.DATASET) else "❌")
    c3.metric("History","✅" if db_storage.exists(ResourceRegistry.HISTORY) else "❌")
    c4.metric("Threshold","✅" if db_storage.exists(ResourceRegistry.THRESHOLD) else "❌")

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    load_css()
    session = SessionManager()

    app_storage = LocalStorageProvider()

    token = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
    if token:
        repo_full = os.getenv("GITHUB_REPOSITORY") or st.secrets.get("GITHUB_REPOSITORY", "teknisiery/football-ai-db")
        owner, repo = repo_full.split("/") if "/" in repo_full else ("teknisiery", repo_full)
        branch = os.getenv("GITHUB_BRANCH") or st.secrets.get("GITHUB_BRANCH", "main")
        db_storage = GitHubStorageProvider(owner, repo, branch, token)
    else:
        db_storage = LocalStorageProvider()
        st.sidebar.warning("Mode Offline: GitHub token tidak ditemukan. Data hanya disimpan lokal.")

    @st.cache_resource
    def get_predictor():
        if app_storage.exists(ResourceRegistry.MODEL):
            bundle = app_storage.load_pickle(ResourceRegistry.MODEL)
            try:
                league_profile = db_storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
            except:
                league_profile = pd.DataFrame()
            return Predictor(bundle, league_profile=league_profile)
        return None

    predictor = get_predictor()

    with st.sidebar:
        st.markdown("<div style='text-align:center;font-size:3rem;'>⚽</div>", unsafe_allow_html=True)
        st.title(APP_TITLE)
        st.caption(f"v{APP_VERSION}")

        ev_over, ev_under = ThresholdService.get_thresholds(db_storage)
        st.metric("Threshold Over", f"{ev_over:.2f}")
        st.metric("Threshold Under", f"{ev_under:.2f}")

        new_over = st.number_input("Min EV Over", value=ev_over, step=0.01)
        new_under = st.number_input("Min EV Under", value=ev_under, step=0.01)
        if st.button("Simpan Threshold"):
            db_storage.save_json(ResourceRegistry.THRESHOLD, {'ev_over': new_over, 'ev_under': new_under})
            st.success("Threshold disimpan")

        st.markdown("---")
        st.subheader("🎯 Threshold BTTS")
        try:
            profil_league = db_storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
            league_list = profil_league['league_name'].tolist()
            selected_league = st.selectbox("Liga", league_list)
            current_btts_thr = profil_league.loc[profil_league['league_name'] == selected_league, 'btts_threshold'].values
            if len(current_btts_thr) > 0:
                current_btts_thr = current_btts_thr[0]
            else:
                current_btts_thr = 0.22
            new_btts_thr = st.number_input("Threshold BTTS", min_value=0.1, max_value=0.6, value=float(current_btts_thr), step=0.01)
            if st.button("Simpan Threshold BTTS"):
                profil_league.loc[profil_league['league_name'] == selected_league, 'btts_threshold'] = new_btts_thr
                db_storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profil_league)
                st.cache_resource.clear()
                st.success("Threshold BTTS disimpan.")
                st.rerun()
            if st.button("Reset ke 0.22"):
                profil_league.loc[profil_league['league_name'] == selected_league, 'btts_threshold'] = 0.22
                db_storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profil_league)
                st.cache_resource.clear()
                st.success("Threshold direset ke 0.22.")
                st.rerun()
        except:
            st.info("File profil liga tidak tersedia.")

        with st.expander("➕ Tambah Liga Baru"):
            with st.form("form_add_league"):
                new_code = st.number_input("Kode Liga", min_value=1, step=1)
                new_name = st.text_input("Nama Liga")
                col1, col2 = st.columns(2)
                with col1:
                    new_teams = st.number_input("Jumlah Tim", min_value=2, step=1)
                with col2:
                    new_match_per_round = st.number_input("Match per Round", min_value=1, step=1)
                st.markdown("**Statistik Awal**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    new_avg = st.number_input("Avg Goals", min_value=0.0, value=2.5, step=0.1)
                with c2:
                    new_over25 = st.number_input("Over 2.5 %", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                with c3:
                    new_btts = st.number_input("BTTS %", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                with c4:
                    new_under35 = st.number_input("Under 3.5 %", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
                submitted = st.form_submit_button("✅ Tambahkan Liga")
                if submitted:
                    if not new_name.strip():
                        st.error("Nama liga wajib diisi.")
                    else:
                        success, msg = add_new_league(
                            int(new_code), new_name.strip(), new_avg, new_over25, new_btts, new_under35,
                            int(new_teams), int(new_match_per_round),
                            db_storage, app_storage
                        )
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        try:
            hist_df = db_storage.load_dataframe(ResourceRegistry.HISTORY)
        except:
            hist_df = pd.DataFrame()

        total_profit, profit_df, summary, monthly_data, profit_by_status = compute_detailed_profits(hist_df)

        st.metric("Total Profit (All Time)", f"Rp {total_profit:+,.0f}")

        time_col_main = '_valid_time' if '_valid_time' in profit_df.columns else ('kickoff_time' if 'kickoff_time' in profit_df.columns else 'settlement_time')
        if not profit_df.empty and time_col_main in profit_df.columns:
            profit_df_sorted = profit_df.dropna(subset=[time_col_main]).sort_values(time_col_main)
            profit_df_sorted['cumulative_profit'] = profit_df_sorted['profit'].cumsum()
            if not profit_df_sorted.empty:
                fig = px.line(profit_df_sorted, x=time_col_main, y='cumulative_profit',
                              labels={time_col_main: 'Kickoff', 'cumulative_profit': 'Profit Kumulatif'})
                fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
                fig.update_xaxes(tickformat='%d/%m')
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Statistik All Time**")
        cols = st.columns(6)
        cols[0].metric("Total Bets", summary['total_bets'])
        cols[1].metric("Full Win", summary['full_win'])
        cols[2].metric("Half Win", summary['half_win'])
        cols[3].metric("Push", summary['push'])
        cols[4].metric("Half Lose", summary['half_lose'])
        cols[5].metric("Full Lose", summary['full_lose'])

        st.markdown("**Profit per Status**")
        for status in ['FULL WIN', 'HALF WIN', 'PUSH', 'HALF LOSE', 'FULL LOSE']:
            st.markdown(f"{status}: Rp {profit_by_status.get(status, 0):+,.0f}")

        if monthly_data:
            st.markdown("**Profit per Bulan**")
            bulan_map = {
                '01':'Januari','02':'Februari','03':'Maret','04':'April','05':'Mei','06':'Juni',
                '07':'Juli','08':'Agustus','09':'September','10':'Oktober','11':'November','12':'Desember'
            }
            for bulan_key in sorted(monthly_data.keys(), reverse=True):
                data = monthly_data[bulan_key]
                if bulan_key == 'Tanpa Tanggal': label_bulan = 'Tanpa Tanggal'
                else:
                    year, month = bulan_key.split('-')
                    label_bulan = f"{bulan_map.get(month, month)} {year}"
                with st.expander(f"{label_bulan} - Rp {data['profit']:+,.0f}"):
                    time_col = data.get('time_col', '_valid_time')
                    df_bulan = data['df'].sort_values(time_col).dropna(subset=[time_col])
                    if not df_bulan.empty:
                        df_bulan['cumulative_profit'] = df_bulan['profit'].cumsum()
                        fig_bulan = px.line(df_bulan, x=time_col, y='cumulative_profit',
                                            labels={time_col: 'Kickoff', 'cumulative_profit': 'Profit Kumulatif'})
                        fig_bulan.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0))
                        fig_bulan.update_xaxes(tickformat='%d/%m')
                        st.plotly_chart(fig_bulan, use_container_width=True)
                    bulan_sum = data['summary']
                    cols_b = st.columns(6)
                    cols_b[0].metric("Total Bets", bulan_sum['total_bets'])
                    cols_b[1].metric("Full Win", bulan_sum['full_win'])
                    cols_b[2].metric("Half Win", bulan_sum['half_win'])
                    cols_b[3].metric("Push", bulan_sum['push'])
                    cols_b[4].metric("Half Lose", bulan_sum['half_lose'])
                    cols_b[5].metric("Full Lose", bulan_sum['full_lose'])
                    profit_by_status_month = df_bulan.groupby('result')['profit'].sum().to_dict()
                    st.markdown("**Profit per Status**")
                    for status in ['FULL WIN', 'HALF WIN', 'PUSH', 'HALF LOSE', 'FULL LOSE']:
                        st.markdown(f"{status}: Rp {profit_by_status_month.get(status, 0):+,.0f}")
        else:
            st.caption("Belum ada data profit bulanan.")

        cached_league = load_profit_summary()
        if cached_league:
            league_data_display = cached_league
        else:
            league_data_full = compute_profits_by_league(hist_df)
            league_data_display = {k: {'profit': v['profit'], 'summary': v['summary']} for k, v in league_data_full.items()}

        if league_data_display:
            with st.expander("📊 Performa per Liga (klik untuk lihat)"):
                for league in sorted(league_data_display.keys()):
                    data = league_data_display[league]
                    with st.expander(f"{league} - Rp {data['profit']:+,.0f}"):
                        liga_sum = data['summary']
                        cols_l = st.columns(6)
                        cols_l[0].metric("Total Bets", liga_sum['total_bets'])
                        cols_l[1].metric("Full Win", liga_sum['full_win'])
                        cols_l[2].metric("Half Win", liga_sum['half_win'])
                        cols_l[3].metric("Push", liga_sum['push'])
                        cols_l[4].metric("Half Lose", liga_sum['half_lose'])
                        cols_l[5].metric("Full Lose", liga_sum['full_lose'])

    st.title(f"⚽ {APP_TITLE}")

    if session.uploaded_file is None:
        render_upload_section(session)
    else:
        st.markdown(f"**📤 {safe_html(session.uploaded_file.name)}**")
        if st.button("🔄 Ganti File"):
            session.reset_upload_and_prediction()
            st.rerun()

        if predictor is None:
            st.warning("Model tidak ditemukan. Silakan latih model di tab Learning atau pastikan model.pkl tersedia.")
        else:
            if session.analysis_needed:
                session.analysis_needed = False
                with st.spinner("Menganalisis..."):
                    run_analysis(session, db_storage, predictor)
                st.rerun()

            if not session.get_prediction_state().processed:
                if st.button("🚀 ANALYZE MATCH", type="primary"):
                    run_analysis(session, db_storage, predictor)
                    st.rerun()

        if session.get_prediction_state().processed:
            st.markdown("---")
            odds_file = st.file_uploader("📊 Upload Odds Correct Score (CSV)", type=["csv"])
            if odds_file:
                try:
                    odds_dict = parse_odds_csv(odds_file.getvalue())
                    session.uploaded_odds = odds_dict
                    st.success("Odds correct score berhasil diunggah.")
                except Exception as e:
                    st.error(f"Gagal membaca file odds: {e}")
                    session.uploaded_odds = None
            else:
                session.uploaded_odds = None

            summary = get_match_prediction_summary(session, db_storage)
            if summary:
                render_prediction_card(summary)
                if st.button("💾 Save Pending"):
                    ps = session.get_prediction_state()
                    row = ps.prediction_dataframe.iloc[0].to_dict()
                    serialized = serialize_prediction(PredictionResult(**ps.prediction_result))
                    row.update(serialized)
                    row['kickoff_time'] = session.uploaded_df['kickoff_time'].iloc[0] if 'kickoff_time' in session.uploaded_df.columns else datetime.now().strftime("%Y-%m-%d %H:%M")
                    row['match_uid'] = f"{summary['home']}|{summary['away']}|{row['kickoff_time']}"
                    row['prediction_status'] = 'PENDING'
                    row['home_team'] = summary['home']
                    row['away_team'] = summary['away']
                    row['prediction'] = f"{summary['ou_pred']} {summary['ou_line']}"
                    row['recommendation'] = summary['recommendation']
                    row['stake'] = summary['stake']
                    for col in EXPECTED_FEATURES + ['league_name']:
                        if col not in row: row[col] = np.nan

                    cs_recs = summary.get('cs_recommendations')
                    if cs_recs:
                        stakes = [30000, 40000, 50000]
                        for i, rec in enumerate(cs_recs):
                            row[f'cs_score_{i+1}'] = f"{rec[0]}:{rec[1]}"
                            row[f'cs_odds_{i+1}'] = rec[2]
                            row[f'cs_stake_{i+1}'] = stakes[i]
                    else:
                        for i in range(1, 4):
                            row[f'cs_score_{i}'] = None
                            row[f'cs_odds_{i}'] = None
                            row[f'cs_stake_{i}'] = None

                    db = DatabaseManager(db_storage)
                    existing_hist = db.load_history()
                    if not existing_hist.empty and 'match_uid' in existing_hist.columns:
                        if row['match_uid'] in existing_hist['match_uid'].values:
                            st.warning("Pertandingan ini sudah ada di History.")
                            st.stop()

                    pend = db.load_pending()
                    if not pend.empty and row['match_uid'] in pend['match_uid'].values:
                        st.warning("Pertandingan sudah ada di Pending.")
                    else:
                        pend = pd.concat([pend, pd.DataFrame([row])], ignore_index=True)
                        db.save_pending(pend)
                        st.success("Disimpan ke Pending!")
                    st.rerun()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📋 Pending", "📝 Settlement", "📜 History", "🧠 Learning",
            "🗄️ Database", "📋 Trans Log", "🐞 Debug", "📊 Perf", "🎯 Correct Score"
        ])
        with tab1: render_pending(session, db_storage)
        with tab2: render_settlement(session, db_storage)
        with tab3: render_history(session, db_storage)
        with tab4: render_learning(session, db_storage)
        with tab5: render_database(session, db_storage)
        with tab6: st.info("Transaction Log")
        with tab7: render_debug(session)
        with tab8: render_performance(session, app_storage, db_storage)
        with tab9: render_cs_history(session, db_storage)

    if session.get_debug_trace():
        with st.expander("📜 Raw Debug Trace"):
            for line in session.get_debug_trace()[-100:]:
                st.text(line)

if __name__ == "__main__":
    main()