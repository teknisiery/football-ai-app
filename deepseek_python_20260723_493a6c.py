# ============================================================
# FOOTBALL AI V2 – PRODUCTION (DUAL STORAGE) – FINAL
# Perbaikan: append dataset_ou.csv, bukan overwrite
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import base64
import requests
import uuid
from io import BytesIO
from datetime import datetime
from xgboost import XGBRegressor
from scipy.stats import poisson
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import re

# ============================================================
# CONFIGURATION
# ============================================================
APP_TITLE = "Football AI V2"
APP_VERSION = "2.1.0"
BASE_DIR = Path(__file__).resolve().parent
EV_THRESHOLD_FILE = BASE_DIR / "ev_threshold.json"

LEAGUE_ROUND_CONFIG = {
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
}

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
    .stFileUploader > div {
        border-radius: 16px; border: 2px dashed #4b5563; background: #1c1f26; padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# UTILS
# ============================================================
def safe_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#x27;"))

def compute_prob_over(lambda_hat: float, ou_line: float) -> float:
    remainder = round(ou_line % 1, 2)
    if remainder == 0.0:
        return 1 - poisson.cdf(int(ou_line), lambda_hat)
    elif remainder == 0.5:
        return 1 - poisson.cdf(int(ou_line - 0.5), lambda_hat)
    elif remainder == 0.25:
        k = int(ou_line)
        p_win = 1 - poisson.cdf(k+1, lambda_hat)
        p_half_lose = poisson.pmf(k, lambda_hat)
        return p_win + 0.5 * p_half_lose
    elif remainder == 0.75:
        k = int(ou_line)
        p_win = 1 - poisson.cdf(k+2, lambda_hat)
        p_half_win = poisson.pmf(k+1, lambda_hat)
        return p_win + 0.5 * p_half_win
    else:
        return 1 - poisson.cdf(ou_line, lambda_hat)

def calc_kelly(prob: float, odds: float) -> float:
    if odds <= 1.0 or prob <= 0:
        return 0.0
    k = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    return max(0.0, min(0.25, k))

def load_ev_thresholds(storage=None):
    """Baca threshold dari storage (database) jika tersedia, fallback ke file lokal."""
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
    THRESHOLD = Resource("threshold", "dict", "json", "ev_threshold.json")
    MODEL = Resource("model", "pickle", "pickle", "model.pkl")
    FEATURE_COLUMNS = Resource("feature_columns", "dict", "json", "feature_columns.json")
    LEAGUE_THRESHOLD = Resource("league_threshold", "dataframe", "csv", "league_threshold.csv")

OPTIONAL_RESOURCES = {
    ResourceRegistry.HISTORY.id, ResourceRegistry.DATASET.id,
    ResourceRegistry.DATASET_WITH_GOAL.id, ResourceRegistry.PENDING.id,
    ResourceRegistry.LEAGUE_STATS.id, ResourceRegistry.THRESHOLD.id,
    ResourceRegistry.FEATURE_COLUMNS.id, ResourceRegistry.LEAGUE_THRESHOLD.id,
    ResourceRegistry.LEAGUE_PROFILE.id,
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
        if resp.status_code == 409: raise RuntimeError("Conflict")
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
    CORE_COLUMNS = ["match_uid","home_team","away_team","league_name","kickoff_time",
                    "prediction","grade","confidence","prediction_id","prediction_time",
                    "prediction_status","prediction_version","prediction_source","prediction_notes",
                    "home_ht_goals","away_ht_goals","home_goals","away_goals",
                    "actual_over","actual_btts","settlement_time"]
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
        df['over_move'] = df['current_over_odds'] - df['open_over_odds']
    if 'open_under_odds' in cols and 'current_under_odds' in cols:
        df['under_move'] = df['current_under_odds'] - df['open_under_odds']
    if 'over_move' in cols and 'under_move' in cols:
        df['ou_movement'] = df['over_move'] + df['under_move']

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
# LEAGUE PROFILE (cache manual dengan st.session_state)
# ============================================================
def get_league_profile(storage: StorageProvider) -> pd.DataFrame:
    """Ambil profil liga dari storage, cache manual di session_state."""
    cache_key = "league_profile_cache"
    if cache_key not in st.session_state:
        if storage.exists(ResourceRegistry.LEAGUE_PROFILE):
            st.session_state[cache_key] = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
        else:
            st.session_state[cache_key] = pd.DataFrame()
    return st.session_state[cache_key]

def attach_league_profile(storage: StorageProvider, df: pd.DataFrame) -> pd.DataFrame:
    profile = get_league_profile(storage)
    if not profile.empty and 'league_code' in profile.columns and 'league_code' in df.columns:
        df = df.merge(profile, on='league_code', how='left', suffixes=('', '_profile'))
    defaults = {'league_avg_goals':2.5,'league_over25_pct':0.5,'league_btts_pct':0.5,'league_under35_pct':0.7,'league_name':'Unknown League'}
    for col, val in defaults.items():
        if col not in df.columns: df[col] = val
        else: df[col] = df[col].fillna(val)
    return df

def update_league_profile(storage: StorageProvider, league_code: int):
    if not storage.exists(ResourceRegistry.DATASET_WITH_GOAL): return
    df = storage.load_dataframe(ResourceRegistry.DATASET_WITH_GOAL)
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
    new_row = {
        'league_code': league_code,
        'league_name': df_league['league_name'].iloc[0] if 'league_name' in df_league.columns else f'League {league_code}',
        'league_avg_goals': df_league['totalgol_ft'].mean(),
        'league_over25_pct': (df_league['totalgol_ft']>2.5).mean(),
        'league_btts_pct': df_league['btts'].mean(),
        'league_under35_pct': (df_league['totalgol_ft']<3.5).mean(),
        'eg_p25': df_league['totalgol_ft'].quantile(0.25),
        'eg_p75': df_league['totalgol_ft'].quantile(0.75),
        'btts_p25':0.0,'btts_p75':0.0,'ht0_p25':0.0,'ht0_p75':0.0,
        'ev_over_threshold':0.01,'ev_under_threshold':0.02,'total_matches':total
    }
    profile_df = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE) if storage.exists(ResourceRegistry.LEAGUE_PROFILE) else pd.DataFrame()
    if league_code not in profile_df['league_code'].values:
        profile_df = pd.concat([profile_df, pd.DataFrame([new_row])])
    else:
        idx = profile_df[profile_df['league_code']==league_code].index[0]
        for k,v in new_row.items(): profile_df.at[idx,k] = v
    storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profile_df)
    # Bersihkan cache setelah update
    if "league_profile_cache" in st.session_state:
        del st.session_state["league_profile_cache"]

# ============================================================
# PREDICTION ENGINE
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

class Predictor:
    def __init__(self, bundle: dict):
        self.model_ou = bundle.get('model') or bundle.get('model_ou')
        self.model_btts = bundle.get('model_btts')
        self.feature_cols = bundle['feature_cols']
        self.calibrator = bundle.get('calibrator')
        self.btts_available = self.model_btts is not None
    def predict(self, features_df: pd.DataFrame) -> PredictionResult:
        X = features_df[self.feature_cols].fillna(0)
        lam = max(0.0, self.model_ou.predict(X)[0])
        ou_line = features_df['current_ou'].iloc[0]
        over_odds = features_df['current_over_odds'].iloc[0]
        under_odds = features_df['current_under_odds'].iloc[0]
        prob_raw = compute_prob_over(lam, ou_line)
        prob_over = prob_raw
        if self.calibrator: prob_over = np.clip(self.calibrator.predict([[prob_raw]])[0], 0, 1)
        ev_over = prob_over * (over_odds-1) + (1-prob_over)*(-1)
        ev_under = (1-prob_over)*(under_odds-1) + prob_over*(-1)
        k_over = calc_kelly(prob_over, over_odds)
        k_under = calc_kelly(1-prob_over, under_odds)
        prob_btts = None; conf_btts = None; pred_btts = -1
        if self.btts_available:
            prob_btts = self.model_btts.predict_proba(X)[0,1]
            conf_btts = max(prob_btts, 1-prob_btts)
            pred_btts = int(prob_btts>=0.5)
        return PredictionResult(
            expected_goal=lam, prob_over=prob_over, prob_under=1-prob_over,
            prob_over_raw=prob_raw, ev_over=ev_over, ev_under=ev_under,
            kelly_over=k_over, kelly_under=k_under,
            prob_btts=prob_btts, prob_ht0=None,
            confidence_ou=max(prob_over,1-prob_over), confidence_btts=conf_btts, confidence_ht0=None,
            prediction_ou=int(prob_over>=0.5), prediction_btts=pred_btts, prediction_ht0=-1
        )

# ============================================================
# SESSION MANAGER (dengan batasan debug_trace)
# ============================================================
@dataclass
class UploadState:
    uploaded: bool = False; filename: str = ""; rows: int = 0; columns: int = 0
    upload_time: str = ""; uploaded_data: Optional[pd.DataFrame] = None

@dataclass
class PredictionState:
    processed: bool = False; status: str = "NOT_PROCESSED"; prediction_count: int = 0
    prediction_dataframe: Optional[pd.DataFrame] = None
    prediction_result: Optional[dict] = None

class SessionManager:
    def __init__(self):
        if "app_states" not in st.session_state:
            st.session_state.app_states = {"upload": UploadState(), "prediction": PredictionState()}
            st.session_state.debug_trace = []
    def get_upload_state(self): return st.session_state.app_states["upload"]
    def set_upload_state(self, s): st.session_state.app_states["upload"] = s
    def get_prediction_state(self): return st.session_state.app_states["prediction"]
    def set_prediction_state(self, s): st.session_state.app_states["prediction"] = s
    def add_debug(self, msg):
        trace = st.session_state.debug_trace
        trace.append(msg)
        if len(trace) > 500:
            st.session_state.debug_trace = trace[-500:]  # potong ke 500 terakhir

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
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGES
# ============================================================
def render_upload_section(session: SessionManager):
    with st.expander("📤 Upload File CSV", expanded=True):
        f = st.file_uploader("Pilih CSV", type=["csv"])
        if f:
            df = pd.read_csv(f)
            # Validasi kolom minimum
            required = ['home_xg', 'away_xg', 'current_over_odds', 'current_under_odds', 'current_ou']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}. File tidak valid.")
                return
            st.session_state.uploaded_file = f
            st.session_state.uploaded_df = df
            st.rerun()

def run_analysis(session: SessionManager, storage: StorageProvider, predictor: Predictor):
    df = st.session_state.uploaded_df.copy()
    home, away = st.session_state.uploaded_file.name.replace('.csv','').split(' vs ')
    df = attach_league_profile(storage, df)
    df = add_features(df)
    result = predictor.predict(df)
    session.set_prediction_state(PredictionState(processed=True, prediction_dataframe=df, prediction_result=result.__dict__))
    
    # Buat match_uid jika belum ada
    if 'match_uid' not in df.columns:
        kickoff = st.session_state.uploaded_df['kickoff_time'].iloc[0] if 'kickoff_time' in st.session_state.uploaded_df.columns else datetime.now().strftime("%Y-%m-%d %H:%M")
        df['match_uid'] = f"{home}|{away}|{kickoff}"
    
    # Append ke dataset, bukan overwrite
    if storage.exists(ResourceRegistry.DATASET):
        existing = storage.load_dataframe(ResourceRegistry.DATASET)
        # Hindari duplikasi dengan match_uid jika ada
        if not existing.empty and 'match_uid' in existing.columns and 'match_uid' in df.columns:
            new_uid = df['match_uid'].iloc[0]
            if new_uid not in existing['match_uid'].values:
                combined = pd.concat([existing, df], ignore_index=True)
                storage.save_dataframe(ResourceRegistry.DATASET, combined)
        else:
            combined = pd.concat([existing, df], ignore_index=True)
            storage.save_dataframe(ResourceRegistry.DATASET, combined)
    else:
        storage.save_dataframe(ResourceRegistry.DATASET, df)

def get_match_prediction_summary(session: SessionManager, storage: StorageProvider):
    ps = session.get_prediction_state()
    if not ps.processed: return None
    r = ps.prediction_result
    df = ps.prediction_dataframe.iloc[0]
    home, away = st.session_state.uploaded_file.name.replace('.csv','').split(' vs ')
    ou_line = df['current_ou']
    over_odds = df['current_over_odds']
    under_odds = df['current_under_odds']
    ev_th_over, ev_th_under = load_ev_thresholds(storage)
    if r['ev_over'] > ev_th_over and r['kelly_over'] > 0.005:
        ou_pred, rec, rec_color, stake = "OVER", "TARUHAN OVER", "a" if r['ev_over']>0.05 else "c", 100000
    elif r['ev_under'] > ev_th_under and r['kelly_under'] > 0.005:
        ou_pred, rec, rec_color, stake = "UNDER", "TARUHAN UNDER", "a" if r['ev_under']>0.05 else "c", 100000
    else:
        ou_pred, rec, rec_color, stake = ("OVER" if r['prob_over']>=0.5 else "UNDER"), "NO BET", "d", 0
    return {
        "home": home, "away": away, "league": safe_html(str(df.get('league_name','Unknown'))),
        "ou_pred": ou_pred, "ou_line": ou_line, "over_odds": over_odds, "under_odds": under_odds,
        "expected_goal": r['expected_goal'], "confidence_ou": r['confidence_ou'],
        "ev_over": r['ev_over'], "ev_under": r['ev_under'],
        "kelly_over": r['kelly_over'], "kelly_under": r['kelly_under'],
        "recommendation": rec, "rec_color": rec_color, "stake": stake,
        "btts_pred": "YES" if r['prediction_btts']==1 else "NO" if r['prediction_btts']==0 else "N/A",
        "confidence_btts": r['confidence_btts'], "ht0_pred": "N/A", "insights": []
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
        with st.expander(f"▶ {home_safe} vs {away_safe} ({row['prediction_status']})"):
            def safe_int(val):
                if pd.isna(val) or val is None:
                    return 0
                return int(float(val))
            c1,c2 = st.columns(2)
            with c1: ht_h = st.number_input("HT Home", value=safe_int(row.get('home_ht_goals')), key=f"hth{idx}")
            with c2: ht_a = st.number_input("HT Away", value=safe_int(row.get('away_ht_goals')), key=f"hta{idx}")
            c3,c4 = st.columns(2)
            with c3: ft_h = st.number_input("FT Home", value=safe_int(row.get('home_goals')), key=f"fth{idx}")
            with c4: ft_a = st.number_input("FT Away", value=safe_int(row.get('away_goals')), key=f"fta{idx}")
            if st.button("💾 Save Score", key=f"save{idx}"):
                valid, _ = ScoreValidator.validate(row.to_dict(), ht_h, ht_a, ft_h, ft_a)
                df.at[idx,'home_ht_goals'] = ht_h
                df.at[idx,'away_ht_goals'] = ht_a
                df.at[idx,'home_goals'] = ft_h
                df.at[idx,'away_goals'] = ft_a
                if valid:
                    df.at[idx,'prediction_status'] = 'VALIDATED'
                    full_record = df.loc[idx].to_dict()
                    full_record['home_ht_goals'] = ht_h
                    full_record['away_ht_goals'] = ht_a
                    full_record['home_goals'] = ft_h
                    full_record['away_goals'] = ft_a
                    full_record['totalgol_ft'] = ft_h + ft_a
                    full_record['totalgol_ht'] = ht_h + ht_a
                    full_record['settlement_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    hist = db.load_history()
                    hist = pd.concat([hist, pd.DataFrame([full_record])], ignore_index=True)
                    db.save_history(hist)
                    df.drop(idx, inplace=True)
                    db.save_pending(df)
                    update_league_profile(storage, int(row.get('league_code',0)))
                    st.success("Skor disimpan dan dipindahkan ke History.")
                else:
                    db.save_pending(df)
                    st.warning("Skor disimpan dengan catatan.")
                st.rerun()

def render_settlement(session: SessionManager, storage: StorageProvider):
    st.subheader("📝 Settlement Audit")
    db = DatabaseManager(storage)
    try:
        raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except: raw = pd.DataFrame()
    if raw.empty: st.info("Belum ada data."); return
    if 'settlement_time' in raw.columns: raw = raw.sort_values('settlement_time', ascending=False)
    raw = raw.head(20)
    if 'profit' in raw.columns: st.metric("Total Profit (20 terbaru)", f"Rp {raw['profit'].sum():+,.0f}")
    for _, row in raw.iterrows():
        home_safe = safe_html(str(row['home_team']))
        away_safe = safe_html(str(row['away_team']))
        league_safe = safe_html(str(row.get('league_name', '')))
        pred_safe = safe_html(str(row.get('prediction', '')))
        with st.expander(f"▶ {home_safe} vs {away_safe}"):
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"**League:** {league_safe}")
                st.markdown(f"**Prediction:** {pred_safe}")
                st.markdown(f"**Score:** {int(row['home_goals'] or 0)}-{int(row['away_goals'] or 0)}")
            with c2:
                ou = row.get('current_ou',2.5)
                total = (row['home_goals'] or 0)+(row['away_goals'] or 0)
                actual = 1 if total>ou else 0 if total<ou else None
                if actual is None: res="PUSH"; profit=0
                else:
                    pred = 1 if str(row.get('prediction','')).startswith('OVER') else 0
                    win = pred==actual
                    odds = row['current_over_odds'] if pred==1 else row['current_under_odds']
                    stake = 100000
                    profit = stake*(odds-1) if win else -stake
                    res = "WIN" if win else "LOSE"
                st.markdown(f"**Settlement:** :{'green' if res=='WIN' else 'red' if res=='LOSE' else 'grey'}[{res}]")
                st.markdown(f"**Profit:** Rp{profit:+,.0f}")

def render_history(session: SessionManager, storage: StorageProvider):
    st.subheader("📜 History Manager")
    try:
        raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except: raw = pd.DataFrame()
    if raw.empty: st.warning("History kosong."); return
    if 'settlement_time' in raw.columns: raw = raw.sort_values('settlement_time', ascending=False)
    st.dataframe(raw.head(20)[['settlement_time','home_team','away_team','prediction','home_goals','away_goals']])

def render_learning(session: SessionManager, storage: StorageProvider):
    st.subheader("🧠 Learning Center")
    if not storage.exists(ResourceRegistry.HISTORY): st.warning("History kosong."); return
    if st.button("🚀 Latih Ulang Model"):
        with st.spinner("Melatih..."):
            from xgboost import XGBRegressor
            hist = storage.load_dataframe(ResourceRegistry.HISTORY)
            if 'totalgol_ft' not in hist.columns: hist['totalgol_ft'] = hist['home_goals'] + hist['away_goals']
            feats = [c for c in EXPECTED_FEATURES if c in hist.columns]
            X = hist[feats].fillna(0)
            y = hist['totalgol_ft']
            model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, objective='count:poisson', random_state=42)
            model.fit(X, y)
            app_storage = LocalStorageProvider()
            app_storage.save_pickle(ResourceRegistry.MODEL, {'model':model,'feature_cols':feats})
            st.cache_resource.clear()
            st.success(f"Model dilatih dari {len(hist)} data!")

def render_database(session: SessionManager, storage: StorageProvider):
    st.subheader("🗄️ Database Monitor")
    resources = [ResourceRegistry.PENDING, ResourceRegistry.HISTORY, ResourceRegistry.DATASET, ResourceRegistry.DATASET_WITH_GOAL]
    data = []
    for r in resources:
        try:
            df = storage.load_dataframe(r)
            data.append([r.id, len(df), "Active" if len(df)>0 else "Empty"])
        except: data.append([r.id, "Error", "Error"])
    st.table(pd.DataFrame(data, columns=["Resource","Rows","Status"]))

def render_debug(session: SessionManager):
    st.subheader("🐞 Debug Center")
    trace = st.session_state.get("debug_trace",[])
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
        if "/" in repo_full: owner, repo = repo_full.split("/")
        else: owner, repo = "teknisiery", repo_full
        branch = os.getenv("GITHUB_BRANCH") or st.secrets.get("GITHUB_BRANCH", "main")
        db_storage = GitHubStorageProvider(owner, repo, branch, token)
    else:
        db_storage = LocalStorageProvider()
        st.sidebar.warning("Mode Offline: GitHub token tidak ditemukan. Data hanya disimpan lokal.")

    @st.cache_resource
    def get_predictor():
        if app_storage.exists(ResourceRegistry.MODEL):
            return Predictor(app_storage.load_pickle(ResourceRegistry.MODEL))
        return None

    predictor = get_predictor()

    with st.sidebar:
        st.markdown("<div style='text-align:center;font-size:3rem;'>⚽</div>", unsafe_allow_html=True)
        st.title(APP_TITLE)
        st.caption(f"v{APP_VERSION}")

        ev_over, ev_under = load_ev_thresholds(db_storage)
        st.metric("Threshold Over", f"{ev_over:.2f}")
        st.metric("Threshold Under", f"{ev_under:.2f}")

        new_over = st.number_input("Min EV Over", value=ev_over, step=0.01)
        new_under = st.number_input("Min EV Under", value=ev_under, step=0.01)
        if st.button("Simpan Threshold"):
            db_storage.save_json(ResourceRegistry.THRESHOLD, {'ev_over': new_over, 'ev_under': new_under})
            st.success("Threshold disimpan")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    st.title(f"⚽ {APP_TITLE}")

    if 'uploaded_file' not in st.session_state:
        render_upload_section(session)
    else:
        st.markdown(f"**📤 {safe_html(st.session_state.uploaded_file.name)}**")
        if st.button("🔄 Ganti File"):
            del st.session_state.uploaded_file
            st.rerun()

        if predictor is None:
            st.warning("Model tidak ditemukan. Silakan latih model di tab Learning atau pastikan model.pkl tersedia.")
        else:
            if st.button("🚀 ANALYZE MATCH", type="primary"):
                run_analysis(session, db_storage, predictor)
                st.rerun()

        if session.get_prediction_state().processed:
            summary = get_match_prediction_summary(session, db_storage)
            if summary:
                render_prediction_card(summary)
                if st.button("💾 Save Pending"):
                    ps = session.get_prediction_state()
                    row = ps.prediction_dataframe.iloc[0].to_dict()
                    row.update(ps.prediction_result)
                    row['kickoff_time'] = st.session_state.uploaded_df['kickoff_time'].iloc[0] if 'kickoff_time' in st.session_state.uploaded_df.columns else datetime.now().strftime("%Y-%m-%d %H:%M")
                    row['match_uid'] = f"{summary['home']}|{summary['away']}|{row['kickoff_time']}"
                    row['prediction_status'] = 'PENDING'
                    row['home_team'] = summary['home']
                    row['away_team'] = summary['away']
                    row['prediction'] = f"{summary['ou_pred']} {summary['ou_line']}"
                    for col in EXPECTED_FEATURES + ['league_name']:
                        if col not in row:
                            row[col] = np.nan
                    db = DatabaseManager(db_storage)
                    pend = db.load_pending()
                    if not pend.empty and row['match_uid'] in pend['match_uid'].values:
                        st.warning("Pertandingan sudah ada di Pending.")
                    else:
                        pend = pd.concat([pend, pd.DataFrame([row])], ignore_index=True)
                        db.save_pending(pend)
                        st.success("Disimpan ke Pending!")
                    st.rerun()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Pending", "📝 Settlement", "📜 History", "🧠 Learning",
            "🗄️ Database", "📋 Trans Log", "🐞 Debug", "📊 Perf"
        ])
        with tab1: render_pending(session, db_storage)
        with tab2: render_settlement(session, db_storage)
        with tab3: render_history(session, db_storage)
        with tab4: render_learning(session, db_storage)
        with tab5: render_database(session, db_storage)
        with tab6: st.info("Transaction Log")
        with tab7: render_debug(session)
        with tab8: render_performance(session, app_storage, db_storage)

    if st.session_state.get("debug_trace"):
        with st.expander("📜 Raw Debug Trace"):
            for line in st.session_state.debug_trace[-100:]:
                st.text(line)

if __name__ == "__main__":
    main()