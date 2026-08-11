# ============================================================
# FOOTBALL AI V2 – PRODUCTION (DUAL STORAGE) 
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
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
from pathlib import Path
import plotly.express as px

# --- Modular imports ---
from config import APP_TITLE, APP_VERSION, EXPECTED_FEATURES, load_league_round_config
from utils import (
    safe_html, calc_kelly, reorder_columns, get_valid_time, normalize_kickoff,
    parse_odds_csv, get_cs_recommendations, calculate_fair_probs, get_hybrid_top3,
)
from services.settlement import SettlementEngine
from services.feature_eng import add_features
from services.model_evaluator import evaluate_model
from services.profit_calculator import ProfitCalculator
from services.match_pnl import build_match_pnl, apply_1x2_odds_floor, ONE_X_TWO_MIN_ODDS, format_pnl_table_html
from services.btts_filter import evaluate_btts_filter, normalize_btts_record, BTTS_FILTER_VERSION
from services.resource_registry import Resource, ResourceRegistry, OPTIONAL_RESOURCES
from services.storage import (
    StorageProvider,
    LocalStorageProvider,
    GitHubStorageProvider,
    DatabaseManager
)
from ui.components import render_prediction_card
from services.live_bet import calculate_live_recommendation
from services.odds_handler import process_combined_odds
from services.decision_engine import evaluate_ou_decision, compute_1x2_hybrid_and_ev
from services.league_profile import (
    get_league_profile,
    attach_league_profile,
    update_league_profile,
    add_new_league,
)
from services.pending_service import PendingService
from services.threshold_service import ThresholdService
from services.coherence_filter import evaluate_coherence
from services.shadow_predictor import compute_shadow_prediction

# ============================================================
# CONFIGURATION (local overrides)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
EV_THRESHOLD_FILE = BASE_DIR / "ev_threshold.json"

LEAGUE_ROUND_CONFIG = load_league_round_config()

# ============================================================
# CUSTOM CSS
# ============================================================
def load_css():
    with open(BASE_DIR / "assets" / "style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
        "cs_profit",
        "prob_1x2_model_home", "prob_1x2_model_draw", "prob_1x2_model_away",
        "prob_1x2_league_home", "prob_1x2_league_draw", "prob_1x2_league_away",
        "prob_1x2_hybrid_home", "prob_1x2_hybrid_draw", "prob_1x2_hybrid_away",
        "fair_odds_1x2_home", "fair_odds_1x2_draw", "fair_odds_1x2_away",
        "market_odds_1x2_home", "market_odds_1x2_draw", "market_odds_1x2_away",
        "open_odds_1x2_home", "open_odds_1x2_draw", "open_odds_1x2_away",
        "ev_1x2_home", "ev_1x2_draw", "ev_1x2_away",
        "prediction_1x2", "stake_1x2",
        # BTTS
        "recommendation_btts", "stake_btts",
        "market_odds_btts_yes", "market_odds_btts_no",
        "fair_odds_btts_yes", "fair_odds_btts_no",
        "ev_btts_yes", "ev_btts_no",
        "recommendation_btts_raw", "stake_btts_raw",
        "btts_filter_version", "btts_filter_reason", "btts_filtered",
        # Shadow prediction (parallel mode)
        "shadow_prob_over", "shadow_prob_under",
        "shadow_prob_home", "shadow_prob_draw", "shadow_prob_away",
        "shadow_prob_btts",
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
# COLUMN PRIORITY (untuk kerapian CSV)
# ============================================================
PENDING_PRIORITY_COLUMNS = [
    "match_uid", "home_team", "away_team", "kickoff_time", "league_name",
    "prediction", "recommendation", "prediction_status", "stake",
]

def prioritize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pindahkan kolom prioritas ke depan CSV."""
    existing = [c for c in PENDING_PRIORITY_COLUMNS if c in df.columns]
    others = [c for c in df.columns if c not in PENDING_PRIORITY_COLUMNS]
    return df[existing + others]

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
    top3_scores: Optional[List[Tuple[int, int, float]]] = None
    score_probs: Optional[List[Tuple[int, int, float]]] = None
    prob_1x2_model: Optional[Dict[str, float]] = None
    prob_1x2_league: Optional[Dict[str, float]] = None
    prob_1x2_hybrid: Optional[Dict[str, float]] = None
    ev_1x2_home: Optional[float] = None
    ev_1x2_draw: Optional[float] = None
    ev_1x2_away: Optional[float] = None
    prediction_1x2: Optional[str] = None
    stake_1x2: float = 0.0

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

        # --- 1X2 model probabilities ---
        home_prob_model = sum(p for h, a, p in all_score_probs if h > a)
        draw_prob_model = sum(p for h, a, p in all_score_probs if h == a)
        away_prob_model = sum(p for h, a, p in all_score_probs if h < a)
        total_model = home_prob_model + draw_prob_model + away_prob_model
        if total_model > 0:
            prob_1x2_model = {
                'home': home_prob_model / total_model,
                'draw': draw_prob_model / total_model,
                'away': away_prob_model / total_model
            }
        else:
            prob_1x2_model = {'home': 0.40, 'draw': 0.30, 'away': 0.30}

        # --- 1X2 league probabilities ---
        try:
            home_win_pct = float(features_df.get('home_win_pct', pd.Series([0.40])).iloc[0])
            away_win_pct = float(features_df.get('away_win_pct', pd.Series([0.30])).iloc[0])
            draw_pct = float(features_df.get('draw_pct', pd.Series([0.30])).iloc[0])
        except:
            home_win_pct, away_win_pct, draw_pct = 0.40, 0.30, 0.30
        prob_1x2_league = {'home': home_win_pct, 'draw': draw_pct, 'away': away_win_pct}

        # --- 1X2 hybrid (model * league) ---
        hybrid_home = prob_1x2_model['home'] * prob_1x2_league['home']
        hybrid_draw = prob_1x2_model['draw'] * prob_1x2_league['draw']
        hybrid_away = prob_1x2_model['away'] * prob_1x2_league['away']
        total_hybrid = hybrid_home + hybrid_draw + hybrid_away
        if total_hybrid > 0:
            prob_1x2_hybrid = {
                'home': hybrid_home / total_hybrid,
                'draw': hybrid_draw / total_hybrid,
                'away': hybrid_away / total_hybrid
            }
        else:
            prob_1x2_hybrid = {'home': 0.40, 'draw': 0.30, 'away': 0.30}

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
            score_probs=all_score_probs,
            prob_1x2_model=prob_1x2_model,
            prob_1x2_league=prob_1x2_league,
            prob_1x2_hybrid=prob_1x2_hybrid
        )

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
    uploaded_odds_1x2: Optional[dict] = None
    uploaded_open_1x2: Optional[dict] = None
    market_odds_btts: Optional[dict] = None

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
    @property
    def uploaded_odds_1x2(self) -> Optional[dict]:
        return self.state.uploaded_odds_1x2
    @uploaded_odds_1x2.setter
    def uploaded_odds_1x2(self, odds: Optional[dict]):
        self.state.uploaded_odds_1x2 = odds
    @property
    def uploaded_open_1x2(self) -> Optional[dict]:
        return self.state.uploaded_open_1x2
    @uploaded_open_1x2.setter
    def uploaded_open_1x2(self, odds: Optional[dict]):
        self.state.uploaded_open_1x2 = odds
    @property
    def market_odds_btts(self) -> Optional[dict]:
        return self.state.market_odds_btts
    @market_odds_btts.setter
    def market_odds_btts(self, odds: Optional[dict]):
        self.state.market_odds_btts = odds
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
        self.uploaded_odds_1x2 = None
        self.uploaded_open_1x2 = None
        self.market_odds_btts = None

# ============================================================
# PAGES
# ============================================================
def serialize_prediction(result: Any) -> dict:
    """Serialize a PredictionResult or its dict payload without reconstructing the dataclass.

    This is deliberately tolerant of older Session State payloads. A Streamlit rerun can
    preserve an object created by an earlier app version; reconstructing PredictionResult
    from that stale dict was the source of the Save Pending TypeError.
    """
    def get(name, default=None):
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    top3 = get('top3_scores')
    return {
        'expected_goal': get('expected_goal'),
        'prob_over': get('prob_over'),
        'prob_under': get('prob_under'),
        'prob_over_raw': get('prob_over_raw'),
        'ev_over': get('ev_over'),
        'ev_under': get('ev_under'),
        'kelly_over': get('kelly_over'),
        'kelly_under': get('kelly_under'),
        'prob_btts': get('prob_btts'),
        'prob_ht0': get('prob_ht0'),
        'confidence_ou': get('confidence_ou'),
        'confidence_btts': get('confidence_btts'),
        'confidence_ht0': get('confidence_ht0'),
        'prediction_ou': get('prediction_ou'),
        'prediction_btts': get('prediction_btts'),
        'prediction_ht0': get('prediction_ht0'),
        'top3_scores_json': json.dumps(top3) if top3 else None,
        'score_probs': get('score_probs'),
        'prob_1x2_model': get('prob_1x2_model'),
        'prob_1x2_league': get('prob_1x2_league'),
        'prob_1x2_hybrid': get('prob_1x2_hybrid'),
        'ev_1x2_home': get('ev_1x2_home'),
        'ev_1x2_draw': get('ev_1x2_draw'),
        'ev_1x2_away': get('ev_1x2_away'),
        'prediction_1x2': get('prediction_1x2'),
        'stake_1x2': get('stake_1x2', 0.0),
    }


def render_upload_section(session: SessionManager):
    with st.expander("📤 Upload File CSV", expanded=True):
        f = st.file_uploader("Pilih CSV", type=["csv"])
        if f:
            df = pd.read_csv(f)
            if 'kickoff_time' in df.columns:
                df['kickoff_time'] = df['kickoff_time'].apply(normalize_kickoff)
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
        kickoff = df['kickoff_time'].iloc[0] if 'kickoff_time' in df.columns else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df['match_uid'] = f"{home}|{away}|{kickoff}"
    if 'home_team' not in df.columns:
        df['home_team'] = home
    if 'away_team' not in df.columns:
        df['away_team'] = away

    df_clean = df.copy()
    if 'league_name' in df_clean.columns:
        df_clean = df_clean.drop(columns=['league_name'])
    df_clean = reorder_columns(df_clean)
    if 'kickoff_time' in df_clean.columns:
        df_clean['kickoff_time'] = df_clean['kickoff_time'].apply(normalize_kickoff)

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

    # --- OU decision via decision engine ---
    ou_decision = evaluate_ou_decision(
        prob_over, ev_over, ev_under,
        ev_th_over=ev_th_over, ev_th_under=ev_th_under
    )
    ou_pred = ou_decision['ou_pred']
    rec = ou_decision['recommendation']
    rec_color = ou_decision['rec_color']
    stake = ou_decision['stake']

    cs_recs = None
    odds_dict = session.uploaded_odds
    hybrid_top3 = None

    if odds_dict:
        fair_probs = calculate_fair_probs(odds_dict)
        if fair_probs and r.get('score_probs'):
            hybrid_top3 = get_hybrid_top3(r['score_probs'], fair_probs)

        scores_for_recs = hybrid_top3 if hybrid_top3 else r.get('top3_scores')
        if scores_for_recs:
            cs_recs = get_cs_recommendations(scores_for_recs, odds_dict)

    prob_1x2_model = r.get('prob_1x2_model', {})
    prob_1x2_league = r.get('prob_1x2_league', {})
    prob_1x2_hybrid = r.get('prob_1x2_hybrid', {})

    # --- 1X2 decision via decision engine ---
    odds_1x2_dict = session.uploaded_odds_1x2
    open_odds_dict = session.uploaded_open_1x2

    if odds_1x2_dict:
        decision_1x2 = compute_1x2_hybrid_and_ev(
            prob_1x2_model, prob_1x2_league, odds_1x2_dict,
            ev_threshold=ev_th_over,
            target_net_profit=100000.0
        )
        prob_1x2_hybrid_final = decision_1x2["prob_1x2_hybrid_final"]
        fair_odds_1x2 = decision_1x2["fair_odds_1x2"]
        ev_1x2_home = decision_1x2["ev_home"]
        ev_1x2_draw = decision_1x2["ev_draw"]
        ev_1x2_away = decision_1x2["ev_away"]
        prediction_1x2 = decision_1x2["prediction_1x2"]
        stake_1x2 = decision_1x2["stake_1x2"]
    else:
        prob_1x2_hybrid_final = prob_1x2_hybrid
        fair_odds_1x2 = None
        ev_1x2_home = ev_1x2_draw = ev_1x2_away = None
        prediction_1x2 = None
        stake_1x2 = 0.0

    movement = {}
    if open_odds_dict and odds_1x2_dict:
        for outcome in ['home', 'draw', 'away']:
            open_odd = open_odds_dict.get(outcome)
            current_odd = odds_1x2_dict.get(outcome)
            if open_odd and current_odd:
                movement[outcome] = round(current_odd - open_odd, 2)
            else:
                movement[outcome] = 0.0
    else:
        movement = {'home': 0.0, 'draw': 0.0, 'away': 0.0}

    # BTTS EV + protected market decision
    recommendation_btts = None
    stake_btts = 0.0
    ev_btts_yes = None
    ev_btts_no = None
    fair_odds_btts_yes = None
    fair_odds_btts_no = None
    market_odds_btts_yes = None
    market_odds_btts_no = None
    btts_filter_reason = None
    btts_filtered = False

    prob_btts = r.get('prob_btts')
    if prob_btts is not None and 0 < prob_btts < 1:
        fair_odds_btts_yes = 1 / prob_btts
        fair_odds_btts_no = 1 / (1 - prob_btts)
        market_odds_dict = st.session_state.get('_market_odds_btts_')
        if market_odds_dict:
            market_odds_btts_yes = market_odds_dict.get('yes')
            market_odds_btts_no = market_odds_dict.get('no')
            if market_odds_btts_yes and market_odds_btts_yes > 1:
                ev_btts_yes = prob_btts * market_odds_btts_yes - 1
            if market_odds_btts_no and market_odds_btts_no > 1:
                ev_btts_no = (1 - prob_btts) * market_odds_btts_no - 1

            btts_decision = evaluate_btts_filter(
                prob_btts,
                market_odds_btts_yes,
                market_odds_btts_no,
                ev_btts_yes,
                ev_btts_no,
            )
            recommendation_btts = btts_decision['recommendation']
            btts_filter_reason = btts_decision['reason']
            btts_filtered = bool(btts_decision['filtered'])

            if recommendation_btts in {'YES', 'NO'}:
                odds_for_stake = market_odds_btts_yes if recommendation_btts == 'YES' else market_odds_btts_no
                if odds_for_stake and odds_for_stake > 1:
                    stake_btts = 100000.0 / (odds_for_stake - 1)
        else:
            recommendation_btts = None
    else:
        prob_btts = None

    r['ev_1x2_home'] = ev_1x2_home
    r['ev_1x2_draw'] = ev_1x2_draw
    r['ev_1x2_away'] = ev_1x2_away
    r['prediction_1x2'] = prediction_1x2
    r['stake_1x2'] = stake_1x2

    # --- Market Coherence Filter ---
    coherence_reason = "Data skor tidak tersedia untuk coherence check."
    score_probs = r.get('score_probs')
    if score_probs:
        league_profile_for_coherence = {
            'league_avg_goals': df.get('league_avg_goals', 2.5),
            'league_over25_pct': df.get('league_over25_pct', 0.5),
            'league_btts_pct': df.get('league_btts_pct', 0.5),
            'league_under35_pct': df.get('league_under35_pct', 0.7),
            'home_win_pct': df.get('home_win_pct', 0.40),
            'away_win_pct': df.get('away_win_pct', 0.30),
            'draw_pct': df.get('draw_pct', 0.30),
        }

        odds_dict_cs = session.uploaded_odds
        fair_probs_cs = calculate_fair_probs(odds_dict_cs) if odds_dict_cs else None

        if odds_1x2_dict:
            implied_1x2 = {k: 1.0 / v for k, v in odds_1x2_dict.items() if v and v > 1.0}
            total_implied = sum(implied_1x2.values())
            fair_1x2 = {k: v / total_implied for k, v in implied_1x2.items()} if total_implied > 0 else None
        else:
            fair_1x2 = None

        passed, reason = evaluate_coherence(
            ou_pred=ou_pred,
            ou_line=ou_line,
            recommendation_btts=recommendation_btts,
            prediction_1x2=prediction_1x2,
            prob_over_model=prob_over,
            score_probs=score_probs,
            fair_probs_cs=fair_probs_cs,
            fair_1x2=fair_1x2,
            league_profile=league_profile_for_coherence,
            over_odds=over_odds,
            under_odds=under_odds,
        )

        if not passed:
            rec = "NO BET"
            rec_color = "d"
            stake = 0.0
            prediction_1x2 = "NO BET"
            stake_1x2 = 0.0
            recommendation_btts = "NO BET"
            stake_btts = 0.0
        coherence_reason = reason

    # --- Shadow Prediction (Parallel Mode) ---
    try:
        league_profile_for_shadow = {
            'league_avg_goals': df.get('league_avg_goals', 2.5),
            'home_win_pct': df.get('home_win_pct', 0.40),
            'away_win_pct': df.get('away_win_pct', 0.30),
            'draw_pct': df.get('draw_pct', 0.30),
            'league_over25_pct': df.get('league_over25_pct', 0.5),
        }
        shadow_result = compute_shadow_prediction(
            r, df, odds_1x2_dict, odds_dict,
            league_profile_for_shadow, storage
        )
        shadow_data = shadow_result
    except Exception as e:
        shadow_data = {'error': str(e)}

    summary = {
        "home": home, "away": away, "league": safe_html(str(df.get('league_name','Unknown'))),
        "ou_pred": ou_pred, "ou_line": ou_line, "over_odds": over_odds, "under_odds": under_odds,
        "expected_goal": r['expected_goal'], "confidence_ou": r['confidence_ou'],
        "ev_over": ev_over, "ev_under": ev_under,
        "kelly_over": r['kelly_over'], "kelly_under": r['kelly_under'],
        "recommendation": rec, "rec_color": rec_color, "stake": stake,
        "btts_pred": "YES" if r.get('prediction_btts', -1) == 1 else ("NO" if r.get('prediction_btts', -1) == 0 else "N/A"),
        "confidence_btts": r.get('confidence_btts'),
        "prob_btts": r.get('prob_btts'),
        "ht0_pred": "N/A", "insights": [],
        "top3_scores": r.get('top3_scores', []),
        "cs_recommendations": cs_recs,
        "hybrid_top3": hybrid_top3,
        "prob_1x2_model": prob_1x2_model,
        "prob_1x2_league": prob_1x2_league,
        "prob_1x2_hybrid": prob_1x2_hybrid,
        "prob_1x2_hybrid_final": prob_1x2_hybrid_final,
        "fair_odds_1x2": fair_odds_1x2,
        "market_odds_1x2": odds_1x2_dict,
        "open_odds_1x2": open_odds_dict,
        "ev_1x2_home": ev_1x2_home,
        "ev_1x2_draw": ev_1x2_draw,
        "ev_1x2_away": ev_1x2_away,
        "prediction_1x2": prediction_1x2,
        "stake_1x2": stake_1x2,
        "movement": movement,
        # BTTS
        "recommendation_btts": recommendation_btts,
        "stake_btts": stake_btts,
        "market_odds_btts_yes": market_odds_btts_yes,
        "market_odds_btts_no": market_odds_btts_no,
        "fair_odds_btts_yes": fair_odds_btts_yes,
        "fair_odds_btts_no": fair_odds_btts_no,
        "ev_btts_yes": ev_btts_yes,
        "ev_btts_no": ev_btts_no,
        "btts_filter_version": BTTS_FILTER_VERSION,
        "btts_filter_reason": btts_filter_reason,
        "btts_filtered": btts_filtered,
        # Coherence
        "coherence_reason": coherence_reason,
        # Shadow
        "shadow_prediction": shadow_data,
    }
    return summary

def render_pending(session: SessionManager, storage: StorageProvider):
    st.subheader("📋 Pending Manager")
    db = DatabaseManager(storage)
    df = db.load_pending()
    if df.empty: st.info("Belum ada data."); return

    filter_col = "btts_filter_version"
    legacy_mask = ~df.get(filter_col, pd.Series(index=df.index, dtype=object)).fillna("").astype(str).eq(BTTS_FILTER_VERSION)
    if legacy_mask.any():
        changed = False
        for idx in df.index[legacy_mask]:
            original = df.loc[idx].to_dict()
            normalized = normalize_btts_record(original)
            for col, value in normalized.items():
                if col not in df.columns or str(df.at[idx, col]) != str(value):
                    df.at[idx, col] = value
                    changed = True
        if changed:
            df = prioritize_columns(df)
            db.save_pending(df)

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

        conf_val = row.get('confidence_ou')
        if conf_val is not None and not pd.isna(conf_val):
            conf_pct = conf_val * 100
            if conf_pct >= 80:
                bg = "#16a34a"; text_color = "white"
            elif conf_pct >= 70:
                bg = "#65a30d"; text_color = "white"
            elif conf_pct >= 60:
                bg = "#eab308"; text_color = "black"
            else:
                bg = "#334155"; text_color = "white"
            conf_badge = f"{conf_pct:.0f}%"
        else:
            bg = "#334155"; text_color = "white"
            conf_badge = "?"

        cols = st.columns([0.15, 0.85])
        with cols[0]:
            st.markdown(
                f'<div style="background-color:{bg}; color:{text_color}; '
                f'padding:10px 5px; border-radius:12px; text-align:center; font-weight:800; '
                f'font-size:0.9rem;">{conf_badge}</div>',
                unsafe_allow_html=True
            )
        with cols[1]:
            with st.expander(f"▶ {home_safe} vs {away_safe}"):
                lines = []
                pred_str = str(row.get('prediction', '')).strip()
                rec_str = str(row.get('recommendation', '')).strip()
                stake_val = row.get('stake')
                if pred_str and rec_str and rec_str != 'NO BET' and stake_val is not None and float(stake_val) > 0:
                    if pred_str.startswith('OVER'):
                        odds = row.get('current_over_odds')
                    else:
                        odds = row.get('current_under_odds')
                    if odds is not None and not pd.isna(odds):
                        lines.append(f"{pred_str} ({rec_str} @{float(odds):.2f} - Rp{int(float(stake_val)):,})")
                    else:
                        lines.append(f"{pred_str} ({rec_str} - Rp{int(float(stake_val)):,})")
                elif pred_str:
                    lines.append(f"{pred_str} (NO BET)")

                pred_1x2 = row.get('prediction_1x2')
                stake_1x2 = row.get('stake_1x2')
                if pred_1x2 and pred_1x2 != 'NO BET' and stake_1x2 is not None and float(stake_1x2) > 0:
                    outcome_key = pred_1x2.lower()
                    odds_1x2 = row.get(f'market_odds_1x2_{outcome_key}')
                    if odds_1x2 is not None and not pd.isna(odds_1x2) and float(odds_1x2) > 0:
                        lines.append(f"1X2 (TARUHAN {pred_1x2} @{float(odds_1x2):.2f} - Rp{int(float(stake_1x2)):,})")
                    else:
                        lines.append(f"1X2 (TARUHAN {pred_1x2} - Rp{int(float(stake_1x2)):,})")
                elif pred_1x2 == 'NO BET':
                    lines.append("1X2 (NO BET)")

                rec_btts = row.get('recommendation_btts')
                stake_btts = row.get('stake_btts')
                if rec_btts and rec_btts != 'NO BET' and stake_btts is not None and float(stake_btts) > 0:
                    odds_col = 'market_odds_btts_yes' if rec_btts == 'YES' else 'market_odds_btts_no'
                    odds_btts = row.get(odds_col)
                    if odds_btts is not None and not pd.isna(odds_btts) and float(odds_btts) > 0:
                        lines.append(f"BTTS (TARUHAN {rec_btts} @{float(odds_btts):.2f} - Rp{int(float(stake_btts)):,})")
                    else:
                        lines.append(f"BTTS (TARUHAN {rec_btts} - Rp{int(float(stake_btts)):,})")
                elif rec_btts == 'NO BET':
                    lines.append("BTTS (NO BET)")

                for i in range(1, 4):
                    score_str = row.get(f'cs_score_{i}')
                    odds_cs = row.get(f'cs_odds_{i}')
                    stake_cs = row.get(f'cs_stake_{i}')
                    if score_str and odds_cs is not None and not pd.isna(odds_cs) and stake_cs is not None and float(stake_cs) > 0:
                        lines.append(f"CS{i} (TARUHAN CS {score_str} @{float(odds_cs):.2f} - Rp{int(float(stake_cs)):,})")

                if lines:
                    for line in lines:
                        st.markdown(f"<p style='margin:2px 0;'>{safe_html(line)}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>Belum ada rekomendasi.</p>", unsafe_allow_html=True)
                st.markdown("---")

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
                        success, message, _ = PendingService.process_settlement(
                            row.to_dict(),
                            ht_h_val, ht_a_val,
                            ft_h_val, ft_a_val,
                            storage,
                            session
                        )
                        if success:
                            st.success(message)
                        else:
                            st.warning(message)
                    else:
                        df.at[idx,'prediction_status'] = 'SCORE_ENTERED'
                        db.save_pending(df)
                        st.warning("Skor disimpan dengan catatan (HT > FT).")

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

    pc = ProfitCalculator()
    raw['profit'] = [pc.calculate(row.to_dict(), 100000)[0] for _, row in raw.iterrows()]
    raw['status'] = [pc.calculate(row.to_dict(), 100000)[1] for _, row in raw.iterrows()]
    accumulated_profit_ou = pd.to_numeric(raw['profit'], errors='coerce').fillna(0.0).sum()
    st.metric("Profit Akumulasi OU", f"Rp {accumulated_profit_ou:+,.0f}")

    start_date = hari_ini - timedelta(days=6)
    mask = (raw['tanggal'] >= start_date) & (raw['tanggal'] <= hari_ini)
    df_recent = raw[mask].copy()
    if df_recent.empty:
        st.info("Tidak ada data settlement dalam 7 hari terakhir.")
        return

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

def render_match_pnl(session: SessionManager, storage: StorageProvider):
    st.subheader("💰 MATCH P&L")
    try:
        raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except Exception as exc:
        st.error(f"Gagal memuat history: {exc}")
        return

    if raw.empty:
        st.info("Belum ada data settlement untuk dihitung.")
        return

    required = {"match_uid", "home_team", "away_team", "home_goals", "away_goals"}
    missing = sorted(required - set(raw.columns))
    if missing:
        st.warning(f"History belum memiliki kolom wajib: {', '.join(missing)}")
        return

    df = raw.copy()
    if "settlement_time" in df.columns:
        df["_match_pnl_dt"] = pd.to_datetime(df["settlement_time"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        cutoff = today - pd.Timedelta(days=6)
        df = df[df["_match_pnl_dt"].dt.normalize() >= cutoff]
        df = df.sort_values("_match_pnl_dt", ascending=False, na_position="last")

    df = df.drop_duplicates(subset=["match_uid"], keep="first")
    if df.empty:
        st.info("Belum ada pertandingan settlement dalam 7 hari terakhir.")
        return

    summaries = []
    for _, row in df.iterrows():
        try:
            result = build_match_pnl(row.to_dict())
            summaries.append(result)
        except Exception:
            continue

    if not summaries:
        st.info("Belum ada pertandingan yang memiliki settlement dan taruhan aktif.")
        return

    by_uid = {str(r.get("match_uid")): r for r in summaries}
    date_by_uid = {}
    for _, row in df.iterrows():
        uid = str(row.get("match_uid"))
        dt = row.get("_match_pnl_dt", pd.NaT)
        date_by_uid[uid] = dt.date() if not pd.isna(dt) else None

    groups = {}
    for result in summaries:
        uid = str(result.get("match_uid"))
        groups.setdefault(date_by_uid.get(uid), []).append(result)

    today_date = pd.Timestamp.now().date()
    selected_uid = st.session_state.get("match_pnl_selected_uid")

    for day_key, day_results in groups.items():
        day_total = sum(float(r.get("net_pnl", 0.0)) for r in day_results)
        if day_key == today_date:
            label = "Hari Ini"
        elif day_key == today_date - timedelta(days=1):
            label = "Kemarin"
        elif day_key is None:
            label = "Tanggal Tidak Diketahui"
        else:
            label = pd.Timestamp(day_key).strftime("%A, %d %B %Y")

        day_id = str(day_key) if day_key is not None else "unknown"
        day_expander = st.expander(
            f"▼ {label} — Rp {day_total:+,.0f}",
            expanded=(day_key == today_date),
            on_change="rerun",
            key=f"match_pnl_day_{day_id}",
        )
        if day_expander.open:
            with day_expander:
                for result in day_results:
                    uid = str(result.get("match_uid"))
                    home = safe_html(str(result.get("home_team", "")))
                    away = safe_html(str(result.get("away_team", "")))
                    score = f"{result['home_goals']}-{result['away_goals']}"
                    net = float(result.get("net_pnl", 0.0))
                    icon = "🟢" if net > 0 else "🔴" if net < 0 else "⚪"
                    if st.button(
                        f"{icon} {home} vs {away} | {score} | Rp {net:+,.0f}",
                        key=f"match_pnl_{uid}",
                        use_container_width=True,
                    ):
                        st.session_state["match_pnl_selected_uid"] = uid
                        selected_uid = uid

    if selected_uid and selected_uid in by_uid:
        result = by_uid[selected_uid]
        home = safe_html(str(result.get("home_team", "")))
        away = safe_html(str(result.get("away_team", "")))
        score = f"{result['home_goals']}-{result['away_goals']}"
        net = float(result.get("net_pnl", 0.0))
        st.markdown(f"### {home} vs {away} | {score} | Rp {net:+,.0f}")
        st.markdown(f"**Total Modal:** Rp {result['total_stake']:,.0f}  \n**Total Return:** Rp {result['total_return']:,.0f}")
        if result["legs"]:
            st.markdown(format_pnl_table_html(result["legs"]), unsafe_allow_html=True)
        else:
            st.info("Tidak ada taruhan aktif pada pertandingan ini.")

def render_1x2_history(session: SessionManager, storage: StorageProvider):
    st.subheader("🎯 1X2 History")
    try:
        raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except:
        raw = pd.DataFrame()
    if raw.empty:
        st.info("Belum ada data history.")
        return

    if 'prediction_1x2' not in raw.columns:
        st.info("Data 1X2 belum tersedia di history.")
        return

    df = raw.dropna(subset=['prediction_1x2']).copy()
    df = df[df['prediction_1x2'] != 'NO BET']
    if df.empty:
        st.info("Belum ada taruhan 1X2 yang direkomendasikan.")
        return

    df['settlement_dt'] = pd.to_datetime(df.get('settlement_time'), errors='coerce')
    df = df.dropna(subset=['settlement_dt'])
    df['tanggal'] = df['settlement_dt'].dt.date

    if 'profit_1x2' not in df.columns:
        df['profit_1x2'] = np.nan
    if 'result_1x2' not in df.columns:
        df['result_1x2'] = None

    for idx, row in df.iterrows():
        if pd.notna(row.get('profit_1x2')):
            continue
        pred = str(row.get('prediction_1x2', '')).strip().upper()
        try:
            stake = float(row.get('stake_1x2', 0) or 0)
        except (TypeError, ValueError):
            stake = 0.0
        try:
            home_goals = int(row.get('home_goals', 0) or 0)
            away_goals = int(row.get('away_goals', 0) or 0)
        except (TypeError, ValueError):
            home_goals = away_goals = 0
        actual = 'HOME' if home_goals > away_goals else ('DRAW' if home_goals == away_goals else 'AWAY')
        try:
            market_odds = float(row.get(f'market_odds_1x2_{pred.lower()}', 0) or 0)
        except (TypeError, ValueError):
            market_odds = 0.0
        if pred in ('HOME', 'DRAW', 'AWAY') and stake > 0 and market_odds > 1:
            profit = stake * (market_odds - 1) if pred == actual else -stake
            df.at[idx, 'profit_1x2'] = profit
            df.at[idx, 'result_1x2'] = 'WIN' if profit > 0 else 'LOSE'

    valid_profit = pd.to_numeric(df['profit_1x2'], errors='coerce').notna()
    df = df[valid_profit].copy()
    if df.empty:
        st.info("Data profit 1X2 belum tersedia di history.")
        return

    df['profit_1x2'] = pd.to_numeric(df['profit_1x2'], errors='coerce').fillna(0.0)
    accumulated_profit_1x2 = df['profit_1x2'].sum()

    hari_ini = datetime.now().date()
    start_date = hari_ini - timedelta(days=6)
    df = df[(df['tanggal'] >= start_date) & (df['tanggal'] <= hari_ini)].copy()
    st.metric("Profit Akumulasi 1X2", f"Rp {accumulated_profit_1x2:+,.0f}")
    if df.empty:
        st.info("Tidak ada data 1X2 dalam 7 hari terakhir.")
        return
    df = df.sort_values('settlement_dt', ascending=False)

    total_profit = df['profit_1x2'].sum()
    total_bets = len(df)
    win_count = (df['result_1x2'].astype(str).str.upper() == 'WIN').sum()
    total_stake = pd.to_numeric(df.get('stake_1x2', 0), errors='coerce').fillna(0).sum()

    st.metric("Total Profit 1X2 (7 Hari)", f"Rp {total_profit:+,.0f}")
    if total_bets > 0:
        win_rate = win_count / total_bets * 100
        roi = total_profit / total_stake * 100 if total_stake > 0 else 0.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Win Rate", f"{win_rate:.1f}%")
        col2.metric("ROI", f"{roi:.1f}%")
        col3.metric("Akurasi", f"{win_count}/{total_bets}")

    grouped = df.groupby('tanggal')
    tanggal_list = sorted(grouped.groups.keys(), reverse=True)
    hari_map = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    bulan_map = {
        1:'Januari',2:'Februari',3:'Maret',4:'April',5:'Mei',6:'Juni',
        7:'Juli',8:'Agustus',9:'September',10:'Oktober',11:'November',12:'Desember'
    }

    for tgl in tanggal_list:
        group = grouped.get_group(tgl)
        total_tgl = group['profit_1x2'].sum()
        if tgl == hari_ini:
            label = "Hari Ini"
        elif tgl == hari_ini - timedelta(days=1):
            label = "Kemarin"
        else:
            nama_hari = hari_map[tgl.weekday()]
            label = f"{nama_hari}, {tgl.day} {bulan_map[tgl.month]} {tgl.year}"

        with st.expander(f"{label} - Rp {total_tgl:+,.0f}"):
            group_sorted = group.sort_values('settlement_dt', ascending=False)
            for _, row in group_sorted.iterrows():
                home = safe_html(str(row.get('home_team','')))
                away = safe_html(str(row.get('away_team','')))
                league = safe_html(str(row.get('league_name', '')))
                pred = row.get('prediction_1x2')
                home_goals = int(row.get('home_goals', 0) or 0)
                away_goals = int(row.get('away_goals', 0) or 0)
                score = f"{home_goals}-{away_goals}"
                profit = row.get('profit_1x2', 0.0)
                result = row.get('result_1x2', 'N/A')
                stake = float(row.get('stake_1x2', 0) or 0)
                pred_lower = str(pred).lower() if pred else ''
                odds = float(row.get(f'market_odds_1x2_{pred_lower}', 0) or 0)

                if result == 'WIN':
                    bg = "#16a34a"
                elif result == 'LOSE':
                    bg = "#ef4444"
                else:
                    bg = "#334155"

                profit_str = f"Rp {profit:+,.0f}"
                st.markdown(
                    f"<div style='background-color:{bg}; padding:8px; border-radius:8px; "
                    f"margin:4px 0; color:white;'>"
                    f"<strong>{home} vs {away}</strong> | {league} | Pred: {pred} | "
                    f"Skor: {score} | Odds: {odds:.2f} | Stake: Rp{stake:,.0f} | {result} | {profit_str}"
                    f"</div>", unsafe_allow_html=True
                )

def render_btts_history(session: SessionManager, storage: StorageProvider):
    st.subheader("🤝 BTTS History")
    try:
        raw = storage.load_dataframe(ResourceRegistry.HISTORY)
    except:
        raw = pd.DataFrame()
    if raw.empty:
        st.info("Belum ada data history.")
        return

    if 'recommendation_btts' not in raw.columns:
        st.info("Data BTTS belum tersedia di history.")
        return

    df = raw.dropna(subset=['recommendation_btts']).copy()
    df = df[df['recommendation_btts'] != 'NO BET']
    if df.empty:
        st.info("Belum ada taruhan BTTS yang direkomendasikan.")
        return

    df['settlement_dt'] = pd.to_datetime(df.get('settlement_time'), errors='coerce')
    df = df.dropna(subset=['settlement_dt'])
    df['tanggal'] = df['settlement_dt'].dt.date

    if 'profit_btts' not in df.columns:
        df['profit_btts'] = np.nan
    if 'result_btts' not in df.columns:
        df['result_btts'] = None

    for idx, row in df.iterrows():
        if pd.notna(row.get('profit_btts')):
            continue
        pred = str(row.get('recommendation_btts', '')).strip().upper()
        try:
            stake = float(row.get('stake_btts', 0) or 0)
        except (TypeError, ValueError):
            stake = 0.0
        try:
            actual_btts = int(row.get('actual_btts', 0) or 0)
        except (TypeError, ValueError):
            actual_btts = 0
        odds_col = 'market_odds_btts_yes' if pred == 'YES' else 'market_odds_btts_no'
        try:
            market_odds = float(row.get(odds_col, 0) or 0)
        except (TypeError, ValueError):
            market_odds = 0.0
        if pred in ('YES', 'NO') and stake > 0 and market_odds > 1:
            won = (pred == 'YES' and actual_btts == 1) or (pred == 'NO' and actual_btts == 0)
            profit = stake * (market_odds - 1) if won else -stake
            df.at[idx, 'profit_btts'] = profit
            df.at[idx, 'result_btts'] = 'WIN' if profit > 0 else 'LOSE'

    valid_profit = pd.to_numeric(df['profit_btts'], errors='coerce').notna()
    df = df[valid_profit].copy()
    if df.empty:
        st.info("Data profit BTTS belum tersedia di history.")
        return

    df['profit_btts'] = pd.to_numeric(df['profit_btts'], errors='coerce').fillna(0.0)
    accumulated_profit_btts = df['profit_btts'].sum()

    hari_ini = datetime.now().date()
    start_date = hari_ini - timedelta(days=6)
    df = df[(df['tanggal'] >= start_date) & (df['tanggal'] <= hari_ini)].copy()
    st.metric("Profit Akumulasi BTTS", f"Rp {accumulated_profit_btts:+,.0f}")
    if df.empty:
        st.info("Tidak ada data BTTS dalam 7 hari terakhir.")
        return
    df = df.sort_values('settlement_dt', ascending=False)

    total_profit = df['profit_btts'].sum()
    total_bets = len(df)
    win_count = (df['result_btts'].astype(str).str.upper() == 'WIN').sum()
    total_stake = pd.to_numeric(df.get('stake_btts', 0), errors='coerce').fillna(0).sum()

    st.metric("Total Profit BTTS (7 Hari)", f"Rp {total_profit:+,.0f}")
    if total_bets > 0:
        win_rate = win_count / total_bets * 100
        roi = total_profit / total_stake * 100 if total_stake > 0 else 0.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Win Rate", f"{win_rate:.1f}%")
        col2.metric("ROI", f"{roi:.1f}%")
        col3.metric("Akurasi", f"{win_count}/{total_bets}")

    grouped = df.groupby('tanggal')
    tanggal_list = sorted(grouped.groups.keys(), reverse=True)
    hari_map = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    bulan_map = {
        1:'Januari',2:'Februari',3:'Maret',4:'April',5:'Mei',6:'Juni',
        7:'Juli',8:'Agustus',9:'September',10:'Oktober',11:'November',12:'Desember'
    }

    for tgl in tanggal_list:
        group = grouped.get_group(tgl)
        total_tgl = group['profit_btts'].sum()
        if tgl == hari_ini:
            label = "Hari Ini"
        elif tgl == hari_ini - timedelta(days=1):
            label = "Kemarin"
        else:
            nama_hari = hari_map[tgl.weekday()]
            label = f"{nama_hari}, {tgl.day} {bulan_map[tgl.month]} {tgl.year}"

        with st.expander(f"{label} - Rp {total_tgl:+,.0f}"):
            group_sorted = group.sort_values('settlement_dt', ascending=False)
            for _, row in group_sorted.iterrows():
                home = safe_html(str(row.get('home_team','')))
                away = safe_html(str(row.get('away_team','')))
                league = safe_html(str(row.get('league_name', '')))
                pred = row.get('recommendation_btts')
                home_goals = int(row.get('home_goals', 0) or 0)
                away_goals = int(row.get('away_goals', 0) or 0)
                score = f"{home_goals}-{away_goals}"
                profit = row.get('profit_btts', 0.0)
                result = row.get('result_btts', 'N/A')
                stake = float(row.get('stake_btts', 0) or 0)
                odds_col = 'market_odds_btts_yes' if str(pred).upper() == 'YES' else 'market_odds_btts_no'
                odds = float(row.get(odds_col, 0) or 0)

                if result == 'WIN':
                    bg = "#16a34a"
                elif result == 'LOSE':
                    bg = "#ef4444"
                else:
                    bg = "#334155"

                profit_str = f"Rp {profit:+,.0f}"
                st.markdown(
                    f"<div style='background-color:{bg}; padding:8px; border-radius:8px; "
                    f"margin:4px 0; color:white;'>"
                    f"<strong>{home} vs {away}</strong> | {league} | Pred: {pred} | "
                    f"Skor: {score} | Odds: {odds:.2f} | Stake: Rp{stake:,.0f} | {result} | {profit_str}"
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

    base_cols = ['settlement_time','home_team','away_team','home_goals','away_goals']
    cs_cols = ['cs_score_1','cs_score_2','cs_score_3','cs_profit']
    available_cs = [c for c in cs_cols if c in raw.columns]
    available_cols = base_cols + available_cs
    if 'cs_profit' not in available_cols:
        st.info("Data correct score belum tersedia di history.")
        return

    df = raw[available_cols].copy()
    df['settlement_time'] = pd.to_datetime(df['settlement_time'], errors='coerce')
    df = df.dropna(subset=['settlement_time'])
    df['tanggal'] = df['settlement_time'].dt.date

    df['cs_profit'] = pd.to_numeric(df['cs_profit'], errors='coerce')
    df = df.dropna(subset=['cs_profit']).copy()
    if df.empty:
        st.info("Data profit Correct Score belum tersedia di history.")
        return

    accumulated_profit_cs = df['cs_profit'].sum()
    st.metric("Profit Akumulasi Correct Score", f"Rp {accumulated_profit_cs:+,.0f}")

    hari_ini = datetime.now().date()
    start_date = hari_ini - timedelta(days=6)
    df = df[(df['tanggal'] >= start_date) & (df['tanggal'] <= hari_ini)].copy()
    if df.empty:
        st.info("Tidak ada data Correct Score dalam 7 hari terakhir.")
        return
    df = df.sort_values('settlement_time', ascending=False)

    total_cs_profit = df['cs_profit'].sum()
    st.metric("Total Profit Correct Score (7 Hari)", f"Rp {total_cs_profit:+,.0f}")

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

    display_cols = [c for c in available_cols if c in df.columns]
    st.dataframe(df[display_cols])

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

        # ---- Edit Statistik Liga ----
        with st.expander("✏️ Edit Statistik Liga"):
            try:
                profil_league = db_storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
            except:
                profil_league = pd.DataFrame()

            for col, default in [('home_win_pct', 0.40), ('away_win_pct', 0.30), ('draw_pct', 0.30)]:
                if col not in profil_league.columns:
                    profil_league[col] = default

            if profil_league.empty:
                st.info("File profil liga tidak tersedia.")
            else:
                profil_league = profil_league.sort_values('league_name', ascending=True)
                league_options = [f"[{int(row['league_code'])}] {row['league_name']}" for _, row in profil_league.iterrows()]
                selected_league_str = st.selectbox("Pilih Liga", league_options, key="edit_league_select")

                if st.button("📂 Load Data Liga"):
                    selected_code = int(selected_league_str.split(']')[0].replace('[', ''))
                    st.session_state['edit_league_code'] = selected_code
                    st.rerun()

                if 'edit_league_code' in st.session_state and st.session_state['edit_league_code'] is not None:
                    selected_code = st.session_state['edit_league_code']
                    mask = profil_league['league_code'] == selected_code
                    if mask.any():
                        liga_row = profil_league[mask].iloc[0]

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

                        if st.button("💾 Simpan Statistik"):
                            profil_league.loc[mask, 'league_avg_goals'] = new_avg
                            profil_league.loc[mask, 'league_over25_pct'] = new_over25
                            profil_league.loc[mask, 'league_btts_pct'] = new_btts_pct
                            profil_league.loc[mask, 'league_under35_pct'] = new_under35
                            profil_league.loc[mask, 'home_win_pct'] = new_home_win
                            profil_league.loc[mask, 'away_win_pct'] = new_away_win
                            profil_league.loc[mask, 'draw_pct'] = new_draw
                            db_storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, profil_league)
                            st.session_state.pop('cached_profil', None)
                            session.invalidate_league_profile_cache()
                            st.cache_resource.clear()
                            st.success("Statistik liga berhasil diperbarui.")
                            st.rerun()
                    else:
                        st.error("Kode liga tidak ditemukan di profil.")
                        st.session_state.pop('edit_league_code', None)
                else:
                    st.info("Pilih liga dari dropdown, lalu klik 'Load Data Liga' untuk mengedit.")

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
            combined_file = st.file_uploader("📊 Upload Odds 1X2 & Correct Score (CSV)", type=["csv"], key="combined_odds")
            if combined_file:
                try:
                    combined_bytes = combined_file.getvalue()
                    combined_sig = (combined_file.name, len(combined_bytes), hash(combined_bytes))
                    combined_changed = st.session_state.get("_combined_odds_sig") != combined_sig

                    if combined_changed:
                        ps = session.get_prediction_state()
                        match_uid = (
                            ps.prediction_dataframe.iloc[0]['match_uid']
                            if ps.prediction_dataframe is not None and not ps.prediction_dataframe.empty
                            else None
                        )
                        combined = process_combined_odds(combined_bytes, match_uid, db_storage)
                        st.session_state["_combined_odds_cache"] = combined
                        st.session_state["_combined_odds_sig"] = combined_sig
                    else:
                        combined = st.session_state.get("_combined_odds_cache", {})

                    if combined.get('open_1x2'):
                        session.uploaded_open_1x2 = combined['open_1x2']
                    else:
                        session.uploaded_open_1x2 = None
                    if combined.get('1x2'):
                        session.uploaded_odds_1x2 = combined['1x2']
                        if combined_changed:
                            st.success("Odds 1X2 berhasil diunggah.")
                    if combined.get('cs'):
                        session.uploaded_odds = combined['cs']
                        if combined_changed:
                            st.success("Odds Correct Score berhasil diunggah.")
                except Exception as e:
                    st.error(f"Gagal membaca file odds: {e}")
                    session.uploaded_odds = None
                    session.uploaded_odds_1x2 = None
                    session.uploaded_open_1x2 = None
            else:
                session.uploaded_odds = None
                session.uploaded_odds_1x2 = None
                session.uploaded_open_1x2 = None

            summary = get_match_prediction_summary(session, db_storage)
            if summary:
                render_prediction_card(summary)
                if st.button("💾 Save Pending"):
                    ps = session.get_prediction_state()
                    row = ps.prediction_dataframe.iloc[0].to_dict()
                    prediction_payload = ps.prediction_result or {}
                    serialized = serialize_prediction(prediction_payload)
                    row.update(serialized)
                    kickoff_raw = session.uploaded_df['kickoff_time'].iloc[0] if 'kickoff_time' in session.uploaded_df.columns else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    kickoff_clean = normalize_kickoff(kickoff_raw)
                    row['kickoff_time'] = kickoff_clean
                    row['match_uid'] = f"{summary['home']}|{summary['away']}|{kickoff_clean}"
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
                        for i, rec in enumerate(cs_recs):
                            row[f'cs_score_{i+1}'] = f"{rec[0]}:{rec[1]}"
                            row[f'cs_odds_{i+1}'] = rec[2]
                            odds = rec[2]
                            row[f'cs_stake_{i+1}'] = 200000.0 / (odds - 1) if odds > 1 else 0
                    else:
                        for i in range(1, 4):
                            row[f'cs_score_{i}'] = None
                            row[f'cs_odds_{i}'] = None
                            row[f'cs_stake_{i}'] = None

                    # Simpan data 1X2
                    if summary.get('prob_1x2_model'):
                        row['prob_1x2_model_home'] = summary['prob_1x2_model'].get('home')
                        row['prob_1x2_model_draw'] = summary['prob_1x2_model'].get('draw')
                        row['prob_1x2_model_away'] = summary['prob_1x2_model'].get('away')
                    if summary.get('prob_1x2_league'):
                        row['prob_1x2_league_home'] = summary['prob_1x2_league'].get('home')
                        row['prob_1x2_league_draw'] = summary['prob_1x2_league'].get('draw')
                        row['prob_1x2_league_away'] = summary['prob_1x2_league'].get('away')
                    if summary.get('prob_1x2_hybrid_final') or summary.get('prob_1x2_hybrid'):
                        hybrid = summary['prob_1x2_hybrid_final'] or summary['prob_1x2_hybrid']
                        row['prob_1x2_hybrid_home'] = hybrid.get('home')
                        row['prob_1x2_hybrid_draw'] = hybrid.get('draw')
                        row['prob_1x2_hybrid_away'] = hybrid.get('away')
                    if summary.get('fair_odds_1x2'):
                        row['fair_odds_1x2_home'] = summary['fair_odds_1x2'].get('home')
                        row['fair_odds_1x2_draw'] = summary['fair_odds_1x2'].get('draw')
                        row['fair_odds_1x2_away'] = summary['fair_odds_1x2'].get('away')
                    if summary.get('market_odds_1x2'):
                        row['market_odds_1x2_home'] = summary['market_odds_1x2'].get('home')
                        row['market_odds_1x2_draw'] = summary['market_odds_1x2'].get('draw')
                        row['market_odds_1x2_away'] = summary['market_odds_1x2'].get('away')
                    if summary.get('open_odds_1x2'):
                        row['open_odds_1x2_home'] = summary['open_odds_1x2'].get('home')
                        row['open_odds_1x2_draw'] = summary['open_odds_1x2'].get('draw')
                        row['open_odds_1x2_away'] = summary['open_odds_1x2'].get('away')

                    row['ev_1x2_home'] = summary.get('ev_1x2_home')
                    row['ev_1x2_draw'] = summary.get('ev_1x2_draw')
                    row['ev_1x2_away'] = summary.get('ev_1x2_away')
                    row['prediction_1x2'] = summary.get('prediction_1x2')
                    row['stake_1x2'] = summary.get('stake_1x2', 0)

                    # Simpan data BTTS
                    row['recommendation_btts'] = summary.get('recommendation_btts')
                    row['stake_btts'] = summary.get('stake_btts', 0)
                    row['market_odds_btts_yes'] = summary.get('market_odds_btts_yes')
                    row['market_odds_btts_no'] = summary.get('market_odds_btts_no')
                    row['fair_odds_btts_yes'] = summary.get('fair_odds_btts_yes')
                    row['fair_odds_btts_no'] = summary.get('fair_odds_btts_no')
                    row['ev_btts_yes'] = summary.get('ev_btts_yes')
                    row['ev_btts_no'] = summary.get('ev_btts_no')
                    row['recommendation_btts_raw'] = row.get('recommendation_btts')
                    row['stake_btts_raw'] = row.get('stake_btts', 0)
                    row['btts_filter_version'] = summary.get('btts_filter_version', BTTS_FILTER_VERSION)
                    row['btts_filter_reason'] = summary.get('btts_filter_reason')
                    row['btts_filtered'] = summary.get('btts_filtered', False)
                    row = normalize_btts_record(row)

                    # --- Shadow prediction data ---
                    shadow_data = summary.get('shadow_prediction', {})
                    if shadow_data and 'error' not in shadow_data:
                        row['shadow_prob_over'] = shadow_data.get('shadow_prob_over')
                        row['shadow_prob_under'] = shadow_data.get('shadow_prob_under')
                        row['shadow_prob_home'] = shadow_data.get('shadow_prob_home')
                        row['shadow_prob_draw'] = shadow_data.get('shadow_prob_draw')
                        row['shadow_prob_away'] = shadow_data.get('shadow_prob_away')
                        row['shadow_prob_btts'] = shadow_data.get('shadow_prob_btts')

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
                        pend = prioritize_columns(pend)
                        db.save_pending(pend)
                        st.success("Disimpan ke Pending!")
                    st.rerun()

        tab1, tab2, tab_match_pnl, tab2_5, tab2_6, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📋 Pending", "📝 Settlement", "💰 MATCH P&L", "🎯 1X2",
            "🤝 BTTS", "🎯 Correct Score", "📜 History", "🧠 Learning",
            "🗄️ Database", "📋 Trans Log", "🐞 Debug", "📊 Perf"
        ], on_change="rerun", key="main_tabs")

        if tab1.open:
            with tab1: render_pending(session, db_storage)
        if tab2.open:
            with tab2: render_settlement(session, db_storage)
        if tab_match_pnl.open:
            with tab_match_pnl: render_match_pnl(session, db_storage)
        if tab2_5.open:
            with tab2_5: render_1x2_history(session, db_storage)
        if tab2_6.open:
            with tab2_6: render_btts_history(session, db_storage)
        if tab3.open:
            with tab3: render_cs_history(session, db_storage)
        if tab4.open:
            with tab4: render_history(session, db_storage)
        if tab5.open:
            with tab5: render_learning(session, db_storage)
        if tab6.open:
            with tab6: render_database(session, db_storage)
        if tab7.open:
            with tab7: st.info("Transaction Log")
        if tab8.open:
            with tab8: render_debug(session)
        if tab9.open:
            with tab9: render_performance(session, app_storage, db_storage)

    if session.get_debug_trace():
        with st.expander("📜 Raw Debug Trace"):
            for line in session.get_debug_trace()[-100:]:
                st.text(line)

if __name__ == "__main__":
    main()