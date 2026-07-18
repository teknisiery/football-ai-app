# ============================================================
# FOOTBALL AI V2 STABLE REV10.2 – Final Fix
# ============================================================
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import copy
import base64
import requests
import uuid
from io import BytesIO
from datetime import datetime, date, timedelta
from xgboost import XGBRegressor
from scipy.stats import poisson
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, roc_auc_score, log_loss,
    mean_absolute_error, mean_squared_error
)
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
from enum import Enum
import time

APP_TITLE = "Football AI V2"
APP_VERSION = "2.0.0"
APP_LAYOUT = "wide"

st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT, initial_sidebar_state="expanded")
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CALIBRATION LEVELS CONFIG
# ============================================================
CALIBRATION_LEVELS = {
    "Excellent": 0.01,
    "Good": 0.05,
    "Needs Improvement": 0.15,
    "Poor": float("inf")
}

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
body { background: #0f1117; color: #ffffff; }
.block-container { padding-top: 2rem; }
.card {
    background: #1c1f26;
    border-radius: 24px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid #2a2e38;
    color: #ffffff;
}
.prediction-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 24px;
    padding: 18px 20px 14px 20px;
    margin: 12px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 2px solid #2563eb;
}
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 40px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-s { background: #0d6e0d; color: #ffffff; }
.badge-a { background: #16a34a; color: #ffffff; }
.badge-b { background: #2563eb; color: #ffffff; }
.badge-c { background: #f97316; color: #ffffff; }
.badge-d { background: #ef4444; color: #ffffff; }
.badge-neutral { background: #334155; color: #ffffff; }

.brain-row, .info-row {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: stretch;
    gap: 10px;
    flex-wrap: nowrap;
    width: 100%;
    margin: 10px 0;
}
.brain-card, .info-card {
    flex: 1;
    border-radius: 16px;
    padding: 12px 6px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 0;
}
.brain-card .icon { font-size: 1.3rem; margin-bottom: 3px; }
.brain-card .label { font-size: 0.6rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.3px; }
.brain-card .badge-value {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 800;
    margin-top: 4px;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 24px;
    font-weight: 700;
    font-size: 1rem;
    transition: all 0.2s;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.6);
}
.stFileUploader > div {
    border-radius: 16px;
    border: 2px dashed #4b5563;
    background: #1c1f26;
    padding: 20px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #1c1f26;
    border-radius: 16px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    color: #a0a0b0;
    background: transparent;
}
.stTabs [aria-selected="true"] { background: #2563eb; color: white; }
@media (max-width: 480px) {
    .prediction-card { padding: 12px 14px 10px 14px; }
    .brain-card .badge-value { font-size: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# RESOURCE REGISTRY – Single Source of Truth
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
    HISTORY = Resource(
        id="history", type="dataframe", serializer="csv",
        default_filename="history_ou.csv", description="Historical Prediction Database"
    )
    DATASET = Resource(
        id="dataset", type="dataframe", serializer="csv",
        default_filename="dataset_ou.csv", description="Training Dataset"
    )
    PENDING = Resource(
        id="pending", type="dataframe", serializer="csv",
        default_filename="pending_predictions.csv", description="Pending Predictions"
    )
    LEAGUE_STATS = Resource(
        id="league_statistics", type="dataframe", serializer="csv",
        default_filename="league_statistics.csv", description="League Statistics"
    )
    LEAGUE_PROFILE = Resource(
        id="league_profile", type="dataframe", serializer="csv",
        default_filename="profil_league.csv", description="League Profile Database"
    )
    THRESHOLD = Resource(
        id="threshold", type="dict", serializer="json",
        default_filename="ev_threshold.json", description="EV Threshold Configuration"
    )
    MODEL = Resource(
        id="model", type="pickle", serializer="pickle",
        default_filename="model.pkl", description="Trained Model Bundle"
    )
    FEATURE_COLUMNS = Resource(
        id="feature_columns", type="dict", serializer="json",
        default_filename="feature_columns.json", description="Feature Columns List"
    )
    LEAGUE_THRESHOLD = Resource(
        id="league_threshold", type="dataframe", serializer="csv",
        default_filename="league_threshold.csv", description="Per-League EV Thresholds"
    )

OPTIONAL_RESOURCES = {
    ResourceRegistry.HISTORY.id,
    ResourceRegistry.DATASET.id,
    ResourceRegistry.PENDING.id,
    ResourceRegistry.LEAGUE_STATS.id,
    ResourceRegistry.THRESHOLD.id,
    ResourceRegistry.FEATURE_COLUMNS.id,
    ResourceRegistry.LEAGUE_THRESHOLD.id,
}

# ============================================================
# IMMUTABLE RESULT OBJECTS
# ============================================================
class AppendStatus(Enum):
    SUCCESS = "success"
    DUPLICATE = "duplicate"
    ERROR = "error"

class VerificationStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

class MatchStatus(Enum):
    PENDING = "PENDING"
    SCORE_ENTERED = "SCORE_ENTERED"
    VALIDATED = "VALIDATED"
    HISTORY_SYNCED = "HISTORY_SYNCED"
    SETTLED = "SETTLED"
    LEARNED = "LEARNED"
    ARCHIVED = "ARCHIVED"

class TransactionAction(Enum):
    ANALYZE = "ANALYZE"
    SAVE_PENDING = "SAVE_PENDING"
    SETTLEMENT = "SETTLEMENT"
    HISTORY_SAVE = "HISTORY_SAVE"
    LEARNING = "LEARNING"
    DATASET_UPDATE = "DATASET_UPDATE"
    DATASET_VALIDATION = "DATASET_VALIDATION"
    SCORE_SAVE = "SCORE_SAVE"
    VALIDATION = "VALIDATION"
    AUTO_CALC = "AUTO_CALC"
    HISTORY_SYNC = "HISTORY_SYNC"

@dataclass
class TransactionLogEntry:
    timestamp: datetime
    transaction_id: str
    match_uid: str
    action: TransactionAction
    status: str
    message: str

@dataclass
class PendingResult:
    success: bool
    message: str
    existing_rows: int = 0
    new_rows: int = 0
    combined_rows: int = 0
    queue_count: int = 0
    verification: str = ""
    prediction_id: str = ""
    match_uid: str = ""
    home_team: str = ""
    away_team: str = ""
    github_commit_success: bool = False

@dataclass(frozen=True)
class LearningResult:
    learning_df: pd.DataFrame
    history_df: pd.DataFrame
    dataset_df: pd.DataFrame
    league_stats_df: pd.DataFrame
    threshold_data: dict
    model_bundle: Any = None
    success: bool = True
    message: str = ""

@dataclass(frozen=True)
class SyncResult:
    success: bool
    duration: float
    provider: str
    files_updated: tuple = field(default_factory=tuple)
    error_message: str = ""
    commit_id: Optional[str] = None

@dataclass
class AppendResult:
    success: bool
    status: AppendStatus
    message: str
    existing_rows: int = 0
    new_rows: int = 0
    combined_rows: int = 0
    match_uid: str = ""

@dataclass
class VerificationResult:
    success: bool
    status: VerificationStatus
    expected_rows: int = 0
    actual_rows: int = 0
    expected_uid: str = ""
    uid_found: bool = False
    message: str = ""

# ============================================================
# STORAGE PROVIDER INTERFACE
# ============================================================
class StorageProvider(ABC):
    @abstractmethod
    def load_dataframe(self, resource: Resource) -> pd.DataFrame:
        pass
    @abstractmethod
    def save_dataframe(self, resource: Resource, df: pd.DataFrame):
        pass
    @abstractmethod
    def load_json(self, resource: Resource) -> dict:
        pass
    @abstractmethod
    def save_json(self, resource: Resource, data: dict):
        pass
    @abstractmethod
    def load_pickle(self, resource: Resource) -> Any:
        pass
    @abstractmethod
    def save_pickle(self, resource: Resource, obj: Any):
        pass
    @abstractmethod
    def exists(self, resource: Resource) -> bool:
        pass
    @abstractmethod
    def delete(self, resource: Resource):
        pass
    def begin_transaction(self): pass
    def commit_transaction(self): pass
    def rollback_transaction(self): pass

# ============================================================
# LOCAL STORAGE PROVIDER
# ============================================================
class LocalStorageProvider(StorageProvider):
    def __init__(self, base_dir: Path = BASE_DIR):
        self.base_dir = base_dir
        self._temp_files = []

    def _get_path(self, resource: Resource) -> Path:
        return self.base_dir / resource.default_filename

    def load_dataframe(self, resource: Resource) -> pd.DataFrame:
        path = self._get_path(resource)
        if not path.exists():
            if resource.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {resource.id} not found at {path}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            if resource.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise

    def save_dataframe(self, resource: Resource, df: pd.DataFrame):
        path = self._get_path(resource)
        temp_path = path.with_suffix('.tmp')
        df.to_csv(temp_path, index=False)
        self._temp_files.append((temp_path, path))

    def load_json(self, resource: Resource) -> dict:
        path = self._get_path(resource)
        if not path.exists():
            if resource.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {resource.id} not found at {path}")
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                if resource.id in OPTIONAL_RESOURCES:
                    return {}
                raise

    def save_json(self, resource: Resource, data: dict):
        path = self._get_path(resource)
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        self._temp_files.append((temp_path, path))

    def load_pickle(self, resource: Resource) -> Any:
        path = self._get_path(resource)
        if not path.exists():
            raise FileNotFoundError(f"Resource {resource.id} not found at {path}")
        return joblib.load(path)

    def save_pickle(self, resource: Resource, obj: Any):
        path = self._get_path(resource)
        temp_path = path.with_suffix('.tmp')
        joblib.dump(obj, temp_path)
        self._temp_files.append((temp_path, path))

    def exists(self, resource: Resource) -> bool:
        return self._get_path(resource).exists()

    def delete(self, resource: Resource):
        path = self._get_path(resource)
        if path.exists():
            path.unlink()

    def begin_transaction(self): self._temp_files.clear()
    def commit_transaction(self):
        for temp_path, target_path in self._temp_files:
            os.replace(temp_path, target_path)
        self._temp_files.clear()
    def rollback_transaction(self):
        for temp_path, _ in self._temp_files:
            if temp_path.exists():
                temp_path.unlink()
        self._temp_files.clear()

# ============================================================
# GITHUB STORAGE PROVIDER
# ============================================================
class GitHubStorageProvider(StorageProvider):
    def __init__(self, repo_owner: str, repo_name: str, branch: str, token: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.token = token
        self.api_base = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"

    def _get_headers(self):
        return {"Authorization": f"token {self.token}", "Accept": "application/vnd.github.v3+json"}

    def _get_file_url(self, resource: Resource) -> str:
        return f"{self.api_base}/{resource.default_filename}?ref={self.branch}"

    def _get_sha(self, resource: Resource) -> Optional[str]:
        url = self._get_file_url(resource)
        r = requests.get(url, headers=self._get_headers())
        if r.status_code == 200:
            return r.json().get("sha")
        return None

    def load_dataframe(self, resource: Resource) -> pd.DataFrame:
        url = self._get_file_url(resource)
        r = requests.get(url, headers=self._get_headers())
        if r.status_code == 404:
            if resource.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {resource.id} not found in GitHub")
        r.raise_for_status()
        content_b64 = r.json().get("content", "")
        decoded = base64.b64decode(content_b64)
        if not decoded.strip():
            if resource.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
        return pd.read_csv(BytesIO(decoded))

    def save_dataframe(self, resource: Resource, df: pd.DataFrame):
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        sha = self._get_sha(resource)
        payload = {
            "message": f"Update {resource.id} via Football AI V2",
            "content": base64.b64encode(csv_bytes).decode('utf-8'),
            "branch": self.branch
        }
        if sha:
            payload["sha"] = sha
        url = self._get_file_url(resource).split('?')[0]
        r = requests.put(url, headers=self._get_headers(), json=payload)
        if r.status_code == 409:
            raise RuntimeError(f"Conflict: {resource.id} has been modified in GitHub. Refresh and retry.")
        r.raise_for_status()

    def load_json(self, resource: Resource) -> dict:
        url = self._get_file_url(resource)
        r = requests.get(url, headers=self._get_headers())
        if r.status_code == 404:
            if resource.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {resource.id} not found in GitHub")
        r.raise_for_status()
        content_b64 = r.json().get("content", "")
        decoded = base64.b64decode(content_b64)
        if not decoded.strip():
            if resource.id in OPTIONAL_RESOURCES:
                return {}
        return json.loads(decoded)

    def save_json(self, resource: Resource, data: dict):
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        sha = self._get_sha(resource)
        payload = {
            "message": f"Update {resource.id} via Football AI V2",
            "content": base64.b64encode(json_bytes).decode('utf-8'),
            "branch": self.branch
        }
        if sha:
            payload["sha"] = sha
        url = self._get_file_url(resource).split('?')[0]
        r = requests.put(url, headers=self._get_headers(), json=payload)
        r.raise_for_status()

    def load_pickle(self, resource: Resource) -> Any:
        url = self._get_file_url(resource)
        r = requests.get(url, headers=self._get_headers())
        if r.status_code == 404:
            raise FileNotFoundError(f"Resource {resource.id} not found in GitHub")
        r.raise_for_status()
        content_b64 = r.json().get("content", "")
        decoded = base64.b64decode(content_b64)
        return joblib.load(BytesIO(decoded))

    def save_pickle(self, resource: Resource, obj: Any):
        buffer = BytesIO()
        joblib.dump(obj, buffer)
        pickle_bytes = buffer.getvalue()
        sha = self._get_sha(resource)
        payload = {
            "message": f"Update {resource.id} via Football AI V2",
            "content": base64.b64encode(pickle_bytes).decode('utf-8'),
            "branch": self.branch
        }
        if sha:
            payload["sha"] = sha
        url = self._get_file_url(resource).split('?')[0]
        r = requests.put(url, headers=self._get_headers(), json=payload)
        r.raise_for_status()

    def exists(self, resource: Resource) -> bool:
        url = self._get_file_url(resource)
        r = requests.get(url, headers=self._get_headers())
        return r.status_code == 200

    def delete(self, resource: Resource):
        sha = self._get_sha(resource)
        if not sha:
            raise FileNotFoundError(f"Cannot delete {resource.id}: not found")
        url = self._get_file_url(resource)
        payload = {
            "message": f"Delete {resource.id} via Football AI V2",
            "sha": sha,
            "branch": self.branch
        }
        r = requests.delete(url, headers=self._get_headers(), json=payload)
        r.raise_for_status()

    def begin_transaction(self): pass
    def commit_transaction(self): pass
    def rollback_transaction(self): pass

# ============================================================
# DATABASE MANAGER
# ============================================================
class DatabaseManager:
    _default_storage: Optional[StorageProvider] = None

    def __init__(self, storage: StorageProvider = None):
        self.storage = storage or DatabaseManager._default_storage or LocalStorageProvider()

    def load_history(self) -> pd.DataFrame:
        return self.storage.load_dataframe(ResourceRegistry.HISTORY)

    def save_history(self, df: pd.DataFrame):
        self.storage.save_dataframe(ResourceRegistry.HISTORY, df)

    def load_dataset(self) -> pd.DataFrame:
        return self.storage.load_dataframe(ResourceRegistry.DATASET)

    def save_dataset(self, df: pd.DataFrame):
        self.storage.save_dataframe(ResourceRegistry.DATASET, df)

    def load_pending(self) -> pd.DataFrame:
        df = self.storage.load_dataframe(ResourceRegistry.PENDING)
        # Normalisasi schema agar sesuai Pending Contract (SSOT)
        df = PendingContract.normalize(df)
        return df

    def save_pending(self, df: pd.DataFrame):
        self.storage.save_dataframe(ResourceRegistry.PENDING, df)

    def load_model(self):
        local = LocalStorageProvider()
        return local.load_pickle(ResourceRegistry.MODEL)

    def save_model(self, bundle):
        local = LocalStorageProvider()
        local.save_pickle(ResourceRegistry.MODEL, bundle)

    def load_threshold(self) -> dict:
        return self.storage.load_json(ResourceRegistry.THRESHOLD)

    def save_threshold(self, data: dict):
        self.storage.save_json(ResourceRegistry.THRESHOLD, data)

    def load_league_statistics(self) -> pd.DataFrame:
        return self.storage.load_dataframe(ResourceRegistry.LEAGUE_STATS)

    def save_league_statistics(self, df: pd.DataFrame):
        self.storage.save_dataframe(ResourceRegistry.LEAGUE_STATS, df)

    def load_league_profile(self) -> pd.DataFrame:
        return self.storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)

    def save_league_profile(self, df: pd.DataFrame):
        self.storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, df)

    def load_league_thresholds(self) -> pd.DataFrame:
        if self.storage.exists(ResourceRegistry.LEAGUE_THRESHOLD):
            return self.storage.load_dataframe(ResourceRegistry.LEAGUE_THRESHOLD)
        return pd.DataFrame()

    def save_league_thresholds(self, df: pd.DataFrame):
        self.storage.save_dataframe(ResourceRegistry.LEAGUE_THRESHOLD, df)

    def load_feature_columns(self) -> list:
        if self.storage.exists(ResourceRegistry.FEATURE_COLUMNS):
            data = self.storage.load_json(ResourceRegistry.FEATURE_COLUMNS)
            return data.get("columns", [])
        return []

    def save_feature_columns(self, columns: list):
        self.storage.save_json(ResourceRegistry.FEATURE_COLUMNS, {"columns": columns})

    def is_model_ready(self):
        local = LocalStorageProvider()
        return local.exists(ResourceRegistry.MODEL)
    def is_dataset_ready(self): return self.storage.exists(ResourceRegistry.DATASET)
    def is_history_ready(self): return self.storage.exists(ResourceRegistry.HISTORY)
    def is_threshold_ready(self): return self.storage.exists(ResourceRegistry.THRESHOLD)
    def is_league_stats_ready(self): return self.storage.exists(ResourceRegistry.LEAGUE_STATS)

    def begin_transaction(self): self.storage.begin_transaction()
    def commit_transaction(self): self.storage.commit_transaction()
    def rollback_transaction(self): self.storage.rollback_transaction()

# ============================================================
# PENDING CONTRACT DEFINITION (SSOT)
# ============================================================
class PendingContract:
    """
    Single Source of Truth untuk schema pending_predictions.csv.
    Mendefinisikan seluruh kolom wajib beserta nilai default.
    """
    CORE_COLUMNS = [
        "match_uid", "home_team", "away_team", "league_name", "kickoff_time",
        "prediction", "grade", "confidence",
        "prediction_id", "prediction_time", "prediction_status",
        "prediction_version", "prediction_source", "prediction_notes",
        "home_ht_goals", "away_ht_goals", "home_goals", "away_goals",
        "actual_over", "actual_btts", "settlement_time"
    ]

    DEFAULT_VALUES = {
        "home_ht_goals": None,
        "away_ht_goals": None,
        "home_goals": None,
        "away_goals": None,
        "actual_over": None,
        "actual_btts": None,
        "settlement_time": None
    }

    @classmethod
    def normalize(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in cls.CORE_COLUMNS:
            if col not in df.columns:
                df[col] = cls.DEFAULT_VALUES.get(col, None)
        # Pastikan tipe object untuk kolom yang akan diisi string/datetime
        df["settlement_time"] = df["settlement_time"].astype(object)
        return df

# ============================================================
# SYNC ENGINE
# ============================================================
class SyncEngine:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def sync(self, result: LearningResult) -> SyncResult:
        start = time.time()
        provider = type(self.db.storage).__name__
        files_updated = []
        try:
            self.db.begin_transaction()
            self.db.save_history(result.history_df)
            files_updated.append(ResourceRegistry.HISTORY.id)
            self.db.save_dataset(result.dataset_df)
            files_updated.append(ResourceRegistry.DATASET.id)
            self.db.save_league_statistics(result.league_stats_df)
            files_updated.append(ResourceRegistry.LEAGUE_STATS.id)
            self.db.save_threshold(result.threshold_data)
            files_updated.append(ResourceRegistry.THRESHOLD.id)
            if result.model_bundle:
                self.db.save_model(result.model_bundle)
                files_updated.append(ResourceRegistry.MODEL.id)
            self.db.commit_transaction()
            duration = time.time() - start
            return SyncResult(
                success=True, duration=round(duration, 3), provider=provider,
                files_updated=tuple(files_updated)
            )
        except Exception as e:
            self.db.rollback_transaction()
            duration = time.time() - start
            return SyncResult(
                success=False, duration=round(duration, 3), provider=provider,
                error_message=str(e)
            )

# ============================================================
# SANITASI CSV
# ============================================================
def sanitize_csv_text(text):
    text = text.replace('\ufeff', '').replace('\u200b', '').replace('\u200e', '').replace('\u200f', '')
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
SESSION_STATE_DEFAULTS = {
    "upload": {
        "uploaded": False, "filename": "", "rows": 0, "columns": 0,
        "upload_time": "", "uploaded_data": None,
    },
    "validation": {
        "validated": False, "valid": False, "errors": [], "warnings": [],
        "summary": {}, "validated_at": "", "validation_count": 0,
    },
    "league_profile": {
        "loaded": False, "status": "NOT_LOADED", "matched_rows": 0,
        "total_rows": 0, "loaded_at": "", "profile": None,
    },
    "feature_engineering": {
        "processed": False, "status": "NOT_PROCESSED", "rows": 0,
        "feature_count": 0, "features_generated": [], "warnings": [],
        "processed_at": "", "feature_dataframe": None,
    },
    "prediction": {
        "processed": False,
        "status": "NOT_PROCESSED",
        "prediction_count": 0,
        "prediction_targets": [],
        "warnings": [],
        "fail_reason": "",
        "processed_at": "",
        "prediction_dataframe": None,
        "calibrator": None,
        "btts_available": False,
        "ht_available": False,
        "calibration_available": False,
    },
    "pending": {
        "processed": False,
        "status": "NOT_PROCESSED",
        "pending_count": 0,
        "processed_at": "",
        "pending_dataframe": None,
        "fail_reason": "",
    },
    "settlement": {
        "processed": False,
        "status": "NOT_PROCESSED",
        "settlement_count": 0,
        "processed_at": "",
        "settlement_dataframe": None,
        "fail_reason": "",
    },
    "learning": {
        "processed": False,
        "learning_status": "NOT_PROCESSED",
        "learning_count": 0,
        "processed_at": "",
        "learning_dataframe": None,
        "fail_reason": "",
        "historical_performance": {
            "S": {"roi": 94.0, "win_rate": 100.0},
            "A": {"roi": 76.42, "win_rate": 91.67},
            "B": {"roi": 44.62, "win_rate": 75.0},
            "C": {"roi": 56.67, "win_rate": 80.56},
            "D": {"roi": 34.32, "win_rate": 62.61},
        }
    },
    "analysis": {
        "prediction_quality": "NOT READY",
        "decision_quality": "NOT READY",
        "learning_quality": "NOT READY",
        "calibration_score": None,
        "ev_summary": {},
        "roi_win_rate": {},
        "last_update": "",
        "sync_result": None
    },
    "app_initialized": False,
    "debug_trace": [],
    "transaction_log": [],
}

def initialize_session_state():
    for key, default in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(default, dict):
                st.session_state[key] = copy.deepcopy(default)
            else:
                st.session_state[key] = default
    if not st.session_state.app_initialized:
        st.session_state.app_initialized = True

# ============================================================
# UPLOAD MANAGER
# ============================================================
def is_data_uploaded() -> bool:
    return st.session_state.upload.get("uploaded", False)

def get_uploaded_filename() -> str:
    return st.session_state.upload.get("filename", "")

def get_uploaded_dataframe() -> pd.DataFrame:
    return st.session_state.upload.get("uploaded_data", None)

def has_uploaded_dataframe() -> bool:
    return get_uploaded_dataframe() is not None

def get_home_team() -> str:
    home, _ = parse_match_filename(get_uploaded_filename())
    return home

def get_away_team() -> str:
    _, away = parse_match_filename(get_uploaded_filename())
    return away

def get_upload_info() -> dict:
    return {
        "uploaded": st.session_state.upload.get("uploaded", False),
        "filename": st.session_state.upload.get("filename", ""),
        "rows": st.session_state.upload.get("rows", 0),
        "columns": st.session_state.upload.get("columns", 0),
        "upload_time": st.session_state.upload.get("upload_time", ""),
        "home_team": get_home_team(),
        "away_team": get_away_team(),
    }

def get_upload_status() -> str:
    if not st.session_state.upload.get("uploaded", False):
        return "NO_FILE"
    if st.session_state.upload.get("uploaded_data") is not None:
        return "SUCCESS"
    return "ERROR"

def parse_match_filename(filename: str) -> tuple:
    if not filename:
        return "", ""
    try:
        name_part = filename.rsplit(".", 1)[0]
        teams = name_part.split(" vs ", 1)
        if len(teams) == 2:
            return teams[0].strip(), teams[1].strip()
        return "", ""
    except Exception:
        return "", ""

def create_upload_metadata(dataframe: pd.DataFrame) -> dict:
    return {
        "uploaded": True,
        "filename": get_uploaded_filename(),
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def reset_upload_state():
    st.session_state.upload["uploaded"] = False
    st.session_state.upload["filename"] = ""
    st.session_state.upload["rows"] = 0
    st.session_state.upload["columns"] = 0
    st.session_state.upload["upload_time"] = ""
    st.session_state.upload["uploaded_data"] = None

def store_upload_result(filename: str, df: pd.DataFrame):
    st.session_state.upload["filename"] = filename
    metadata = create_upload_metadata(df)
    for key, value in metadata.items():
        st.session_state.upload[key] = value
    st.session_state.upload["uploaded_data"] = df

def process_upload(uploaded_file) -> bool:
    if uploaded_file is None:
        if st.session_state.upload["uploaded"]:
            reset_upload_state()
        return False
    filename = uploaded_file.name
    if st.session_state.upload["uploaded"] and st.session_state.upload["filename"] == filename:
        return True
    try:
        df = pd.read_csv(uploaded_file)
        store_upload_result(filename, df)
        return True
    except Exception:
        reset_upload_state()
        return False

# ============================================================
# reset_workflow_state
# ============================================================
def reset_workflow_state(reset_upload: bool = False):
    debug_backup = st.session_state.get('debug_trace', [])
    defaults = SESSION_STATE_DEFAULTS
    if not reset_upload:
        upload_backup = copy.deepcopy(st.session_state.upload)
        for key in defaults:
            if key != "upload":
                st.session_state[key] = copy.deepcopy(defaults[key])
        st.session_state.upload = upload_backup
    else:
        for key in defaults:
            st.session_state[key] = copy.deepcopy(defaults[key])
    st.session_state.debug_trace = debug_backup

# ============================================================
# VALIDATION ENGINE
# ============================================================
REQUIRED_COLUMNS = ["league_code", "kickoff_time", "home_xg", "away_xg"]

def is_validation_ready() -> bool:
    return has_uploaded_dataframe()

def is_dataset_valid() -> bool:
    return st.session_state.validation.get("valid", False)

def get_validation_status() -> str:
    if not st.session_state.validation.get("validated", False):
        return "NOT_VALIDATED"
    return "VALID" if st.session_state.validation.get("valid", False) else "INVALID"

def get_validation_errors() -> list:
    return st.session_state.validation.get("errors", [])

def get_validation_warnings() -> list:
    return st.session_state.validation.get("warnings", [])

def get_validation_summary() -> dict:
    return st.session_state.validation.get("summary", {})

def get_validation_time() -> str:
    return st.session_state.validation.get("validated_at", "")

def get_validation_count() -> int:
    return st.session_state.validation.get("validation_count", 0)

def get_validation_info() -> dict:
    return {
        "validated": st.session_state.validation.get("validated", False),
        "valid": st.session_state.validation.get("valid", False),
        "status": get_validation_status(),
        "error_count": len(get_validation_errors()),
        "warning_count": len(get_validation_warnings()),
        "validated_at": get_validation_time(),
        "rows_validated": get_validation_count(),
    }

def create_validation_metadata(errors: list, warnings: list, summary: dict, valid: bool, dataframe: pd.DataFrame) -> dict:
    return {
        "validated": True,
        "valid": valid,
        "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_count": len(dataframe),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }

def reset_validation_state():
    st.session_state.validation["validated"] = False
    st.session_state.validation["valid"] = False
    st.session_state.validation["errors"] = []
    st.session_state.validation["warnings"] = []
    st.session_state.validation["summary"] = {}
    st.session_state.validation["validated_at"] = ""
    st.session_state.validation["validation_count"] = 0

def store_validation_result(metadata: dict):
    for key, value in metadata.items():
        st.session_state.validation[key] = value

def validate_dataset() -> bool:
    if not has_uploaded_dataframe():
        reset_validation_state()
        return False
    df = get_uploaded_dataframe()
    errors = []
    warnings = []
    summary = {}
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")
    numeric_candidates = [
        "home_xg", "away_xg", "home_xga", "away_xga",
        "last5_home_xg", "last5_away_xg",
        "last5_home_over25", "last5_away_over25",
        "last5_home_btts", "last5_away_btts",
        "open_over_odds", "current_over_odds", "current_ou",
    ]
    for col in numeric_candidates:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Kolom '{col}' harus bertipe numerik.")
    for col in df.columns:
        if df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            warnings.append(f"Kolom '{col}' memiliki {missing_count} missing values.")
    summary["total_rows"] = len(df)
    summary["total_columns"] = len(df.columns)
    summary["missing_columns"] = missing_cols
    valid = len(errors) == 0
    metadata = create_validation_metadata(errors, warnings, summary, valid, df)
    store_validation_result(metadata)
    return valid

# ============================================================
# LEAGUE PROFILE ENGINE
# ============================================================
def is_league_profile_ready() -> bool:
    return has_uploaded_dataframe()

def is_league_profile_loaded() -> bool:
    return st.session_state.league_profile.get("loaded", False)

def get_league_profile_dataframe() -> pd.DataFrame:
    return st.session_state.league_profile.get("profile", None)

def get_league_profile_info() -> dict:
    return {
        "loaded": st.session_state.league_profile.get("loaded", False),
        "status": st.session_state.league_profile.get("status", "NOT_LOADED"),
        "matched_rows": st.session_state.league_profile.get("matched_rows", 0),
        "total_rows": st.session_state.league_profile.get("total_rows", 0),
        "loaded_at": st.session_state.league_profile.get("loaded_at", ""),
    }

def get_league_profile_status() -> str:
    return st.session_state.league_profile.get("status", "NOT_LOADED")

def create_league_profile_metadata(merged_df: pd.DataFrame, total_rows_db: int, matched_rows: int) -> dict:
    return {
        "loaded": True,
        "status": "LOADED",
        "matched_rows": matched_rows,
        "total_rows": total_rows_db,
        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile": merged_df,
    }

def reset_league_profile_state():
    st.session_state.league_profile["loaded"] = False
    st.session_state.league_profile["status"] = "NOT_LOADED"
    st.session_state.league_profile["matched_rows"] = 0
    st.session_state.league_profile["total_rows"] = 0
    st.session_state.league_profile["loaded_at"] = ""
    st.session_state.league_profile["profile"] = None

def set_league_profile_failed():
    st.session_state.league_profile["loaded"] = False
    st.session_state.league_profile["status"] = "FAILED"
    st.session_state.league_profile["matched_rows"] = 0
    st.session_state.league_profile["total_rows"] = 0
    st.session_state.league_profile["loaded_at"] = ""
    st.session_state.league_profile["profile"] = None

def store_league_profile_result(metadata: dict):
    for key, value in metadata.items():
        st.session_state.league_profile[key] = value

def build_league_profile() -> bool:
    if not has_uploaded_dataframe():
        reset_league_profile_state()
        return False
    db = DatabaseManager()
    try:
        league_db = db.load_league_profile()
        if "league_code" not in league_db.columns:
            set_league_profile_failed()
            return False
        match_df = get_uploaded_dataframe()
        if "league_code" not in match_df.columns:
            set_league_profile_failed()
            return False
        merged_df = match_df.merge(league_db, on="league_code", how="left", indicator="merge_profil")
        matched_rows = (merged_df["merge_profil"] == "both").sum()
        merged_df.drop(columns=["merge_profil"], inplace=True)

        if db.is_league_stats_ready():
            stats_df = db.load_league_statistics()
            if "league_code" in stats_df.columns:
                merged_df = merged_df.merge(stats_df, on="league_code", how="left", suffixes=("", "_stat"))
                for col in stats_df.columns:
                    if col != "league_code" and col+"_stat" in merged_df.columns:
                        merged_df[col] = merged_df[col+"_stat"].fillna(merged_df[col])
                        merged_df.drop(columns=[col+"_stat"], inplace=True)

        metadata = create_league_profile_metadata(merged_df, len(league_db), matched_rows)
        store_league_profile_result(metadata)
        return True
    except Exception as e:
        import traceback
        err_msg = f"EXCEPTION in build_league_profile: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        st.session_state.setdefault('debug_trace', []).append(err_msg)
        set_league_profile_failed()
        return False

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def is_feature_engineering_ready() -> bool:
    return is_league_profile_loaded()

def is_feature_engineering_processed() -> bool:
    return st.session_state.feature_engineering.get("processed", False)

def get_feature_engineering_status() -> str:
    if not is_feature_engineering_ready():
        return "NOT_PROCESSED"
    return st.session_state.feature_engineering.get("status", "NOT_PROCESSED")

def get_feature_dataframe() -> pd.DataFrame:
    return st.session_state.feature_engineering.get("feature_dataframe", None)

def get_feature_count() -> int:
    return st.session_state.feature_engineering.get("feature_count", 0)

def get_feature_engineering_info() -> dict:
    return {
        "processed": st.session_state.feature_engineering.get("processed", False),
        "status": st.session_state.feature_engineering.get("status", "NOT_PROCESSED"),
        "rows": st.session_state.feature_engineering.get("rows", 0),
        "feature_count": st.session_state.feature_engineering.get("feature_count", 0),
        "features_generated": st.session_state.feature_engineering.get("features_generated", []),
        "warnings": st.session_state.feature_engineering.get("warnings", []),
        "processed_at": st.session_state.feature_engineering.get("processed_at", ""),
    }

def safe_add_feature(df, name, func, required_cols, generated, warnings):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        warnings.append(f"Feature '{name}' dilewati: kolom {missing} tidak tersedia.")
        return
    try:
        df[name] = func(df)
        generated.append(name)
    except Exception as e:
        warnings.append(f"Feature '{name}' gagal dibuat: {str(e)}")

def create_feature_engineering_metadata(feature_df, features_generated, warnings, feature_count):
    return {
        "processed": True,
        "status": "PROCESSED",
        "rows": len(feature_df),
        "feature_count": feature_count,
        "features_generated": features_generated,
        "warnings": warnings,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_dataframe": feature_df,
    }

def reset_feature_engineering_state():
    st.session_state.feature_engineering["processed"] = False
    st.session_state.feature_engineering["status"] = "NOT_PROCESSED"
    st.session_state.feature_engineering["rows"] = 0
    st.session_state.feature_engineering["feature_count"] = 0
    st.session_state.feature_engineering["features_generated"] = []
    st.session_state.feature_engineering["warnings"] = []
    st.session_state.feature_engineering["processed_at"] = ""
    st.session_state.feature_engineering["feature_dataframe"] = None

def set_feature_engineering_failed(warnings=None):
    st.session_state.feature_engineering["processed"] = False
    st.session_state.feature_engineering["status"] = "FAILED"
    st.session_state.feature_engineering["rows"] = 0
    st.session_state.feature_engineering["feature_count"] = 0
    st.session_state.feature_engineering["features_generated"] = []
    st.session_state.feature_engineering["warnings"] = warnings if warnings else []
    st.session_state.feature_engineering["processed_at"] = ""
    st.session_state.feature_engineering["feature_dataframe"] = None

def store_feature_engineering_result(metadata):
    for key, value in metadata.items():
        st.session_state.feature_engineering[key] = value

def build_features() -> bool:
    if not is_feature_engineering_ready():
        reset_feature_engineering_state()
        return False
    df = get_league_profile_dataframe().copy()
    warnings = []
    generated = []
    cols = df.columns

    safe_add_feature(df, "xg_ratio_home", lambda d: (d["home_xg"] / d["home_xga"].replace(0, np.nan)).fillna(0), ["home_xg", "home_xga"], generated, warnings)
    safe_add_feature(df, "xg_ratio_away", lambda d: (d["away_xg"] / d["away_xga"].replace(0, np.nan)).fillna(0), ["away_xg", "away_xga"], generated, warnings)

    home_avg_col = None
    for col in ["last5_home_avg_goals", "home_avg_goals"]:
        if col in cols:
            home_avg_col = col
            break
    away_avg_col = None
    for col in ["last5_away_avg_goals", "away_avg_goals"]:
        if col in cols:
            away_avg_col = col
            break
    home_conceded_col = None
    for col in ["last5_home_conceded", "home_conceded"]:
        if col in cols:
            home_conceded_col = col
            break
    away_conceded_col = None
    for col in ["last5_away_conceded", "away_conceded"]:
        if col in cols:
            away_conceded_col = col
            break

    if home_avg_col and home_conceded_col:
        safe_add_feature(df, "goal_diff_home", lambda d: d[home_avg_col] - d[home_conceded_col], [home_avg_col, home_conceded_col], generated, warnings)
    else:
        df["goal_diff_home"] = 0.0
        generated.append("goal_diff_home")
    if away_avg_col and away_conceded_col:
        safe_add_feature(df, "goal_diff_away", lambda d: d[away_avg_col] - d[away_conceded_col], [away_avg_col, away_conceded_col], generated, warnings)
    else:
        df["goal_diff_away"] = 0.0
        generated.append("goal_diff_away")

    safe_add_feature(df, "xg_diff_home", lambda d: d["home_xg"] - d["home_xga"], ["home_xg", "home_xga"], generated, warnings)
    safe_add_feature(df, "xg_diff_away", lambda d: d["away_xg"] - d["away_xga"], ["away_xg", "away_xga"], generated, warnings)

    home_btts_col = None
    for col in ["last5_home_btts", "home_btts"]:
        if col in cols:
            home_btts_col = col
            break
    away_btts_col = None
    for col in ["last5_away_btts", "away_btts"]:
        if col in cols:
            away_btts_col = col
            break
    if home_btts_col and away_btts_col:
        safe_add_feature(df, "btts_potential", lambda d: (d[home_btts_col] + d[away_btts_col]) / 2, [home_btts_col, away_btts_col], generated, warnings)
    else:
        df["btts_potential"] = 0.0
        generated.append("btts_potential")

    home_over25_col = None
    for col in ["last5_home_over25", "home_over25"]:
        if col in cols:
            home_over25_col = col
            break
    away_over25_col = None
    for col in ["last5_away_over25", "away_over25"]:
        if col in cols:
            away_over25_col = col
            break
    if home_over25_col and away_over25_col:
        safe_add_feature(df, "over25_potential", lambda d: (d[home_over25_col] + d[away_over25_col]) / 2, [home_over25_col, away_over25_col], generated, warnings)
    else:
        df["over25_potential"] = 0.0
        generated.append("over25_potential")

    safe_add_feature(df, "odds_ratio", lambda d: (d["current_over_odds"] / d["current_under_odds"].replace(0, np.nan)).fillna(0), ["current_over_odds", "current_under_odds"], generated, warnings)

    if home_avg_col:
        safe_add_feature(df, "momentum_home", lambda d: d[home_avg_col].fillna(0), [home_avg_col], generated, warnings)
    else:
        df["momentum_home"] = 0.0
        generated.append("momentum_home")
    if away_avg_col:
        safe_add_feature(df, "momentum_away", lambda d: d[away_avg_col].fillna(0), [away_avg_col], generated, warnings)
    else:
        df["momentum_away"] = 0.0
        generated.append("momentum_away")

    safe_add_feature(df, "xg_interact", lambda d: d["home_xg"] * d["away_xg"], ["home_xg", "away_xg"], generated, warnings)

    if "over_move" in cols and home_avg_col:
        safe_add_feature(df, "odds_momentum", lambda d: d["over_move"] * d[home_avg_col].fillna(0), ["over_move", home_avg_col], generated, warnings)
    else:
        df["odds_momentum"] = 0.0
        generated.append("odds_momentum")

    if "league_avg_goals" in cols:
        safe_add_feature(df, "xghome_x_leagueavg", lambda d: d["home_xg"] * d["league_avg_goals"], ["home_xg", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "xgaway_x_leagueavg", lambda d: d["away_xg"] * d["league_avg_goals"], ["away_xg", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "xgahome_x_leagueavg", lambda d: d["home_xga"] * d["league_avg_goals"], ["home_xga", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "xgaaway_x_leagueavg", lambda d: d["away_xga"] * d["league_avg_goals"], ["away_xga", "league_avg_goals"], generated, warnings)
        if "last5_home_xg" in cols:
            safe_add_feature(df, "last5_home_xg_x_leagueavg", lambda d: d["last5_home_xg"] * d["league_avg_goals"], ["last5_home_xg", "league_avg_goals"], generated, warnings)
        if "last5_away_xg" in cols:
            safe_add_feature(df, "last5_away_xg_x_leagueavg", lambda d: d["last5_away_xg"] * d["league_avg_goals"], ["last5_away_xg", "league_avg_goals"], generated, warnings)
        if "last5_home_xga" in cols:
            safe_add_feature(df, "last5_home_xga_x_leagueavg", lambda d: d["last5_home_xga"] * d["league_avg_goals"], ["last5_home_xga", "league_avg_goals"], generated, warnings)
        if "last5_away_xga" in cols:
            safe_add_feature(df, "last5_away_xga_x_leagueavg", lambda d: d["last5_away_xga"] * d["league_avg_goals"], ["last5_away_xga", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "ou_line_x_leagueavg", lambda d: d["current_ou"] * d["league_avg_goals"], ["current_ou", "league_avg_goals"], generated, warnings)
    if "league_over25_pct" in cols and "last5_home_over25" in cols:
        safe_add_feature(df, "over25_x_leaguepct", lambda d: d["last5_home_over25"] * d["league_over25_pct"], ["last5_home_over25", "league_over25_pct"], generated, warnings)
    if "league_btts_pct" in cols and "last5_home_btts" in cols:
        safe_add_feature(df, "btts_x_leaguepct", lambda d: d["last5_home_btts"] * d["league_btts_pct"], ["last5_home_btts", "league_btts_pct"], generated, warnings)

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)

    if len(generated) == 0:
        set_feature_engineering_failed(warnings)
        return False

    feature_count = len(generated)
    metadata = create_feature_engineering_metadata(df, generated, warnings, feature_count)
    store_feature_engineering_result(metadata)
    return True

# ============================================================
# STANDALONE ADD FEATURES
# ============================================================
def add_features(df):
    warnings = []
    generated = []
    cols = df.columns

    safe_add_feature(df, "xg_ratio_home", lambda d: (d["home_xg"] / d["home_xga"].replace(0, np.nan)).fillna(0), ["home_xg", "home_xga"], generated, warnings)
    safe_add_feature(df, "xg_ratio_away", lambda d: (d["away_xg"] / d["away_xga"].replace(0, np.nan)).fillna(0), ["away_xg", "away_xga"], generated, warnings)

    home_avg_col = None
    for col in ["last5_home_avg_goals", "home_avg_goals"]:
        if col in cols:
            home_avg_col = col
            break
    away_avg_col = None
    for col in ["last5_away_avg_goals", "away_avg_goals"]:
        if col in cols:
            away_avg_col = col
            break
    home_conceded_col = None
    for col in ["last5_home_conceded", "home_conceded"]:
        if col in cols:
            home_conceded_col = col
            break
    away_conceded_col = None
    for col in ["last5_away_conceded", "away_conceded"]:
        if col in cols:
            away_conceded_col = col
            break

    if home_avg_col and home_conceded_col:
        safe_add_feature(df, "goal_diff_home", lambda d: d[home_avg_col] - d[home_conceded_col], [home_avg_col, home_conceded_col], generated, warnings)
    else:
        df["goal_diff_home"] = 0.0
        generated.append("goal_diff_home")
    if away_avg_col and away_conceded_col:
        safe_add_feature(df, "goal_diff_away", lambda d: d[away_avg_col] - d[away_conceded_col], [away_avg_col, away_conceded_col], generated, warnings)
    else:
        df["goal_diff_away"] = 0.0
        generated.append("goal_diff_away")

    safe_add_feature(df, "xg_diff_home", lambda d: d["home_xg"] - d["home_xga"], ["home_xg", "home_xga"], generated, warnings)
    safe_add_feature(df, "xg_diff_away", lambda d: d["away_xg"] - d["away_xga"], ["away_xg", "away_xga"], generated, warnings)

    home_btts_col = None
    for col in ["last5_home_btts", "home_btts"]:
        if col in cols:
            home_btts_col = col
            break
    away_btts_col = None
    for col in ["last5_away_btts", "away_btts"]:
        if col in cols:
            away_btts_col = col
            break
    if home_btts_col and away_btts_col:
        safe_add_feature(df, "btts_potential", lambda d: (d[home_btts_col] + d[away_btts_col]) / 2, [home_btts_col, away_btts_col], generated, warnings)
    else:
        df["btts_potential"] = 0.0
        generated.append("btts_potential")

    home_over25_col = None
    for col in ["last5_home_over25", "home_over25"]:
        if col in cols:
            home_over25_col = col
            break
    away_over25_col = None
    for col in ["last5_away_over25", "away_over25"]:
        if col in cols:
            away_over25_col = col
            break
    if home_over25_col and away_over25_col:
        safe_add_feature(df, "over25_potential", lambda d: (d[home_over25_col] + d[away_over25_col]) / 2, [home_over25_col, away_over25_col], generated, warnings)
    else:
        df["over25_potential"] = 0.0
        generated.append("over25_potential")

    safe_add_feature(df, "odds_ratio", lambda d: (d["current_over_odds"] / d["current_under_odds"].replace(0, np.nan)).fillna(0), ["current_over_odds", "current_under_odds"], generated, warnings)

    if home_avg_col:
        safe_add_feature(df, "momentum_home", lambda d: d[home_avg_col].fillna(0), [home_avg_col], generated, warnings)
    else:
        df["momentum_home"] = 0.0
        generated.append("momentum_home")
    if away_avg_col:
        safe_add_feature(df, "momentum_away", lambda d: d[away_avg_col].fillna(0), [away_avg_col], generated, warnings)
    else:
        df["momentum_away"] = 0.0
        generated.append("momentum_away")

    safe_add_feature(df, "xg_interact", lambda d: d["home_xg"] * d["away_xg"], ["home_xg", "away_xg"], generated, warnings)

    if "over_move" in cols and home_avg_col:
        safe_add_feature(df, "odds_momentum", lambda d: d["over_move"] * d[home_avg_col].fillna(0), ["over_move", home_avg_col], generated, warnings)
    else:
        df["odds_momentum"] = 0.0
        generated.append("odds_momentum")

    if "league_avg_goals" in cols:
        safe_add_feature(df, "xghome_x_leagueavg", lambda d: d["home_xg"] * d["league_avg_goals"], ["home_xg", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "xgaway_x_leagueavg", lambda d: d["away_xg"] * d["league_avg_goals"], ["away_xg", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "xgahome_x_leagueavg", lambda d: d["home_xga"] * d["league_avg_goals"], ["home_xga", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "xgaaway_x_leagueavg", lambda d: d["away_xga"] * d["league_avg_goals"], ["away_xga", "league_avg_goals"], generated, warnings)
        if "last5_home_xg" in cols:
            safe_add_feature(df, "last5_home_xg_x_leagueavg", lambda d: d["last5_home_xg"] * d["league_avg_goals"], ["last5_home_xg", "league_avg_goals"], generated, warnings)
        if "last5_away_xg" in cols:
            safe_add_feature(df, "last5_away_xg_x_leagueavg", lambda d: d["last5_away_xg"] * d["league_avg_goals"], ["last5_away_xg", "league_avg_goals"], generated, warnings)
        if "last5_home_xga" in cols:
            safe_add_feature(df, "last5_home_xga_x_leagueavg", lambda d: d["last5_home_xga"] * d["league_avg_goals"], ["last5_home_xga", "league_avg_goals"], generated, warnings)
        if "last5_away_xga" in cols:
            safe_add_feature(df, "last5_away_xga_x_leagueavg", lambda d: d["last5_away_xga"] * d["league_avg_goals"], ["last5_away_xga", "league_avg_goals"], generated, warnings)
        safe_add_feature(df, "ou_line_x_leagueavg", lambda d: d["current_ou"] * d["league_avg_goals"], ["current_ou", "league_avg_goals"], generated, warnings)
    if "league_over25_pct" in cols and "last5_home_over25" in cols:
        safe_add_feature(df, "over25_x_leaguepct", lambda d: d["last5_home_over25"] * d["league_over25_pct"], ["last5_home_over25", "league_over25_pct"], generated, warnings)
    if "league_btts_pct" in cols and "last5_home_btts" in cols:
        safe_add_feature(df, "btts_x_leaguepct", lambda d: d["last5_home_btts"] * d["league_btts_pct"], ["last5_home_btts", "league_btts_pct"], generated, warnings)

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df

# ============================================================
# PREDICTION ENGINE
# ============================================================
def is_prediction_ready() -> bool:
    return is_feature_engineering_processed()

def is_prediction_processed() -> bool:
    return st.session_state.prediction.get("processed", False)

def get_prediction_dataframe() -> pd.DataFrame:
    return st.session_state.prediction.get("prediction_dataframe", None)

def get_prediction_status() -> str:
    if not is_prediction_ready():
        return "NOT_PROCESSED"
    return st.session_state.prediction.get("status", "NOT_PROCESSED")

def get_prediction_count() -> int:
    return st.session_state.prediction.get("prediction_count", 0)

def get_prediction_targets() -> list:
    return st.session_state.prediction.get("prediction_targets", [])

def get_prediction_processed_at() -> str:
    return st.session_state.prediction.get("processed_at", "")

def get_prediction_fail_reason() -> str:
    return st.session_state.prediction.get("fail_reason", "")

def get_prediction_warnings() -> list:
    return st.session_state.prediction.get("warnings", [])

def get_prediction_info() -> dict:
    return {
        "processed": is_prediction_processed(),
        "status": get_prediction_status(),
        "prediction_count": get_prediction_count(),
        "prediction_targets": get_prediction_targets(),
        "processed_at": get_prediction_processed_at(),
    }

def create_prediction_metadata(prediction_df, prediction_targets, warnings=None, fail_reason=""):
    return {
        "processed": True,
        "status": "PROCESSED",
        "prediction_count": len(prediction_df),
        "prediction_targets": prediction_targets,
        "warnings": warnings if warnings else [],
        "fail_reason": fail_reason,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_dataframe": prediction_df,
    }

def create_prediction_failed_metadata(reason, warnings=None):
    return {
        "processed": False,
        "status": "FAILED",
        "prediction_count": 0,
        "prediction_targets": [],
        "warnings": warnings if warnings else [],
        "fail_reason": reason,
        "processed_at": "",
        "prediction_dataframe": None,
    }

def create_prediction_default_metadata():
    return {
        "processed": False,
        "status": "NOT_PROCESSED",
        "prediction_count": 0,
        "prediction_targets": [],
        "warnings": [],
        "fail_reason": "",
        "processed_at": "",
        "prediction_dataframe": None,
    }

def reset_prediction_state():
    store_prediction_result(create_prediction_default_metadata())

def set_prediction_failed(reason, warnings=None):
    store_prediction_result(create_prediction_failed_metadata(reason, warnings))

def store_prediction_result(metadata):
    for key, value in metadata.items():
        st.session_state.prediction[key] = value

def _validate_prediction_input(df):
    errors = []
    for col in ["current_ou", "current_over_odds", "current_under_odds"]:
        if col not in df.columns:
            errors.append(f"Kolom '{col}' tidak tersedia dalam data pertandingan.")
    if not errors:
        for col in ["current_ou", "current_over_odds", "current_under_odds"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Kolom '{col}' harus bertipe numerik.")
    return errors

def build_prediction() -> bool:
    if not is_prediction_ready():
        reset_prediction_state()
        return False

    feature_df = get_feature_dataframe()
    if feature_df is None or len(feature_df) == 0:
        set_prediction_failed("Feature dataframe kosong atau tidak tersedia.")
        return False

    errors = _validate_prediction_input(feature_df)
    if errors:
        reason = "INVALID INPUT: " + "; ".join(errors)
        set_prediction_failed(reason)
        return False

    db = DatabaseManager()
    if not db.is_model_ready():
        set_prediction_failed("File model.pkl tidak ditemukan.")
        return False

    try:
        bundle = db.load_model()
    except Exception as e:
        set_prediction_failed(f"Gagal memuat model: {str(e)}")
        return False

    if isinstance(bundle, dict):
        model_ou = bundle.get("model_ou")
        model_btts = bundle.get("model_btts")
        model_ht0 = bundle.get("model_ht0")
        calibrator = bundle.get("calibrator")
        expected_features = bundle.get("feature_cols")
        if model_ou is None or expected_features is None:
            set_prediction_failed("Bundle tidak memiliki 'model_ou' atau 'feature_cols'.")
            return False
        st.session_state.prediction["btts_available"] = model_btts is not None
        st.session_state.prediction["ht_available"] = model_ht0 is not None
        st.session_state.prediction["calibration_available"] = calibrator is not None
        st.session_state.prediction["calibrator"] = calibrator
        for key in bundle.keys():
            if key.startswith("calibrator_"):
                st.session_state.prediction[key] = bundle[key]
    else:
        model_ou = bundle
        model_btts = None
        model_ht0 = None
        calibrator = None
        expected_features = getattr(model_ou, "feature_names_in_", None)
        if expected_features is None:
            feat_cols = db.load_feature_columns()
            if not feat_cols:
                set_prediction_failed("Model tidak memiliki feature_names_in_ dan feature_columns tidak ditemukan.")
                return False
            expected_features = feat_cols
        st.session_state.prediction["btts_available"] = False
        st.session_state.prediction["ht_available"] = False
        st.session_state.prediction["calibration_available"] = False
        st.session_state.prediction["calibrator"] = None

    if isinstance(expected_features, np.ndarray):
        expected_features = expected_features.tolist()

    missing = [f for f in expected_features if f not in feature_df.columns]
    if missing:
        set_prediction_failed(f"Fitur tidak ditemukan: {', '.join(missing)}.")
        return False

    X = feature_df[list(expected_features)].fillna(0)
    warnings = []

    try:
        lambda_pred = model_ou.predict(X)
        if lambda_pred.ndim == 1:
            lambda_pred = lambda_pred.reshape(-1, 1)
        lambda_pred = np.maximum(0.0, lambda_pred.flatten())
    except Exception as e:
        set_prediction_failed(f"Gagal prediksi expected goal: {str(e)}", warnings)
        return False

    ou_line = feature_df["current_ou"].values
    over_odds = feature_df["current_over_odds"].values
    under_odds = feature_df["current_under_odds"].values

    prob_over_raw = np.array([
        1.0 - poisson.cdf(np.floor(ou), lam) if (ou % 1 == 0 or ou % 0.5 == 0)
        else (1 - poisson.cdf(np.floor(ou), lam) + 1 - poisson.cdf(np.ceil(ou), lam)) / 2
        for ou, lam in zip(ou_line, lambda_pred)
    ])

    prob_over_cal = prob_over_raw.copy()
    if st.session_state.prediction["calibration_available"]:
        league_codes = feature_df.get("league_code", pd.Series(np.zeros(len(feature_df))))
        for i, lc in enumerate(league_codes):
            key = f"calibrator_{int(lc)}"
            cal = st.session_state.prediction.get(key) or calibrator
            if cal is not None:
                try:
                    prob_over_cal[i] = np.clip(cal.predict(np.array([prob_over_raw[i]]))[0], 0.0, 1.0)
                except Exception:
                    pass

    ev_over = prob_over_cal * (over_odds - 1) + (1 - prob_over_cal) * (-1)
    ev_under = (1 - prob_over_cal) * (under_odds - 1) + prob_over_cal * (-1)

    def calc_kelly(p, odds):
        if odds <= 1.0:
            return 0.0
        k = (p * (odds - 1) - (1 - p)) / (odds - 1)
        return max(0.0, min(0.25, k))

    kelly_over = np.array([calc_kelly(p, o) for p, o in zip(prob_over_cal, over_odds)])
    kelly_under = np.array([calc_kelly(1-p, u) for p, u in zip(prob_over_cal, under_odds)])

    prob_btts = np.full(len(feature_df), np.nan)
    if st.session_state.prediction["btts_available"]:
        try:
            prob_btts = model_btts.predict_proba(X)[:, 1]
            cal_btts = st.session_state.prediction.get("calibrator_btts")
            if cal_btts is not None:
                prob_btts = np.clip(cal_btts.predict(prob_btts.reshape(-1,1)), 0.0, 1.0)
        except Exception:
            pass

    prob_ht0 = np.full(len(feature_df), np.nan)
    if st.session_state.prediction["ht_available"]:
        try:
            prob_ht0 = model_ht0.predict_proba(X)[:, 1]
            cal_ht0 = st.session_state.prediction.get("calibrator_ht0")
            if cal_ht0 is not None:
                prob_ht0 = np.clip(cal_ht0.predict(prob_ht0.reshape(-1,1)), 0.0, 1.0)
        except Exception:
            pass

    confidence_ou = np.maximum(prob_over_cal, 1-prob_over_cal)
    confidence_btts = np.where(np.isnan(prob_btts), np.nan, np.maximum(prob_btts, 1-prob_btts))
    confidence_ht0 = np.where(np.isnan(prob_ht0), np.nan, np.maximum(prob_ht0, 1-prob_ht0))

    pred_df = pd.DataFrame({
        "expected_goal": lambda_pred,
        "prob_over": prob_over_cal,
        "prob_under": 1 - prob_over_cal,
        "prob_over_raw": prob_over_raw,
        "ev_over": ev_over,
        "ev_under": ev_under,
        "kelly_over": kelly_over,
        "kelly_under": kelly_under,
        "prob_btts": prob_btts,
        "prob_ht0": prob_ht0,
        "confidence_ou": confidence_ou,
        "confidence_btts": confidence_btts,
        "confidence_ht0": confidence_ht0,
        "prediction_ou": (prob_over_cal >= 0.5).astype(int),
        "prediction_btts": np.where(np.isnan(prob_btts), -1, (prob_btts >= 0.5).astype(int)),
        "prediction_ht0": np.where(np.isnan(prob_ht0), -1, (prob_ht0 >= 0.5).astype(int)),
    }, index=feature_df.index)

    prediction_targets = list(pred_df.columns)
    metadata = create_prediction_metadata(pred_df, prediction_targets, warnings)
    store_prediction_result(metadata)
    return True

# ============================================================
# DECISION ENGINE
# ============================================================
def get_decision_grade(confidence, edge, ev_signal):
    if confidence >= 0.80 and edge > 0.10 and ev_signal == "strong":
        return "S"
    elif confidence >= 0.70 and edge > 0.10 and ev_signal in ("strong", "good"):
        return "A"
    elif confidence >= 0.80 and edge >= 0.05 and ev_signal != "avoid":
        return "B"
    elif confidence >= 0.70 and edge >= 0.02 and ev_signal != "avoid":
        return "C"
    else:
        return "D"

def get_stake_recommendation(grade):
    mapping = {"S": "HIGH BET", "A": "STRONG BET", "B": "MEDIUM BET", "C": "SMALL BET", "D": "NO BET"}
    return mapping.get(grade, "NO BET")

def calculate_decision_engine(feat_row, pred_row, ev_thresholds, league_thresholds_df, historical_performance):
    prob_over = pred_row["prob_over"]
    ev_over = pred_row["ev_over"]
    ev_under = pred_row["ev_under"]
    kelly_over = pred_row["kelly_over"]
    kelly_under = pred_row["kelly_under"]
    confidence_ou = pred_row["confidence_ou"]
    expected_goal = pred_row["expected_goal"]

    th_over, th_under = ev_thresholds
    lc = int(feat_row.get("league_code", 0))
    if not league_thresholds_df.empty:
        mask = league_thresholds_df['league_code'] == lc
        if mask.any():
            th_over = league_thresholds_df.loc[mask, 'ev_over'].values[0]
            th_under = league_thresholds_df.loc[mask, 'ev_under'].values[0]

    take_over = ev_over > th_over and kelly_over > 0.005
    take_under = ev_under > th_under and kelly_under > 0.005
    if take_over and take_under:
        ev_signal = "strong" if ev_over >= ev_under else "good"
    elif take_over:
        ev_signal = "strong" if ev_over > 0.05 else "good"
    elif take_under:
        ev_signal = "strong" if ev_under > 0.05 else "good"
    else:
        ev_signal = "avoid"

    over_odds = feat_row.get("current_over_odds")
    if over_odds is None or over_odds <= 1.0:
        edge = 0.0
    else:
        implied = 1.0 / over_odds
        edge = max(0.0, prob_over - implied)

    ou_line = feat_row.get("current_ou")
    over25_pot = feat_row.get("over25_potential", 0.5)
    btts_pot = feat_row.get("btts_potential", 0.5)
    league_over25 = feat_row.get("league_over25_pct", 0.5)
    league_btts = feat_row.get("league_btts_pct", 0.5)
    league_avg_goals = feat_row.get("league_avg_goals", 2.5)
    momentum_home = feat_row.get("momentum_home", 0)
    momentum_away = feat_row.get("momentum_away", 0)
    xg_ratio_home = feat_row.get("xg_ratio_home", 0)
    xg_ratio_away = feat_row.get("xg_ratio_away", 0)

    eg_score = min(100, (expected_goal / 5.0) * 100)
    football_brain = int(eg_score*0.25 + over25_pot*100*0.35 + btts_pot*100*0.20 +
                         ((momentum_home+momentum_away)/2 + 1)*50*0.10 +
                         ((xg_ratio_home+xg_ratio_away)/2)*50*0.10)
    football_brain = max(0, min(100, football_brain))

    league_brain = int(league_over25*100*0.5 + league_btts*100*0.3 + (league_avg_goals/4.0)*100*0.2)
    league_brain = max(0, min(100, league_brain))

    edge_score = min(100, edge * 1000)
    ou_line_factor = max(0, min(100, (ou_line - 2.0) * 50))
    market_brain = int(confidence_ou*100*0.5 + edge_score*0.3 + ou_line_factor*0.2)
    market_brain = max(0, min(100, market_brain))

    base_score = (football_brain + league_brain + market_brain) / 3
    ev_score = min(100, (ev_over + 0.2) * 200) if ev_over > 0 else 0
    kelly_score = min(100, kelly_over * 400) if kelly_over > 0 else 0
    ai_score = int(base_score * 0.5 + ev_score * 0.3 + kelly_score * 0.2)
    ai_score = max(0, min(100, ai_score))

    grade = get_decision_grade(confidence_ou, edge, ev_signal)
    recommendation = get_stake_recommendation(grade)
    rec_color = grade.lower()
    stake_mapping = {"S": "Rp100.000", "A": "Rp80.000", "B": "Rp50.000", "C": "Rp25.000", "D": "Rp0"}
    stake = stake_mapping.get(grade, "Rp0")
    hist = historical_performance.get(grade, {"roi": 0.0, "win_rate": 0.0})

    insights = []
    home_xg = feat_row.get("home_xg", 0)
    away_xg = feat_row.get("away_xg", 0)
    if home_xg > 1.5: insights.append(f"✅ Home xG tinggi ({home_xg:.1f})")
    if away_xg > 1.5: insights.append(f"✅ Away xG tinggi ({away_xg:.1f})")
    insights.append(f"📊 Expected Goal: {expected_goal:.2f}")
    insights.append(f"📈 Confidence OU: {confidence_ou:.0%}")

    return {
        "expected_goal": expected_goal,
        "prob_over": prob_over,
        "ou_line": ou_line,
        "confidence_ou": confidence_ou,
        "edge": edge,
        "ev_over": ev_over,
        "ev_under": ev_under,
        "kelly_over": kelly_over,
        "kelly_under": kelly_under,
        "grade": grade,
        "historical_roi": hist["roi"],
        "historical_win_rate": hist["win_rate"],
        "recommendation": recommendation,
        "rec_color": rec_color,
        "stake": stake,
        "football_brain": football_brain,
        "league_brain": league_brain,
        "market_brain": market_brain,
        "ai_score": ai_score,
        "btts_pred": "YES" if pred_row.get("prediction_btts", -1) == 1 else ("NO" if pred_row.get("prediction_btts", -1) == 0 else "N/A"),
        "prob_btts": pred_row.get("prob_btts", np.nan),
        "confidence_btts": pred_row.get("confidence_btts", np.nan),
        "ht0_pred": "OVER" if pred_row.get("prediction_ht0", -1) == 1 else ("UNDER" if pred_row.get("prediction_ht0", -1) == 0 else "N/A"),
        "prob_ht0": pred_row.get("prob_ht0", np.nan),
        "confidence_ht0": pred_row.get("confidence_ht0", np.nan),
        "insights": insights,
    }

# ============================================================
# PENDING ENGINE (HARDENED)
# ============================================================
class PendingSchema:
    MATCH_UID = "match_uid"
    HOME_TEAM = "home_team"
    AWAY_TEAM = "away_team"
    LEAGUE_NAME = "league_name"
    KICKOFF_TIME = "kickoff_time"
    PREDICTION = "prediction"
    GRADE = "grade"
    CONFIDENCE = "confidence"
    PREDICTION_ID = "prediction_id"
    PREDICTION_TIME = "prediction_time"
    PREDICTION_STATUS = "prediction_status"
    PREDICTION_VERSION = "prediction_version"
    PREDICTION_SOURCE = "prediction_source"
    PREDICTION_NOTES = "prediction_notes"

    @staticmethod
    def display_columns():
        return [
            PendingSchema.MATCH_UID, PendingSchema.HOME_TEAM, PendingSchema.AWAY_TEAM,
            PendingSchema.LEAGUE_NAME, PendingSchema.KICKOFF_TIME, PendingSchema.PREDICTION,
            PendingSchema.GRADE, PendingSchema.CONFIDENCE, PendingSchema.PREDICTION_TIME,
            PendingSchema.PREDICTION_STATUS,
        ]

class MatchUidBuilder:
    @staticmethod
    def build(home, away, kickoff):
        return f"{home}|{away}|{kickoff}"

def generate_match_uid(home: str, away: str, kickoff: str) -> str:
    return MatchUidBuilder.build(home, away, kickoff)

@dataclass
class PredictionContext:
    home_team: str
    away_team: str
    kickoff_time: str
    league_name: str
    prediction: str
    grade: str
    confidence: float

def build_prediction_context() -> PredictionContext:
    home = get_home_team()
    away = get_away_team()
    uploaded_df = get_uploaded_dataframe()
    kickoff = uploaded_df["kickoff_time"].iloc[0] if "kickoff_time" in uploaded_df.columns else ""
    summary = get_match_prediction_summary()
    league = summary.get("league", "") if summary else ""
    prediction_str = (summary.get("ou_pred", "") + " " + str(summary.get("ou_line", ""))) if summary else ""
    grade = summary.get("grade", "") if summary else ""
    confidence = summary.get("confidence_ou", "") if summary else ""
    return PredictionContext(
        home_team=home, away_team=away, kickoff_time=kickoff,
        league_name=league, prediction=prediction_str, grade=grade, confidence=confidence,
    )

def create_match_identity(ctx: PredictionContext) -> dict:
    return {
        PendingSchema.MATCH_UID: generate_match_uid(ctx.home_team, ctx.away_team, ctx.kickoff_time),
        PendingSchema.HOME_TEAM: ctx.home_team,
        PendingSchema.AWAY_TEAM: ctx.away_team,
        PendingSchema.LEAGUE_NAME: ctx.league_name,
        PendingSchema.KICKOFF_TIME: ctx.kickoff_time,
        PendingSchema.PREDICTION: ctx.prediction,
        PendingSchema.GRADE: ctx.grade,
        PendingSchema.CONFIDENCE: ctx.confidence,
    }

def merge_match_identity(pending_df: pd.DataFrame, identity: dict) -> pd.DataFrame:
    df = pending_df.copy()
    for col, val in identity.items():
        df[col] = val
    return df

def load_existing_pending() -> pd.DataFrame:
    db = DatabaseManager()
    try:
        return db.load_pending()
    except Exception:
        return pd.DataFrame()

def detect_duplicate(existing: pd.DataFrame, match_uid: str) -> bool:
    if existing.empty:
        return False
    if PendingSchema.MATCH_UID not in existing.columns or PendingSchema.PREDICTION_STATUS not in existing.columns:
        return False
    mask = (existing[PendingSchema.MATCH_UID] == match_uid) & (existing[PendingSchema.PREDICTION_STATUS] == MatchStatus.PENDING.value)
    return mask.any()

def merge_pending(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new_df
    combined = pd.concat([existing, new_df], ignore_index=True)
    if PendingSchema.MATCH_UID in combined.columns:
        combined.drop_duplicates(subset=[PendingSchema.MATCH_UID], keep="last", inplace=True)
    return combined

def save_pending_df(df: pd.DataFrame) -> None:
    db = DatabaseManager()
    db.save_pending(df)

def append_pending_queue(final_df: pd.DataFrame, match_uid: str) -> AppendResult:
    existing = load_existing_pending()
    if detect_duplicate(existing, match_uid):
        return AppendResult(False, AppendStatus.DUPLICATE, f"Duplicate match (match_uid: {match_uid})",
                            existing_rows=len(existing), new_rows=len(final_df),
                            combined_rows=len(existing), match_uid=match_uid)
    combined = merge_pending(existing, final_df)
    existing_rows = len(existing)
    new_rows = len(final_df)
    combined_rows = len(combined)
    try:
        save_pending_df(combined)
        return AppendResult(True, AppendStatus.SUCCESS, "OK", existing_rows=existing_rows,
                            new_rows=new_rows, combined_rows=combined_rows, match_uid=match_uid)
    except Exception as e:
        return AppendResult(False, AppendStatus.ERROR, str(e), existing_rows=existing_rows,
                            new_rows=new_rows, combined_rows=combined_rows, match_uid=match_uid)

def verify_pending_queue(expected_rows: int, expected_uid: str) -> VerificationResult:
    reloaded = load_existing_pending()
    actual_rows = len(reloaded)
    uid_found = False
    if not reloaded.empty and PendingSchema.MATCH_UID in reloaded.columns:
        uid_found = expected_uid in reloaded[PendingSchema.MATCH_UID].values
    ok = (actual_rows == expected_rows) and uid_found
    return VerificationResult(ok, VerificationStatus.SUCCESS if ok else VerificationStatus.FAILED,
                              expected_rows=expected_rows, actual_rows=actual_rows,
                              expected_uid=expected_uid, uid_found=uid_found,
                              message="OK" if ok else "Mismatch in row count or UID not found.")

def build_pending(ctx: PredictionContext) -> PendingResult:
    if not is_prediction_ready():
        reset_pending_state()
        return PendingResult(False, "Prediksi belum siap disimpan.")
    feature_df = get_feature_dataframe()
    prediction_df = get_prediction_dataframe()
    if feature_df is None or prediction_df is None:
        set_pending_failed("Feature atau prediction dataframe tidak tersedia.")
        return PendingResult(False, "Dataframe fitur atau prediksi tidak tersedia.")
    if len(feature_df) != len(prediction_df):
        msg = "Jumlah baris feature dan prediksi tidak cocok."
        set_pending_failed(msg)
        return PendingResult(False, msg)

    pending_df = pd.concat([feature_df.reset_index(drop=True), prediction_df.reset_index(drop=True)], axis=1)
    now_ts = datetime.now()
    timestamp_str = now_ts.strftime("%Y%m%d%H%M%S%f")
    prediction_ids = [f"pred_{timestamp_str}_{i}" for i in range(len(pending_df))]
    pending_df[PendingSchema.PREDICTION_ID] = prediction_ids
    pending_df[PendingSchema.PREDICTION_TIME] = now_ts.strftime("%Y-%m-%d %H:%M:%S")
    pending_df[PendingSchema.PREDICTION_STATUS] = MatchStatus.PENDING.value
    pending_df[PendingSchema.PREDICTION_VERSION] = APP_VERSION
    pending_df[PendingSchema.PREDICTION_SOURCE] = "MODEL"
    pending_df[PendingSchema.PREDICTION_NOTES] = ""

    metadata = create_pending_metadata(pending_df)
    store_pending_result(metadata)

    identity = create_match_identity(ctx)
    final_df = merge_match_identity(pending_df, identity)

    # Contract Enforcement
    final_df = PendingContract.normalize(final_df)

    append_result = append_pending_queue(final_df, identity[PendingSchema.MATCH_UID])
    if not append_result.success:
        if append_result.status == AppendStatus.DUPLICATE:
            return PendingResult(False, append_result.message, home_team=ctx.home_team,
                                 away_team=ctx.away_team, match_uid=append_result.match_uid)
        else:
            set_pending_failed(append_result.message)
            return PendingResult(False, append_result.message)

    ver_result = verify_pending_queue(append_result.combined_rows, identity[PendingSchema.MATCH_UID])
    debug_lines = [
        "===== PENDING APPEND =====",
        f"Existing Rows : {append_result.existing_rows}",
        f"New Rows : {append_result.new_rows}",
        f"Combined Rows : {append_result.combined_rows}",
        f"Reload Rows : {ver_result.actual_rows}",
        f"Match UID Found : {ver_result.uid_found}",
        f"Append Verification : {ver_result.status.value}",
    ]
    for line in debug_lines:
        st.session_state.setdefault('debug_trace', []).append(line)

    if ver_result.status == VerificationStatus.FAILED:
        set_pending_failed(ver_result.message)
        return PendingResult(False, "Verifikasi append gagal.",
                             existing_rows=append_result.existing_rows, new_rows=append_result.new_rows,
                             combined_rows=append_result.combined_rows,
                             queue_count=ver_result.actual_rows,
                             verification=ver_result.status.value,
                             prediction_id=prediction_ids[0],
                             match_uid=identity[PendingSchema.MATCH_UID],
                             home_team=ctx.home_team, away_team=ctx.away_team,
                             github_commit_success=False)

    add_transaction(TransactionAction.SAVE_PENDING, identity[PendingSchema.MATCH_UID],
                    "SUCCESS", f"Pending saved for {ctx.home_team} vs {ctx.away_team}")
    return PendingResult(True, f"✅ {ctx.home_team} vs {ctx.away_team} berhasil ditambahkan ke Pending Queue",
                         existing_rows=append_result.existing_rows, new_rows=append_result.new_rows,
                         combined_rows=append_result.combined_rows, queue_count=ver_result.actual_rows,
                         verification=ver_result.status.value,
                         prediction_id=prediction_ids[0],
                         match_uid=identity[PendingSchema.MATCH_UID],
                         home_team=ctx.home_team, away_team=ctx.away_team,
                         github_commit_success=True)

def is_pending_ready() -> bool:
    return is_prediction_processed()

def is_pending_processed() -> bool:
    return st.session_state.pending.get("processed", False)

def get_pending_dataframe() -> pd.DataFrame:
    return st.session_state.pending.get("pending_dataframe", None)

def get_pending_status() -> str:
    if not is_pending_ready():
        return "NOT_PROCESSED"
    return st.session_state.pending.get("status", "NOT_PROCESSED")

def get_pending_count() -> int:
    return st.session_state.pending.get("pending_count", 0)

def get_pending_processed_at() -> str:
    return st.session_state.pending.get("processed_at", "")

def get_pending_info() -> dict:
    return {
        "processed": is_pending_processed(),
        "status": get_pending_status(),
        "pending_count": get_pending_count(),
        "processed_at": get_pending_processed_at(),
    }

def get_pending_fail_reason() -> str:
    return st.session_state.pending.get("fail_reason", "")

def create_pending_metadata(pending_df: pd.DataFrame, fail_reason: str = "") -> dict:
    return {
        "processed": True,
        "status": "PROCESSED",
        "pending_count": len(pending_df),
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pending_dataframe": pending_df,
        "fail_reason": fail_reason,
    }

def create_pending_failed_metadata(reason: str) -> dict:
    return {
        "processed": False,
        "status": "FAILED",
        "pending_count": 0,
        "processed_at": "",
        "pending_dataframe": None,
        "fail_reason": reason,
    }

def create_pending_default_metadata() -> dict:
    return {
        "processed": False,
        "status": "NOT_PROCESSED",
        "pending_count": 0,
        "processed_at": "",
        "pending_dataframe": None,
        "fail_reason": "",
    }

def reset_pending_state():
    store_pending_result(create_pending_default_metadata())

def set_pending_failed(reason: str):
    store_pending_result(create_pending_failed_metadata(reason))

def store_pending_result(metadata: dict):
    for key, value in metadata.items():
        st.session_state.pending[key] = value

# ============================================================
# SCORE VALIDATOR
# ============================================================
class ScoreValidator:
    @staticmethod
    def validate(row: dict, ht_home: Optional[int], ht_away: Optional[int],
                 ft_home: Optional[int], ft_away: Optional[int]) -> Tuple[bool, List[str]]:
        errors = []
        required = {"match_uid":"Match UID","league_name":"League","prediction":"Prediction",
                    "prediction_id":"Prediction ID","prediction_time":"Prediction Time",
                    "prediction_version":"Prediction Version","prediction_source":"Prediction Source"}
        for field, label in required.items():
            if not row.get(field): errors.append(f"{label} kosong.")
        if row.get("current_over_odds",0) <= 1.0: errors.append("Over odds <= 1.0 atau kosong.")
        if row.get("confidence_ou") is None or row.get("confidence_ou") == "": errors.append("Confidence kosong.")
        if ht_home is None or ht_away is None: errors.append("HT Score kosong.")
        if ft_home is None or ft_away is None: errors.append("FT Score kosong.")
        if ht_home is not None and ft_home is not None and (ht_home+ht_away) > (ft_home+ft_away):
            errors.append("HT Goals lebih besar dari FT Goals.")
        if row.get("prediction_status") not in [MatchStatus.PENDING.value, MatchStatus.SCORE_ENTERED.value]:
            errors.append("Status tidak valid.")
        return len(errors)==0, errors

# ============================================================
# HISTORY SYNC ENGINE (fix settlement_time dtype)
# ============================================================
class HistorySyncEngine:
    def __init__(self, db: DatabaseManager): self.db = db
    def sync_all_validated(self) -> int:
        pending = self.db.load_pending()
        validated = pending[pending[PendingSchema.PREDICTION_STATUS] == MatchStatus.VALIDATED.value]
        if validated.empty: return 0
        try: history = self.db.load_history()
        except: history = pd.DataFrame()
        synced = 0
        for idx, row in validated.iterrows():
            if not history.empty and not history[(history['match_uid']==row['match_uid'])&(history['prediction_id']==row['prediction_id'])].empty:
                continue
            history = pd.concat([history, pd.DataFrame([row.to_dict()])], ignore_index=True)
            pending.at[idx, PendingSchema.PREDICTION_STATUS] = MatchStatus.HISTORY_SYNCED.value
            pending.loc[idx, 'settlement_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            synced += 1
            add_transaction(TransactionAction.HISTORY_SYNC, row['match_uid'], "SUCCESS", "Synced to history")
        if synced > 0:
            self.db.save_history(history)
            history_synced_mask = pending[PendingSchema.PREDICTION_STATUS] == MatchStatus.HISTORY_SYNCED.value
            pending.loc[history_synced_mask, PendingSchema.PREDICTION_STATUS] = MatchStatus.SETTLED.value
            self.db.save_pending(pending)
            for _, row in pending[history_synced_mask].iterrows():
                add_transaction(TransactionAction.SETTLEMENT, row['match_uid'], "SETTLED", "Auto settlement")
        return synced

# ============================================================
# SETTLEMENT ENGINE (Auto Calculation Center)
# ============================================================
def calculate_auto_results(home_ht: int, away_ht: int, home_ft: int, away_ft: int, ou_line: float) -> dict:
    total_ht = home_ht+away_ht; total_ft = home_ft+away_ft
    return {
        "HT Goals": total_ht, "FT Goals": total_ft,
        "BTTS": "YES" if (home_ft>0 and away_ft>0) else "NO",
        "OU": "OVER" if total_ft > ou_line else "UNDER",
        "HT Over 0.5": "YES" if total_ht>0 else "NO",
        "HT Over 1.5": "YES" if total_ht>1 else "NO",
        "Home Win": "YES" if home_ft>away_ft else "NO",
        "Away Win": "YES" if away_ft>home_ft else "NO",
        "Draw": "YES" if home_ft==away_ft else "NO",
    }

def _settlement_status_color(result: str) -> str:
    return {"WIN":"#16a34a","LOSE":"#ef4444","HALF WIN":"#86efac","HALF LOSE":"#f97316","PUSH":"#6b7280","VOID":"#9ca3af"}.get(result.upper(),"#334155")

def _settlement_badge(result: str) -> str:
    color = _settlement_status_color(result)
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:20px;font-weight:bold;">{result.upper()}</span>'

def render_settlement():
    st.subheader("📝 Auto Calculation Center (Settlement)")
    db = DatabaseManager()
    sync_engine = HistorySyncEngine(db)
    synced = sync_engine.sync_all_validated()
    if synced > 0:
        st.success(f"{synced} pertandingan disinkronkan ke History.")
        return

    pending = db.load_pending()
    validated = pending[pending[PendingSchema.PREDICTION_STATUS] == MatchStatus.VALIDATED.value]
    if validated.empty:
        st.info("Tidak ada pertandingan VALIDATED. Semua sudah tersinkron.")
        return

    for idx, row in validated.iterrows():
        home = row['home_team']; away = row['away_team']
        ht_home = int(row.get('home_ht_goals',0)); ht_away = int(row.get('away_ht_goals',0))
        ft_home = int(row.get('home_goals',0)); ft_away = int(row.get('away_goals',0))
        ou_line = row.get('current_ou',2.5)
        auto = calculate_auto_results(ht_home, ht_away, ft_home, ft_away, ou_line)

        st.markdown(f"### ⚽ {home} vs {away}")
        indicators = [("🟦 HT GOALS",auto["HT Goals"]),("🟩 FT GOALS",auto["FT Goals"]),
                      ("🤝 BTTS",auto["BTTS"]),("🎯 OU RESULT",auto["OU"])]
        cols = st.columns(4)
        for col,(label,val) in zip(cols,indicators):
            with col: st.markdown(f"""<div style="background:#1e293b; border-radius:16px; padding:12px; text-align:center;"><div style="font-size:0.8rem; color:#a0a0b0;">{label}</div><div style="font-size:1.5rem; font-weight:700;">{val}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        statuses = [("⏱ HT OVER 0.5",auto["HT Over 0.5"]),("⏱ HT OVER 1.5",auto["HT Over 1.5"]),
                    ("🏠 HOME WIN",auto["Home Win"]),("✈ AWAY WIN",auto["Away Win"]),("🤝 DRAW",auto["Draw"])]
        cols = st.columns(5)
        for col,(label,status) in zip(cols,statuses):
            with col: st.markdown(f"""<div style="background:#1e293b; border-radius:16px; padding:12px; text-align:center;"><div style="font-size:0.8rem; color:#a0a0b0;">{label}</div><div style="font-size:1.2rem; font-weight:700;">{status}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🎲 Market Settlement")
        pred_ou = int(row.get('prediction_ou',0))
        actual_over = 1 if ft_home+ft_away > ou_line else 0
        ou_result = "WIN" if pred_ou==actual_over else ("LOSE" if pred_ou!=actual_over else "PUSH")
        if (ft_home+ft_away)==ou_line: ou_result="PUSH"
        pred_btts = int(row.get('prediction_btts',0))
        actual_btts = 1 if (ft_home>0 and ft_away>0) else 0
        btts_result = "WIN" if pred_btts==actual_btts else "LOSE"
        pred_ht0 = int(row.get('prediction_ht0',0)) if 'prediction_ht0' in row else 0
        actual_ht0 = 1 if (ht_home+ht_away)>0 else 0
        ht0_result = "WIN" if pred_ht0==actual_ht0 else "LOSE"

        audit = [("OU",f"{'OVER' if pred_ou==1 else 'UNDER'} {ou_line}",f"{'OVER' if actual_over==1 else 'UNDER'}",ou_result),
                 ("BTTS",f"{'YES' if pred_btts==1 else 'NO'}",f"{'YES' if actual_btts==1 else 'NO'}",btts_result),
                 ("HT O0.5",f"{'OVER' if pred_ht0==1 else 'UNDER'}",f"{'OVER' if actual_ht0==1 else 'UNDER'}",ht0_result)]
        st.markdown("| Item | Prediction | Actual | Result |")
        st.markdown("|------|------------|--------|--------|")
        for item,pred,act,res in audit:
            badge = _settlement_badge(res)
            st.markdown(f"| {item} | {pred} | {act} | {badge} |", unsafe_allow_html=True)

# ============================================================
# PENDING MANAGER (Editable Table – Fixed)
# ============================================================
def render_pending():
    st.subheader("📋 Pending Manager (Input Skor)")
    db = DatabaseManager()
    pending_all = db.load_pending()
    if pending_all.empty:
        st.info("Belum ada data Pending."); return

    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status", ["ALL"]+[s.value for s in MatchStatus])
    with col2:
        leagues = ["ALL"]+sorted(pending_all['league_name'].dropna().unique().tolist())
        league_filter = st.selectbox("League", leagues)
    with col3:
        conf_filter = st.selectbox("Confidence", ["ALL",">70%",">80%",">90%"])

    df = pending_all.copy()
    if status_filter!="ALL": df = df[df[PendingSchema.PREDICTION_STATUS]==status_filter]
    if league_filter!="ALL": df = df[df['league_name']==league_filter]
    if conf_filter==">70%": df = df[df['confidence_ou']>0.7]
    elif conf_filter==">80%": df = df[df['confidence_ou']>0.8]
    elif conf_filter==">90%": df = df[df['confidence_ou']>0.9]

    if df.empty: st.info("Tidak ada data."); return

    editable_status = [MatchStatus.PENDING.value, MatchStatus.SCORE_ENTERED.value]
    df['is_editable'] = df[PendingSchema.PREDICTION_STATUS].isin(editable_status)
    # Kolom skor sudah dijamin ada oleh PendingContract
    df['ht_home_input'] = df['home_ht_goals'].apply(lambda x: int(x) if pd.notna(x) else None)
    df['ht_away_input'] = df['away_ht_goals'].apply(lambda x: int(x) if pd.notna(x) else None)
    df['ft_home_input'] = df['home_goals'].apply(lambda x: int(x) if pd.notna(x) else None)
    df['ft_away_input'] = df['away_goals'].apply(lambda x: int(x) if pd.notna(x) else None)

    editor_columns = [
        'match_uid', 'home_team', 'away_team', 'prediction', 'prediction_status',
        'ht_home_input', 'ht_away_input', 'ft_home_input', 'ft_away_input'
    ]
    editor_df = df[editor_columns].copy()

    # Semua kolom input tidak di-disabled secara kondisional agar tidak ada Series
    column_config = {
        "match_uid": st.column_config.TextColumn("Match UID", disabled=True),
        "home_team": st.column_config.TextColumn("Home", disabled=True),
        "away_team": st.column_config.TextColumn("Away", disabled=True),
        "prediction": st.column_config.TextColumn("Prediction", disabled=True),
        "prediction_status": st.column_config.TextColumn("Status", disabled=True),
        "ht_home_input": st.column_config.NumberColumn("HT Home", min_value=0, step=1, disabled=False),
        "ht_away_input": st.column_config.NumberColumn("HT Away", min_value=0, step=1, disabled=False),
        "ft_home_input": st.column_config.NumberColumn("FT Home", min_value=0, step=1, disabled=False),
        "ft_away_input": st.column_config.NumberColumn("FT Away", min_value=0, step=1, disabled=False),
    }

    edited_df = st.data_editor(
        editor_df,
        column_config=column_config,
        width='stretch',
        key="pending_score_editor"
    )

    if st.button("💾 Save Changes"):
        changed = False
        for idx, row in edited_df.iterrows():
            if idx not in pending_all.index:
                continue
            original = pending_all.loc[idx]
            if not original[PendingSchema.PREDICTION_STATUS] in editable_status:
                continue
            ht_home = row['ht_home_input']
            ht_away = row['ht_away_input']
            ft_home = row['ft_home_input']
            ft_away = row['ft_away_input']
            if (ht_home == original.get('home_ht_goals') and ht_away == original.get('away_ht_goals') and
                ft_home == original.get('home_goals') and ft_away == original.get('away_goals')):
                continue
            valid, errors = ScoreValidator.validate(original.to_dict(), ht_home, ht_away, ft_home, ft_away)
            pending_all.at[idx, 'home_ht_goals'] = ht_home
            pending_all.at[idx, 'away_ht_goals'] = ht_away
            pending_all.at[idx, 'home_goals'] = ft_home
            pending_all.at[idx, 'away_goals'] = ft_away
            ou_line = original.get('current_ou',2.5)
            total_ft = (ft_home or 0)+(ft_away or 0)
            pending_all.at[idx, 'actual_over'] = 1 if total_ft > ou_line else 0
            pending_all.at[idx, 'actual_btts'] = 1 if ((ft_home or 0)>0 and (ft_away or 0)>0) else 0
            if valid:
                pending_all.at[idx, 'prediction_status'] = MatchStatus.VALIDATED.value
                add_transaction(TransactionAction.VALIDATION, original['match_uid'], "VALIDATED", "Score valid")
            else:
                pending_all.at[idx, 'prediction_status'] = MatchStatus.SCORE_ENTERED.value
                add_transaction(TransactionAction.VALIDATION, original['match_uid'], "SCORE_ENTERED", f"Validation failed: {'; '.join(errors)}")
            changed = True
        if changed:
            db.save_pending(pending_all)
            st.success("Perubahan disimpan.")
            st.rerun()

# ============================================================
# HISTORY MANAGER
# ============================================================
def render_history_manager():
    st.subheader("📜 History Manager")
    db = DatabaseManager()
    try: history = db.load_history()
    except: history = pd.DataFrame()
    if history.empty: st.info("History kosong."); return
    st.dataframe(history, use_container_width=True)
    st.download_button("Export CSV", history.to_csv(index=False).encode('utf-8'), "history.csv","text/csv")

# ============================================================
# DATABASE MONITOR
# ============================================================
def render_database_monitor():
    st.subheader("🗄️ Database Monitor")
    db = DatabaseManager()
    resources = [ResourceRegistry.PENDING, ResourceRegistry.HISTORY, ResourceRegistry.DATASET,
                 ResourceRegistry.LEAGUE_STATS, ResourceRegistry.LEAGUE_THRESHOLD]
    data = []
    for res in resources:
        try:
            df = db.storage.load_dataframe(res)
            rows = len(df)
            last_mod = "N/A"
            if isinstance(db.storage, LocalStorageProvider):
                path = db.storage._get_path(res)
                last_mod = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if path.exists() else "N/A"
            status = "Active" if not df.empty else "Empty"
        except: rows, last_mod, status = "Error", "Error", "Error"
        data.append([res.id, rows, last_mod, status])
    st.table(pd.DataFrame(data, columns=["Resource","Rows","Last Update","Status"]))

# ============================================================
# TRANSACTION LOG
# ============================================================
def add_transaction(action: TransactionAction, match_uid: str, status: str, message: str):
    entry = TransactionLogEntry(
        timestamp=datetime.now(),
        transaction_id=f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}",
        match_uid=match_uid, action=action, status=status, message=message
    )
    st.session_state.transaction_log.append(entry)
    if len(st.session_state.transaction_log) > 500: st.session_state.transaction_log = st.session_state.transaction_log[-500:]

def render_transaction_log():
    st.subheader("📋 Transaction Log")
    logs = st.session_state.get('transaction_log', [])
    if not logs: st.info("Belum ada transaksi."); return
    for entry in reversed(logs):
        st.text(f"{entry.timestamp} | {entry.transaction_id} | {entry.match_uid} | {entry.action.value} | {entry.status} | {entry.message}")

# ============================================================
# DEBUG CENTER
# ============================================================
def render_debug_center():
    st.subheader("🐞 Debug Center")
    trace = st.session_state.get('debug_trace', [])
    if not trace: st.info("No trace."); return
    keywords = {"Prediction":["CHECKPOINT","prediction"],"Pending":["PENDING","append"],"Settlement":["Settlement","settlement"],
                "History":["History"],"Learning":["Learning"],"GitHub":["GITHUB","WRITE","HTTP"]}
    for cat, filters in keywords.items():
        lines = [l for l in trace if any(f.lower() in l.lower() for f in filters)]
        if lines:
            with st.expander(f"{cat} ({len(lines)})"):
                for l in lines: st.text(l)

# ============================================================
# PERFORMANCE CENTER
# ============================================================
def render_performance_center():
    st.subheader("📊 Performance Center")
    db = DatabaseManager()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Model","✅" if db.is_model_ready() else "❌")
    c2.metric("Dataset","✅" if db.is_dataset_ready() else "❌")
    c3.metric("History","✅" if db.is_history_ready() else "❌")
    c4.metric("Threshold","✅" if db.is_threshold_ready() else "❌")

# ============================================================
# PREDICTION CARD & UI
# ============================================================
def render_horizontal_metric_row(cards):
    divs = []
    for icon, label, value, bg in cards:
        divs.append(f'<div class="brain-card" style="background: {bg};"><div class="icon">{icon}</div><div class="label">{label}</div><div class="badge-value" style="background: rgba(0,0,0,0.2); color: white;">{value}</div></div>')
    html = f'<div class="brain-row">{"".join(divs)}</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_prediction_card(summary):
    if not summary: return
    st.markdown(f"""
    <div class="prediction-card">
        <div style="text-align:center;">
            <h3 style="margin:0 0 2px 0; font-size:1.6rem;">⚽ {summary['home']} vs {summary['away']}</h3>
            <p style="color:#a0a0b0; margin:0 0 8px 0; font-size:0.8rem;">{summary['league']}</p>
        </div>
    """, unsafe_allow_html=True)
    col_ou, col_grade, col_rec = st.columns(3)
    with col_ou: st.markdown(f"<div style='font-size:1.8rem; font-weight:900; margin-bottom:6px;'>{summary['ou_pred']} {summary['ou_line']}</div>", unsafe_allow_html=True)
    with col_grade: st.markdown(f'<span class="badge badge-{summary["rec_color"]}" style="font-size:1rem;">Grade {summary["grade"]}</span>', unsafe_allow_html=True)
    with col_rec: st.markdown(f'<span class="badge badge-{summary["rec_color"]}">{summary["recommendation"]}</span>', unsafe_allow_html=True)
    render_horizontal_metric_row([
        ("🏅", "Grade", summary['grade'], "#334155"),
        ("📈", "Confidence", f"{summary['confidence_ou']:.0%}", "#16a34a"),
        ("📏", "Edge", f"{summary['edge']:.4f}", "#7c3aed"),
        ("⚽", "Expected Goal", f"{summary['expected_goal']:.2f}", "#2563eb"),
    ])
    if summary.get('ev_over') is not None:
        render_horizontal_metric_row([
            ("💰", "EV Over", f"{summary['ev_over']:+.3f}", "#16a34a" if summary['ev_over'] > 0 else "#ef4444"),
            ("💰", "EV Under", f"{summary['ev_under']:+.3f}", "#16a34a" if summary['ev_under'] > 0 else "#ef4444"),
            ("📊", "Kelly Over", f"{summary.get('kelly_over', 0):.1%}", "#2563eb"),
            ("📊", "Kelly Under", f"{summary.get('kelly_under', 0):.1%}", "#2563eb"),
        ])
    render_horizontal_metric_row([
        ("⚽", "Football Brain", summary['football_brain'], "#2563eb"),
        ("🏆", "League Brain", summary['league_brain'], "#f97316"),
        ("📈", "Market Brain", summary['market_brain'], "#7c3aed"),
        ("🧠", "AI Score", summary['ai_score'], "#16a34a"),
    ])
    render_horizontal_metric_row([
        ("🤝", "BTTS", f"{summary['btts_pred']} ({summary['confidence_btts']:.0%})" if not np.isnan(summary.get('confidence_btts')) else "N/A", "#334155"),
        ("⏱", "HT Over 0.5", f"{summary['ht0_pred']} ({summary['confidence_ht0']:.0%})" if not np.isnan(summary.get('confidence_ht0')) else "N/A", "#334155"),
        ("💰", "Historical ROI", f"{summary['historical_roi']:.1f}%", "#16a34a"),
        ("📊", "Historical Win Rate", f"{summary['historical_win_rate']:.1f}%", "#16a34a"),
    ])
    stake_level = {"HIGH BET": 1.0, "STRONG BET": 0.75, "MEDIUM BET": 0.50, "SMALL BET": 0.25, "NO BET": 0.0}
    level = stake_level.get(summary['recommendation'], 0)
    st.markdown(f"""
    <div style="margin: 8px 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
            <span>{summary['recommendation']}</span>
            <span>{summary['stake']}</span>
        </div>
        <div style="background: #334155; border-radius: 8px; height: 10px; margin-top: 4px;">
            <div style="width: {level*100}%; background: #2563eb; border-radius: 8px; height: 100%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if summary.get("insights"):
        with st.expander("🔍 Analysis Summary", expanded=False):
            for ins in summary["insights"]: st.markdown(f"- {ins}")
    st.markdown("</div>", unsafe_allow_html=True)

def get_match_prediction_summary():
    if not is_prediction_processed(): return None
    feature_df = get_feature_dataframe()
    pred_df = get_prediction_dataframe()
    if feature_df is None or len(feature_df)==0 or pred_df is None or len(pred_df)==0: return None
    feat_row = feature_df.iloc[0]; pred_row = pred_df.iloc[0]
    home = get_home_team(); away = get_away_team()
    league = feat_row.get("league_name","Unknown League")
    if not league and is_league_profile_loaded():
        lp_df = get_league_profile_dataframe()
        if lp_df is not None and "league_name" in lp_df.columns: league = lp_df["league_name"].iloc[0]
    db = DatabaseManager()
    try:
        th = db.load_threshold()
        default_over = th.get('ev_over',0.01); default_under = th.get('ev_under',0.20)
    except: default_over, default_under = 0.01, 0.20
    league_thresholds_df = db.load_league_thresholds()
    historical_performance = st.session_state.learning["historical_performance"]
    dec = calculate_decision_engine(feat_row, pred_row, (default_over, default_under), league_thresholds_df, historical_performance)
    return {
        "home":home,"away":away,"league":league,
        "ou_pred":"OVER" if dec["prob_over"]>=0.5 else "UNDER","ou_line":dec["ou_line"],
        "expected_goal":dec["expected_goal"],"confidence_ou":dec["confidence_ou"],
        "edge":dec["edge"],"ev_over":dec["ev_over"],"ev_under":dec["ev_under"],
        "grade":dec["grade"],"historical_roi":dec["historical_roi"],"historical_win_rate":dec["historical_win_rate"],
        "recommendation":dec["recommendation"],"rec_color":dec["rec_color"],"stake":dec["stake"],
        "football_brain":dec["football_brain"],"league_brain":dec["league_brain"],"market_brain":dec["market_brain"],
        "ai_score":dec["ai_score"],"btts_pred":dec["btts_pred"],"prob_btts":dec["prob_btts"],"confidence_btts":dec["confidence_btts"],
        "ht0_pred":dec["ht0_pred"],"prob_ht0":dec["prob_ht0"],"confidence_ht0":dec["confidence_ht0"],
        "insights":dec["insights"],"kelly_over":dec.get("kelly_over",0),"kelly_under":dec.get("kelly_under",0),
    }

# ============================================================
# PIPELINE STEPS
# ============================================================
class PipelineStep(ABC):
    name: str = ""
    @abstractmethod
    def execute(self, context: dict) -> dict: pass
    def is_enabled(self) -> bool: return True

class ValidationStep(PipelineStep):
    name = "validation"
    def execute(self, context): context["validation_success"] = validate_dataset(); return context

class LeagueProfileStep(PipelineStep):
    name = "league_profile"
    def execute(self, context):
        if context.get("validation_success"): build_league_profile()
        return context

class FeatureEngineeringStep(PipelineStep):
    name = "feature_engineering"
    def execute(self, context):
        if context.get("validation_success"): build_features()
        return context

class PredictionStep(PipelineStep):
    name = "prediction"
    def execute(self, context):
        if context.get("validation_success"): build_prediction()
        return context

class ApplicationController:
    def __init__(self, pipeline=None): self.pipeline = pipeline
    def run_analysis(self, uploaded_file) -> dict:
        if 'debug_trace' not in st.session_state: st.session_state.debug_trace = []
        add = lambda msg: st.session_state.debug_trace.append(msg)
        add("CHECKPOINT 1: masuk run_analysis")
        context = {"uploaded_file":uploaded_file}
        add("CHECKPOINT 2: mulai validasi")
        val_step = ValidationStep()
        context = val_step.execute(context)
        if context.get("validation_success",False): add("CHECKPOINT 2: validation OK")
        else:
            errors = get_validation_errors()
            add(f"CHECKPOINT 2: validation FAILED - errors: {errors}")
            return context
        add("CHECKPOINT 3: mulai league profile")
        lp_step = LeagueProfileStep()
        context = lp_step.execute(context)
        if is_league_profile_loaded(): add("CHECKPOINT 3: league profile OK")
        else:
            reason = st.session_state.league_profile.get("status","FAILED")
            add(f"CHECKPOINT 3: league profile FAILED - status: {reason}")
            return context
        add("CHECKPOINT 4: mulai feature engineering")
        fe_step = FeatureEngineeringStep()
        context = fe_step.execute(context)
        if is_feature_engineering_processed(): add("CHECKPOINT 4: feature engineering OK")
        else:
            fail_reason = st.session_state.feature_engineering.get("fail_reason","unknown")
            warnings = st.session_state.feature_engineering.get("warnings",[])
            add(f"CHECKPOINT 4: feature engineering FAILED - reason: {fail_reason}, warnings: {warnings}")
            return context
        add("CHECKPOINT 5: mulai prediction")
        pred_step = PredictionStep()
        context = pred_step.execute(context)
        if is_prediction_processed(): add("CHECKPOINT 5: prediction OK")
        else:
            fail_reason = get_prediction_fail_reason()
            add(f"CHECKPOINT 5: prediction FAILED - reason: {fail_reason}")
            return context
        add("CHECKPOINT 6: semua step berhasil")
        return context

# ============================================================
# MAIN APP
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("<div style='text-align:center;'><span style='font-size:3rem;'>⚽</span></div>", unsafe_allow_html=True)
        st.title("Football AI V2")
        st.caption(f"v{APP_VERSION}")
        st.markdown("---")
        st.subheader("AI Status")
        db = DatabaseManager()
        st.write("Model: ", "✅" if db.is_model_ready() else "❌")
        st.write("Profil liga: ", "✅" if db.is_threshold_ready() else "❌")
        st.markdown("---")
        st.subheader("EV Thresholds")
        try:
            th = db.load_threshold()
            ev_over = th.get('ev_over',0.01); ev_under = th.get('ev_under',0.20)
        except: ev_over, ev_under = 0.01, 0.20
        new_over = st.number_input("Min EV Over", value=ev_over, step=0.01)
        new_under = st.number_input("Min EV Under", value=ev_under, step=0.01)
        if st.button("Simpan Threshold"): db.save_threshold({'ev_over':new_over,'ev_under':new_under}); st.success("Threshold disimpan")
        st.markdown("---")
        st.subheader("Pending")
        st.metric("Jumlah", st.session_state.pending.get("pending_count",0))

def render_upload_section():
    if not has_uploaded_dataframe():
        with st.expander("📤 Upload File CSV", expanded=True):
            uploaded_file = st.file_uploader("Pilih file CSV pertandingan", type=["csv"], key="file_uploader")
            if uploaded_file is not None: process_upload(uploaded_file); st.rerun()
    else:
        info = get_upload_info()
        st.markdown(f"**📤 {info['home_team']} vs {info['away_team']}** | *{info['filename']}*")
        if st.button("🔄 Ganti File"): reset_workflow_state(reset_upload=True); st.rerun()

def main():
    initialize_session_state()
    render_sidebar()
    st.title(f"⚽ {APP_TITLE}")
    render_upload_section()

    token = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
    if token:
        owner = os.getenv("GITHUB_USERNAME") or st.secrets.get("GITHUB_USERNAME")
        repo = os.getenv("GITHUB_REPOSITORY") or st.secrets.get("GITHUB_REPOSITORY")
        if not owner or not repo:
            repo = os.getenv("GITHUB_REPO") or st.secrets.get("GITHUB_REPO","username/repo")
            parts = repo.split("/"); owner, repo = (parts[0], parts[1]) if len(parts)==2 else ("username","repo")
        branch = os.getenv("GITHUB_BRANCH") or st.secrets.get("GITHUB_BRANCH","main")
        DatabaseManager._default_storage = GitHubStorageProvider(owner, repo, branch, token)
    else:
        DatabaseManager._default_storage = LocalStorageProvider()

    if has_uploaded_dataframe():
        col_btn, _, _ = st.columns([2,1,1])
        with col_btn:
            if st.button("🚀 ANALYZE MATCH", type="primary", use_container_width=True):
                st.session_state.debug_trace = []
                add = lambda msg: st.session_state.debug_trace.append(msg)
                add("ENTRY: Tombol Analyze Match diklik")
                with st.spinner("Menganalisis..."):
                    reset_workflow_state(reset_upload=False)
                    controller = ApplicationController()
                    controller.run_analysis(None)
                if is_prediction_processed(): add("✅ Prediction berhasil, rerun untuk menampilkan UI."); st.rerun()
                else:
                    st.error("Workflow Analyze Match gagal. Lihat debug trace di bawah.")
                    if st.session_state.debug_trace:
                        with st.expander("🔍 Debug Trace (klik untuk melihat)", expanded=True):
                            for line in st.session_state.debug_trace: st.text(line)

        if is_prediction_processed():
            st.markdown("---")
            summary = get_match_prediction_summary()
            if summary:
                render_prediction_card(summary)
                if st.button("💾 Save Pending", use_container_width=True):
                    ctx = build_prediction_context()
                    result = build_pending(ctx)
                    if result.success: st.success(f"{result.message}\n\nQueue: {result.queue_count} pertandingan\nUID: {result.match_uid}")
                    else: st.error(f"❌ Gagal menyimpan pending\n\n{result.message}")
                    st.rerun()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Pending", "📝 Settlement", "📜 History", "🧠 Learning",
            "🗄️ Database", "📋 Transaction Log", "🐞 Debug", "📊 Performance"
        ])
        with tab1: render_pending()
        with tab2: render_settlement()
        with tab3: render_history_manager()
        with tab4:
            if st.button("🧠 Build Learning Dataset"):
                st.info("Learning engine akan membaca History yang sudah tervalidasi.")
        with tab5: render_database_monitor()
        with tab6: render_transaction_log()
        with tab7: render_debug_center()
        with tab8: render_performance_center()

    if st.session_state.get("debug_trace"):
        with st.expander("📜 Raw Debug Trace"):
            for line in st.session_state.debug_trace: st.text(line)

if __name__ == "__main__":
    main()