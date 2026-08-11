# services/threshold_service.py
"""Threshold service untuk EV Over/Under dan BTTS."""
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Any

BASE_DIR = Path(__file__).resolve().parent.parent
EV_THRESHOLD_FILE = BASE_DIR / "ev_threshold.json"


class ThresholdService:
    @staticmethod
    def get_thresholds(storage: Optional[Any] = None) -> Tuple[float, float]:
        from services.resource_registry import ResourceRegistry

        if storage and storage.exists(ResourceRegistry.THRESHOLD):
            data = storage.load_json(ResourceRegistry.THRESHOLD)
            return data.get('ev_over', 0.01), data.get('ev_under', 0.02)
        if os.path.exists(EV_THRESHOLD_FILE):
            try:
                with open(EV_THRESHOLD_FILE) as f:
                    data = json.load(f)
                    return data.get('ev_over', 0.01), data.get('ev_under', 0.02)
            except Exception:
                pass
        return 0.01, 0.02

    @staticmethod
    def get_btts_threshold(storage, league_code):
        from services.resource_registry import ResourceRegistry

        if storage and storage.exists(ResourceRegistry.LEAGUE_PROFILE):
            df = storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE)
            if 'btts_threshold' in df.columns:
                row = df[df['league_code'] == league_code]
                if not row.empty:
                    return row['btts_threshold'].values[0]
        return 0.22
