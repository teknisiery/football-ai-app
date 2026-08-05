# test_storage.py
import pandas as pd
import json
import pytest
from pathlib import Path
from services.storage import LocalStorageProvider, DatabaseManager, ResourceRegistry, Resource

# Resource dummy untuk pengujian
@pytest.fixture
def test_resource():
    return Resource("test", "dataframe", "csv", "test.csv")

@pytest.fixture
def json_resource():
    return Resource("test_json", "dict", "json", "test.json")

@pytest.fixture
def storage(tmp_path):
    return LocalStorageProvider(base_dir=tmp_path)


def test_local_storage_dataframe(storage, test_resource):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    storage.save_dataframe(test_resource, df)
    loaded = storage.load_dataframe(test_resource)
    pd.testing.assert_frame_equal(loaded, df)


def test_local_storage_json(storage, json_resource):
    data = {"key": "value", "number": 42}
    storage.save_json(json_resource, data)
    loaded = storage.load_json(json_resource)
    assert loaded == data


def test_database_manager_pending(storage):
    db = DatabaseManager(storage)
    pending_df = pd.DataFrame([{
        "match_uid": "abc|def|2024-01-01",
        "home_team": "A", "away_team": "B",
        "league_name": "Liga", "kickoff_time": "2024-01-01",
        "prediction": "OVER 2.5", "recommendation": "TARUHAN OVER", "stake": 100000,
        "prediction_status": "PENDING",
        "home_ht_goals": 0, "away_ht_goals": 0,
        "home_goals": None, "away_goals": None,
        "confidence_ou": 0.75
    }])
    db.save_pending(pending_df)
    loaded = db.load_pending()
    assert len(loaded) == 1
    assert loaded.iloc[0]['home_team'] == 'A'


def test_database_manager_history(storage):
    db = DatabaseManager(storage)
    history_df = pd.DataFrame([{
        "match_uid": "xyz|uvw|2024-01-01",
        "home_team": "C", "away_team": "D",
        "league_name": "Liga", "kickoff_time": "2024-01-01",
        "prediction": "UNDER 2.5", "recommendation": "TARUHAN UNDER", "stake": 100000,
        "home_goals": 1, "away_goals": 0,
        "home_ht_goals": 0, "away_ht_goals": 0,
        "confidence_ou": 0.65,
        "settlement_time": "2024-01-01 22:00",
        "profit": 85000.0,
        "result": "FULL WIN"
    }])
    # Tambahkan kolom profit dan result (sudah ada di atas)
    db.save_history(history_df)
    loaded = db.load_history()
    assert len(loaded) == 1
    assert loaded.iloc[0]['profit'] == 85000.0
