# tests/test_match_summary.py
import sys
from unittest.mock import MagicMock, patch
import pytest

# Mock streamlit sebelum import app
sys.modules['streamlit'] = MagicMock()

import pandas as pd
import numpy as np
from app import get_match_prediction_summary, SessionManager, AppState, PredictionState, UploadState
from services.resource_registry import ResourceRegistry
from services.storage import LocalStorageProvider

@pytest.fixture
def mock_session():
    # Setup session state yang diperlukan
    session = SessionManager()
    # Set prediction state processed dengan data dummy
    ps = PredictionState(processed=True)
    # Buat dataframe hasil prediksi dengan minimal fitur
    df = pd.DataFrame([{
        'current_ou': 2.5,
        'current_over_odds': 1.90,
        'current_under_odds': 1.85,
        'league_name': 'Test League',
        'match_uid': 'A|B|2024-01-01 20:00',
        'home_team': 'A',
        'away_team': 'B',
        'kickoff_time': '2024-01-01 20:00',
    }])
    ps.prediction_dataframe = df
    ps.prediction_result = {
        'expected_goal': 2.8,
        'prob_over': 0.65,
        'prob_under': 0.35,
        'prob_over_raw': 0.65,
        'ev_over': 0.15,
        'ev_under': -0.10,
        'kelly_over': 0.08,
        'kelly_under': 0.0,
        'prob_btts': 0.58,
        'confidence_ou': 0.65,
        'confidence_btts': 0.58,
        'prediction_ou': 1,
        'prediction_btts': 1,
        'top3_scores': [(2,1,0.07),(1,1,0.065),(3,1,0.04)],
        'score_probs': [(2,1,0.07),(1,1,0.065),(3,1,0.04)],
        'prob_1x2_model': {'home':0.4, 'draw':0.3, 'away':0.3},
        'prob_1x2_league': {'home':0.4, 'draw':0.3, 'away':0.3},
        'prob_1x2_hybrid': {'home':0.4, 'draw':0.3, 'away':0.3},
    }
    session.set_prediction_state(ps)
    # Mock uploaded_file name
    session.uploaded_file = MagicMock()
    session.uploaded_file.name = 'A vs B.csv'
    # Tidak ada odds 1X2/CS diupload
    session.uploaded_odds = None
    session.uploaded_odds_1x2 = None
    session.uploaded_open_1x2 = None
    return session

@pytest.fixture
def mock_storage(tmp_path):
    # Gunakan LocalStorageProvider dengan direktori sementara
    storage = LocalStorageProvider(base_dir=tmp_path)
    # Simpan threshold default
    from services.resource_registry import ResourceRegistry
    storage.save_json(ResourceRegistry.THRESHOLD, {"ev_over": 0.01, "ev_under": 0.02})
    return storage

class TestGetMatchPredictionSummary:
    def test_summary_without_odds(self, mock_session, mock_storage):
        # Simulasikan tidak ada odds 1X2/CS di session
        mock_session.uploaded_odds = None
        mock_session.uploaded_odds_1x2 = None

        summary = get_match_prediction_summary(mock_session, mock_storage)
        assert summary is not None
        assert summary["home"] == "A"
        assert summary["away"] == "B"
        # Karena prob_over >= 0.10 dan ev_over > threshold (0.01) maka rekomendasi OVER
        assert summary["recommendation"] == "TARUHAN OVER"
        assert summary["stake"] == 100000
        # Tidak ada data 1X2
        assert summary["prediction_1x2"] is None
        # BTTS: tidak ada market odds di session, jadi recommendation_btts None
        # Di dalam fungsi, market_odds_dict diambil dari st.session_state.get('_market_odds_btts_'), 
        # tapi karena streamlit sudah di-mock, st.session_state adalah MagicMock, akan return None.
        # Jadi recommendation_btts tetap None.
        assert summary["recommendation_btts"] is None

    def test_summary_with_1x2_odds(self, mock_session, mock_storage):
        # Berikan odds 1X2 via session
        mock_session.uploaded_odds_1x2 = {"home": 2.0, "draw": 3.5, "away": 4.0}
        # Tidak ada open odds
        mock_session.uploaded_open_1x2 = None

        summary = get_match_prediction_summary(mock_session, mock_storage)
        # Sekarang harus ada prediksi 1X2
        assert summary["prediction_1x2"] is not None
        # Karena ev_1x2 akan dihitung, dengan prob_hybrid 0.4, odds home=2.0 -> ev = 0.4*2.0-1 = -0.2 -> no bet? 
        # Prob hybrid default 0.4, 0.3, 0.3. ev list: home -0.2, draw 0.05, away 0.2. Threshold ev_over = 0.01.
        # draw ev 0.05 > 0.01, away ev 0.2 > 0.01, terbesar away -> prediction_1x2 = "AWAY"
        # Stake: target 100000 / (4.0-1) = 33333.33
        assert summary["prediction_1x2"] == "AWAY"
        assert abs(summary["stake_1x2"] - 33333.33) < 1

    def test_no_bet_ou(self, mock_session, mock_storage):
        # Ubah prediction result agar ev_over rendah -> NO BET
        ps = mock_session.get_prediction_state()
        ps.prediction_result = {
            **ps.prediction_result,
            'prob_over': 0.05,  # di bawah 0.10 threshold
            'ev_over': 0.005,
            'ev_under': 0.02,
        }
        mock_session.set_prediction_state(ps)
        summary = get_match_prediction_summary(mock_session, mock_storage)
        assert summary["recommendation"] == "NO BET"
        assert summary["stake"] == 0

    def test_correct_score_with_odds(self, mock_session, mock_storage):
        # Sediakan odds correct score
        mock_session.uploaded_odds = {
            "2:1": 7.0,
            "1:1": 6.0,
            "3:1": 10.0,
            "OTHER": 50.0,
        }
        summary = get_match_prediction_summary(mock_session, mock_storage)
        # Harus ada cs_recommendations
        assert summary["cs_recommendations"] is not None
        # top3_scores: (2,1,0.07) -> key "2:1" ada -> (2,1,7.0)
        assert len(summary["cs_recommendations"]) == 3
        # urutan pertama harus (2,1,7.0)
        assert summary["cs_recommendations"][0] == (2, 1, 7.0)
