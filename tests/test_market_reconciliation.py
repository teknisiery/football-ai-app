# tests/test_market_reconciliation.py
import pytest
from services.market_reconciliation import (
    de_vig_correct_score,
    estimate_tail_from_model,
    reconcile_cs_with_1x2,
)
from services.probability_fusion import marginalize


class TestDeVigCorrectScore:
    def test_basic_conversion(self):
        cs_odds = {"1:0": 8.2, "0:0": 7.0, "2:0": 15.0}
        result = de_vig_correct_score(cs_odds, method='basic')
        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_with_other(self):
        cs_odds = {"1:0": 8.2, "0:0": 7.0, "OTHER": 50.0}
        result = de_vig_correct_score(cs_odds, method='basic')
        # OTHER tidak dimasukkan ke distribusi spesifik, tapi tidak mempengaruhi perhitungan
        assert (1, 0) in result
        assert (0, 0) in result
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_empty_odds(self):
        result = de_vig_correct_score({})
        assert result == {}

    def test_with_model_fallback(self):
        cs_odds = {"1:0": 8.2}  # hanya satu skor, banyak tail
        model_scores = [(0, 0, 0.1), (1, 0, 0.2), (0, 1, 0.15), (1, 1, 0.12),
                        (2, 0, 0.08), (0, 2, 0.07), (2, 1, 0.05)]
        result = de_vig_correct_score(cs_odds, method='poisson_tail', model_score_probs=model_scores)
        assert len(result) > 1
        assert abs(sum(result.values()) - 1.0) < 1e-9
        # (1,0) harus tetap ada
        assert (1, 0) in result


class TestEstimateTailFromModel:
    def test_tail_filling(self):
        cs_dist = {(1, 0): 0.3, (0, 0): 0.4}  # total baru 0.7
        model_scores = [(0, 1, 0.1), (1, 1, 0.15), (2, 0, 0.05)]
        result = estimate_tail_from_model(cs_dist, model_scores)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert result[(0, 1)] > 0
        assert result[(1, 1)] > 0
        assert result[(2, 0)] > 0

    def test_tail_no_model(self):
        cs_dist = {(1, 0): 0.3, (0, 0): 0.4}
        result = estimate_tail_from_model(cs_dist, expected_goals=(1.2, 0.8))
        assert abs(sum(result.values()) - 1.0) < 1e-9
        # Harus mengisi semua skor yang belum ada
        assert len(result) > 2


class TestReconcileCSWith1X2:
    def test_ipf_convergence(self):
        # Distribusi awal
        cs_dist = {(1, 0): 0.5, (0, 0): 0.3, (0, 1): 0.2}
        # Target 1X2
        fair_1x2 = {'home': 0.4, 'draw': 0.35, 'away': 0.25}
        result = reconcile_cs_with_1x2(cs_dist, fair_1x2)
        assert abs(sum(result.values()) - 1.0) < 1e-9

        # Hitung marginal 1X2 hasil
        marg = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
        for (h, a), p in result.items():
            if h > a:
                marg['home'] += p
            elif h == a:
                marg['draw'] += p
            else:
                marg['away'] += p

        # Marginal hasil harus mendekati target
        assert abs(marg['home'] - 0.4) < 1e-6
        assert abs(marg['draw'] - 0.35) < 1e-6
        assert abs(marg['away'] - 0.25) < 1e-6

    def test_ipf_preserves_positive(self):
        cs_dist = {(1, 0): 0.5, (0, 0): 0.3, (0, 1): 0.2}
        fair_1x2 = {'home': 0.6, 'draw': 0.2, 'away': 0.2}
        result = reconcile_cs_with_1x2(cs_dist, fair_1x2)
        # Semua probabilitas harus positif
        for p in result.values():
            assert p >= 0
