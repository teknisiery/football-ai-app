# tests/test_probability_fusion.py
import pytest
from services.probability_fusion import (
    normalize_score_distribution,
    fuse_score_distributions,
    marginalize,
    prob_over,
    prob_under,
    apply_btts_evidence,
)


class TestNormalizeScoreDistribution:
    def test_basic_normalization(self):
        dist = {(1, 0): 0.2, (0, 1): 0.3, (1, 1): 0.5}
        result = normalize_score_distribution(dist)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_already_normalized(self):
        dist = {(1, 0): 0.4, (0, 1): 0.6}
        result = normalize_score_distribution(dist)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_empty_dist(self):
        result = normalize_score_distribution({})
        assert result == {}


class TestFuseScoreDistributions:
    def test_two_distributions_equal_weight(self):
        d1 = {(0, 0): 0.5, (1, 0): 0.5}
        d2 = {(0, 0): 0.8, (1, 0): 0.2}
        fused = fuse_score_distributions([d1, d2], [1.0, 1.0])
        # Logarithmic pooling: geometric mean
        # (0,0): exp(0.5*ln0.5 + 0.5*ln0.8) = exp(-0.458) ≈ 0.632
        # (1,0): exp(0.5*ln0.5 + 0.5*ln0.2) = exp(-1.151) ≈ 0.316
        # After normalization: 0.667, 0.333
        assert abs(fused[(0, 0)] - 0.6667) < 0.01
        assert abs(fused[(1, 0)] - 0.3333) < 0.01

    def test_weighted_fusion(self):
        d1 = {(0, 0): 0.5, (1, 0): 0.5}
        d2 = {(0, 0): 0.8, (1, 0): 0.2}
        fused = fuse_score_distributions([d1, d2], [0.8, 0.2])
        assert abs(sum(fused.values()) - 1.0) < 1e-9

    def test_three_distributions(self):
        d1 = {(0, 0): 0.6, (1, 0): 0.4}
        d2 = {(0, 0): 0.7, (1, 0): 0.3}
        d3 = {(0, 0): 0.5, (1, 0): 0.5}
        fused = fuse_score_distributions([d1, d2, d3], [1.0, 1.0, 1.0])
        assert abs(sum(fused.values()) - 1.0) < 1e-9

    def test_missing_scores(self):
        d1 = {(0, 0): 0.5, (1, 0): 0.5}
        d2 = {(0, 0): 0.8, (2, 0): 0.2}  # (1,0) tidak ada
        fused = fuse_score_distributions([d1, d2], [1.0, 1.0])
        # (1,0) dari d2 akan menggunakan EPS
        assert (1, 0) in fused
        assert fused[(1, 0)] > 0
        assert (2, 0) in fused

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            fuse_score_distributions([], [])

    def test_zero_weights(self):
        d1 = {(0, 0): 0.5, (1, 0): 0.5}
        d2 = {(0, 0): 0.8, (1, 0): 0.2}
        with pytest.raises(ValueError):
            fuse_score_distributions([d1, d2], [0.0, 0.0])


class TestMarginalize:
    def test_home_marginal(self):
        dist = {(0, 0): 0.2, (1, 0): 0.3, (0, 1): 0.4, (1, 1): 0.1}
        result = marginalize(dist, 'home')
        assert abs(result['0'] - 0.6) < 1e-9  # (0,0)+(0,1)
        assert abs(result['1'] - 0.4) < 1e-9  # (1,0)+(1,1)

    def test_1x2_marginal(self):
        dist = {(1, 0): 0.3, (0, 0): 0.2, (0, 1): 0.5}
        result = marginalize(dist, '1x2')
        assert abs(result['home'] - 0.3) < 1e-9
        assert abs(result['draw'] - 0.2) < 1e-9
        assert abs(result['away'] - 0.5) < 1e-9

    def test_btts_marginal(self):
        dist = {(0, 0): 0.2, (1, 1): 0.3, (1, 0): 0.5}
        result = marginalize(dist, 'btts')
        assert abs(result['yes'] - 0.3) < 1e-9
        assert abs(result['no'] - 0.7) < 1e-9

    def test_over_marginal_bulat(self):
        dist = {(0, 0): 0.1, (1, 0): 0.2, (1, 1): 0.3, (2, 1): 0.4}  # total: 0,1,2,3
        result = marginalize(dist, 'over', line=2.5)
        assert abs(result['over'] - 0.4) < 1e-9  # hanya 2:1 (total 3)
        assert abs(result['under'] - 0.6) < 1e-9  # 0:0,1:0,1:1
        assert result['push'] == 0.0

    def test_over_marginal_bulat_push(self):
        dist = {(0, 0): 0.1, (1, 0): 0.2, (1, 1): 0.3, (2, 0): 0.4}  # total:0,1,2,2
        result = marginalize(dist, 'over', line=2.0)
        assert abs(result['over'] - 0.0) < 1e-9
        assert abs(result['under'] - 0.3) < 1e-9
        assert abs(result['push'] - 0.7) < 1e-9


class TestProbOverUnder:
    def test_prob_over_bulat(self):
        dist = {(0, 0): 0.1, (1, 0): 0.2, (1, 1): 0.3, (2, 0): 0.4}
        # over >2.0: 0, under <2.0: 0.3, push=2.0: 0.7
        p_over = prob_over(dist, 2.0)
        assert abs(p_over - 0.35) < 1e-9  # 0 + 0.5*0.7

    def test_prob_under_bulat(self):
        dist = {(0, 0): 0.1, (1, 0): 0.2, (1, 1): 0.3, (2, 0): 0.4}
        p_under = prob_under(dist, 2.0)
        assert abs(p_under - 0.65) < 1e-9  # 0.3 + 0.5*0.7

    def test_prob_over_quarter(self):
        dist = {(0, 0): 0.1, (1, 0): 0.2, (1, 1): 0.3, (2, 1): 0.4}  # total:0,1,2,3
        p_over = prob_over(dist, 2.25)
        # Untuk 2.25, push tidak relevan karena quarter tidak ada push di sini (line 2.25 tidak mungkin total=2.25)
        # Jadi over hanya >2.25 -> total 3: 0.4
        assert abs(p_over - 0.4) < 1e-9

    def test_prob_over_quarter_no_push(self):
        dist = {(0, 0): 0.1, (1, 1): 0.2, (2, 0): 0.3, (3, 0): 0.4}
        p_over = prob_over(dist, 2.5)
        assert abs(p_over - 0.4) < 1e-9  # >2.5 hanya 3:0


class TestApplyBttsEvidence:
    def test_apply_btts_yes(self):
        dist = {(0, 0): 0.2, (1, 1): 0.3, (1, 0): 0.5}
        result = apply_btts_evidence(dist, 0.7, weight=0.3)
        # BTTS YES prob 0.7
        # (0,0): tidak btts -> factor = (1-0.7)^0.3 = 0.3^0.3 ≈ 0.696
        # (1,1): btts -> factor = 0.7^0.3 ≈ 0.891
        # (1,0): tidak btts -> factor = 0.3^0.3 ≈ 0.696
        # hasil: (0,0): 0.139, (1,1): 0.267, (1,0): 0.348 -> setelah normalisasi: 0.184, 0.355, 0.461
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert result[(1, 1)] > result[(0, 0)]

    def test_apply_btts_no(self):
        dist = {(0, 0): 0.2, (1, 1): 0.3, (1, 0): 0.5}
        result = apply_btts_evidence(dist, 0.3, weight=0.3)
        # BTTS YES prob 0.3 (artinya prob NO tinggi)
        # (1,1) akan turun, (0,0) dan (1,0) akan naik
        assert result[(1, 1)] < 0.3  # turun dari 0.3

    def test_invalid_btts_prob(self):
        dist = {(0, 0): 1.0}
        with pytest.raises(ValueError):
            apply_btts_evidence(dist, 0.0)
        with pytest.raises(ValueError):
            apply_btts_evidence(dist, 1.0)

    def test_invalid_weight(self):
        dist = {(0, 0): 1.0}
        with pytest.raises(ValueError):
            apply_btts_evidence(dist, 0.5, weight=0.0)
        with pytest.raises(ValueError):
            apply_btts_evidence(dist, 0.5, weight=1.0)
