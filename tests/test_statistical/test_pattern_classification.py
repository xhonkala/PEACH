import numpy as np
import pytest

from peach._core.utils.pattern_classification import (
    classify_single_feature,
    classify_all_features,
)


class TestClassifySingleFeature:
    def test_flat_nonsignificant(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.3,
            f_pvalue_fdr=0.10,
        )
        assert result["pattern"] == "flat"
        assert result["details"]["reason"] == "nonsignificant"

    def test_flat_nan_r2(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=float("nan"),
            f_pvalue_fdr=0.001,
        )
        assert result["pattern"] == "flat"
        assert result["details"]["reason"] == "nan_r2"

    def test_flat_nan_pvalue(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.5,
            f_pvalue_fdr=float("nan"),
        )
        assert result["pattern"] == "flat"
        assert result["details"]["reason"] == "nan_pvalue"

    def test_exclusive_dominant_positive(self):
        result = classify_single_feature(
            vertex_betas=np.array([10.0, 0.5, 0.3]),
            r2=0.8,
            f_pvalue_fdr=0.001,
        )
        assert result["pattern"] == "archetype-exclusive"
        assert result["details"]["dominant_archetype"] == 0

    def test_exclusive_with_negative_betas(self):
        result = classify_single_feature(
            vertex_betas=np.array([-10.0, 0.5, 0.3]),
            r2=0.8,
            f_pvalue_fdr=0.001,
        )
        assert result["pattern"] == "archetype-exclusive"
        assert result["details"]["dominant_archetype"] == 0

    def test_exclusive_dominant_not_largest_raw(self):
        # Dominant by absolute magnitude even though negative
        result = classify_single_feature(
            vertex_betas=np.array([1.0, -12.0, 0.5]),
            r2=0.7,
            f_pvalue_fdr=0.001,
        )
        assert result["pattern"] == "archetype-exclusive"
        assert result["details"]["dominant_archetype"] == 1

    def test_not_exclusive_similar_magnitudes(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.6,
            f_pvalue_fdr=0.001,
        )
        assert result["pattern"] != "archetype-exclusive"

    def test_interaction_significant(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.6,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=0.01,
        )
        assert result["pattern"] == "interaction"

    def test_interaction_not_significant_falls_to_structured(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.6,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=0.10,
        )
        assert result["pattern"] == "structured"

    def test_interaction_none_falls_to_structured(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.6,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=None,
        )
        assert result["pattern"] == "structured"

    def test_structured_fallback_has_dominant_archetype(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.6,
            f_pvalue_fdr=0.001,
        )
        assert result["pattern"] == "structured"
        assert result["details"]["dominant_archetype"] == 0

    def test_custom_fdr_threshold(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.6,
            f_pvalue_fdr=0.08,
            fdr_threshold=0.10,
        )
        assert result["pattern"] != "flat"

    def test_custom_exclusive_ratio(self):
        # With ratio=1.5, 10/5=2.0 >= 1.5 -> exclusive
        result = classify_single_feature(
            vertex_betas=np.array([10.0, 5.0, 1.0]),
            r2=0.7,
            f_pvalue_fdr=0.001,
            exclusive_ratio=1.5,
        )
        assert result["pattern"] == "archetype-exclusive"

        # With default ratio=2.0, 10/5=2.0 >= 2.0 -> exclusive
        result2 = classify_single_feature(
            vertex_betas=np.array([10.0, 5.0, 1.0]),
            r2=0.7,
            f_pvalue_fdr=0.001,
            exclusive_ratio=2.0,
        )
        assert result2["pattern"] == "archetype-exclusive"

        # With ratio=3.0, 10/5=2.0 < 3.0 -> not exclusive
        result3 = classify_single_feature(
            vertex_betas=np.array([10.0, 5.0, 1.0]),
            r2=0.7,
            f_pvalue_fdr=0.001,
            exclusive_ratio=3.0,
        )
        assert result3["pattern"] != "archetype-exclusive"

    def test_r2_stored_in_result(self):
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.42,
            f_pvalue_fdr=0.001,
        )
        assert result["r2"] == 0.42

    def test_exclusive_beats_interaction(self):
        # Exclusive check comes before interaction check
        result = classify_single_feature(
            vertex_betas=np.array([20.0, 0.5, 0.3]),
            r2=0.9,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=0.001,
        )
        assert result["pattern"] == "archetype-exclusive"


class TestClassifyAllFeatures:
    def test_batch_classification(self):
        vertex_coefficients = np.array([
            [10.0, 0.5, 0.3],   # exclusive
            [5.0, 5.0, 5.0],    # will be flat if pvalue is high, or structured
            [5.0, 4.0, 3.0],    # structured
            [5.0, 4.0, 3.0],    # interaction (if interaction p is significant)
        ])
        r_squared = np.array([0.8, 0.1, 0.6, 0.6])
        f_pvalue_fdr = np.array([0.001, 0.80, 0.001, 0.001])
        interaction_fdr = np.array([0.50, 0.50, 0.50, 0.01])

        results = classify_all_features(
            vertex_coefficients=vertex_coefficients,
            r_squared=r_squared,
            f_pvalue_fdr=f_pvalue_fdr,
            interaction_f_pvalue_fdr=interaction_fdr,
        )

        assert len(results) == 4
        assert results[0]["pattern"] == "archetype-exclusive"
        assert results[1]["pattern"] == "flat"
        assert results[2]["pattern"] == "structured"
        assert results[3]["pattern"] == "interaction"

    def test_batch_no_interaction(self):
        vertex_coefficients = np.array([
            [10.0, 0.5, 0.3],
            [5.0, 4.0, 3.0],
        ])
        r_squared = np.array([0.8, 0.6])
        f_pvalue_fdr = np.array([0.001, 0.001])

        results = classify_all_features(
            vertex_coefficients=vertex_coefficients,
            r_squared=r_squared,
            f_pvalue_fdr=f_pvalue_fdr,
            interaction_f_pvalue_fdr=None,
        )

        assert len(results) == 2
        assert results[0]["pattern"] == "archetype-exclusive"
        assert results[1]["pattern"] == "structured"

    def test_nan_handling_in_batch(self):
        vertex_coefficients = np.array([
            [10.0, 0.5, 0.3],
            [5.0, 4.0, 3.0],
        ])
        r_squared = np.array([float("nan"), 0.6])
        f_pvalue_fdr = np.array([0.001, float("nan")])

        results = classify_all_features(
            vertex_coefficients=vertex_coefficients,
            r_squared=r_squared,
            f_pvalue_fdr=f_pvalue_fdr,
        )

        assert results[0]["pattern"] == "flat"
        assert results[0]["details"]["reason"] == "nan_r2"
        assert results[1]["pattern"] == "flat"
        assert results[1]["details"]["reason"] == "nan_pvalue"


class TestSEAwareClassification:
    def test_exclusive_requires_significance_when_ses_provided(self):
        """A large but noisy coefficient should NOT be classified as exclusive."""
        result = classify_single_feature(
            vertex_betas=np.array([10.0, 1.0, 0.5]),
            r2=0.5,
            f_pvalue_fdr=0.001,
            vertex_ses=np.array([50.0, 0.1, 0.01]),  # SE on dominant is 50!
        )
        assert result["pattern"] != "archetype-exclusive", (
            "Should not classify as exclusive when dominant coefficient SE is huge"
        )

    def test_exclusive_passes_when_se_is_small(self):
        """A large coefficient with small SE should still be exclusive."""
        result = classify_single_feature(
            vertex_betas=np.array([10.0, 1.0, 0.5]),
            r2=0.5,
            f_pvalue_fdr=0.001,
            vertex_ses=np.array([0.1, 0.1, 0.01]),  # SE is small
        )
        assert result["pattern"] == "archetype-exclusive"

    def test_exclusive_without_ses_unchanged(self):
        """Without SEs, exclusive classification works as before."""
        result = classify_single_feature(
            vertex_betas=np.array([10.0, 1.0, 0.5]),
            r2=0.5,
            f_pvalue_fdr=0.001,
            vertex_ses=None,
        )
        assert result["pattern"] == "archetype-exclusive"

    def test_classify_all_passes_ses_through(self):
        """classify_all_features should pass vertex_ses to each feature."""
        vertex_coefficients = np.array([
            [10.0, 0.5, 0.3],   # would be exclusive, but SE is huge
            [10.0, 0.5, 0.3],   # SE is small -> exclusive
        ])
        r_squared = np.array([0.8, 0.8])
        f_pvalue_fdr = np.array([0.001, 0.001])
        vertex_ses = np.array([
            [50.0, 0.1, 0.01],  # huge SE on dominant
            [0.1, 0.1, 0.01],   # small SE
        ])

        results = classify_all_features(
            vertex_coefficients=vertex_coefficients,
            r_squared=r_squared,
            f_pvalue_fdr=f_pvalue_fdr,
            vertex_ses=vertex_ses,
        )

        assert results[0]["pattern"] != "archetype-exclusive"
        assert results[1]["pattern"] == "archetype-exclusive"
