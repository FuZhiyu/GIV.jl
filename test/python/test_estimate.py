# python3 -m pytest -p no:faulthandler test/python/test_estimate.py -v

import pytest
import numpy as np
from src.python.giv_api import giv

def assert_allclose(actual, expected, atol=1e-4):
    np.testing.assert_allclose(actual, expected, atol=atol)


def test_homogeneous_elasticity(simdata):
    """Test basic estimation with scalar search algorithm"""
    model = giv(
        simdata,
        "q + endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="scalar_search",
        guess={"Aggregate": 2.0}
    )

    # Test coefficient values
    expected_coef = [2.5341730 / 2]  # Adjusted from Julia output
    assert_allclose(model.coef, expected_coef, atol=1e-4)

    # Test standard errors
    expected_se = [0.2407 / 2]
    assert_allclose(np.sqrt(np.diag(model.vcov)), expected_se, atol=1e-4)


def test_algorithm_equivalence(simdata):
    """Verify different algorithms produce consistent results"""
    # IV algorithm
    model_iv = giv(
        simdata,
        "q + endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="iv",
        guess={"Constant": 1.0}
    )

    # IV-VCOV algorithm
    model_vcov = giv(
        simdata,
        "q + endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="iv_vcov",
        guess={"Constant": 1.0}
    )

    # Debiased OLS algorithm
    model_ols = giv(
        simdata,
        "q + endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="debiased_ols",
        guess={"Constant": 1.0}
    )

    # Cross-algorithm validation
    assert_allclose(model_iv.coef, model_vcov.coef, atol=1e-6)
    assert_allclose(model_iv.coef, model_ols.coef, atol=1e-6)


def test_heterogeneous_elasticity(simdata):
    """Test models with entity-specific elasticities"""
    model = giv(
        simdata,
        "q + id & endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="scalar_search",
        guess={"Aggregate": 2.5}
    )

    expected_coef = [1.59636, 1.657, 1.29643, 3.33497, 0.58443]
    assert_allclose(model.coef, expected_coef, atol=1e-4)

    expected_se = [1.7824, 0.4825, 0.3911, 0.3846, 0.1732]
    assert_allclose(np.sqrt(np.diag(model.vcov)), expected_se, atol=1e-4)


def test_excluded_entities(simdata):
    """Test estimation with excluded entities across algorithms"""
    subdf = simdata[simdata["id"] > 1].copy()

    # Get number of remaining entities for guess initialization
    n_entities = subdf["id"].nunique()

    # Original scalar search test
    model_scalar = giv(
        subdf,
        "q + id & endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="scalar_search",
        guess={"Aggregate": 2.0}
    )
    expected_coef_scalar = [1.9772, 1.4518, 3.4499, 0.7464]
    assert_allclose(model_scalar.coef, expected_coef_scalar, atol=1e-4)

    # 1. Test debiased OLS consistency
    model_debiased = giv(
        subdf,
        "q + id & endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="debiased_ols",
        guess={"id": [1.0] * n_entities}
    )
    assert_allclose(model_debiased.coef, model_scalar.coef, atol=1e-6)

    # 2. Test IV algorithm specific values
    model_iv = giv(
        subdf,
        "q + id & endog(p) ~ id & (η1 + η2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="iv",
        guess={"id": [1.0] * n_entities}
    )
    expected_coef_iv = [1.0442, 0.9967, 4.2707, 0.7597]
    assert_allclose(model_iv.coef, expected_coef_iv, atol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])