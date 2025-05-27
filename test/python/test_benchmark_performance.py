# python3 -m pytest -p no:faulthandler test/python/test_benchmark_performance.py -v

import pytest
import numpy as np
import pandas as pd
import time
from src.python.giv_api import giv
from simulation import SimModel


def preprocess_dataframe(df, formula, id_col, t_col, weight_col, quiet=False):
    """Python version of Julia's preprocess_dataframe"""
    # Check balanced panel
    n_entities = df[id_col].nunique()
    n_periods = df[t_col].nunique()
    if len(df) != n_entities * n_periods:
        raise ValueError("Only balanced panels supported")

    # Check unique ID-time pairs
    if df.duplicated([id_col, t_col]).any():
        raise ValueError("Duplicate ID-time observations")

    # Sort and validate weights
    df = df.sort_values([t_col, id_col]).copy()
    if weight_col is not None and (df[weight_col] < 0).any():
        raise ValueError("Weights must be non-negative")

    return df


@pytest.fixture(scope="module")
def benchmark_data():
    # Set seed matching Julia's Random.seed!(6)
    np.random.seed(6)

    # Create simulated data (Python version)
    model = SimModel(
        T=400,
        N=100,
        usupplyshare=0.0,
        h=0.3,
        sigma_u_curv=0.2,
        zeta_s=0.0,
        NC=2,
        M=0.5,
        sigma_zeta=0.0
    )

    # Convert to DataFrame and create groups
    df = model.data.to_dataframe()
    df["group"] = (df["id"] - 1) % 10 + 1  # Match Julia's mod.(df.id, 10)

    # Preprocess like Julia version
    df = preprocess_dataframe(
        df,
        formula="q + group & endog(p) ~ id & (eta1 + eta2)",
        id_col="id",
        t_col="t",
        weight_col="absS",
        quiet=True
    )

    return df


def test_iv_estimation(benchmark_data):
    """Test IV algorithm through Julia bridge"""
    # Time estimation
    start = time.time()
    model = giv(
        benchmark_data,
        "q + group & endog(p) ~ id & (eta1 + eta2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="iv",
        guess={"group": [1.0] * 10}
    )
    elapsed = time.time() - start

    # Basic validation
    assert model.converged, "Model failed to converge"
    assert len(model.coef) >= 10, "Insufficient coefficients"
    assert not np.isnan(model.coef).any(), "NaN values in coefficients"
    assert np.all(np.diag(model.vcov) > 0), "Invalid variances"

    print(f"\nIV estimation completed in {elapsed:.2f} seconds")


def test_iv_vcov_estimation(benchmark_data):
    """Test IV VCOV algorithm through Julia bridge"""
    start = time.time()
    model = giv(
        benchmark_data,
        "q + group & endog(p) ~ id & (eta1 + eta2)",
        id="id",
        t="t",
        weight="absS",
        algorithm="iv_vcov",
        guess={"group": [1.0] * 10}
    )
    elapsed = time.time() - start

    # Additional VCOV checks
    eigenvalues = np.linalg.eigvalsh(model.vcov)
    assert np.all(eigenvalues > 0), "VCOV matrix not positive definite"

    print(f"IV VCOV estimation completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    # For direct execution without pytest
    df = benchmark_data()
    test_iv_estimation(df)
    test_iv_vcov_estimation(df)