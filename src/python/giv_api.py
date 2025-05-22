"""giv_api.py
========================================
Python bridge for **GIV.jl** using *juliacall*.
Place this file in ``src/python`` and import with

```python
from src.python.giv_api import giv, coef_vector, coef_dataframe
```"""

# one time julia in the enviroment:
# export PATH="$HOME/.julia/environments/pyjuliapkg/pyjuliapkg/install/bin:$PATH"

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from juliacall import Main as jl

# ---------------------------------------------------------------------------
# Paths & one‑time Julia boot
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parents[2]  # ~/GIV.jl repo root
_SRC_DIR = _PROJECT_DIR / "src"                       # contains GIV.jl
_julia_ready = False


def _init_julia() -> None:
    """Boot Julia, activate repo environment, import dependencies (once)."""
    global _julia_ready
    if _julia_ready:
        return

    jl.seval("using Pkg")
    jl.seval(f'Pkg.activate("{_PROJECT_DIR.as_posix()}")')
    jl.seval(f'push!(LOAD_PATH, "{_SRC_DIR.as_posix()}")')
    jl.seval("using DataFrames, StatsModels, Tables, GIV")

    _julia_ready = True

# ---------------------------------------------------------------------------
# Helper – convert guess dict
# ---------------------------------------------------------------------------

def _py_to_julia_guess(guess: dict):
    """Convert Python ``dict`` ➜ Julia ``Dict{String,Float64}``."""
    return jl.Dict([(str(k), float(v)) for k, v in guess.items()])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def giv(
    df: pd.DataFrame,
    formula: str,
    *,
    id: str,
    t: str,
    weight: Optional[str] = None,
    **kwargs: Any,
):
    """Estimate a GIV model from pandas data."""

    _init_julia()

    # --- Convert inputs ------------------------------------------------------
    jdf = jl.DataFrame(df)
    jformula = jl.seval(f"@formula({formula})")
    jid, jt = jl.Symbol(id), jl.Symbol(t)
    jweight = jl.Symbol(weight) if weight is not None else jl.nothing

    # --- Keyword tweaks ------------------------------------------------------
    if isinstance(kwargs.get("algorithm"), str):
        kwargs["algorithm"] = jl.Symbol(kwargs["algorithm"])
    if isinstance(kwargs.get("guess"), dict):
        kwargs["guess"] = _py_to_julia_guess(kwargs["guess"])

    # --- Call Julia ----------------------------------------------------------
    return jl.giv(jdf, jformula, jid, jt, jweight, **kwargs)

# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def coef_vector(model) -> np.ndarray:
    """Return ζ̂ as a NumPy array."""
    return np.asarray(jl.Vector(model.coef))


def coefficient_table(model) -> pd.DataFrame:
    """Get full statistical summary matching Julia's display"""
    _init_julia()
    
    # Get Julia's formatted coefficient table
    ct = jl.seval("coeftable")(model)
    
    # Extract components using Julia-side operations
    cols = jl.seval("""
    function getcols(ct)
        cols = [ct.cols[i] for i in 1:length(ct.cols)]  # Ensure list of vectors
        (; cols=cols, colnms=ct.colnms, rownms=ct.rownms)
    end
    """)(ct)
    
    # Convert to Python types
    colnames = list(cols.colnms)
    data = np.column_stack(cols.cols)  # Stack columns horizontally
    
    # Build DataFrame with proper types
    df = pd.DataFrame(data, columns=colnames)
    
    # Add term names if available
    if len(cols.rownms) > 0:
        df.insert(0, "Term", list(cols.rownms))
    
    # Convert p-values to float
    if "Pr(>|t|)" in df.columns:
        df["Pr(>|t|)"] = df["Pr(>|t|)"].astype(float)
    
    return df

# ---------------------------------------------------------------------------
# Self‑test (run `python src/python/giv_api.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n, T = 4, 6
    rng = np.random.default_rng(0)

    df_example = pd.DataFrame(
        {
            "id": np.repeat(np.arange(1, n + 1), T),
            "t": np.tile(np.arange(1, T + 1), n),
            "q": rng.standard_normal(n * T),
            "p": np.tile(rng.standard_normal(T), n),  # constant across ids per t
            "w": 1.0,  # constant weight column (required by scalar_search)
        }
    )

    model = giv(
        df_example,
        "q + id & endog(p) ~ 0",
        id="id",
        t="t",
        weight="w",
        algorithm="scalar_search",
        guess={"Aggregate": 2.0},
    )

    print("Estimated ζ̂:", coef_vector(model))
    print("\nCoefficient table:\n", coefficient_table(model).head())