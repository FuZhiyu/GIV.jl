# File: src/python/giv_api.py

"""
giv_api.py
========================================
Python bridge for **GIV.jl** using *juliacall*.
Place this file in ``src/python`` and import with

    from src.python.giv_api import giv, GIVModel
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from juliacall import Main as jl

# ---------------------------------------------------------------------------
# Paths & one-time Julia boot
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parents[2]   # ~/GIV.jl repo root
_SRC_DIR     = _PROJECT_DIR / "src"                  # contains GIV.jl
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

_init_julia()

# ---------------------------------------------------------------------------
# Helper – convert guess dict
# ---------------------------------------------------------------------------
def _py_to_julia_guess(guess: dict) -> Any:
    """Handle nested guesses for categorical terms"""
    jl_dict = jl.Dict()
    for term, value in guess.items():
        if isinstance(value, dict):
            jl_subdict = jl.Dict()
            for k, v in value.items():
                jl_subdict[str(k)] = float(v)
            jl_dict[term] = jl_subdict
        elif isinstance(value, (list, np.ndarray)):
            jl_dict[term] = jl.convert(jl.Vector[jl.Float64],
                                        [float(x) for x in value])
        else:
            jl_dict[term] = float(value)
    return jl_dict

# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------
class GIVModel:
    """Python-native wrapper for Julia GIV results"""

    def __init__(self, jl_model: Any):
        self._jl_model = jl_model

        self.coef              = np.asarray(jl_model.coef)
        self.vcov              = np.asarray(jl_model.vcov)
        self.factor_coef       = np.asarray(jl_model.factor_coef)
        self.factor_vcov       = np.asarray(jl_model.factor_vcov)
        agg = jl_model.agg_coef
        try:
            self.agg_coef = float(agg)
        except (TypeError, ValueError):
            self.agg_coef = np.asarray(agg)
        self.formula           = str(jl_model.formula)
        self.price_factor_coef = np.asarray(jl_model.price_factor_coef)
        self.residual_variance = np.asarray(jl_model.residual_variance)
        self.responsename      = str(jl_model.responsename)
        self.endogname         = str(jl_model.endogname)
        self.coefnames         = list(jl_model.coefnames)
        self.factor_coefnames  = list(jl_model.factor_coefnames)
        self.idvar             = str(jl_model.idvar)
        self.tvar              = str(jl_model.tvar)
        wv = jl_model.weightvar
        self.weightvar         = str(wv) if wv is not jl.nothing else None
        self.exclude_pairs     = [(p.first, p.second)
                                  for p in jl_model.exclude_pairs]
        self.converged         = bool(jl_model.converged)
        self.N                 = int(jl_model.N)
        self.T                 = int(jl_model.T)
        self.nobs              = int(jl_model.nobs)
        self.dof               = int(jl_model.dof)
        self.dof_residual      = int(jl_model.dof_residual)

        # Helper to extract Julia DataFrame columns
        get_col = jl.seval("(df, col) -> df[!, Symbol(col)]")

        # coefdf
        j_coefdf    = jl_model.coefdf
        j_coef_names = jl.seval("names")(j_coefdf)
        coefdf_dict = {
            str(nm): np.asarray(get_col(j_coefdf, nm))
            for nm in j_coef_names
        }
        self.coefdf = pd.DataFrame(coefdf_dict)

        # full df (if saved)
        j_df = jl_model.df
        if j_df is not jl.nothing:
            j_names = jl.seval("names")(j_df)
            df_dict = {
                str(nm): np.asarray(get_col(j_df, nm))
                for nm in j_names
            }
            self.df = pd.DataFrame(df_dict)
        else:
            self.df = None

    def coefficient_table(self) -> pd.DataFrame:
        """Return the full coefficient table as DataFrame"""
        return coefficient_table(self._jl_model)

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
) -> GIVModel:
    """Estimate a GIV model from pandas data."""

    jdf      = jl.DataFrame(df)
    jformula = jl.seval(f"@formula({formula})")
    jid      = jl.Symbol(id)
    jt       = jl.Symbol(t)
    jweight  = jl.Symbol(weight) if weight else jl.nothing

    # Handle keyword arguments
    if isinstance(kwargs.get("algorithm"), str):
        kwargs["algorithm"] = jl.Symbol(kwargs["algorithm"])
    if isinstance(kwargs.get("guess"), dict):
        kwargs["guess"] = _py_to_julia_guess(kwargs["guess"])

    return GIVModel(jl.giv(jdf, jformula, jid, jt, jweight, **kwargs))

# ---------------------------------------------------------------------------
# Coefficient Table Generator
# ---------------------------------------------------------------------------
def coefficient_table(jl_model: Any) -> pd.DataFrame:
    """Get full statistical summary from Julia model"""
    ct = jl.seval("GIV.coeftable")(jl_model)
    cols = jl.seval("""
    function getcols(ct)
        cols = [ct.cols[i] for i in 1:length(ct.cols)]
        (; cols=cols, colnms=ct.colnms, rownms=ct.rownms)
    end
    """)(ct)

    df = pd.DataFrame(
        np.column_stack(cols.cols),
        columns=list(cols.colnms)
    )
    if cols.rownms:
        df.insert(0, "Term", list(cols.rownms))
    if "Pr(>|t|)" in df.columns:
        df["Pr(>|t|)"] = df["Pr(>|t|)"].astype(float)
    return df

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n, T = 4, 6
    import numpy as _np
    rng = _np.random.default_rng(0)

    df_example = pd.DataFrame({
        "id": _np.repeat(_np.arange(1, n+1), T),
        "t":  _np.tile(_np.arange(1, T+1),   n),
        "q":  rng.standard_normal(n*T),
        "p":  _np.tile(rng.standard_normal(T), n),
        "w":  1.0
    })

    model = giv(
        df_example,
        "q + id & endog(p) ~ 0",
        id="id", t="t", weight="w",
        algorithm="scalar_search",
        guess={"Aggregate": 2.0}
    )

    print("Estimated coefficients:", model.coef)
    print("\nCoefficient table:")
    print(model.coefficient_table().head())
