import pytest
from src.python.giv_api import _init_julia
import pandas as pd
import os

# python3 -m pytest -v

@pytest.fixture(scope="session", autouse=True)
def init_julia_once():
    """
    Ensure that Julia is booted and GIV.jl is loaded exactly once
    before any of the tests run.
    """
    _init_julia()

@pytest.fixture(scope="session")
def simdata():
    """
    Load the simdata1.csv only once per test session.
    """
    here     = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(here, "../../examples/simdata1.csv")
    return pd.read_csv(csv_path)