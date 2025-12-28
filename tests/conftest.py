"""
Pytest configuration and shared fixtures.
"""

import pytest
import math


@pytest.fixture
def standard_params():
    """Standard at-the-money option parameters."""
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "q": 0.0,
    }


@pytest.fixture
def itm_call_params():
    """In-the-money call parameters."""
    return {
        "S": 110.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "q": 0.0,
    }


@pytest.fixture
def otm_put_params():
    """Out-of-the-money put parameters."""
    return {
        "S": 110.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "q": 0.0,
    }


@pytest.fixture
def with_dividend_params():
    """Parameters with non-zero dividend yield."""
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "q": 0.02,
    }
