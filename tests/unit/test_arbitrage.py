"""Unit tests for arbitrage diagnostics."""

import pytest
from src.diagnostics.arbitrage import (
    check_price_bounds,
    check_put_call_parity,
    check_strike_monotonicity,
    check_butterfly_arbitrage,
    check_calendar_spread,
    OptionData,
)


def test_price_bounds_valid():
    """Valid prices should pass bounds check."""
    result = check_price_bounds(
        call_price=10.0, put_price=5.0, S=100, K=100, T=1.0, r=0.05, q=0.0
    )
    assert result.is_valid


def test_put_call_parity_valid():
    """Valid prices should satisfy put-call parity."""
    # Use consistent prices that satisfy parity
    import math
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    call_price = 10.0
    put_price = call_price - (S * math.exp(-q*T) - K * math.exp(-r*T))

    result = check_put_call_parity(
        call_price=call_price, put_price=put_price, S=S, K=K, T=T, r=r, q=q
    )
    assert result.is_valid


def test_strike_monotonicity_valid():
    """Properly ordered prices should pass monotonicity check."""
    calls = [
        OptionData(strike=95, price=12.0, option_type="call"),
        OptionData(strike=100, price=10.0, option_type="call"),
        OptionData(strike=105, price=8.0, option_type="call"),
    ]
    result = check_strike_monotonicity(calls)
    assert result.is_valid


def test_butterfly_no_arbitrage():
    """Valid butterfly spread should pass."""
    result = check_butterfly_arbitrage(
        K1=95, K2=100, K3=105, C1=12.0, C2=10.0, C3=8.0
    )
    assert result.is_valid


def test_calendar_spread_valid():
    """Far option should be >= near option."""
    result = check_calendar_spread(
        near_price=5.0, far_price=7.0, option_type="call"
    )
    assert result.is_valid
