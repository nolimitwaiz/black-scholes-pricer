"""
Unit tests for Black-Scholes pricing and Greeks calculations.

This module validates:
1. Known analytical solutions from textbooks
2. Put-call parity relationship
3. Edge cases (T→0, σ→0, deep ITM/OTM)
4. Greeks accuracy via finite-difference comparison
5. Monotonicity properties
"""

import pytest
import math
from src.core.black_scholes import (
    d1,
    d2,
    black_scholes_call,
    black_scholes_put,
    black_scholes_price,
    delta,
    gamma,
    vega,
    theta,
    rho,
    calculate_greeks,
)


# ===========================
# Known Solutions Tests
# ===========================


def test_atm_call_known_solution(standard_params):
    """
    Test against known solution from Hull's "Options, Futures, and Other Derivatives".
    Example: S=100, K=100, T=1, r=5%, σ=20%, q=0 → Call ≈ 10.4506
    """
    price = black_scholes_call(**standard_params)
    assert abs(price - 10.4506) < 0.01, f"Expected ~10.4506, got {price}"


def test_atm_put_known_solution(standard_params):
    """
    Known solution for ATM put.
    S=100, K=100, T=1, r=5%, σ=20%, q=0 → Put ≈ 5.5735
    """
    price = black_scholes_put(**standard_params)
    assert abs(price - 5.5735) < 0.01, f"Expected ~5.5735, got {price}"


def test_itm_call_known_solution():
    """
    Deep ITM call should be approximately intrinsic value + small time value.
    S=120, K=100, T=0.5, r=5%, σ=20% → Call ≈ 22.95
    """
    price = black_scholes_call(S=120, K=100, T=0.5, r=0.05, sigma=0.20, q=0.0)
    assert 22.5 < price < 23.5, f"Expected ~22.95, got {price}"


def test_otm_put_known_solution():
    """
    Deep OTM put should have small value.
    S=120, K=100, T=0.5, r=5%, σ=20% → Put ≈ 0.48
    """
    price = black_scholes_put(S=120, K=100, T=0.5, r=0.05, sigma=0.20, q=0.0)
    assert 0.3 < price < 0.7, f"Expected ~0.48, got {price}"


# ===========================
# Put-Call Parity Tests
# ===========================


def test_put_call_parity_atm(standard_params):
    """
    Verify put-call parity: C - P = S·e^(-qT) - K·e^(-rT)
    """
    S, K, T, r, sigma, q = (
        standard_params["S"],
        standard_params["K"],
        standard_params["T"],
        standard_params["r"],
        standard_params["sigma"],
        standard_params["q"],
    )

    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)

    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    assert abs(lhs - rhs) < 1e-6, f"Put-call parity violated: {lhs} != {rhs}"


def test_put_call_parity_with_dividend(with_dividend_params):
    """Test put-call parity with non-zero dividend yield."""
    S, K, T, r, sigma, q = (
        with_dividend_params["S"],
        with_dividend_params["K"],
        with_dividend_params["T"],
        with_dividend_params["r"],
        with_dividend_params["sigma"],
        with_dividend_params["q"],
    )

    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)

    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    assert abs(lhs - rhs) < 1e-6


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q",
    [
        (100, 100, 1.0, 0.05, 0.20, 0.0),  # ATM
        (110, 100, 1.0, 0.05, 0.20, 0.0),  # ITM call
        (90, 100, 1.0, 0.05, 0.20, 0.0),  # OTM call
        (100, 100, 0.25, 0.05, 0.30, 0.0),  # High vol, short expiry
        (100, 100, 2.0, 0.03, 0.15, 0.01),  # Long expiry with dividend
    ],
)
def test_put_call_parity_parametrized(S, K, T, r, sigma, q):
    """Test put-call parity across various parameter combinations."""
    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)

    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    assert abs(lhs - rhs) < 1e-5


# ===========================
# Edge Cases Tests
# ===========================


def test_call_at_expiration():
    """Call at expiration should equal intrinsic value."""
    S, K = 105.0, 100.0
    price = black_scholes_call(S, K, T=1e-8, r=0.05, sigma=0.20, q=0.0)
    intrinsic = max(S - K, 0.0)
    assert abs(price - intrinsic) < 0.01


def test_put_at_expiration():
    """Put at expiration should equal intrinsic value."""
    S, K = 95.0, 100.0
    price = black_scholes_put(S, K, T=1e-8, r=0.05, sigma=0.20, q=0.0)
    intrinsic = max(K - S, 0.0)
    assert abs(price - intrinsic) < 0.01


def test_zero_volatility_itm_call():
    """With zero vol, ITM call should be discounted forward minus strike."""
    S, K, T, r = 110.0, 100.0, 1.0, 0.05
    price = black_scholes_call(S, K, T, r, sigma=1e-8, q=0.0)
    expected = (S * math.exp(r * T) - K) * math.exp(-r * T)
    assert abs(price - expected) < 0.01


def test_zero_volatility_otm_call():
    """With zero vol, OTM call should be worthless."""
    S, K, T, r = 90.0, 100.0, 1.0, 0.05
    price = black_scholes_call(S, K, T, r, sigma=1e-8, q=0.0)
    assert price < 0.01


def test_deep_itm_call():
    """Deep ITM call should approximate intrinsic value."""
    S, K = 200.0, 100.0
    price = black_scholes_call(S, K, T=1.0, r=0.05, sigma=0.20, q=0.0)
    intrinsic = S * math.exp(-0.0 * 1.0) - K * math.exp(-0.05 * 1.0)
    # Should be close to intrinsic (within a few percent)
    assert abs(price - intrinsic) < 5.0


def test_deep_otm_call():
    """Deep OTM call should be nearly worthless."""
    S, K = 50.0, 100.0
    price = black_scholes_call(S, K, T=1.0, r=0.05, sigma=0.20, q=0.0)
    assert price < 1.0


# ===========================
# d1 and d2 Tests
# ===========================


def test_d1_d2_relationship(standard_params):
    """Verify d2 = d1 - σ√T."""
    S, K, T, r, sigma, q = (
        standard_params["S"],
        standard_params["K"],
        standard_params["T"],
        standard_params["r"],
        standard_params["sigma"],
        standard_params["q"],
    )

    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)

    expected_d2 = d1_val - sigma * math.sqrt(T)
    assert abs(d2_val - expected_d2) < 1e-10


def test_d1_symmetry():
    """
    d1 for S>K should have opposite sign to d1 for S<K (roughly).
    Actually, d1(S, K, ...) with S=120, K=100 should be positive.
    """
    d1_itm = d1(S=120, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    d1_otm = d1(S=80, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)

    assert d1_itm > 0
    assert d1_otm < 0


# ===========================
# Pricing Function Tests
# ===========================


def test_black_scholes_price_call(standard_params):
    """Test generic pricing function for call."""
    price_generic = black_scholes_price(**standard_params, option_type="call")
    price_call = black_scholes_call(**standard_params)
    assert abs(price_generic - price_call) < 1e-10


def test_black_scholes_price_put(standard_params):
    """Test generic pricing function for put."""
    price_generic = black_scholes_price(**standard_params, option_type="put")
    price_put = black_scholes_put(**standard_params)
    assert abs(price_generic - price_put) < 1e-10


def test_black_scholes_price_invalid_type(standard_params):
    """Test that invalid option_type raises ValueError."""
    with pytest.raises(ValueError):
        black_scholes_price(**standard_params, option_type="straddle")


# ===========================
# Input Validation Tests
# ===========================


def test_negative_spot_raises():
    """Negative spot price should raise ValueError."""
    with pytest.raises(ValueError):
        black_scholes_call(S=-100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)


def test_negative_strike_raises():
    """Negative strike should raise ValueError."""
    with pytest.raises(ValueError):
        black_scholes_call(S=100, K=-100, T=1.0, r=0.05, sigma=0.20, q=0.0)


def test_negative_time_raises():
    """Negative time should raise ValueError."""
    with pytest.raises(ValueError):
        black_scholes_call(S=100, K=100, T=-1.0, r=0.05, sigma=0.20, q=0.0)


def test_negative_volatility_raises():
    """Negative volatility should raise ValueError."""
    with pytest.raises(ValueError):
        black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=-0.20, q=0.0)


# ===========================
# Greeks Tests
# ===========================


def test_call_delta_range(standard_params):
    """Call delta should be in [0, 1]."""
    delta_val = delta(**standard_params, option_type="call")
    assert 0.0 <= delta_val <= 1.0


def test_put_delta_range(standard_params):
    """Put delta should be in [-1, 0]."""
    delta_val = delta(**standard_params, option_type="put")
    assert -1.0 <= delta_val <= 0.0


def test_gamma_always_positive(standard_params):
    """Gamma should always be non-negative."""
    gamma_val = gamma(**standard_params)
    assert gamma_val >= 0.0


def test_vega_always_positive(standard_params):
    """Vega should always be non-negative."""
    vega_val = vega(**standard_params)
    assert vega_val >= 0.0


def test_call_theta_typically_negative(standard_params):
    """Long call theta is usually negative (time decay)."""
    theta_val = theta(**standard_params, option_type="call")
    # For ATM call with r > q, theta is usually negative
    assert theta_val < 0.0


def test_call_rho_positive(standard_params):
    """Call rho should be positive (calls benefit from higher rates)."""
    rho_val = rho(**standard_params, option_type="call")
    assert rho_val > 0.0


def test_put_rho_negative(standard_params):
    """Put rho should be negative."""
    rho_val = rho(**standard_params, option_type="put")
    assert rho_val < 0.0


# ===========================
# Greeks Finite-Difference Validation
# ===========================


def test_delta_finite_difference_call(standard_params):
    """Validate call delta against finite-difference approximation."""
    S = standard_params["S"]
    h = 0.01  # Small perturbation

    delta_analytical = delta(**standard_params, option_type="call")

    # Finite difference: (f(S+h) - f(S-h)) / (2h)
    params_up = {**standard_params, "S": S + h}
    params_down = {**standard_params, "S": S - h}

    price_up = black_scholes_call(**params_up)
    price_down = black_scholes_call(**params_down)

    delta_numerical = (price_up - price_down) / (2 * h)

    assert abs(delta_analytical - delta_numerical) < 1e-4


def test_gamma_finite_difference(standard_params):
    """Validate gamma against finite-difference approximation."""
    S = standard_params["S"]
    h = 0.01

    gamma_analytical = gamma(**standard_params)

    # Gamma = ∂²V/∂S² ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
    params_up = {**standard_params, "S": S + h}
    params_down = {**standard_params, "S": S - h}

    price = black_scholes_call(**standard_params)
    price_up = black_scholes_call(**params_up)
    price_down = black_scholes_call(**params_down)

    gamma_numerical = (price_up - 2 * price + price_down) / (h * h)

    assert abs(gamma_analytical - gamma_numerical) < 1e-3


def test_vega_finite_difference(standard_params):
    """Validate vega against finite-difference approximation."""
    sigma = standard_params["sigma"]
    h = 0.001  # 0.1% volatility change

    vega_analytical = vega(**standard_params)

    params_up = {**standard_params, "sigma": sigma + h}
    params_down = {**standard_params, "sigma": sigma - h}

    price_up = black_scholes_call(**params_up)
    price_down = black_scholes_call(**params_down)

    # Vega reported per 1% change, so multiply by 100
    vega_numerical = (price_up - price_down) / (2 * h)

    # Vega analytical is per 1% (0.01) change, numerical is per h change
    # Vega = ∂V/∂σ, but we report per 1% so we don't need scaling here
    assert abs(vega_analytical - vega_numerical) < 0.01


def test_theta_finite_difference(standard_params):
    """Validate theta against finite-difference approximation."""
    T = standard_params["T"]
    h = 1.0 / 365.0  # 1 day

    theta_analytical = theta(**standard_params, option_type="call")

    params_down = {**standard_params, "T": T - h}

    price = black_scholes_call(**standard_params)
    price_down = black_scholes_call(**params_down)

    # Theta analytical is per day, so we compare to daily change
    # Theta = change in value per day = (V(T-h) - V(T)) / (days)
    # where h is 1 day, so theta_numerical = price_down - price
    theta_numerical = price_down - price

    # Allow larger tolerance for theta as it's more sensitive
    assert abs(theta_analytical - theta_numerical) < 0.01


def test_rho_finite_difference(standard_params):
    """Validate rho against finite-difference approximation."""
    r = standard_params["r"]
    h = 0.01  # 1% change to match rho definition

    rho_analytical = rho(**standard_params, option_type="call")

    params_up = {**standard_params, "r": r + h}
    params_down = {**standard_params, "r": r - h}

    price_up = black_scholes_call(**params_up)
    price_down = black_scholes_call(**params_down)

    # Rho is reported per 1% change, so h = 0.01
    # ∂V/∂r ≈ (V(r+0.01) - V(r-0.01)) / (2*0.01) = (price_up - price_down) / 0.02
    # This gives change per 1% rate change, matching rho definition
    rho_numerical = (price_up - price_down) / (2 * h)

    assert abs(rho_analytical - rho_numerical) < 0.5


# ===========================
# calculate_greeks() Tests
# ===========================


def test_calculate_greeks_returns_all(standard_params):
    """Test that calculate_greeks returns all five Greeks."""
    greeks = calculate_greeks(**standard_params, option_type="call")

    assert hasattr(greeks, "delta")
    assert hasattr(greeks, "gamma")
    assert hasattr(greeks, "vega")
    assert hasattr(greeks, "theta")
    assert hasattr(greeks, "rho")


def test_calculate_greeks_consistency(standard_params):
    """Verify calculate_greeks matches individual functions."""
    greeks = calculate_greeks(**standard_params, option_type="call")

    assert abs(greeks.delta - delta(**standard_params, option_type="call")) < 1e-10
    assert abs(greeks.gamma - gamma(**standard_params)) < 1e-10
    assert abs(greeks.vega - vega(**standard_params)) < 1e-10
    assert abs(greeks.theta - theta(**standard_params, option_type="call")) < 1e-10
    assert abs(greeks.rho - rho(**standard_params, option_type="call")) < 1e-10


# ===========================
# Monotonicity Tests
# ===========================


def test_call_price_increases_with_spot():
    """Call price should increase as spot price increases."""
    base_price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    higher_price = black_scholes_call(S=105, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    assert higher_price > base_price


def test_call_price_decreases_with_strike():
    """Call price should decrease as strike increases."""
    base_price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    lower_price = black_scholes_call(S=100, K=105, T=1.0, r=0.05, sigma=0.20, q=0.0)
    assert lower_price < base_price


def test_call_price_increases_with_volatility():
    """Call price should increase with volatility."""
    base_price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    higher_price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.25, q=0.0)
    assert higher_price > base_price


def test_call_price_increases_with_time():
    """Call price should increase with time to expiration (for r > q)."""
    base_price = black_scholes_call(S=100, K=100, T=0.5, r=0.05, sigma=0.20, q=0.0)
    higher_price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    assert higher_price > base_price


def test_put_price_decreases_with_spot():
    """Put price should decrease as spot price increases."""
    base_price = black_scholes_put(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    lower_price = black_scholes_put(S=105, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    assert lower_price < base_price


def test_put_price_increases_with_strike():
    """Put price should increase as strike increases."""
    base_price = black_scholes_put(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    higher_price = black_scholes_put(S=100, K=105, T=1.0, r=0.05, sigma=0.20, q=0.0)
    assert higher_price > base_price
