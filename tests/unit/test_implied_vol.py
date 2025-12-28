"""
Unit tests for implied volatility solver.

This module validates:
1. Round-trip accuracy (solve IV from synthetic prices)
2. Arbitrage bounds validation
3. Convergence behavior for various scenarios
4. Newton-Raphson vs Brent fallback logic
5. Initial guess generation
"""

import pytest
import math
from src.core.black_scholes import black_scholes_call, black_scholes_put
from src.solvers.implied_vol import (
    implied_volatility,
    implied_volatility_vectorized,
    brenner_subrahmanyam_approximation,
    get_initial_guess,
    validate_arbitrage_bounds,
)
from src.solvers.newton_raphson import newton_raphson_iv
from src.solvers.brent import brent_iv


# ===========================
# Round-Trip Tests
# ===========================


def test_roundtrip_atm_call():
    """Solve for IV from synthetic ATM call price, should recover original volatility."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    # Generate synthetic market price
    market_price = black_scholes_call(S, K, T, r, true_sigma, q)

    # Solve for IV
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success, f"Solver failed: {result.message}"
    assert abs(result.volatility - true_sigma) < 1e-6, f"Expected {true_sigma}, got {result.volatility}"


def test_roundtrip_atm_put():
    """Solve for IV from synthetic ATM put price."""
    true_sigma = 0.30
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_put(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="put")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-6


@pytest.mark.parametrize(
    "true_sigma",
    [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00],
)
def test_roundtrip_various_volatilities(true_sigma):
    """Test round-trip across wide range of volatilities."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success, f"Failed for sigma={true_sigma}: {result.message}"
    assert abs(result.volatility - true_sigma) < 1e-5


@pytest.mark.parametrize(
    "moneyness",
    [0.8, 0.9, 1.0, 1.1, 1.2],  # OTM to ITM
)
def test_roundtrip_various_strikes(moneyness):
    """Test round-trip across different moneyness levels."""
    true_sigma = 0.25
    S, T, r, q = 100.0, 1.0, 0.05, 0.0
    K = S / moneyness  # Adjust strike for desired moneyness

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success, f"Failed for K={K}: {result.message}"
    assert abs(result.volatility - true_sigma) < 1e-5


def test_roundtrip_with_dividend():
    """Test round-trip with non-zero dividend yield."""
    true_sigma = 0.20
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-6


def test_roundtrip_short_expiry():
    """Test round-trip for short expiry (1 week)."""
    true_sigma = 0.30
    S, K, T, r, q = 100.0, 100.0, 7.0 / 365.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-4  # Slightly looser tolerance


def test_roundtrip_long_expiry():
    """Test round-trip for long expiry (2 years)."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 2.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-6


# ===========================
# Arbitrage Bounds Tests
# ===========================


def test_call_price_below_intrinsic_raises():
    """Call price below intrinsic value should raise ValueError."""
    S, K, T, r, q = 110.0, 100.0, 1.0, 0.05, 0.0
    intrinsic = S - K  # ~10 for this case
    market_price = 5.0  # Below intrinsic

    with pytest.raises(ValueError, match="Arbitrage violation"):
        implied_volatility(market_price, S, K, T, r, q, option_type="call")


def test_call_price_above_spot_raises():
    """Call price above spot price should raise ValueError."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    market_price = 110.0  # Above spot

    with pytest.raises(ValueError, match="Arbitrage violation"):
        implied_volatility(market_price, S, K, T, r, q, option_type="call")


def test_put_price_above_strike_raises():
    """Put price above strike present value should raise ValueError."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    pv_strike = K * math.exp(-r * T)  # ~95.12
    market_price = 100.0  # Above PV(strike)

    with pytest.raises(ValueError, match="Arbitrage violation"):
        implied_volatility(market_price, S, K, T, r, q, option_type="put")


def test_validate_arbitrage_bounds_call_valid():
    """Valid call price should pass arbitrage check."""
    market_price = 10.0
    violation = validate_arbitrage_bounds(
        market_price, S=100, K=100, T=1.0, r=0.05, q=0.0, option_type="call"
    )
    assert violation is None


def test_validate_arbitrage_bounds_call_invalid_low():
    """Call price below lower bound should fail."""
    market_price = 3.0  # Too low for ITM call
    violation = validate_arbitrage_bounds(
        market_price, S=110, K=100, T=1.0, r=0.05, q=0.0, option_type="call"
    )
    assert violation is not None
    assert "below lower bound" in violation


def test_validate_arbitrage_bounds_call_invalid_high():
    """Call price above spot should fail."""
    market_price = 110.0
    violation = validate_arbitrage_bounds(
        market_price, S=100, K=100, T=1.0, r=0.05, q=0.0, option_type="call"
    )
    assert violation is not None
    assert "above upper bound" in violation


# ===========================
# Convergence Tests
# ===========================


def test_newton_raphson_convergence_fast():
    """Newton-Raphson should converge in < 10 iterations for normal case."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = newton_raphson_iv(
        market_price, S, K, T, r, q, "call", initial_guess=0.30
    )

    assert result.success
    assert result.iterations < 10
    assert abs(result.volatility - true_sigma) < 1e-6


def test_newton_raphson_with_good_guess():
    """NR with good initial guess should converge in ~3-5 iterations."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = newton_raphson_iv(
        market_price, S, K, T, r, q, "call", initial_guess=0.23  # Close guess
    )

    assert result.success
    assert result.iterations <= 5


def test_brent_always_converges():
    """Brent should converge even with no initial guess."""
    true_sigma = 0.40
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = brent_iv(market_price, S, K, T, r, q, "call")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-6


def test_auto_method_tries_newton_first():
    """Auto method should try Newton-Raphson first and succeed."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(
        market_price, S, K, T, r, q, option_type="call", method="auto"
    )

    assert result.success
    assert result.method == "newton-raphson"


def test_method_newton_only():
    """Explicit 'newton' method shouldn't fall back to Brent."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(
        market_price, S, K, T, r, q, option_type="call", method="newton"
    )

    assert result.success
    assert result.method == "newton-raphson"


def test_method_brent_only():
    """Explicit 'brent' method should use Brent directly."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(
        market_price, S, K, T, r, q, option_type="call", method="brent"
    )

    assert result.success
    assert result.method == "brent"


# ===========================
# Initial Guess Tests
# ===========================


def test_brenner_subrahmanyam_atm():
    """Brenner-Subrahmanyam should give reasonable guess for ATM."""
    # ATM call with sigma=0.20, T=1
    S, K, T = 100.0, 100.0, 1.0
    true_sigma = 0.20
    market_price = black_scholes_call(S, K, T, 0.05, true_sigma, 0.0)

    guess = brenner_subrahmanyam_approximation(market_price, S, K, T)

    # Should be within ~50% of true value (rough initial guess)
    assert 0.10 < guess < 0.40


def test_get_initial_guess_atm():
    """get_initial_guess should use Brenner-Subrahmanyam for ATM."""
    market_price = 10.0
    guess = get_initial_guess(market_price, S=100, K=100, T=1.0, option_type="call")

    # Should return a reasonable value
    assert 0.05 < guess < 1.0


def test_get_initial_guess_otm():
    """get_initial_guess should use fixed guess for deep OTM."""
    market_price = 1.0
    guess = get_initial_guess(market_price, S=100, K=130, T=1.0, option_type="call")

    # Should fall back to IV_INITIAL_GUESS (0.25)
    assert 0.20 < guess < 0.30


# ===========================
# Vectorized Solver Tests
# ===========================


def test_implied_volatility_vectorized():
    """Test vectorized IV solver across smile."""
    S, T, r, q = 100.0, 1.0, 0.05, 0.0
    true_sigma = 0.25

    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    market_prices = [black_scholes_call(S, K, T, r, true_sigma, q) for K in strikes]

    results = implied_volatility_vectorized(
        market_prices, S, strikes, T, r, q, option_type="call"
    )

    assert len(results) == len(strikes)
    for result in results:
        assert result.success
        assert abs(result.volatility - true_sigma) < 1e-5


def test_vectorized_mismatched_lengths():
    """Vectorized solver should raise if lengths don't match."""
    with pytest.raises(ValueError, match="must have same length"):
        implied_volatility_vectorized(
            market_prices=[10.0, 5.0],
            S=100,
            strikes=[100.0],  # Length mismatch
            T=1.0,
            r=0.05,
        )


# ===========================
# Edge Cases Tests
# ===========================


def test_very_low_volatility():
    """Test IV solver with very low volatility (5%)."""
    true_sigma = 0.05
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-5


def test_very_high_volatility():
    """Test IV solver with very high volatility (100%)."""
    true_sigma = 1.00
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-5


def test_deep_itm_call():
    """Test IV solver for deep ITM call."""
    true_sigma = 0.25
    S, K, T, r, q = 130.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-5


def test_deep_otm_put():
    """Test IV solver for deep OTM put."""
    true_sigma = 0.30
    S, K, T, r, q = 130.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_put(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="put")

    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-5


def test_custom_initial_guess():
    """Test that custom initial guess is used."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(
        market_price,
        S,
        K,
        T,
        r,
        q,
        option_type="call",
        initial_guess=0.50,  # Far from true value
    )

    # Should still converge despite poor guess
    assert result.success
    assert abs(result.volatility - true_sigma) < 1e-6


# ===========================
# Result Metadata Tests
# ===========================


def test_result_contains_metadata():
    """Test that result contains all expected metadata."""
    true_sigma = 0.25
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0

    market_price = black_scholes_call(S, K, T, r, true_sigma, q)
    result = implied_volatility(market_price, S, K, T, r, q, option_type="call")

    assert hasattr(result, "volatility")
    assert hasattr(result, "iterations")
    assert hasattr(result, "method")
    assert hasattr(result, "success")
    assert hasattr(result, "message")

    assert result.method in ("newton-raphson", "brent")
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
