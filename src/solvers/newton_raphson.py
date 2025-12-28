"""
Newton-Raphson method for implied volatility calculation.

This module implements the Newton-Raphson algorithm for solving
the Black-Scholes equation for volatility given a market price.
The method uses vega (∂V/∂σ) as the derivative for fast convergence.
"""

import math
from typing import Optional

from src.core.black_scholes import black_scholes_price, vega
from src.utils.constants import (
    IV_MAX_ITERATIONS,
    IV_MIN_VEGA,
    IV_MIN_VOL,
    IV_MAX_VOL,
    IV_PRICE_TOLERANCE,
    IV_VOL_TOLERANCE,
)
from src.utils.types import OptionType, ImpliedVolResult


def newton_raphson_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    initial_guess: float,
    max_iterations: int = IV_MAX_ITERATIONS,
    price_tolerance: float = IV_PRICE_TOLERANCE,
    vol_tolerance: float = IV_VOL_TOLERANCE,
) -> ImpliedVolResult:
    """
    Solve for implied volatility using Newton-Raphson method.

    The Newton-Raphson update is:
        σ_{n+1} = σ_n - (BS(σ_n) - market_price) / vega(σ_n)

    This method has quadratic convergence near the solution but can
    diverge if the initial guess is poor or vega is too small.

    Args:
        market_price: Observed market price of the option
        S, K, T, r, q: Standard Black-Scholes parameters
        option_type: "call" or "put"
        initial_guess: Starting volatility estimate
        max_iterations: Maximum number of iterations
        price_tolerance: Convergence tolerance for price difference
        vol_tolerance: Convergence tolerance for volatility change

    Returns:
        ImpliedVolResult with volatility, iterations, method, success flag

    Notes:
        - Returns success=False if vega drops below MIN_VEGA (switch to Brent)
        - Returns success=False if σ goes out of bounds [MIN_VOL, MAX_VOL]
        - Returns success=True if converged within tolerances
    """
    sigma = initial_guess
    iterations = 0

    for i in range(max_iterations):
        iterations += 1

        # Compute Black-Scholes price at current volatility
        bs_price = black_scholes_price(S, K, T, r, sigma, q, option_type)

        # Check convergence on price
        price_diff = abs(bs_price - market_price)
        if price_diff < price_tolerance:
            return ImpliedVolResult(
                volatility=sigma,
                iterations=iterations,
                method="newton-raphson",
                success=True,
                message=f"Converged in {iterations} iterations (price tol)",
            )

        # Compute vega for Newton-Raphson update
        vega_value = vega(S, K, T, r, sigma, q)

        # Check if vega is too small (numerical instability risk)
        if abs(vega_value) < IV_MIN_VEGA:
            return ImpliedVolResult(
                volatility=sigma,
                iterations=iterations,
                method="newton-raphson",
                success=False,
                message=f"Vega too small ({vega_value:.2e}) at iteration {iterations}, need fallback",
            )

        # Newton-Raphson update
        sigma_new = sigma - (bs_price - market_price) / vega_value

        # Check bounds
        if sigma_new < IV_MIN_VOL or sigma_new > IV_MAX_VOL:
            return ImpliedVolResult(
                volatility=sigma,
                iterations=iterations,
                method="newton-raphson",
                success=False,
                message=f"Stepped out of bounds (σ={sigma_new:.4f}) at iteration {iterations}",
            )

        # Check convergence on volatility change
        vol_diff = abs(sigma_new - sigma)
        if vol_diff < vol_tolerance:
            return ImpliedVolResult(
                volatility=sigma_new,
                iterations=iterations,
                method="newton-raphson",
                success=True,
                message=f"Converged in {iterations} iterations (vol tol)",
            )

        sigma = sigma_new

    # Max iterations reached without convergence
    return ImpliedVolResult(
        volatility=sigma,
        iterations=iterations,
        method="newton-raphson",
        success=False,
        message=f"Max iterations ({max_iterations}) reached without convergence",
    )
