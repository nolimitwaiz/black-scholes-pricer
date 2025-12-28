"""
Brent's method for implied volatility calculation.

This module implements Brent's method (a hybrid bisection/inverse quadratic
interpolation algorithm) as a robust fallback when Newton-Raphson fails.
Brent's method is guaranteed to converge if a solution exists within the
specified bounds, though it's slower than Newton-Raphson.
"""

from scipy.optimize import brentq

from src.core.black_scholes import black_scholes_price
from src.utils.constants import IV_MIN_VOL, IV_MAX_VOL, IV_PRICE_TOLERANCE
from src.utils.types import OptionType, ImpliedVolResult


def brent_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    vol_lower: float = IV_MIN_VOL,
    vol_upper: float = IV_MAX_VOL,
    tolerance: float = IV_PRICE_TOLERANCE,
) -> ImpliedVolResult:
    """
    Solve for implied volatility using Brent's method.

    Brent's method is a root-finding algorithm that combines:
    - Bisection (reliable but slow)
    - Inverse quadratic interpolation (fast when applicable)
    - Secant method (intermediate speed/reliability)

    It's slower than Newton-Raphson but guaranteed to converge if
    the objective function has opposite signs at the bounds.

    Args:
        market_price: Observed market price of the option
        S, K, T, r, q: Standard Black-Scholes parameters
        option_type: "call" or "put"
        vol_lower: Lower bound for volatility search
        vol_upper: Upper bound for volatility search
        tolerance: Convergence tolerance for price difference

    Returns:
        ImpliedVolResult with volatility, iterations, method, success flag

    Raises:
        ValueError: If the objective function doesn't bracket a root
                   (i.e., BS(vol_lower) and BS(vol_upper) have same sign)
    """

    def objective(sigma: float) -> float:
        """
        Objective function: BS(σ) - market_price.
        We seek σ such that this equals zero.
        """
        return black_scholes_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        # Brent's method from scipy
        # brentq requires that objective(a) and objective(b) have opposite signs
        implied_vol = brentq(
            objective,
            vol_lower,
            vol_upper,
            xtol=tolerance,  # Tolerance on sigma
            rtol=1e-8,  # Relative tolerance
            maxiter=100,  # Maximum iterations
            full_output=False,
        )

        # Verify the solution
        final_price = black_scholes_price(S, K, T, r, implied_vol, q, option_type)
        price_error = abs(final_price - market_price)

        return ImpliedVolResult(
            volatility=implied_vol,
            iterations=0,  # brentq doesn't expose iteration count
            method="brent",
            success=True,
            message=f"Converged with price error {price_error:.2e}",
        )

    except ValueError as e:
        # This occurs if objective function doesn't bracket a root
        # Check values at bounds to provide useful error message
        obj_lower = objective(vol_lower)
        obj_upper = objective(vol_upper)

        error_msg = (
            f"Brent method failed: objective function doesn't bracket a root. "
            f"obj({vol_lower:.4f}) = {obj_lower:.4f}, "
            f"obj({vol_upper:.4f}) = {obj_upper:.4f}. "
            f"Market price {market_price} may violate arbitrage bounds."
        )

        return ImpliedVolResult(
            volatility=0.0,
            iterations=0,
            method="brent",
            success=False,
            message=error_msg,
        )
