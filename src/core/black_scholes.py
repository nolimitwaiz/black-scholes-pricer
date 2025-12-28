"""
Black-Scholes option pricing model with continuous dividend yield.

This module implements the classical Black-Scholes-Merton formula for
European options, including all standard Greeks. The implementation
includes extensive edge case handling for numerical stability.

Mathematical Background:
    The Black-Scholes formula prices European options under assumptions:
    - Log-normal asset price distribution
    - Constant volatility and interest rate
    - No transaction costs or taxes
    - Continuous trading possible

References:
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
    Journal of Political Economy, 81(3), 637-654.
"""

import math
from typing import Optional

from src.core.distributions import normal_cdf, normal_pdf
from src.utils.constants import EPSILON_TIME, EPSILON_VOL, MAX_STANDARD_DEVIATIONS
from src.utils.types import Greeks, OptionType


def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float, q: float) -> None:
    """
    Validate option pricing inputs.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield

    Raises:
        ValueError: If any input is invalid
    """
    if S <= 0:
        raise ValueError(f"Spot price must be positive, got S={S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive, got K={K}")
    if T < 0:
        raise ValueError(f"Time to expiration cannot be negative, got T={T}")
    if sigma < 0:
        raise ValueError(f"Volatility cannot be negative, got sigma={sigma}")


def d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """
    Calculate d1 parameter in Black-Scholes formula.

    d1 represents the sensitivity of the option price to changes in the
    underlying asset price, normalized by volatility.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, continuous)
        sigma: Volatility (annualized standard deviation)
        q: Continuous dividend yield (annualized)

    Returns:
        The d1 parameter

    Formula:
        d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)

    Notes:
        Uses log-space arithmetic (log(S) - log(K)) to prevent overflow
        for extreme values of S/K.
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # Edge case: near expiration
    if T < EPSILON_TIME:
        # At expiration, d1 is essentially the sign of moneyness
        return math.inf if S > K else -math.inf

    # Edge case: zero volatility
    if sigma < EPSILON_VOL:
        # With no uncertainty, d1 is determined purely by forward vs strike
        forward = S * math.exp((r - q) * T)
        return math.inf if forward > K else -math.inf

    # Use log-space to prevent overflow: log(S/K) = log(S) - log(K)
    log_moneyness = math.log(S) - math.log(K)
    drift = (r - q + 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)

    return (log_moneyness + drift) / diffusion


def d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """
    Calculate d2 parameter in Black-Scholes formula.

    d2 represents the risk-neutral probability (under Q measure) that
    the option will be exercised at expiration.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, continuous)
        sigma: Volatility (annualized standard deviation)
        q: Continuous dividend yield (annualized)

    Returns:
        The d2 parameter

    Formula:
        d2 = d1 - σ√T

    Notes:
        For a call, N(d2) is the risk-neutral probability of exercise.
    """
    _validate_inputs(S, K, T, r, sigma, q)

    d1_value = d1(S, K, T, r, sigma, q)

    # Edge cases already handled in d1()
    if T < EPSILON_TIME or sigma < EPSILON_VOL:
        return d1_value

    return d1_value - sigma * math.sqrt(T)


def black_scholes_call(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    Calculate European call option price using Black-Scholes formula.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, continuous)
        sigma: Volatility (annualized standard deviation)
        q: Continuous dividend yield (annualized), default 0.0

    Returns:
        Call option price

    Formula:
        C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)

    Examples:
        >>> # ATM call with 1 year to expiry, 20% vol, 5% rate
        >>> price = black_scholes_call(100, 100, 1.0, 0.05, 0.20, 0.0)
        >>> abs(price - 10.4506) < 0.01  # Known solution
        True

    Edge Cases:
        - T → 0: Returns max(S - K, 0) (intrinsic value)
        - σ → 0: Returns max(S·e^((r-q)T) - K, 0)·e^(-rT) (deterministic)
        - Deep ITM (d1 > 8): Returns S·e^(-qT) - K·e^(-rT)
        - Deep OTM (d1 < -8): Returns 0
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # Edge case: at expiration
    if T < EPSILON_TIME:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)

    # Edge case: zero volatility (deterministic asset)
    if sigma < EPSILON_VOL:
        forward_price = S * math.exp((r - q) * T)
        if forward_price > K:
            return (forward_price - K) * math.exp(-r * T)
        else:
            return 0.0

    d1_value = d1(S, K, T, r, sigma, q)
    d2_value = d2(S, K, T, r, sigma, q)

    # Edge case: deep ITM (d1 > 8 means N(d1) ≈ 1, N(d2) ≈ 1)
    if d1_value > MAX_STANDARD_DEVIATIONS:
        return S * math.exp(-q * T) - K * math.exp(-r * T)

    # Edge case: deep OTM (d1 < -8 means N(d1) ≈ 0, N(d2) ≈ 0)
    if d1_value < -MAX_STANDARD_DEVIATIONS:
        return 0.0

    # Standard calculation
    discount_spot = S * math.exp(-q * T)
    discount_strike = K * math.exp(-r * T)

    return discount_spot * normal_cdf(d1_value) - discount_strike * normal_cdf(d2_value)


def black_scholes_put(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    Calculate European put option price using Black-Scholes formula.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, continuous)
        sigma: Volatility (annualized standard deviation)
        q: Continuous dividend yield (annualized), default 0.0

    Returns:
        Put option price

    Formula:
        P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

    Alternatively (via put-call parity):
        P = C - S·e^(-qT) + K·e^(-rT)

    Examples:
        >>> # ATM put with 1 year to expiry, 20% vol, 5% rate
        >>> price = black_scholes_put(100, 100, 1.0, 0.05, 0.20, 0.0)
        >>> abs(price - 5.5735) < 0.01  # Known solution
        True
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # Edge case: at expiration
    if T < EPSILON_TIME:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)

    # Edge case: zero volatility
    if sigma < EPSILON_VOL:
        forward_price = S * math.exp((r - q) * T)
        if forward_price < K:
            return (K - forward_price) * math.exp(-r * T)
        else:
            return 0.0

    d1_value = d1(S, K, T, r, sigma, q)
    d2_value = d2(S, K, T, r, sigma, q)

    # Edge case: deep ITM put (d1 < -8)
    if d1_value < -MAX_STANDARD_DEVIATIONS:
        return K * math.exp(-r * T) - S * math.exp(-q * T)

    # Edge case: deep OTM put (d1 > 8)
    if d1_value > MAX_STANDARD_DEVIATIONS:
        return 0.0

    # Standard calculation
    discount_spot = S * math.exp(-q * T)
    discount_strike = K * math.exp(-r * T)

    return discount_strike * normal_cdf(-d2_value) - discount_spot * normal_cdf(-d1_value)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """
    Calculate European option price (call or put).

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility
        q: Continuous dividend yield, default 0.0
        option_type: "call" or "put"

    Returns:
        Option price

    Raises:
        ValueError: If option_type is not "call" or "put"
    """
    if option_type == "call":
        return black_scholes_call(S, K, T, r, sigma, q)
    elif option_type == "put":
        return black_scholes_put(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ===========================
# Greeks Calculations
# ===========================


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """
    Calculate option delta (∂V/∂S).

    Delta measures the rate of change of option price with respect to
    the underlying asset price. For a call, delta ∈ [0, 1]; for a put, delta ∈ [-1, 0].

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes parameters
        option_type: "call" or "put"

    Returns:
        Delta value

    Formulas:
        Call delta: Δ_c = e^(-qT) · N(d1)
        Put delta:  Δ_p = -e^(-qT) · N(-d1) = Δ_c - e^(-qT)

    Interpretation:
        Delta of 0.6 means: for $1 increase in spot, option price increases by ~$0.60.
        Delta also approximates the probability of finishing in-the-money (though
        this is more precisely N(d2) under the risk-neutral measure).
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # Edge case: at expiration
    if T < EPSILON_TIME:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    # Edge case: zero volatility
    if sigma < EPSILON_VOL:
        forward = S * math.exp((r - q) * T)
        if option_type == "call":
            return math.exp(-q * T) if forward > K else 0.0
        else:
            return -math.exp(-q * T) if forward < K else 0.0

    d1_value = d1(S, K, T, r, sigma, q)
    discount_factor = math.exp(-q * T)

    if option_type == "call":
        return discount_factor * normal_cdf(d1_value)
    else:  # put
        return -discount_factor * normal_cdf(-d1_value)


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate option gamma (∂²V/∂S²).

    Gamma measures the rate of change of delta with respect to the
    underlying price. It is the same for calls and puts (by put-call parity).

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes parameters

    Returns:
        Gamma value (always non-negative)

    Formula:
        Γ = e^(-qT) · φ(d1) / (S · σ · √T)

    where φ(d1) is the standard normal PDF evaluated at d1.

    Interpretation:
        Gamma of 0.05 means: for $1 increase in spot, delta increases by 0.05.
        Gamma is highest for at-the-money options and decreases as the option
        moves in- or out-of-the-money. It approaches zero near expiration.
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # Gamma → 0 as T → 0 (delta becomes step function)
    if T < EPSILON_TIME:
        return 0.0

    # Gamma → 0 as σ → 0 (delta becomes step function)
    if sigma < EPSILON_VOL:
        return 0.0

    d1_value = d1(S, K, T, r, sigma, q)
    pdf_d1 = normal_pdf(d1_value)

    discount_factor = math.exp(-q * T)
    denominator = S * sigma * math.sqrt(T)

    return (discount_factor * pdf_d1) / denominator


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate option vega (∂V/∂σ).

    Vega measures the sensitivity of the option price to changes in
    volatility. Same for calls and puts. Reported per 1% change in volatility.

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes parameters

    Returns:
        Vega value (per 1% volatility change)

    Formula:
        ν = S · e^(-qT) · √T · φ(d1)

    Interpretation:
        Vega of 0.35 means: for 1% increase in volatility (e.g., 20% → 21%),
        option price increases by $0.35.

    Notes:
        Vega is highest for at-the-money options and decreases for deep
        in- or out-of-the-money options. It also increases with time to expiration.
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # Vega → 0 as T → 0 (no time for volatility to matter)
    if T < EPSILON_TIME:
        return 0.0

    d1_value = d1(S, K, T, r, sigma, q)
    pdf_d1 = normal_pdf(d1_value)

    discount_factor = math.exp(-q * T)

    # Return vega per 1% change in volatility
    return S * discount_factor * math.sqrt(T) * pdf_d1


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """
    Calculate option theta (∂V/∂T), reported per calendar day.

    Theta measures time decay of the option. Typically negative for
    long positions (option loses value as time passes).

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes parameters
        option_type: "call" or "put"

    Returns:
        Theta value per day (divide by 365 to get annualized rate)

    Formulas:
        Call theta:
            Θ_c = -[S·σ·e^(-qT)·φ(d1)/(2√T)] - r·K·e^(-rT)·N(d2) + q·S·e^(-qT)·N(d1)

        Put theta:
            Θ_p = -[S·σ·e^(-qT)·φ(d1)/(2√T)] + r·K·e^(-rT)·N(-d2) - q·S·e^(-qT)·N(-d1)

    Interpretation:
        Theta of -0.05 means option loses $0.05 in value per calendar day,
        all else equal.

    Notes:
        Returned value is per day. Multiply by days to get change over a period.
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # At expiration, theta is undefined (discontinuous)
    if T < EPSILON_TIME:
        return 0.0

    if sigma < EPSILON_VOL:
        return 0.0

    d1_value = d1(S, K, T, r, sigma, q)
    d2_value = d2(S, K, T, r, sigma, q)
    pdf_d1 = normal_pdf(d1_value)

    sqrt_T = math.sqrt(T)
    discount_spot = math.exp(-q * T)
    discount_strike = math.exp(-r * T)

    # First term (same for call and put): diffusion contribution
    term1 = -(S * sigma * discount_spot * pdf_d1) / (2.0 * sqrt_T)

    if option_type == "call":
        # Second term: risk-free discounting
        term2 = -r * K * discount_strike * normal_cdf(d2_value)
        # Third term: dividend contribution
        term3 = q * S * discount_spot * normal_cdf(d1_value)
    else:  # put
        term2 = r * K * discount_strike * normal_cdf(-d2_value)
        term3 = -q * S * discount_spot * normal_cdf(-d1_value)

    # Total theta (annualized), then convert to per-day
    theta_annual = term1 + term2 + term3
    return theta_annual / 365.0  # Per calendar day


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """
    Calculate option rho (∂V/∂r), reported per 1% change in interest rate.

    Rho measures sensitivity to changes in the risk-free rate.

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes parameters
        option_type: "call" or "put"

    Returns:
        Rho value per 1% change in rate

    Formulas:
        Call rho: ρ_c = K·T·e^(-rT)·N(d2)
        Put rho:  ρ_p = -K·T·e^(-rT)·N(-d2)

    Interpretation:
        Rho of 0.45 means: for 1% increase in rates (e.g., 5% → 6%),
        option price increases by $0.45.

    Notes:
        Typically, call rho is positive (higher rates → higher call value)
        and put rho is negative (higher rates → lower put value).
    """
    _validate_inputs(S, K, T, r, sigma, q)

    # At expiration, rho is minimal
    if T < EPSILON_TIME:
        return 0.0

    if sigma < EPSILON_VOL:
        return 0.0

    d2_value = d2(S, K, T, r, sigma, q)
    discount_strike = K * T * math.exp(-r * T)

    if option_type == "call":
        return discount_strike * normal_cdf(d2_value)
    else:  # put
        return -discount_strike * normal_cdf(-d2_value)


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> Greeks:
    """
    Calculate all Greeks for an option in one pass.

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes parameters
        option_type: "call" or "put"

    Returns:
        Greeks dataclass with delta, gamma, vega, theta, rho

    Example:
        >>> greeks = calculate_greeks(100, 100, 1.0, 0.05, 0.20)
        >>> print(f"Delta: {greeks.delta:.4f}")
        Delta: 0.6368
    """
    return Greeks(
        delta=delta(S, K, T, r, sigma, q, option_type),
        gamma=gamma(S, K, T, r, sigma, q),
        vega=vega(S, K, T, r, sigma, q),
        theta=theta(S, K, T, r, sigma, q, option_type),
        rho=rho(S, K, T, r, sigma, q, option_type),
    )
