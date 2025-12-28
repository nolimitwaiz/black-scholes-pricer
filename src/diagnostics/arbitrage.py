"""
Arbitrage diagnostics for option pricing validation.

This module implements comprehensive no-arbitrage checks including:
- Price bounds validation
- Put-call parity
- Strike monotonicity
- Butterfly convexity
- Calendar spread arbitrage
"""

import math
from typing import Optional
from dataclasses import dataclass

from src.utils.constants import ARBITRAGE_TOLERANCE, PARITY_TOLERANCE
from src.utils.types import ArbitrageCheck


@dataclass
class OptionData:
    """Container for option data with all required fields."""

    strike: float
    price: float
    option_type: str  # "call" or "put"
    expiry: Optional[float] = None  # Time to expiration (for calendar spreads)


def check_price_bounds(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tolerance: float = ARBITRAGE_TOLERANCE,
) -> ArbitrageCheck:
    """
    Validate option prices against no-arbitrage bounds.

    Checks:
    1. Call lower bound: C >= max(S·e^(-qT) - K·e^(-rT), 0)
    2. Call upper bound: C <= S·e^(-qT)
    3. Put lower bound: P >= max(K·e^(-rT) - S·e^(-qT), 0)
    4. Put upper bound: P <= K·e^(-rT)

    Args:
        call_price, put_price: Observed option prices
        S: Spot price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        q: Dividend yield
        tolerance: Tolerance for floating point comparisons

    Returns:
        ArbitrageCheck with validation results
    """
    violations = []
    details = {}

    discount_spot = S * math.exp(-q * T)
    discount_strike = K * math.exp(-r * T)

    # Call lower bound
    call_lower = max(discount_spot - discount_strike, 0.0)
    call_lower_ok = call_price >= call_lower - tolerance
    details["call_lower_bound"] = call_lower_ok
    if not call_lower_ok:
        violations.append(
            f"Call price {call_price:.4f} below lower bound {call_lower:.4f}"
        )

    # Call upper bound
    call_upper_ok = call_price <= discount_spot + tolerance
    details["call_upper_bound"] = call_upper_ok
    if not call_upper_ok:
        violations.append(
            f"Call price {call_price:.4f} above upper bound {discount_spot:.4f}"
        )

    # Put lower bound
    put_lower = max(discount_strike - discount_spot, 0.0)
    put_lower_ok = put_price >= put_lower - tolerance
    details["put_lower_bound"] = put_lower_ok
    if not put_lower_ok:
        violations.append(f"Put price {put_price:.4f} below lower bound {put_lower:.4f}")

    # Put upper bound
    put_upper_ok = put_price <= discount_strike + tolerance
    details["put_upper_bound"] = put_upper_ok
    if not put_upper_ok:
        violations.append(
            f"Put price {put_price:.4f} above upper bound {discount_strike:.4f}"
        )

    is_valid = len(violations) == 0
    return ArbitrageCheck(is_valid=is_valid, violations=violations, details=details)


def check_put_call_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tolerance: float = PARITY_TOLERANCE,
) -> ArbitrageCheck:
    """
    Validate put-call parity relationship.

    Put-call parity:
        C - P = S·e^(-qT) - K·e^(-rT)

    Args:
        call_price, put_price: Option prices
        S, K, T, r, q: Standard parameters
        tolerance: Tolerance for parity check

    Returns:
        ArbitrageCheck with validation results
    """
    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    diff = abs(lhs - rhs)
    is_valid = diff < tolerance

    violations = []
    if not is_valid:
        violations.append(
            f"Put-call parity violated: C - P = {lhs:.6f}, "
            f"S·e^(-qT) - K·e^(-rT) = {rhs:.6f}, diff = {diff:.6f}"
        )

    details = {"parity_lhs": lhs, "parity_rhs": rhs, "difference": diff}

    return ArbitrageCheck(is_valid=is_valid, violations=violations, details=details)


def check_strike_monotonicity(
    options: list[OptionData], tolerance: float = ARBITRAGE_TOLERANCE
) -> ArbitrageCheck:
    """
    Check monotonicity in strike: calls decrease, puts increase.

    For calls: C(K1) >= C(K2) if K1 < K2
    For puts: P(K1) <= P(K2) if K1 < K2

    Args:
        options: List of OptionData sorted by strike
        tolerance: Tolerance for price comparisons

    Returns:
        ArbitrageCheck with validation results
    """
    violations = []
    details = {}

    # Separate calls and puts
    calls = sorted([opt for opt in options if opt.option_type == "call"], key=lambda x: x.strike)
    puts = sorted([opt for opt in options if opt.option_type == "put"], key=lambda x: x.strike)

    # Check call monotonicity (decreasing in K)
    for i in range(len(calls) - 1):
        if calls[i].price < calls[i + 1].price - tolerance:
            violations.append(
                f"Call monotonicity violated: C(K={calls[i].strike}) = {calls[i].price:.4f} "
                f"< C(K={calls[i+1].strike}) = {calls[i+1].price:.4f}"
            )
    details["call_monotonic"] = len(violations) == 0

    # Check put monotonicity (increasing in K)
    initial_violations = len(violations)
    for i in range(len(puts) - 1):
        if puts[i].price > puts[i + 1].price + tolerance:
            violations.append(
                f"Put monotonicity violated: P(K={puts[i].strike}) = {puts[i].price:.4f} "
                f"> P(K={puts[i+1].strike}) = {puts[i+1].price:.4f}"
            )
    details["put_monotonic"] = len(violations) == initial_violations

    is_valid = len(violations) == 0
    return ArbitrageCheck(is_valid=is_valid, violations=violations, details=details)


def check_butterfly_arbitrage(
    K1: float,
    K2: float,
    K3: float,
    C1: float,
    C2: float,
    C3: float,
    tolerance: float = ARBITRAGE_TOLERANCE,
) -> ArbitrageCheck:
    """
    Check butterfly spread no-arbitrage condition.

    A butterfly spread consists of:
    - Buy 1 call at K1
    - Sell 2 calls at K2
    - Buy 1 call at K3
    where K1 < K2 < K3 and K2 - K1 = K3 - K2 (equally spaced)

    No-arbitrage requires:
        C1 + C3 >= 2·C2

    More generally for unequal spacing:
        w1·C1 + w3·C3 >= C2
    where w1 = (K3-K2)/(K3-K1), w3 = (K2-K1)/(K3-K1)

    Args:
        K1, K2, K3: Strike prices (K1 < K2 < K3)
        C1, C2, C3: Corresponding call prices
        tolerance: Tolerance for check

    Returns:
        ArbitrageCheck with validation results
    """
    if not (K1 < K2 < K3):
        raise ValueError(f"Strikes must be ordered: K1 < K2 < K3, got {K1}, {K2}, {K3}")

    # Compute weights for convexity check
    w1 = (K3 - K2) / (K3 - K1)
    w3 = (K2 - K1) / (K3 - K1)

    # Convexity condition
    lhs = w1 * C1 + w3 * C3
    rhs = C2

    is_valid = lhs >= rhs - tolerance

    violations = []
    if not is_valid:
        violations.append(
            f"Butterfly arbitrage: w1·C1 + w3·C3 = {lhs:.4f} < C2 = {rhs:.4f}. "
            f"Arbitrage: sell butterfly spread."
        )

    details = {"lhs": lhs, "rhs": rhs, "w1": w1, "w3": w3}

    return ArbitrageCheck(is_valid=is_valid, violations=violations, details=details)


def check_calendar_spread(
    near_price: float,
    far_price: float,
    option_type: str,
    tolerance: float = ARBITRAGE_TOLERANCE,
) -> ArbitrageCheck:
    """
    Check calendar spread no-arbitrage condition.

    For the same strike, longer-dated options must be worth at least
    as much as shorter-dated options (time value is non-negative).

    Condition: C_far >= C_near (same for puts)

    Args:
        near_price: Price of near-expiry option
        far_price: Price of far-expiry option
        option_type: "call" or "put"
        tolerance: Tolerance for check

    Returns:
        ArbitrageCheck with validation results
    """
    is_valid = far_price >= near_price - tolerance

    violations = []
    if not is_valid:
        violations.append(
            f"Calendar spread arbitrage: {option_type} far price {far_price:.4f} "
            f"< near price {near_price:.4f}"
        )

    details = {"near_price": near_price, "far_price": far_price}

    return ArbitrageCheck(is_valid=is_valid, violations=violations, details=details)
