"""
Data types and structures for options pricing.

This module defines dataclasses and types used throughout the toolkit
for representing options, Greeks, and solver results.
"""

from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionParams:
    """
    Immutable container for option parameters.

    Attributes:
        S: Current spot price of the underlying asset
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, continuous compounding)
        q: Continuous dividend yield (annualized)
        option_type: Either "call" or "put"
    """
    S: float
    K: float
    T: float
    r: float
    q: float
    option_type: OptionType

    def __post_init__(self) -> None:
        """Validate parameters are positive where required."""
        if self.S <= 0:
            raise ValueError(f"Spot price must be positive, got S={self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got K={self.K}")
        if self.T < 0:
            raise ValueError(f"Time to expiration must be non-negative, got T={self.T}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"Option type must be 'call' or 'put', got {self.option_type}")


@dataclass
class Greeks:
    """
    Container for option Greeks.

    Attributes:
        delta: Rate of change of option price with respect to spot price (∂V/∂S)
        gamma: Rate of change of delta with respect to spot price (∂²V/∂S²)
        vega: Rate of change of option price with respect to volatility (∂V/∂σ), per 1% vol
        theta: Rate of change of option price with respect to time (∂V/∂T), per day
        rho: Rate of change of option price with respect to interest rate (∂V/∂r), per 1% rate
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class ImpliedVolResult:
    """
    Result from implied volatility solver.

    Attributes:
        volatility: Solved implied volatility (annualized)
        iterations: Number of iterations required for convergence
        method: Method used ('newton-raphson' or 'brent')
        success: Whether the solver converged successfully
        message: Additional information about convergence
    """
    volatility: float
    iterations: int
    method: Literal["newton-raphson", "brent"]
    success: bool
    message: str = ""


@dataclass
class ArbitrageCheck:
    """
    Result from arbitrage validation.

    Attributes:
        is_valid: Whether the price satisfies no-arbitrage conditions
        violations: List of specific violations detected
        details: Dictionary with detailed check results
    """
    is_valid: bool
    violations: list[str]
    details: dict[str, bool]
