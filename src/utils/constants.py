"""
Numerical constants and tolerances for options pricing calculations.

This module defines critical thresholds for edge case detection and
solver convergence criteria. All values are calibrated for numerical
stability while maintaining accuracy requirements.
"""

# Edge case detection thresholds
EPSILON_TIME = 1e-6  # ~0.1 seconds; below this, use intrinsic value
EPSILON_VOL = 1e-6  # ~0.0001% annualized; below this, deterministic pricing
EPSILON_PRICE = 1e-10  # Minimum price threshold (effectively zero)

# Normal distribution bounds
MAX_STANDARD_DEVIATIONS = 8.0  # Beyond ±8σ, CDF is effectively 0 or 1

# Implied volatility solver parameters
IV_PRICE_TOLERANCE = 1e-6  # $0.000001 price accuracy
IV_VOL_TOLERANCE = 1e-8  # Volatility convergence tolerance
IV_MAX_ITERATIONS = 50  # Maximum Newton-Raphson iterations
IV_MIN_VEGA = 1e-6  # Below this, switch to Brent method
IV_INITIAL_GUESS = 0.25  # Default 25% volatility if no better guess
IV_MIN_VOL = 0.001  # 0.1% minimum volatility
IV_MAX_VOL = 10.0  # 1000% maximum volatility

# Arbitrage diagnostics tolerances
ARBITRAGE_TOLERANCE = 1e-4  # $0.0001 tolerance for bounds checks
PARITY_TOLERANCE = 1e-6  # Put-call parity tolerance

# Greeks finite-difference step sizes
FD_STEP_SPOT = 0.01  # $0.01 for delta/gamma finite differences
FD_STEP_VOL = 0.0001  # 1 basis point for vega
FD_STEP_RATE = 0.0001  # 1 basis point for rho
FD_STEP_TIME = 1.0 / 365.0  # 1 day for theta

# Cache parameters
DEFAULT_CACHE_TTL = 300  # 5 minutes in seconds
