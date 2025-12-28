"""
Statistical distributions with numerical safeguards.

This module provides numerically stable implementations of the
standard normal cumulative distribution function (CDF) and probability
density function (PDF), with special handling for extreme values.
"""

import math
from scipy.stats import norm

from src.utils.constants import MAX_STANDARD_DEVIATIONS


def normal_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function with bounds clamping.

    For |x| > 8, the CDF is effectively 0 (x < -8) or 1 (x > 8) due to
    floating point precision limits. We clamp to these values to prevent
    underflow and improve numerical stability.

    Args:
        x: Value at which to evaluate the CDF

    Returns:
        Probability that a standard normal random variable is less than x

    Examples:
        >>> normal_cdf(0.0)  # Median
        0.5
        >>> normal_cdf(1.96)  # ~97.5th percentile
        0.975
        >>> normal_cdf(10.0)  # Deep in tail
        1.0
    """
    if x > MAX_STANDARD_DEVIATIONS:
        return 1.0
    if x < -MAX_STANDARD_DEVIATIONS:
        return 0.0

    return norm.cdf(x)


def normal_pdf(x: float) -> float:
    """
    Standard normal probability density function with overflow protection.

    For |x| > 10, the PDF is negligible (< 2e-22) and can be safely
    approximated as zero to prevent numerical issues in subsequent calculations.

    Args:
        x: Value at which to evaluate the PDF

    Returns:
        Probability density at x for standard normal distribution

    Examples:
        >>> abs(normal_pdf(0.0) - 0.3989) < 0.001  # Peak at zero
        True
        >>> normal_pdf(3.0) < 0.01  # Small in tails
        True
        >>> normal_pdf(15.0)  # Effectively zero
        0.0

    Notes:
        The standard normal PDF is given by:
            φ(x) = (1/√(2π)) * exp(-x²/2)
    """
    if abs(x) > 10.0:
        return 0.0

    # Explicit calculation for clarity and to avoid scipy overhead
    # (though scipy.stats.norm.pdf is vectorized and optimized)
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)
