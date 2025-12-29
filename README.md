# Black-Scholes Pricer

A production-grade Black-Scholes options pricer with implied volatility solver, Greeks calculation, and comprehensive arbitrage diagnostics.

## Features

- **Black-Scholes Pricing**: European call/put options with continuous dividend yield
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho with numerical stability
- **Implied Volatility**: Newton-Raphson with Brent fallback for robust convergence
- **Arbitrage Diagnostics**: Put-call parity, price bounds, strike monotonicity, butterfly spreads
- **CLI & Web UI**: Professional command-line and Streamlit interfaces
- **Comprehensive Tests**: 91+ tests with 85%+ coverage

## Installation

```bash
git clone https://github.com/nolimitwaiz/black-scholes-pricer.git
cd black-scholes-pricer
pip install -e .
```

## Quick Start

### Python API

```python
from src.core.black_scholes import black_scholes_call, calculate_greeks
from src.solvers.implied_vol import implied_volatility

# Price a call option
price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
print(f"Call price: ${price:.2f}")  # Call price: $10.45

# Calculate Greeks
greeks = calculate_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
print(f"Delta: {greeks.delta:.4f}")  # Delta: 0.6368

# Solve for implied volatility
result = implied_volatility(market_price=10.45, S=100, K=100, T=1.0, r=0.05)
print(f"IV: {result.volatility:.2%}")  # IV: 20.00%
```

### CLI

```bash
# Price an option
options-toolkit price -S 100 -K 100 -T 1.0 -r 0.05 -v 0.20 --type call

# Calculate Greeks
options-toolkit greeks -S 100 -K 100 -T 1.0 -r 0.05 -v 0.20

# Solve implied volatility
options-toolkit iv -p 10.45 -S 100 -K 100 -T 1.0 -r 0.05
```

### Streamlit UI

```bash
streamlit run interfaces/streamlit_app.py
```

## Architecture

**Functional Core**: Pure functions for all mathematical operations (pricing, Greeks) for maximum testability and clarity.

**Hybrid Design**: Optional thin class wrappers for convenience while maintaining functional purity in core logic.

**Numerical Stability**: Comprehensive edge case handling for T→0, σ→0, deep ITM/OTM scenarios.

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/unit/test_black_scholes.py -v
```

## Project Structure

```
black-scholes-pricer/
├── src/
│   ├── core/               # Black-Scholes pricing & Greeks
│   ├── solvers/            # Implied volatility (Newton-Raphson + Brent)
│   ├── diagnostics/        # Arbitrage checks
│   └── utils/              # Constants, types, distributions
├── interfaces/
│   ├── cli.py              # Command-line interface
│   └── streamlit_app.py    # Web UI
├── tests/                  # Comprehensive test suite
└── docs/                   # Mathematical documentation
```

## Implementation Roadmap

**Phase 1** (Core): Black-Scholes pricing + Greeks + tests
**Phase 2** (Solver): Implied volatility with dual-method approach
**Phase 3** (Validation): Arbitrage diagnostics suite
**Phase 4** (Interfaces): CLI and Streamlit UI
**Phase 5** (Polish): Documentation and optimization

## Mathematical Background

See `docs/MATHEMATICS.md` for:
- Black-Scholes PDE derivation
- Greeks formulas and interpretations
- Implied volatility solver theory
- No-arbitrage conditions

## License

MIT License - see LICENSE file

## Author

Waiz Khan - [GitHub](https://github.com/nolimitwaiz)
