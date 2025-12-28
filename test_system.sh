#!/bin/bash

echo "=================================================="
echo "COMPLETE SYSTEM TEST - Options Pricing Toolkit"
echo "=================================================="
echo ""

# Test 1: Python imports
echo "[1/6] Testing Python imports..."
python3 -c "
from src.core.black_scholes import black_scholes_call, calculate_greeks
from src.solvers.implied_vol import implied_volatility
from src.diagnostics.arbitrage import check_price_bounds
print('✓ All imports successful')
" || { echo "✗ FAILED: Import errors"; exit 1; }

# Test 2: Run test suite
echo ""
echo "[2/6] Running full test suite (91 tests)..."
python3 -m pytest tests/ -q --tb=line || { echo "✗ FAILED: Tests failed"; exit 1; }

# Test 3: Test pricing
echo ""
echo "[3/6] Testing Black-Scholes pricing..."
python3 -c "
from src.core.black_scholes import black_scholes_call
price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
assert 10.40 < price < 10.50, f'Price {price} out of expected range'
print(f'✓ Call price: \${price:.4f} (expected ~\$10.45)')
" || { echo "✗ FAILED: Pricing failed"; exit 1; }

# Test 4: Test Greeks
echo ""
echo "[4/6] Testing Greeks calculation..."
python3 -c "
from src.core.black_scholes import calculate_greeks
greeks = calculate_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
assert 0.6 < greeks.delta < 0.7, 'Delta out of range'
assert greeks.gamma > 0, 'Gamma should be positive'
assert greeks.vega > 0, 'Vega should be positive'
print(f'✓ Delta={greeks.delta:.4f}, Gamma={greeks.gamma:.6f}, Vega={greeks.vega:.2f}')
" || { echo "✗ FAILED: Greeks failed"; exit 1; }

# Test 5: Test IV solver
echo ""
echo "[5/6] Testing implied volatility solver..."
python3 -c "
from src.solvers.implied_vol import implied_volatility
result = implied_volatility(market_price=10.45, S=100, K=100, T=1.0, r=0.05)
assert result.success, 'IV solver failed to converge'
assert abs(result.volatility - 0.20) < 0.01, f'IV {result.volatility} not near 0.20'
print(f'✓ IV={result.volatility:.4f}, Method={result.method}, Iterations={result.iterations}')
" || { echo "✗ FAILED: IV solver failed"; exit 1; }

# Test 6: Test CLI
echo ""
echo "[6/6] Testing CLI interface..."
python3 -m interfaces.cli price -S 100 -K 100 -T 1.0 -r 0.05 -v 0.20 --type call > /dev/null || { echo "✗ FAILED: CLI failed"; exit 1; }
echo "✓ CLI working"

echo ""
echo "=================================================="
echo "✓✓✓ ALL TESTS PASSED - SYSTEM FULLY FUNCTIONAL ✓✓✓"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - 91 unit tests: PASS"
echo "  - Pricing engine: PASS"
echo "  - Greeks calculation: PASS"
echo "  - IV solver: PASS"
echo "  - CLI interface: PASS"
echo ""
echo "Next steps:"
echo "  1. Test web UI: streamlit run interfaces/streamlit_app.py"
echo "  2. Review code: open ~/Desktop/options-pricing-toolkit"
echo "  3. Read docs: cat README.md"
echo ""
