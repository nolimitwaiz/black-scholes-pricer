"""
Command-line interface for options pricing toolkit.

This CLI provides access to:
- Option pricing (Black-Scholes)
- Greeks calculation
- Implied volatility solving
- Arbitrage diagnostics
"""

import click
from datetime import datetime, timedelta
from src.core.black_scholes import black_scholes_price, calculate_greeks
from src.solvers.implied_vol import implied_volatility


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Options Pricing Toolkit - Professional Black-Scholes implementation."""
    pass


@cli.command()
@click.option("--spot", "-S", type=float, required=True, help="Spot price")
@click.option("--strike", "-K", type=float, required=True, help="Strike price")
@click.option("--time", "-T", type=float, required=True, help="Time to expiry (years)")
@click.option("--rate", "-r", type=float, required=True, help="Risk-free rate")
@click.option("--vol", "-v", type=float, required=True, help="Volatility (annualized)")
@click.option("--div", "-q", type=float, default=0.0, help="Dividend yield")
@click.option("--type", "-t", type=click.Choice(["call", "put"]), default="call")
def price(spot, strike, time, rate, vol, div, type):
    """Calculate option price using Black-Scholes."""
    price_value = black_scholes_price(spot, strike, time, rate, vol, div, type)
    click.echo(f"\n{type.capitalize()} Option Price: ${price_value:.4f}")


@cli.command()
@click.option("--spot", "-S", type=float, required=True, help="Spot price")
@click.option("--strike", "-K", type=float, required=True, help="Strike price")
@click.option("--time", "-T", type=float, required=True, help="Time to expiry (years)")
@click.option("--rate", "-r", type=float, required=True, help="Risk-free rate")
@click.option("--vol", "-v", type=float, required=True, help="Volatility")
@click.option("--div", "-q", type=float, default=0.0, help="Dividend yield")
@click.option("--type", "-t", type=click.Choice(["call", "put"]), default="call")
def greeks(spot, strike, time, rate, vol, div, type):
    """Calculate all option Greeks."""
    greeks_values = calculate_greeks(spot, strike, time, rate, vol, div, type)

    click.echo(f"\nGreeks for {type.capitalize()} Option:")
    click.echo(f"  Delta:  {greeks_values.delta:>10.6f}")
    click.echo(f"  Gamma:  {greeks_values.gamma:>10.6f}")
    click.echo(f"  Vega:   {greeks_values.vega:>10.6f}")
    click.echo(f"  Theta:  {greeks_values.theta:>10.6f} (per day)")
    click.echo(f"  Rho:    {greeks_values.rho:>10.6f}")


@cli.command()
@click.option("--market-price", "-p", type=float, required=True, help="Market price")
@click.option("--spot", "-S", type=float, required=True, help="Spot price")
@click.option("--strike", "-K", type=float, required=True, help="Strike price")
@click.option("--time", "-T", type=float, required=True, help="Time to expiry (years)")
@click.option("--rate", "-r", type=float, required=True, help="Risk-free rate")
@click.option("--div", "-q", type=float, default=0.0, help="Dividend yield")
@click.option("--type", "-t", type=click.Choice(["call", "put"]), default="call")
def iv(market_price, spot, strike, time, rate, div, type):
    """Solve for implied volatility."""
    try:
        result = implied_volatility(market_price, spot, strike, time, rate, div, type)

        if result.success:
            click.echo(f"\nImplied Volatility: {result.volatility:.4f} ({result.volatility*100:.2f}%)")
            click.echo(f"Method: {result.method}")
            click.echo(f"Iterations: {result.iterations}")
        else:
            click.echo(f"\nSolver failed: {result.message}", err=True)
    except ValueError as e:
        click.echo(f"\nError: {e}", err=True)


if __name__ == "__main__":
    cli()
