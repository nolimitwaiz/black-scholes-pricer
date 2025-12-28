"""
Streamlit web interface for options pricing toolkit.

Interactive UI with tabs for:
- Option pricing calculator
- Greeks visualization
- Implied volatility solver
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.core.black_scholes import black_scholes_price, calculate_greeks, delta, gamma, vega
from src.solvers.implied_vol import implied_volatility

st.set_page_config(page_title="Options Pricing Toolkit", layout="wide")

st.title("Options Pricing Toolkit")
st.markdown("Professional Black-Scholes options pricing and analysis")

# Sidebar parameters
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Spot Price (S)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=0.01)
T = st.sidebar.slider("Time to Expiry (years)", 0.01, 5.0, 1.0)
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 20.0, 5.0) / 100
q = st.sidebar.slider("Dividend Yield (%)", 0.0, 10.0, 0.0) / 100
sigma = st.sidebar.slider("Volatility (%)", 1.0, 200.0, 25.0) / 100
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# Main tabs
tab1, tab2, tab3 = st.tabs(["Pricing & Greeks", "Greeks Sensitivity", "Implied Volatility"])

with tab1:
    st.header("Option Valuation")

    col1, col2 = st.columns(2)

    with col1:
        price = black_scholes_price(S, K, T, r, sigma, q, option_type)
        st.metric(label=f"{option_type.capitalize()} Price", value=f"${price:.4f}")

        greeks_vals = calculate_greeks(S, K, T, r, sigma, q, option_type)

        st.subheader("Greeks")
        greeks_df = pd.DataFrame({
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [
                f"{greeks_vals.delta:.6f}",
                f"{greeks_vals.gamma:.6f}",
                f"{greeks_vals.vega:.6f}",
                f"{greeks_vals.theta:.6f}",
                f"{greeks_vals.rho:.6f}"
            ],
            "Description": [
                "Price change per $1 spot move",
                "Delta change per $1 spot move",
                "Price change per 1% vol move",
                "Price change per day",
                "Price change per 1% rate move"
            ]
        })
        st.table(greeks_df)

with tab2:
    st.header("Greeks Sensitivity Analysis")

    # Delta vs Spot
    spot_range = np.linspace(S*0.7, S*1.3, 50)
    delta_values = [delta(s, K, T, r, sigma, q, option_type) for s in spot_range]

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=spot_range, y=delta_values, name="Delta"))
    fig_delta.update_layout(title="Delta vs Spot Price", xaxis_title="Spot Price", yaxis_title="Delta")
    st.plotly_chart(fig_delta, use_container_width=True)

    # Gamma vs Spot
    gamma_values = [gamma(s, K, T, r, sigma, q) for s in spot_range]

    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Scatter(x=spot_range, y=gamma_values, name="Gamma", line=dict(color="orange")))
    fig_gamma.update_layout(title="Gamma vs Spot Price", xaxis_title="Spot Price", yaxis_title="Gamma")
    st.plotly_chart(fig_gamma, use_container_width=True)

with tab3:
    st.header("Implied Volatility Solver")

    market_price = st.number_input("Market Price", value=price, min_value=0.01)

    if st.button("Solve for Implied Volatility"):
        try:
            result = implied_volatility(market_price, S, K, T, r, q, option_type)

            if result.success:
                st.success(f"Implied Volatility: {result.volatility:.4f} ({result.volatility*100:.2f}%)")
                st.info(f"Method: {result.method} | Iterations: {result.iterations}")
            else:
                st.error(f"Solver failed: {result.message}")
        except ValueError as e:
            st.error(f"Error: {e}")
