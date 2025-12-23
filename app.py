#!/usr/bin/env python3
"""
Streamlit web app for stock forecast lookup (3-month horizon).
Free deployment: streamlit.io/cloud
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from quant_lite import run_one, Config

# Streamlit caching for model results (24 hour TTL)
@st.cache_data(ttl=86400, show_spinner=False)
def get_forecast_cached(ticker: str, horizon_months: int, default_suffix: str):
    """Cached wrapper around run_one to avoid re-training models."""
    cfg = Config(
        horizon_months=horizon_months,
        default_suffix=default_suffix,
        cache_dir=Path("data_cache"),
    )
    return run_one(ticker, cfg=cfg, refresh=False)

# Page config
st.set_page_config(
    page_title="Indian Stock 3M Forecaster",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Indian Stock 3-Month Forecaster")
st.markdown("""
Enter an Indian stock ticker (e.g., `RELIANCE.NS`, `TCS`, `INFY.NS`) to get:
- **3-month return forecast** (median + p10/p90 bands)
- **Trend probability** (P_Up)
- **Realized volatility** (horizon-scaled)

The model automatically uses **full-history** (≥500 days) or **short-history** mode based on data availability.
""")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Settings")
    default_suffix = st.selectbox("Default suffix", [".NS", ".BO", ""], index=0)
    fallback_suffix = st.text_input("Fallback suffix (if primary fails)", ".BO")
    horizon_months = st.slider("Forecast horizon (months)", 1, 6, 3)
    
    st.markdown("---")
    st.markdown("**Model**: Gradient Boosting + Quantile Regression")
    st.markdown("**Features**: Stationary (returns, RSI, MACD, ATR, momentum)")

# Main input
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "Stock Ticker", 
        placeholder="e.g., RELIANCE, TCS.NS, INFY.NS",
        help="Enter NSE/BSE ticker. If no suffix, .NS is tried first, then .BO"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Forecast", type="primary", use_container_width=True)

if run_btn and ticker_input:
    with st.spinner(f"Fetching data and training model for {ticker_input}..."):
        # Try primary ticker
        ticker = ticker_input.strip().upper()
        if not ticker.endswith((".NS", ".BO")):
            ticker = f"{ticker}{default_suffix}"
        
        result = get_forecast_cached(ticker, horizon_months, default_suffix)
        
        # Fallback to .BO if .NS fails
        if result is None and fallback_suffix and ticker.endswith(default_suffix):
            base = ticker[:-len(default_suffix)]
            ticker_fallback = f"{base}{fallback_suffix}"
            st.info(f"Primary ticker failed, trying fallback: {ticker_fallback}")
            result = get_forecast_cached(ticker_fallback, horizon_months, "")  # No default suffix for fallback
            if result:
                ticker = ticker_fallback
        
        if result is None:
            st.error(f"❌ Could not fetch or model data for **{ticker}**. Possible reasons:\n"
                     f"- Ticker not found on Yahoo Finance\n"
                     f"- Insufficient history (<40 trading days)\n"
                     f"- Data quality issues")
        else:
            st.success(f"✅ Forecast generated for **{result['Stock']}**")
            
            # Display model regime
            regime = result.get("Model_Regime", "unknown")
            data_rows = result.get("Data_Rows", 0)
            if regime == "full":
                st.info(f"🟢 **Full-history model** (≥500 days, {data_rows} rows available)")
            elif regime == "medium":
                st.warning(f"🟡 **Medium-history model** (180–500 days, {data_rows} rows available)")
            else:
                st.warning(f"🟠 **Short-history model** (<180 days, {data_rows} rows available) — higher uncertainty")
            
            # Key metrics in columns
            st.markdown("### 📊 Key Metrics")
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(
                    "Current Price",
                    f"₹{result['Price']:.2f}",
                )
            
            with m2:
                forecast_med = result['Forecast_Med_%']
                st.metric(
                    f"{horizon_months}M Forecast (Median)",
                    f"{forecast_med:+.2f}%",
                    delta=None,
                )
            
            with m3:
                p_up = result['P_Up'] * 100
                st.metric(
                    "Trend Probability (P_Up)",
                    f"{p_up:.1f}%",
                    delta="Bullish" if p_up > 55 else ("Bearish" if p_up < 45 else "Neutral"),
                    delta_color="normal" if p_up > 55 else "inverse",
                )
            
            with m4:
                vol = result['Risk_Vol_%']
                st.metric(
                    f"Risk (Realized Vol, {horizon_months}M)",
                    f"{vol:.2f}%",
                )
            
            # Forecast band visualization
            st.markdown("### 📈 Forecast Distribution")
            
            med = result['Forecast_Med_%']
            p10 = result['Forecast_P10_%']
            p90 = result['Forecast_P90_%']
            
            fig = go.Figure()
            
            # Add uncertainty band
            fig.add_trace(go.Scatter(
                x=[p10, med, p90],
                y=[0.5, 0.5, 0.5],
                mode='markers+lines',
                name='Forecast Range',
                marker=dict(size=[12, 16, 12], color=['red', 'blue', 'green']),
                line=dict(color='gray', width=2, dash='dash'),
                text=[f'P10: {p10:.2f}%', f'Median: {med:.2f}%', f'P90: {p90:.2f}%'],
                hoverinfo='text',
            ))
            
            # Zero line
            fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="black")
            
            fig.update_layout(
                title=f"{ticker} — {horizon_months}-Month Return Forecast",
                xaxis_title="Return (%)",
                yaxis_visible=False,
                height=250,
                showlegend=False,
                hovermode='closest',
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Detailed metrics table
            with st.expander("📋 Detailed Metrics"):
                details = pd.DataFrame({
                    "Metric": [
                        "Stock",
                        "Current Price (₹)",
                        f"Forecast Horizon (days)",
                        "P_Up (probability return > 0)",
                        "Forecast Median (%)",
                        "Forecast P10 (%)",
                        "Forecast P90 (%)",
                        "Uncertainty Band Width (%)",
                        f"Risk (Realized Vol, {horizon_months}M) (%)",
                        "Edge (prob-weighted) (%)",
                        "Risk-Adj Score",
                        "Model Regime",
                        "Data Rows Available",
                    ],
                    "Value": [
                        str(result['Stock']),
                        f"{result['Price']:.2f}",
                        str(result['Horizon_Days']),
                        f"{result['P_Up']:.4f}",
                        f"{result['Forecast_Med_%']:.2f}",
                        f"{result['Forecast_P10_%']:.2f}",
                        f"{result['Forecast_P90_%']:.2f}",
                        f"{result['Uncertainty_Band_%']:.2f}",
                        f"{result['Risk_Vol_%']:.2f}",
                        f"{result['Edge_%']:.2f}",
                        f"{result['Risk_Adj_Score']:.4f}",
                        str(result.get('Model_Regime', 'N/A')),
                        str(result.get('Data_Rows', 'N/A')),
                    ]
                })
                st.dataframe(details, width="stretch", hide_index=True)
            
            # Interpretation guide
            with st.expander("💡 How to Interpret"):
                st.markdown("""
                **Forecast Median**: Model's best estimate of return in {0} months.
                
                **P10 / P90**: 10th and 90th percentile forecasts. Wide bands = high uncertainty.
                
                **P_Up**: Probability the return will be positive. >60% = bullish, <40% = bearish.
                
                **Risk (Realized Vol)**: Historical volatility scaled to the forecast horizon. Higher = riskier.
                
                **Edge**: Forecast scaled by conviction (2×P_Up - 1). Downweights low-probability signals.
                
                **Risk-Adj Score**: Edge / Vol. Higher = better risk-adjusted expected return.
                
                **Model Regime**:
                - **Full**: ≥500 days data, uses all technical indicators
                - **Medium**: 180–500 days, simplified features
                - **Short**: <180 days (recent IPOs), higher overfitting risk
                """.format(horizon_months))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
Built with Streamlit | Model: Gradient Boosting + Quantile Regression | 
<a href='https://github.com' target='_blank'>GitHub</a> | Not financial advice
</div>
""", unsafe_allow_html=True)

