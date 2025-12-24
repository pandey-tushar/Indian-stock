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

# Page config
st.set_page_config(
    page_title="StockForecast AI | Indian Stock Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .forecast-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4 0%, #0d47a1 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0d47a1 0%, #1f77b4 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 StockForecast AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get AI-powered 3-month return forecasts for any Indian stock</p>', unsafe_allow_html=True)

# Main input section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### 🔍 Enter Stock Ticker")
    ticker_input = st.text_input(
        "", 
        placeholder="e.g., RELIANCE, TCS, INFY, OLAELEC",
        label_visibility="collapsed",
        help="Enter NSE/BSE ticker. If no suffix, .NS is tried first, then .BO"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        run_btn = st.button("🚀 Get Forecast", type="primary", use_container_width=True)

# Sidebar (collapsed by default)
with st.sidebar:
    st.header("⚙️ Settings")
    default_suffix = st.selectbox("Default exchange", [".NS (NSE)", ".BO (BSE)"], index=0).split()[0]
    horizon_months = st.slider("Forecast horizon (months)", 1, 6, 3)
    
    st.markdown("---")
    st.markdown("**Powered by**: Gradient Boosting ML")
    st.markdown("**Features**: Technical indicators, momentum, volatility")
    st.markdown("**Auto-detects**: Full/short history mode")

# Main logic
if run_btn and ticker_input:
    with st.spinner(f"🔮 Analyzing {ticker_input}... This may take 10-15 seconds for first-time lookups."):
        # Try primary ticker
        ticker = ticker_input.strip().upper()
        if not ticker.endswith((".NS", ".BO")):
            ticker = f"{ticker}{default_suffix}"
        
        result = get_forecast_cached(ticker, horizon_months, default_suffix)
        
        # Fallback to .BO if .NS fails
        if result is None and default_suffix == ".NS":
            base = ticker[:-3] if ticker.endswith(".NS") else ticker
            ticker_fallback = f"{base}.BO"
            st.info(f"💡 Trying BSE exchange: {ticker_fallback}")
            result = get_forecast_cached(ticker_fallback, horizon_months, "")
            if result:
                ticker = ticker_fallback
        
        if result is None:
            st.error(f"""
            ❌ **Could not analyze {ticker_input}**
            
            **Possible reasons:**
            - Ticker not found on Yahoo Finance
            - Insufficient trading history (<40 days)
            - Data quality issues
            
            **Try:** Adding `.NS` or `.BO` suffix, or check the ticker symbol.
            """)
        else:
            st.success(f"✅ **Analysis complete for {result['Stock']}**")
            
            # Model regime indicator
            regime = result.get("Model_Regime", "unknown")
            data_rows = result.get("Data_Rows", 0)
            if regime == "full":
                st.info(f"🟢 **Full History Model** — {data_rows:,} trading days analyzed")
            elif regime == "medium":
                st.warning(f"🟡 **Medium History Model** — {data_rows:,} trading days (limited indicators)")
            else:
                st.warning(f"🟠 **Short History Model** — {data_rows:,} trading days (recent IPO, higher uncertainty)")
            
            # Key metrics in prominent cards
            st.markdown("---")
            st.markdown("### 📊 Forecast Summary")
            
            m1, m2, m3 = st.columns(3)
            
            forecast_med = result['Forecast_Med_%']
            p_up = result['P_Up'] * 100
            vol = result['Risk_Vol_%']
            
            with m1:
                st.metric(
                    f"{horizon_months}-Month Forecast",
                    f"{forecast_med:+.2f}%",
                    delta=f"{'Bullish' if forecast_med > 0 else 'Bearish'} outlook",
                    delta_color="normal" if forecast_med > 0 else "inverse",
                )
            
            with m2:
                st.metric(
                    "Trend Probability",
                    f"{p_up:.1f}%",
                    delta="High confidence" if p_up > 60 else ("Low confidence" if p_up < 40 else "Neutral"),
                    delta_color="normal" if p_up > 60 else ("inverse" if p_up < 40 else "off"),
                )
            
            with m3:
                st.metric(
                    "Risk (Volatility)",
                    f"{vol:.2f}%",
                    delta="Low risk" if vol < 15 else ("High risk" if vol > 25 else "Moderate risk"),
                    delta_color="normal" if vol < 15 else ("inverse" if vol > 25 else "off"),
                )
            
            # Current price
            st.markdown(f"**Current Price:** ₹{result['Price']:,.2f}")
            
            # Forecast band visualization
            st.markdown("---")
            st.markdown("### 📈 Expected Return Range")
            
            med = result['Forecast_Med_%']
            p10 = result['Forecast_P10_%']
            p90 = result['Forecast_P90_%']
            band_width = result['Uncertainty_Band_%']
            
            fig = go.Figure()
            
            # Add uncertainty band (shaded area)
            fig.add_trace(go.Scatter(
                x=[p10, p90, p90, p10, p10],
                y=[0.3, 0.3, 0.7, 0.7, 0.3],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
            ))
            
            # Add forecast points
            fig.add_trace(go.Scatter(
                x=[p10, med, p90],
                y=[0.5, 0.5, 0.5],
                mode='markers+lines',
                name='Forecast Range',
                marker=dict(
                    size=[15, 20, 15], 
                    color=['#d62728', '#1f77b4', '#2ca02c'],
                    line=dict(width=2, color='white')
                ),
                line=dict(color='#1f77b4', width=3),
                text=[f'Pessimistic: {p10:.1f}%', f'Expected: {med:.1f}%', f'Optimistic: {p90:.1f}%'],
                hoverinfo='text',
            ))
            
            # Zero line
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black", opacity=0.5)
            
            # Add annotations
            fig.add_annotation(x=p10, y=0.3, text=f"{p10:.1f}%", showarrow=False, font=dict(color='#d62728', size=12))
            fig.add_annotation(x=med, y=0.2, text=f"{med:.1f}%", showarrow=False, font=dict(color='#1f77b4', size=14, weight='bold'))
            fig.add_annotation(x=p90, y=0.3, text=f"{p90:.1f}%", showarrow=False, font=dict(color='#2ca02c', size=12))
            
            fig.update_layout(
                title=f"{result['Stock']} — {horizon_months}-Month Return Forecast",
                xaxis_title="Expected Return (%)",
                yaxis_visible=False,
                height=300,
                showlegend=False,
                hovermode='closest',
                plot_bgcolor='white',
                xaxis=dict(gridcolor='lightgray', zeroline=True),
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Interpretation
            st.markdown(f"""
            <div class="forecast-box">
                <h4>💡 What This Means</h4>
                <p><strong>Expected Return:</strong> {med:+.2f}% over {horizon_months} months</p>
                <p><strong>Confidence Range:</strong> {p10:.1f}% to {p90:.1f}% (80% probability)</p>
                <p><strong>Uncertainty:</strong> {band_width:.1f}% band width {'(High uncertainty)' if band_width > 50 else '(Moderate uncertainty)' if band_width > 30 else '(Low uncertainty)'}</p>
                <p><strong>Trend Direction:</strong> {'🟢 Bullish' if p_up > 60 else '🔴 Bearish' if p_up < 40 else '🟡 Neutral'} ({p_up:.1f}% probability of positive return)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Advanced metrics (collapsible)
            with st.expander("📋 Advanced Metrics & Model Details"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Forecast Metrics**")
                    st.write(f"- **Median Forecast**: {result['Forecast_Med_%']:.2f}%")
                    st.write(f"- **P10 (Pessimistic)**: {result['Forecast_P10_%']:.2f}%")
                    st.write(f"- **P90 (Optimistic)**: {result['Forecast_P90_%']:.2f}%")
                    st.write(f"- **Uncertainty Band**: {result['Uncertainty_Band_%']:.2f}%")
                
                with col_b:
                    st.markdown("**Risk Metrics**")
                    st.write(f"- **Realized Volatility**: {result['Risk_Vol_%']:.2f}%")
                    st.write(f"- **Edge (Prob-weighted)**: {result['Edge_%']:.2f}%")
                    st.write(f"- **Risk-Adjusted Score**: {result['Risk_Adj_Score']:.4f}")
                    st.write(f"- **Model Regime**: {result.get('Model_Regime', 'N/A')}")
                
                if 'oos_dir_acc' in result:
                    st.markdown("**Model Validation**")
                    st.write(f"- **Out-of-Sample Direction Accuracy**: {result['oos_dir_acc']:.1%}")
                    st.write(f"- **Out-of-Sample MAE**: {result['oos_mae_logret']:.4f}")
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #999; font-size: 0.85em; padding: 1rem;'>
            ⚠️ <strong>Disclaimer:</strong> This forecast is generated by machine learning models for informational purposes only. 
            Past performance does not guarantee future results. Not financial advice. Always do your own research.
            </div>
            """, unsafe_allow_html=True)

# If no input yet, show example
elif not run_btn:
    st.markdown("---")
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        st.markdown("""
        ### 🏢 Large Caps
        - **RELIANCE** — Reliance Industries
        - **TCS** — Tata Consultancy
        - **INFY** — Infosys
        - **HDFCBANK** — HDFC Bank
        """)
    
    with col_ex2:
        st.markdown("""
        ### 🚀 Recent IPOs
        - **OLAELEC** — Ola Electric
        - **PAYTM** — Paytm
        - **FIRSTCRY** — FirstCry
        - **KRONOX** — Kronox Lab
        """)
    
    with col_ex3:
        st.markdown("""
        ### 📊 ETFs
        - **AXISNIFTY** — Axis Nifty ETF
        - **NIFTYBEES** — Nifty ETF
        """)
    
    st.markdown("---")
    st.markdown("""
    ### ✨ How It Works
    
    1. **Enter a ticker** — Type any Indian stock symbol (NSE or BSE)
    2. **AI Analysis** — Our ML model analyzes historical patterns, technical indicators, and volatility
    3. **Get Forecast** — Receive a 3-month return forecast with confidence bands
    
    **Model automatically adapts** to data availability:
    - **Full history** (≥500 days): Uses all technical indicators
    - **Short history** (<180 days): Simplified model for recent IPOs
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 2rem 0;'>
    <p>Built with ❤️ using Streamlit & Machine Learning</p>
    <p>© 2024 StockForecast AI | Not affiliated with any financial institution</p>
</div>
""", unsafe_allow_html=True)

# Cached function
@st.cache_data(ttl=86400, show_spinner=False)
def get_forecast_cached(ticker: str, horizon_months: int, default_suffix: str):
    """Cached wrapper around run_one to avoid re-training models."""
    cfg = Config(
        horizon_months=horizon_months,
        default_suffix=default_suffix,
        cache_dir=Path("data_cache"),
    )
    return run_one(ticker, cfg=cfg, refresh=False)
