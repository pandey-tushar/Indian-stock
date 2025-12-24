"""
Home Page - Main Forecast Interface
"""
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from quant_lite import run_one, Config

# Page config (ensure sidebar is visible)
st.set_page_config(
    page_title="Home | StockForecast Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.95;
        margin-bottom: 2rem;
    }
    
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 2rem auto;
        max-width: 700px;
    }
    
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #667eea;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .forecast-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.9rem;
        border-radius: 12px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .badge-success {
        background: #10b981;
        color: white;
    }
    
    .badge-warning {
        background: #f59e0b;
        color: white;
    }
    
    .badge-info {
        background: #3b82f6;
        color: white;
    }
    
    .example-ticker {
        display: inline-block;
        background: #f0f0f0;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        color: #667eea;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-ticker:hover {
        background: #667eea;
        color: white;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">StockForecast Pro</div>
    <div class="hero-subtitle">AI-Powered Stock Analysis & Return Predictions</div>
    <p style="font-size: 1rem; opacity: 0.9;">Powered by advanced machine learning • Real-time market analysis • Professional-grade forecasts</p>
</div>
""", unsafe_allow_html=True)

# Main Search Section
st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown("### 🔍 Enter Stock Ticker")

# Month slider (1-12 months)
col_slider, col_info = st.columns([2, 1])
with col_slider:
    horizon_months = st.slider(
        "Forecast Period (Months)",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
        help="Select how many months ahead you want to forecast"
    )
with col_info:
    st.markdown(f"<br><div style='text-align: center; color: #667eea; font-weight: 600; font-size: 1.2rem;'>{horizon_months} {'Month' if horizon_months == 1 else 'Months'}</div>", unsafe_allow_html=True)

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = st.text_input(
        "",
        placeholder="RELIANCE, TCS, INFY, OLAELEC, PAYTM...",
        label_visibility="collapsed",
        key="ticker_input"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Analyze →", type="primary", use_container_width=True)

# Example tickers
st.markdown("""
<div style="margin-top: 1rem;">
    <strong>Popular tickers:</strong><br>
    <span class="example-ticker">RELIANCE</span>
    <span class="example-ticker">TCS</span>
    <span class="example-ticker">INFY</span>
    <span class="example-ticker">HDFCBANK</span>
    <span class="example-ticker">OLAELEC</span>
    <span class="example-ticker">PAYTM</span>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.markdown("## StockForecast Pro")
    st.caption("AI-powered return forecasts with uncertainty bands")
    st.markdown("---")
    st.page_link("pages/1_🏠_Home.py", label="Home", icon="🏠")
    st.page_link("pages/2_📊_About_Model.py", label="About the Model", icon="📊")
    st.markdown("---")

    st.markdown("### ⚙️ Quick Settings")
    default_suffix = st.selectbox("Exchange", [".NS (NSE)", ".BO (BSE)"], index=0).split()[0]
    
    st.markdown("---")
    st.markdown("""
    ### 📊 About
    **StockForecast Pro** uses advanced ML models to predict stock returns.
    
    - ✅ Real-time analysis
    - ✅ Confidence intervals
    - ✅ Risk assessment
    """)

# Main Logic
if run_btn and ticker_input:
    with st.spinner("🔄 Analyzing market data and generating forecast..."):
        ticker = ticker_input.strip().upper()
        if not ticker.endswith((".NS", ".BO")):
            ticker = f"{ticker}{default_suffix}"
        
        result = get_forecast_cached(ticker, horizon_months, default_suffix)
        
        if result is None and default_suffix == ".NS":
            base = ticker[:-3] if ticker.endswith(".NS") else ticker
            ticker_fallback = f"{base}.BO"
            st.info(f"🔄 Trying BSE exchange: {ticker_fallback}")
            result = get_forecast_cached(ticker_fallback, horizon_months, "")
            if result:
                ticker = ticker_fallback
        
        if result is None:
            st.error(f"""
            ### ❌ Analysis Unavailable
            
            Unable to analyze **{ticker_input}**. This may be due to:
            - Ticker not found on Yahoo Finance
            - Insufficient trading history (<40 days)
            - Data quality issues
            
            **Suggestions:**
            - Verify the ticker symbol
            - Try adding `.NS` or `.BO` suffix
            - Check if the stock is actively traded
            """)
        else:
            # Success Header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
                <h2 style="color: white; margin: 0;">✅ Analysis Complete: {result['Stock']}</h2>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">Current Price: <strong>₹{result['Price']:,.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model Status
            regime = result.get("Model_Regime", "unknown")
            data_rows = result.get("Data_Rows", 0)
            if regime == "full":
                st.markdown('<span class="status-badge badge-success">✓ Full History Model</span>', unsafe_allow_html=True)
                st.caption(f"Analyzed {data_rows:,} trading days with comprehensive technical indicators")
            elif regime == "medium":
                st.markdown('<span class="status-badge badge-warning">⚠ Medium History Model</span>', unsafe_allow_html=True)
                st.caption(f"Analyzed {data_rows:,} trading days (limited indicators)")
            else:
                st.markdown('<span class="status-badge badge-info">ℹ Short History Model</span>', unsafe_allow_html=True)
                st.caption(f"Analyzed {data_rows:,} trading days (recent IPO, higher uncertainty)")
            
            st.markdown("---")
            
            # Key Metrics
            st.markdown('<div class="section-title">📊 Forecast Summary</div>', unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            
            forecast_med = result['Forecast_Med_%']
            p_up = result['P_Up'] * 100
            vol = result['Risk_Vol_%']
            
            with m1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Expected Return</div>
                    <div class="metric-value" style="color: {};">{:.2f}%</div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                        {} outlook
                    </div>
                </div>
                """.format(
                    "#10b981" if forecast_med > 0 else "#ef4444",
                    forecast_med,
                    "Bullish" if forecast_med > 0 else "Bearish"
                ), unsafe_allow_html=True)
            
            with m2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Trend Probability</div>
                    <div class="metric-value" style="color: {};">{:.1f}%</div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                        {} confidence
                    </div>
                </div>
                """.format(
                    "#10b981" if p_up > 60 else "#ef4444" if p_up < 40 else "#f59e0b",
                    p_up,
                    "High" if p_up > 60 else "Low" if p_up < 40 else "Moderate"
                ), unsafe_allow_html=True)
            
            with m3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Risk (Volatility)</div>
                    <div class="metric-value" style="color: {};">{:.2f}%</div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                        {} risk
                    </div>
                </div>
                """.format(
                    "#10b981" if vol < 15 else "#ef4444" if vol > 25 else "#f59e0b",
                    vol,
                    "Low" if vol < 15 else "High" if vol > 25 else "Moderate"
                ), unsafe_allow_html=True)
            
            # Forecast Chart
            st.markdown("---")
            st.markdown(f'<div class="section-title">📈 {horizon_months}-Month Return Forecast Distribution</div>', unsafe_allow_html=True)
            
            med = result['Forecast_Med_%']
            p10 = result['Forecast_P10_%']
            p90 = result['Forecast_P90_%']
            band_width = result['Uncertainty_Band_%']
            
            fig = go.Figure()
            
            # Uncertainty band
            fig.add_trace(go.Scatter(
                x=[p10, p90, p90, p10, p10],
                y=[0.3, 0.3, 0.7, 0.7, 0.3],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
            ))
            
            # Forecast points
            fig.add_trace(go.Scatter(
                x=[p10, med, p90],
                y=[0.5, 0.5, 0.5],
                mode='markers+lines',
                marker=dict(
                    size=[18, 24, 18],
                    color=['#ef4444', '#667eea', '#10b981'],
                    line=dict(width=3, color='white')
                ),
                line=dict(color='#667eea', width=4),
                hovertext=[f'Pessimistic: {p10:.1f}%', f'Expected: {med:.1f}%', f'Optimistic: {p90:.1f}%'],
                hoverinfo='text',
                showlegend=False,
            ))
            
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#999", opacity=0.5)
            
            fig.update_layout(
                title=f"{result['Stock']} — {horizon_months}-Month Return Forecast",
                xaxis_title="Expected Return (%)",
                yaxis_visible=False,
                height=350,
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f0f0f0', zeroline=True, zerolinecolor='#999'),
                font=dict(family="Inter, sans-serif", size=12),
                title_font=dict(size=18, color="#1a1a1a"),
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Interpretation
            uncertainty_desc = "High uncertainty" if band_width > 50 else "Moderate uncertainty" if band_width > 30 else "Low uncertainty"
            trend_icon = "🟢" if p_up > 60 else "🔴" if p_up < 40 else "🟡"
            trend_text = "Bullish" if p_up > 60 else "Bearish" if p_up < 40 else "Neutral"
            
            st.markdown(f"""
            <div class="forecast-box">
                <h3 style="margin-top: 0; color: #1a1a1a;">💡 Forecast Interpretation</h3>
                <div style="line-height: 1.8; color: #333;">
                    <p><strong>Expected Return:</strong> <span style="color: #667eea; font-weight: 600;">{med:.2f}%</span> over the next {horizon_months} months</p>
                    <p><strong>Confidence Range:</strong> {p10:.1f}% to {p90:.1f}% (80% probability the return will fall within this range)</p>
                    <p><strong>Uncertainty Level:</strong> {band_width:.1f}% band width — {uncertainty_desc}</p>
                    <p><strong>Trend Direction:</strong> {trend_icon} {trend_text} ({p_up:.1f}% probability of positive return)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
            <div style="background: #fef3c7; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b; margin-top: 2rem;">
                <strong>⚠️ Important Disclaimer</strong><br>
                This forecast is generated by machine learning models for informational purposes only. 
                Past performance does not guarantee future results. This is not financial advice. 
                Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
            </div>
            """, unsafe_allow_html=True)

# Landing Page (when no input)
elif not run_btn:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🏢 Large Cap Stocks
        - RELIANCE
        - TCS
        - INFY
        - HDFCBANK
        - ICICIBANK
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Growth Stocks
        - OLAELEC
        - PAYTM
        - FIRSTCRY
        - ZOMATO
        """)
    
    with col3:
        st.markdown("""
        ### 📊 ETFs & Index Funds
        - AXISNIFTY
        - NIFTYBEES
        - BANKBEES
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 16px; margin: 2rem 0;">
        <h3 style="color: #1a1a1a;">✨ How It Works</h3>
        <ol style="line-height: 2; color: #333;">
            <li><strong>Enter Ticker</strong> — Type any Indian stock symbol (NSE or BSE)</li>
            <li><strong>Select Period</strong> — Choose forecast horizon from 1 to 12 months</li>
            <li><strong>AI Analysis</strong> — Our advanced ML models analyze historical patterns, technical indicators, and market volatility</li>
            <li><strong>Get Forecast</strong> — Receive a professional-grade return forecast with confidence intervals</li>
        </ol>
        <p style="margin-top: 1rem; color: #666;">
            <strong>Smart Adaptation:</strong> The model automatically adjusts based on data availability — 
            full history for established stocks, optimized models for recent IPOs.
        </p>
    </div>
    """, unsafe_allow_html=True)

