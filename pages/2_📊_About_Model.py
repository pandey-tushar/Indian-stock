"""
About Page - Model Details & Technical Information
"""
import streamlit as st

st.set_page_config(
    page_title="About the Model | StockForecast Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .section-box {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .tech-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Explicit sidebar navigation
with st.sidebar:
    st.markdown("## StockForecast Pro")
    st.caption("AI-powered return forecasts with uncertainty bands")
    st.markdown("---")
    st.page_link("pages/1_🏠_Home.py", label="Home", icon="🏠")
    st.page_link("pages/2_📊_About_Model.py", label="About the Model", icon="📊")
    st.markdown("---")
    st.caption("Not financial advice.")

st.title("📊 About the Model")
st.markdown("---")

# Overview
st.markdown("""
<div class="section-box">
    <h2>🎯 Model Overview</h2>
    <p><strong>StockForecast Pro</strong> uses advanced machine learning to predict stock returns over customizable time horizons (1-12 months). 
    The system employs <strong>Gradient Boosting</strong> models trained on stationary technical indicators to generate probabilistic forecasts 
    with confidence intervals.</p>
</div>
""", unsafe_allow_html=True)

# Architecture
st.markdown("""
<div class="section-box">
    <h2>🏗️ Model Architecture</h2>
    <p>The forecasting system consists of <strong>4 specialized models</strong> working together:</p>
    <ul>
        <li><strong>Gradient Boosting Classifier</strong> — Predicts probability of positive return (P_Up)</li>
        <li><strong>Gradient Boosting Regressor (Median)</strong> — Predicts expected return (50th percentile)</li>
        <li><strong>Quantile Regressor (P10)</strong> — Predicts pessimistic scenario (10th percentile)</li>
        <li><strong>Quantile Regressor (P90)</strong> — Predicts optimistic scenario (90th percentile)</li>
    </ul>
    <p>The combination of these models provides both <strong>point estimates</strong> and <strong>uncertainty quantification</strong>.</p>
</div>
""", unsafe_allow_html=True)

# Features
st.markdown("""
<div class="section-box">
    <h2>🔧 Technical Features</h2>
    <p>The model uses <strong>stationary features</strong> (ratios and returns, not raw prices) to ensure robustness across different price levels:</p>
    
    <h3>Trend Indicators</h3>
    <ul>
        <li><strong>Distance from Moving Averages</strong> — Price relative to SMA(20, 50, 200)</li>
        <li><strong>RSI (Relative Strength Index)</strong> — Momentum oscillator (0-100, normalized)</li>
        <li><strong>MACD Histogram</strong> — Trend-following momentum indicator</li>
    </ul>
    
    <h3>Volatility & Risk</h3>
    <ul>
        <li><strong>ATR (Average True Range)</strong> — Normalized volatility measure</li>
        <li><strong>Realized Volatility</strong> — Historical volatility over multiple windows</li>
        <li><strong>Volatility Ratios</strong> — Short-term vs long-term volatility</li>
    </ul>
    
    <h3>Momentum & Volume</h3>
    <ul>
        <li><strong>Rate of Change (ROC)</strong> — 21-day price momentum</li>
        <li><strong>Momentum Ratio</strong> — Recent vs longer-term momentum</li>
        <li><strong>Volume-Price Divergence</strong> — Volume trend vs price trend</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Adaptive Regimes
st.markdown("""
<div class="section-box">
    <h2>🔄 Adaptive Model Regimes</h2>
    <p>The system automatically selects the appropriate model complexity based on available data:</p>
    
    <h3>🟢 Full History Model (≥500 trading days)</h3>
    <ul>
        <li>Uses all technical indicators (SMA 50/200, full feature set)</li>
        <li>Includes classifier for trend probability</li>
        <li>Best accuracy and lowest uncertainty</li>
        <li>Recommended for established stocks</li>
    </ul>
    
    <h3>🟡 Medium History Model (180-500 days)</h3>
    <ul>
        <li>Uses simplified feature set (SMA 50, core indicators)</li>
        <li>Still includes classifier</li>
        <li>Good accuracy for mid-cap stocks</li>
    </ul>
    
    <h3>🟠 Short History Model (<180 days)</h3>
    <ul>
        <li>Minimal feature set (SMA 20, basic indicators)</li>
        <li>No classifier (uses regression only)</li>
        <li>Designed for recent IPOs</li>
        <li>Higher uncertainty (wider confidence bands)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Training & Validation
st.markdown("""
<div class="section-box">
    <h2>✅ Training & Validation</h2>
    <h3>Hyperparameters</h3>
    <ul>
        <li><strong>Learning Rate:</strong> 0.03 (conservative, prevents overfitting)</li>
        <li><strong>Max Depth:</strong> 2-3 (shallow trees for generalization)</li>
        <li><strong>Min Samples Split:</strong> 20 (regularization)</li>
        <li><strong>Subsample:</strong> 0.8 (stochastic gradient boosting)</li>
        <li><strong>N Estimators:</strong> 150-200 (ensemble size)</li>
    </ul>
    
    <h3>Validation Strategy</h3>
    <ul>
        <li><strong>Time-Series Split:</strong> 5-fold walk-forward validation</li>
        <li><strong>No Look-Ahead Bias:</strong> Models only trained on past data</li>
        <li><strong>Out-of-Sample Metrics:</strong> Directional accuracy, MAE on log-returns</li>
        <li><strong>Cross-Validation:</strong> Respects temporal order (no random shuffling)</li>
    </ul>
    
    <h3>Performance Metrics</h3>
    <ul>
        <li><strong>Directional Accuracy:</strong> 45-69% (varies by stock volatility)</li>
        <li><strong>MAE (Log-Return):</strong> 0.07-0.41 (lower is better)</li>
        <li><strong>Best Performance:</strong> Large-cap stocks with ≥2 years history</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Risk-Adjusted Scoring
st.markdown("""
<div class="section-box">
    <h2>📈 Risk-Adjusted Scoring</h2>
    <p>The model doesn't just predict returns—it also provides risk-aware rankings:</p>
    
    <h3>Edge Calculation</h3>
    <p><strong>Edge = Forecast × (2×P_Up - 1)</strong></p>
    <p>This downweights low-conviction signals. A 10% forecast with 90% probability is weighted higher than a 15% forecast with 50% probability.</p>
    
    <h3>Risk-Adjusted Score</h3>
    <p><strong>Score = Edge / Realized Volatility</strong></p>
    <p>This creates a pseudo-Sharpe ratio. Stocks with high expected return and low volatility rank higher.</p>
    
    <h3>Uncertainty Bands</h3>
    <p>The P10-P90 range provides an 80% confidence interval. Wide bands indicate high model uncertainty (e.g., recent IPOs, volatile sectors).</p>
</div>
""", unsafe_allow_html=True)

# Limitations
st.markdown("""
<div class="section-box">
    <h2>⚠️ Model Limitations</h2>
    <ul>
        <li><strong>No Fundamental Analysis:</strong> Model is purely technical (price/volume patterns)</li>
        <li><strong>No News/Sentiment:</strong> Doesn't incorporate earnings, news, or market sentiment</li>
        <li><strong>No Sector/Industry Context:</strong> Treats all stocks equally (no sector-specific models)</li>
        <li><strong>Assumes Stationarity:</strong> May break down during regime changes (crashes, bubbles)</li>
        <li><strong>Short History Risk:</strong> Recent IPOs have higher uncertainty and potential overfitting</li>
        <li><strong>Market-Wide Events:</strong> Cannot predict black swan events or systemic shocks</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Technology Stack
st.markdown("""
<div class="section-box">
    <h2>💻 Technology Stack</h2>
    <p>
        <span class="tech-badge">Python 3.13</span>
        <span class="tech-badge">scikit-learn</span>
        <span class="tech-badge">Gradient Boosting</span>
        <span class="tech-badge">yfinance</span>
        <span class="tech-badge">pandas</span>
        <span class="tech-badge">numpy</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Plotly</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("---")
st.markdown("""
<div style="background: #fef3c7; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b;">
    <strong>⚠️ Important Disclaimer</strong><br>
    This model is for <strong>educational and informational purposes only</strong>. It does not constitute financial advice. 
    Past performance does not guarantee future results. Always conduct your own research and consult with qualified financial advisors 
    before making investment decisions. The model has known limitations and should not be the sole basis for investment decisions.
</div>
""", unsafe_allow_html=True)

