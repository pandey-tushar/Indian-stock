#!/usr/bin/env python3
"""
StockForecast Pro - Main App Entry Point
Multi-page Streamlit application
"""
import streamlit as st

# Page config
st.set_page_config(
    page_title="StockForecast Pro | AI-Powered Stock Predictions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS (applied globally)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding-top: 1rem;
    }
    
    .footer {
        text-align: center;
        padding: 3rem 0;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 4rem;
    }
    
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Explicit sidebar navigation (so it's always obvious)
with st.sidebar:
    st.markdown("## StockForecast Pro")
    st.caption("AI-powered return forecasts with uncertainty bands")
    st.markdown("---")
    try:
        st.page_link("pages/1_🏠_Home.py", label="Home", icon="🏠")
        st.page_link("pages/2_📊_About_Model.py", label="About the Model", icon="📊")
    except Exception:
        # Fallback if Streamlit version lacks page_link (shouldn't happen on >=1.52)
        st.markdown("- **Home**: select `🏠 Home` from the Pages menu")
        st.markdown("- **About**: select `📊 About Model` from the Pages menu")
    st.markdown("---")
    st.caption("Not financial advice.")

# Main landing content (if user lands on app.py directly, redirect to home)
st.title("📈 StockForecast Pro")
st.markdown("""
### Welcome to StockForecast Pro

Navigate using the sidebar to:
- **🏠 Home** — Get stock forecasts
- **📊 About Model** — Learn about our AI technology

Or use the navigation menu at the top of the page.
""")

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.1rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.5rem;">StockForecast Pro</p>
    <p style="margin: 0.5rem 0;">Powered by Advanced Machine Learning & Real-Time Market Data</p>
    <p style="margin-top: 1rem; color: #999;">© 2024 StockForecast Pro. All rights reserved. | Not affiliated with any financial institution.</p>
    <p style="color: #999; font-size: 0.85rem;">This platform is for informational purposes only and does not constitute financial advice.</p>
</div>
""", unsafe_allow_html=True)
