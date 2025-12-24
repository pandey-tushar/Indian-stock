#!/usr/bin/env python3
"""
StockForecast Pro — customer-facing Streamlit app.
Landing page always includes the stock search + month slider.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from quant_lite import Config, run_one


@st.cache_data(ttl=86400, show_spinner=False)
def get_forecast_cached(ticker: str, horizon_months: int, default_suffix: str, use_multi_horizon: bool = True) -> dict | None:
    """
    Cached wrapper around run_one/run_one_multi_horizon to avoid retraining on repeat queries.
    Cache key includes (ticker, horizon_months, default_suffix, use_multi_horizon).
    """
    from quant_lite import run_one_multi_horizon
    
    cfg = Config(
        horizon_months=horizon_months,
        default_suffix=default_suffix,
        cache_dir=Path("data_cache"),
        use_multi_horizon=use_multi_horizon,
    )
    if use_multi_horizon:
        return run_one_multi_horizon(ticker, cfg=cfg, refresh=False)
    else:
        return run_one(ticker, cfg=cfg, refresh=False)


st.set_page_config(
    page_title="StockForecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 18px;
    color: #fff;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
  }
  .hero h1 { margin: 0; font-weight: 800; letter-spacing: -0.02em; }
  .hero p { margin: 0.5rem 0 0 0; opacity: 0.95; }
  .card {
    background: #fff;
    border-radius: 14px;
    padding: 1.25rem 1.25rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }
  .muted { color: #666; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## StockForecast Pro")
    st.caption("AI-powered return forecasts with uncertainty bands")
    st.markdown("---")
    default_suffix = st.selectbox("Default exchange", [".NS (NSE)", ".BO (BSE)"], index=0).split()[0]
    fallback_suffix = st.text_input("Fallback exchange", ".BO")
    use_multi_horizon = st.checkbox("Multi-horizon ensemble (1M/3M/6M)", value=True, help="Ensemble predictions from multiple horizons for better accuracy")
    st.markdown("---")
    st.caption("Not financial advice.")

st.markdown(
    """
<div class="hero">
  <h1>📈 StockForecast Pro</h1>
  <p>Enter any Indian stock ticker to get an expected return and an uncertainty band.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_forecast, tab_about = st.tabs(["Forecast", "About"])

with tab_forecast:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        ticker_input = st.text_input("Stock ticker", placeholder="RELIANCE, TCS, INFY, OLAELEC, PAYTM…")
    with col_b:
        horizon_months = st.slider("Horizon (months)", min_value=1, max_value=12, value=3, step=1)

    run_btn = st.button("Get forecast", type="primary")

    if run_btn and ticker_input:
        with st.spinner("Analyzing… First lookup for a ticker can take ~5–15s. Next ones are cached."):
            raw = ticker_input.strip().upper()
            ticker = raw if raw.endswith((".NS", ".BO")) else f"{raw}{default_suffix}"

            result = get_forecast_cached(ticker, horizon_months, default_suffix, use_multi_horizon=use_multi_horizon)

            # Fallback if primary fails
            if result is None and fallback_suffix:
                if ticker.endswith(default_suffix):
                    base = ticker[: -len(default_suffix)]
                else:
                    base = raw
                alt = f"{base}{fallback_suffix}"
                result = get_forecast_cached(alt, horizon_months, default_suffix="", use_multi_horizon=use_multi_horizon)
                if result is not None:
                    ticker = alt

        if result is None:
            st.error(
                "Could not fetch/model this ticker. Try adding `.NS` / `.BO`, or the stock may be too new (< ~40 trading days)."
            )
        else:
            st.success(f"Forecast ready: **{result['Stock']}**")

            forecast_med = float(result["Forecast_Med_%"])
            p10 = float(result["Forecast_P10_%"])
            p90 = float(result["Forecast_P90_%"])
            p_up = float(result["P_Up"]) * 100.0
            vol = float(result["Risk_Vol_%"])

            m1, m2, m3 = st.columns(3)
            m1.metric(f"{horizon_months}M expected return (median)", f"{forecast_med:+.2f}%")
            m2.metric("Trend probability (P_Up)", f"{p_up:.1f}%")
            m3.metric(f"Risk (realized vol, {horizon_months}M)", f"{vol:.2f}%")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[p10, p90, p90, p10, p10],
                    y=[0.35, 0.35, 0.65, 0.65, 0.35],
                    fill="toself",
                    fillcolor="rgba(102, 126, 234, 0.18)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[p10, forecast_med, p90],
                    y=[0.5, 0.5, 0.5],
                    mode="markers+lines",
                    marker=dict(size=[14, 18, 14], color=["#ef4444", "#667eea", "#10b981"], line=dict(width=2, color="white")),
                    line=dict(color="#667eea", width=3),
                    hovertext=[f"P10: {p10:.2f}%", f"Median: {forecast_med:.2f}%", f"P90: {p90:.2f}%"],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#999", opacity=0.6)
            fig.update_layout(
                title=f"{ticker} — {horizon_months}M return forecast (P10–P90 band)",
                xaxis_title="Return (%)",
                yaxis_visible=False,
                height=320,
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Details"):
                details = {
                    "Price": result.get("Price"),
                    "Model_Regime": result.get("Model_Regime"),
                    "Data_Rows": result.get("Data_Rows"),
                    "Uncertainty_Band_%": result.get("Uncertainty_Band_%"),
                }
                if "Ensemble_Horizons" in result:
                    details["Ensemble_Horizons"] = result.get("Ensemble_Horizons")
                st.write(details)

with tab_about:
    st.markdown("### About the model")
    st.markdown(
        """
This app predicts forward returns using **Gradient Boosting** on stationary technical features (returns/ratios, not raw prices).

**Multi-horizon ensemble mode** (default):
- Trains separate models for 1M, 3M, and 6M horizons
- Ensembles predictions with confidence weighting
- Better captures different time-scale dynamics
- Reduces overfitting by not forcing one model to fit all horizons

**Single-horizon mode**:
- Trains one model for the selected horizon
- Faster but may be less robust

It outputs:
- **Median forecast** plus **P10/P90 bands** (uncertainty)
- **P_Up**: probability return is positive
- **Risk (volatility)** scaled to your chosen horizon

It automatically adapts to available history (full vs short-history mode).
"""
    )

st.markdown("---")
st.caption("© StockForecast Pro • Not financial advice")
