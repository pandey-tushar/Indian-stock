#!/usr/bin/env python3
"""
Enhanced reporting with separate, focused visualizations and research notes.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def plot_risk_return_profile(df: pd.DataFrame, outdir: Path):
    """Bar chart: forecast vs risk for each portfolio stock."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df["Forecast_Med_%"], width, label="3M Forecast (%)", color="steelblue")
    ax.bar(x + width/2, df["Risk_Vol_%"], width, label="Realized Vol (%)", color="coral")
    
    ax.set_xlabel("Stock")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Portfolio: Forecast vs Risk (3-month horizon)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Stock"], rotation=45, ha="right")
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(outdir / "risk_return_profile.png", dpi=150)
    plt.close()


def plot_probability_distribution(df: pd.DataFrame, outdir: Path):
    """Histogram of P_Up across all stocks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["P_Up"], bins=20, color="mediumseagreen", edgecolor="black", alpha=0.7)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="P_Up=0.5 (neutral)")
    ax.set_xlabel("P_Up (probability of positive return)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Trend Probability Across All Stocks")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "probability_distribution.png", dpi=150)
    plt.close()


def plot_forecast_bands(port: pd.DataFrame, outdir: Path):
    """Error bar plot showing median + p10/p90 bands for portfolio stocks."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    stocks = port["Stock"].tolist()
    med = port["Forecast_Med_%"].tolist()
    p10 = port["Forecast_P10_%"].tolist()
    p90 = port["Forecast_P90_%"].tolist()
    
    # Error bars from p10 to p90 (absolute distances)
    lower_err = [abs(m - p) for m, p in zip(med, p10)]
    upper_err = [abs(p - m) for m, p in zip(med, p90)]
    
    ax.errorbar(
        range(len(stocks)), med, 
        yerr=[lower_err, upper_err],
        fmt='o', markersize=8, capsize=5, capthick=2,
        color='darkblue', ecolor='gray', elinewidth=2, alpha=0.7
    )
    
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha="right")
    ax.set_ylabel("Return Forecast (%)")
    ax.set_title("Portfolio: 3M Forecast with Uncertainty Bands (P10–P90)")
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "forecast_bands.png", dpi=150)
    plt.close()


def plot_weight_contribution(port: pd.DataFrame, outdir: Path):
    """Stacked bar: weight vs expected contribution to portfolio return."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: weights
    colors = ["steelblue" if b == "core_full_history" else "orange" 
              for b in port.get("Bucket", ["core_full_history"]*len(port))]
    ax1.barh(port["Stock"], port["Weight"]*100, color=colors)
    ax1.set_xlabel("Portfolio Weight (%)")
    ax1.set_title("Portfolio Allocation")
    ax1.invert_yaxis()
    
    # Right: median forecast % (unweighted)
    forecast = port["Forecast_Med_%"]
    ax2.barh(port["Stock"], forecast, color=colors)
    ax2.set_xlabel("Median Forecast Return (%)")
    ax2.set_title("Expected 3M Return by Stock (unweighted)")
    ax2.invert_yaxis()
    ax2.axvline(0, color='black', linewidth=0.5)
    
    # Legend
    blue_patch = mpatches.Patch(color='steelblue', label='Core (full history)')
    orange_patch = mpatches.Patch(color='orange', label='New (short history)')
    fig.legend(handles=[blue_patch, orange_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(outdir / "weight_contribution.png", dpi=150)
    plt.close()


def plot_uncertainty_vs_forecast(df: pd.DataFrame, outdir: Path):
    """Scatter: forecast vs uncertainty band (width of p10–p90)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(
        df["Uncertainty_Band_%"], 
        df["Forecast_Med_%"],
        s=100, alpha=0.6, c=df["P_Up"], cmap="RdYlGn", edgecolors="black"
    )
    
    ax.set_xlabel("Uncertainty Band Width (P90 - P10, %)")
    ax.set_ylabel("Forecast Median (%)")
    ax.set_title("Forecast vs Uncertainty (color = P_Up)")
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("P_Up")
    plt.tight_layout()
    plt.savefig(outdir / "uncertainty_vs_forecast.png", dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--portfolio", required=True)
    p.add_argument("--outdir", default="reports_v2")
    args = p.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    df = pd.read_csv(args.results)
    port = pd.read_csv(args.portfolio)
    
    print(f"Creating enhanced visualizations in {outdir}...")
    
    plot_probability_distribution(df, outdir)
    plot_uncertainty_vs_forecast(df, outdir)
    plot_risk_return_profile(port, outdir)
    plot_forecast_bands(port, outdir)
    plot_weight_contribution(port, outdir)
    
    print(f"[OK] Wrote 5 plots to {outdir}/")


if __name__ == "__main__":
    raise SystemExit(main())

