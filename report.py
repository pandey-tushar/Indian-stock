"""
Quant-Lite reporting / visualization

Reads:
- results.csv
- results_portfolio.csv (optional, created when running quant_lite.py --portfolio)

Writes:
- reports/summary.md
- reports/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def classify_row(r: pd.Series) -> tuple[str, str]:
    """
    Returns (label, rationale) with a simple, interpretable policy.
    """
    p = float(r.get("P_Up", np.nan))
    med = float(r.get("Forecast_Med_%", np.nan))
    band = float(r.get("Uncertainty_Band_%", np.nan))
    vol = float(r.get("Risk_Vol_%", np.nan))

    high_unc = (not np.isnan(band)) and band >= 60.0
    high_risk = (not np.isnan(vol)) and vol >= 25.0

    # Directional stance
    if p >= 0.60 and med >= 0:
        stance = "BUY/ACCUMULATE"
    elif p >= 0.52 and med >= 0:
        stance = "HOLD/SMALL BUY"
    elif p <= 0.45 and med <= 0:
        stance = "BEARISH (AVOID/HEDGE)"
    else:
        stance = "NEUTRAL/WATCH"

    notes = []
    notes.append(f"P_Up={p:.2f}, Med={med:.2f}%")
    if high_unc:
        notes.append("high uncertainty (wide p10–p90 band)")
    if high_risk:
        notes.append("high realized vol")
    return stance, "; ".join(notes)


def ensure_reports_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def plot_scatter(df: pd.DataFrame, outdir: Path) -> None:
    # Risk vs median forecast, colored by P_Up
    x = df.copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["Risk_Vol_%", "Forecast_Med_%", "P_Up"])

    plt.figure(figsize=(11, 7))
    sc = plt.scatter(
        x["Risk_Vol_%"],
        x["Forecast_Med_%"],
        c=x["P_Up"],
        cmap="viridis",
        s=np.clip(np.abs(x.get("Risk_Adj_Score", 0.0).astype(float)) * 120, 30, 250),
        alpha=0.85,
        edgecolors="none",
    )
    plt.axhline(0, color="black", linewidth=1, alpha=0.4)
    plt.xlabel("Risk_Vol_% (horizon-scaled realized volatility)")
    plt.ylabel("Forecast_Med_% (median 3M return)")
    plt.title("Risk vs Forecast (bubble=size ~ |Risk_Adj_Score|, color=P_Up)")
    cb = plt.colorbar(sc)
    cb.set_label("P_Up")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_risk_vs_forecast.png", dpi=160)
    plt.close()


def plot_prob_vs_score(df: pd.DataFrame, outdir: Path) -> None:
    x = df.copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["P_Up", "Risk_Adj_Score", "Edge_%"])

    plt.figure(figsize=(11, 7))
    plt.scatter(
        x["P_Up"],
        x["Risk_Adj_Score"],
        c=np.sign(x["Edge_%"].astype(float)),
        cmap="coolwarm",
        s=60,
        alpha=0.85,
        edgecolors="none",
    )
    plt.axvline(0.5, color="black", linewidth=1, alpha=0.35)
    plt.axhline(0, color="black", linewidth=1, alpha=0.35)
    plt.xlabel("P_Up")
    plt.ylabel("Risk_Adj_Score")
    plt.title("Conviction vs Risk-Adjusted Score (color indicates Edge sign)")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_prob_vs_score.png", dpi=160)
    plt.close()


def plot_portfolio_weights(port: pd.DataFrame, outdir: Path) -> None:
    x = port.copy()
    if x.empty or "Weight" not in x.columns:
        return
    x = x.sort_values("Weight", ascending=True)

    plt.figure(figsize=(10, max(4, 0.5 * len(x))))
    if "Bucket" in x.columns:
        colors = x["Bucket"].map(
            {
                "core_full_history": "#2a6fdb",
                "new_short_history": "#f59e0b",
            }
        ).fillna("#2a6fdb")
        plt.barh(x["Stock"], x["Weight"] * 100.0, color=list(colors))
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color="#2a6fdb", lw=8, label="core_full_history"),
                plt.Line2D([0], [0], color="#f59e0b", lw=8, label="new_short_history"),
            ],
            loc="lower right",
        )
    else:
        plt.barh(x["Stock"], x["Weight"] * 100.0, color="#2a6fdb")
    plt.xlabel("Weight (%)")
    plt.title("Portfolio Weights")
    plt.tight_layout()
    plt.savefig(outdir / "portfolio_weights.png", dpi=160)
    plt.close()


def write_summary(
    df: pd.DataFrame,
    port: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
    top_n: int,
) -> None:
    lines: list[str] = []
    lines.append("## Quant-Lite Report")
    lines.append("")

    # Core caveat: score isn't strictly long-only
    lines.append("### Key interpretation notes")
    lines.append("- **P_Up** is the model’s probability the horizon return is positive.")
    lines.append("- **Risk_Adj_Score** is based on **Edge_% / Risk_Vol_%** where **Edge_% = Forecast_Med_% × (2·P_Up − 1)**.")
    lines.append("- If **Forecast_Med_% < 0** and **P_Up < 0.5**, the model is effectively bearish; that can still produce a *positive* score (a *short/avoid* signal).")
    lines.append("")

    # Top signals table
    cols = [
        "Stock",
        "P_Up",
        "Forecast_Med_%",
        "Forecast_P10_%",
        "Forecast_P90_%",
        "Uncertainty_Band_%",
        "Risk_Vol_%",
        "Edge_%",
        "Risk_Adj_Score",
    ]
    t = df[cols].head(top_n).copy()
    t["Action"], t["Action_Notes"] = zip(*t.apply(classify_row, axis=1))
    lines.append(f"### Top {top_n} signals (ranked by Risk_Adj_Score)")
    lines.append("")
    lines.append(t.to_markdown(index=False))
    lines.append("")

    # Portfolio section
    lines.append("### Portfolio (if enabled)")
    if port.empty:
        lines.append("- No portfolio file found or portfolio is empty.")
    else:
        pview_cols = [
            "Stock",
            "Weight",
            "P_Up",
            "Forecast_Med_%",
            "Forecast_P10_%",
            "Forecast_P90_%",
            "Risk_Vol_%",
            "Risk_Adj_Score",
        ]
        p = port[pview_cols].copy()
        p["Action"], p["Action_Notes"] = zip(*p.apply(classify_row, axis=1))
        lines.append("")
        lines.append(p.to_markdown(index=False))
        if not summary.empty:
            lines.append("")
            lines.append("**Portfolio summary**")
            lines.append("")
            lines.append(summary.to_markdown(index=False))
    lines.append("")

    # Simple “what to do” guidance
    lines.append("### Suggested workflow (practical)")
    lines.append("- **Long-only**: focus on names with **P_Up > 0.55** and **Forecast_Med_% > 0**, then size down if uncertainty band or vol is high.")
    lines.append("- **Avoid/Hedge**: names with **P_Up < 0.45** and **Forecast_Med_% < 0** (bearish regime per model).")
    lines.append("- **Rebalance cadence**: weekly or bi-weekly; re-run the script and compare changes in P_Up, bands, and vol.")
    lines.append("")

    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def export_all_signals(
    results: pd.DataFrame,
    tickers_file: str,
    outdir: Path,
) -> Path:
    """
    Export one row per raw ticker in tickers_file.
    If a ticker wasn't successfully modeled, metrics are left blank and Status explains why.
    """
    raw_lines = Path(tickers_file).read_text(encoding="utf-8").splitlines()
    raw_tickers = []
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        raw_tickers.append(s)

    # Map by base ticker (strip suffix) for best-effort matching.
    # Example: "NCC" -> row "NCC.NS"
    res = results.copy()
    res["Base"] = res["Stock"].astype(str).str.split(".").str[0]
    res_by_base = res.set_index("Base")

    rows = []
    for t in raw_tickers:
        base = t.split(".")[0]
        if base in res_by_base.index:
            r = res_by_base.loc[base]
            p_up = float(r.get("P_Up", np.nan))
            rows.append(
                {
                    "Ticker": t,
                    "Resolved_Symbol": str(r.get("Stock", "")),
                    "Status": "OK",
                    "P_Up": p_up,
                    "P_Down": (1.0 - p_up) if not np.isnan(p_up) else np.nan,
                    "Forecast_Med_%": float(r.get("Forecast_Med_%", np.nan)),
                    "Forecast_P10_%": float(r.get("Forecast_P10_%", np.nan)),
                    "Forecast_P90_%": float(r.get("Forecast_P90_%", np.nan)),
                    "Risk_Vol_%": float(r.get("Risk_Vol_%", np.nan)),
                    "Risk_Adj_Score": float(r.get("Risk_Adj_Score", np.nan)),
                }
            )
        else:
            rows.append(
                {
                    "Ticker": t,
                    "Resolved_Symbol": "",
                    "Status": "NO_DATA",
                    "P_Up": np.nan,
                    "P_Down": np.nan,
                    "Forecast_Med_%": np.nan,
                    "Forecast_P10_%": np.nan,
                    "Forecast_P90_%": np.nan,
                    "Risk_Vol_%": np.nan,
                    "Risk_Adj_Score": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out_path = outdir / "signals_all.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv", help="Path to results.csv")
    ap.add_argument("--portfolio", default="results_portfolio.csv", help="Path to results_portfolio.csv (optional)")
    ap.add_argument("--portfolio-summary", default="results_portfolio_summary.csv", help="Path to results_portfolio_summary.csv (optional)")
    ap.add_argument("--outdir", default="reports", help="Output directory for charts + summary.md")
    ap.add_argument("--top", type=int, default=25, help="How many top-ranked signals to include in summary")
    ap.add_argument("--tickers-file", default="stocks.txt", help="Ticker list to create a full signals table (including missing).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_reports_dir(outdir)

    df = pd.read_csv(args.results)
    df = df.sort_values("Risk_Adj_Score", ascending=False).reset_index(drop=True)

    port_path = Path(args.portfolio)
    port = pd.read_csv(port_path) if port_path.exists() else pd.DataFrame()
    summ_path = Path(args.portfolio_summary)
    summ = pd.read_csv(summ_path) if summ_path.exists() else pd.DataFrame()

    sns.set_theme(style="whitegrid")
    plot_scatter(df, outdir)
    plot_prob_vs_score(df, outdir)
    plot_portfolio_weights(port, outdir)
    write_summary(df, port, summ, outdir, top_n=args.top)
    sig_path = export_all_signals(df, tickers_file=args.tickers_file, outdir=outdir)

    print(f"Wrote: {outdir / 'summary.md'}")
    print(f"Wrote: {outdir / 'scatter_risk_vs_forecast.png'}")
    print(f"Wrote: {outdir / 'scatter_prob_vs_score.png'}")
    if not port.empty:
        print(f"Wrote: {outdir / 'portfolio_weights.png'}")
    print(f"Wrote: {sig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


