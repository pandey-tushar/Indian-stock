## Quant-Lite Forecaster (Indian stocks)

This repo contains a small **volatility-aware, stationarity-first** forecaster that ranks tickers by a risk-adjusted score instead of emitting a single noisy price target.

### What it outputs

- **P_Up**: probability the forward return over the horizon is positive (trend probability)
- **Forecast bands**: median + p10/p90 return forecasts (uncertainty-aware)
- **Risk_Vol_%**: realized volatility over the horizon window (risk-aware)
- **Risk_Adj_Score**: combines return, probability, and risk to rank candidates

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

Uses `stocks.txt` (one ticker per line). If the ticker has no suffix, the script appends `.NS` by default.

```bash
python quant_lite.py
```

Common options:

```bash
python quant_lite.py --horizon-months 3 --suffix .NS --top 20 --export results.csv
python quant_lite.py --refresh
```

If `.NS` fails for a symbol, the script can automatically try BSE symbols (Yahoo uses `.BO`):

```bash
python quant_lite.py --fallback-suffixes .BO --portfolio --export results.csv
```

### Notes

- Uses only **stationary features** (returns/ratios), not raw prices.
- Uses **time-ordered validation** (`TimeSeriesSplit`) to provide lightweight out-of-sample sanity metrics.
- Caches downloads under `data_cache/` to reduce API calls.


