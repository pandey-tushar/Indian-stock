"""
Quant-Lite forecaster
---------------------
Goal: actionable, volatility-aware ranking rather than a noisy single price point.

Outputs per ticker:
- Trend probability: P(ForwardReturn > 0)
- Return forecast: median (and p10/p90 bands via quantile GBMs)
- Risk: realized volatility over the horizon window
- Risk-adjusted score: combines expected return, trend probability, and risk

Design choices:
- Stationary features only (returns/ratios), no raw price levels.
- Strict time-order splits (TimeSeriesSplit) for sanity-check metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


EPS = 1e-9


def get_model_cache_key(ticker: str, cfg: Config, feature_cols: list[str], data_hash: str) -> str:
    """Generate unique cache key for trained models."""
    key_parts = [
        ticker,
        str(cfg.horizon_months),
        str(sorted(feature_cols)),
        data_hash[:8],  # First 8 chars of data hash
    ]
    return hashlib.md5("__".join(key_parts).encode()).hexdigest()


def save_models_to_cache(
    ticker: str,
    cfg: Config,
    feature_cols: list[str],
    data_hash: str,
    clf,
    reg_med,
    reg_p10,
    reg_p90,
    feat_imp: dict,
):
    """Save trained models to disk cache."""
    cfg.model_cache_dir.mkdir(exist_ok=True)
    cache_key = get_model_cache_key(ticker, cfg, feature_cols, data_hash)
    cache_path = cfg.model_cache_dir / f"{cache_key}.pkl"
    
    with open(cache_path, "wb") as f:
        pickle.dump({
            "ticker": ticker,
            "timestamp": datetime.now(),
            "clf": clf,
            "reg_med": reg_med,
            "reg_p10": reg_p10,
            "reg_p90": reg_p90,
            "feat_imp": feat_imp,
            "feature_cols": feature_cols,
        }, f)


def load_models_from_cache(
    ticker: str,
    cfg: Config,
    feature_cols: list[str],
    data_hash: str,
):
    """Load cached models if available and recent."""
    cache_key = get_model_cache_key(ticker, cfg, feature_cols, data_hash)
    cache_path = cfg.model_cache_dir / f"{cache_key}.pkl"
    
    if not cache_path.exists():
        return None
    
    # Check age
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    if age.total_seconds() > cfg.model_cache_max_age_hours * 3600:
        return None
    
    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached
    except Exception:
        return None


def hash_dataframe(df: pd.DataFrame) -> str:
    """Quick hash of dataframe for cache invalidation."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df[["Close", "Volume"]].iloc[-min(100, len(df)):], index=False).values
    ).hexdigest()


EPS = 1e-9


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def read_tickers(path: str | Path) -> list[str]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8").splitlines()
    tickers: list[str] = []
    for line in raw:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(s)
    # de-dupe while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def has_suffix(t: str) -> bool:
    return "." in t


def ticker_candidates(t: str, primary_suffix: str, fallback_suffixes: list[str]) -> list[str]:
    """
    If ticker already has a suffix (e.g. RELIANCE.NS), return [ticker].
    Otherwise return [ticker+primary_suffix, ticker+fallback1, ...].
    """
    if has_suffix(t):
        return [t]
    cands = [f"{t}{primary_suffix}"]
    for sfx in fallback_suffixes:
        if sfx and sfx != primary_suffix:
            cands.append(f"{t}{sfx}")
    # de-dupe preserving order
    seen = set()
    out: list[str] = []
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / (roll_down + EPS)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(close, span=fast) - ema(close, span=slow)
    sig = ema(macd_line, span=signal)
    return macd_line - sig


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


def realized_vol(log_rets: pd.Series, window: int) -> pd.Series:
    # horizon-scaled (not annualized): std * sqrt(window)
    return log_rets.rolling(window=window, min_periods=window).std() * math.sqrt(window)


@dataclass(frozen=True)
class Config:
    horizon_months: int = 3
    trading_days_per_month: int = 21
    # Prefer long history, but don't drop recent IPOs entirely.
    min_rows_long: int = 500
    min_rows_short: int = 160
    default_suffix: str = ".NS"
    cache_dir: Path = Path("data_cache")
    cache_max_age_hours: int = 24
    model_cache_dir: Path = Path("model_cache")
    model_cache_max_age_hours: int = 24  # Re-train if model is older than this
    use_multi_horizon: bool = True  # Use multi-horizon ensemble by default

    @property
    def horizon_days(self) -> int:
        return self.horizon_months * self.trading_days_per_month


def normal_cdf(x: float) -> float:
    # Phi(x) using erf (no scipy dependency).
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_cached_or_download(
    ticker: str,
    cfg: Config,
    refresh: bool = False,
    period: str = "max",
) -> pd.DataFrame | None:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("/", "_")
    fpath = cfg.cache_dir / f"{safe}.csv"

    def _read_csv() -> pd.DataFrame | None:
        try:
            df = pd.read_csv(fpath, parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            return df
        except Exception:
            return None

    if fpath.exists() and not refresh:
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
        if _utcnow() - mtime < timedelta(hours=cfg.cache_max_age_hours):
            df = _read_csv()
            if df is not None and not df.empty:
                return df

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Basic hygiene: drop 0 volume + NA rows
        if "Volume" in df.columns:
            df = df[df["Volume"] > 0]
        df = df.dropna()
        df = df.sort_index()
        out = df.reset_index()
        out.to_csv(fpath, index=False)
        return df
    except Exception:
        return None


def get_feature_cols_for_horizon(df: pd.DataFrame, horizon_days: int, data_length: int) -> list[str]:
    """
    Get horizon-appropriate feature columns based on data length and horizon.
    Shorter horizons use shorter-window features; longer horizons use longer-window features.
    Only returns features that actually exist in the dataframe.
    """
    # Base features that are always available
    base_features = ["RSI_14", "MACD_Norm", "ATR_Norm", "LogRet_1", "LogRet_5", "LogRet_21"]
    
    if data_length >= 500:  # Full history
        if horizon_days <= 21:  # 1M: focus on short-term momentum
            candidates = [
                "Dist_SMA20",
                "Dist_SMA50",
                "RSI_14",
                "MACD_Norm",
                "ATR_Norm",
                "LogRet_1",
                "LogRet_5",
                "Vol_20",
                "ROC_21",
                "Mom_Ratio",
                "Vol_Ratio_Price",
            ]
        elif horizon_days <= 63:  # 3M: balanced
            candidates = [
                "Dist_SMA50",
                "Dist_SMA200",
                "RSI_14",
                "MACD_Norm",
                "ATR_Norm",
                "LogRet_5",
                "LogRet_21",
                "Vol_Ratio",
                "ROC_21",
                "Mom_Ratio",
                "Vol_Ratio_Price",
            ]
        else:  # 6M+: focus on longer-term trends
            candidates = [
                "Dist_SMA50",
                "Dist_SMA200",
                "RSI_14",
                "MACD_Norm",
                "ATR_Norm",
                "LogRet_21",
                "Vol_Ratio",
                "ROC_21",
                "Mom_Ratio",
                "Vol_Ratio_Price",
            ]
    elif data_length >= 180:  # Medium history
        if horizon_days <= 21:
            candidates = [
                "Dist_SMA20",
                "Dist_SMA50",
                "RSI_14",
                "MACD_Norm",
                "ATR_Norm",
                "LogRet_5",
                "Vol_20",
                "ROC_21",
                "Mom_Ratio",
                "Vol_Ratio_Price",
            ]
        else:
            candidates = [
                "Dist_SMA50",
                "RSI_14",
                "MACD_Norm",
                "ATR_Norm",
                "LogRet_5",
                "LogRet_21",
                "Vol_Ratio",
                "ROC_21",
                "Mom_Ratio",
                "Vol_Ratio_Price",
            ]
    else:  # Short history
        candidates = [
            "Dist_SMA20",
            "RSI_14",
            "MACD_Norm",
            "ATR_Norm",
            "LogRet_5",
            "Vol_20",
            "ROC_21",
        ]
    
    # Filter to only include features that exist in the dataframe
    available = [f for f in candidates if f in df.columns]
    return available if available else base_features


def engineer_features(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Build stationary features and target.
    - Features are computed for all dates where indicators exist.
    - Target exists only where future close exists (shift(-horizon_days)).
    - Feature windows are adapted based on horizon_days for better horizon-specific modeling.
    """
    x = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in x.columns:
            raise ValueError(f"Missing column: {col}")

    # Adaptive window sizes based on horizon
    # For shorter horizons, use shorter windows; for longer, use longer windows
    if horizon_days <= 21:  # 1 month or less
        sma_short, sma_med, sma_long = 10, 20, 50
        ret_short, ret_med, ret_long = 1, 5, 10
        vol_window = 10
    elif horizon_days <= 63:  # 3 months or less
        sma_short, sma_med, sma_long = 20, 50, 200
        ret_short, ret_med, ret_long = 1, 5, 21
        vol_window = 20
    else:  # 6+ months
        sma_short, sma_med, sma_long = 50, 100, 200
        ret_short, ret_med, ret_long = 5, 21, 63
        vol_window = 40

    x["SMA_50"] = x["Close"].rolling(50, min_periods=50).mean()
    x["SMA_20"] = x["Close"].rolling(20, min_periods=20).mean()
    x["SMA_200"] = x["Close"].rolling(200, min_periods=200).mean()
    x["Dist_SMA20"] = (x["Close"] / (x["SMA_20"] + EPS)) - 1.0
    x["Dist_SMA50"] = (x["Close"] / (x["SMA_50"] + EPS)) - 1.0
    x["Dist_SMA200"] = (x["Close"] / (x["SMA_200"] + EPS)) - 1.0

    x["RSI_14"] = rsi(x["Close"], length=14) / 100.0  # 0-1
    x["MACD_H"] = macd_hist(x["Close"])
    x["MACD_Norm"] = x["MACD_H"] / (x["Close"] + EPS)

    x["ATR_14"] = atr(x["High"], x["Low"], x["Close"], length=14)
    x["ATR_Norm"] = x["ATR_14"] / (x["Close"] + EPS)

    x["LogRet_1"] = log_return(x["Close"], periods=1)
    x["LogRet_5"] = log_return(x["Close"], periods=5)
    x["LogRet_21"] = log_return(x["Close"], periods=21)
    
    # Horizon-adaptive returns
    x[f"LogRet_{ret_short}"] = log_return(x["Close"], periods=ret_short)
    x[f"LogRet_{ret_med}"] = log_return(x["Close"], periods=ret_med)
    if ret_long != ret_med:
        x[f"LogRet_{ret_long}"] = log_return(x["Close"], periods=ret_long)

    x["Vol_20"] = realized_vol(x["LogRet_1"], window=vol_window)
    x["Vol_H"] = realized_vol(x["LogRet_1"], window=horizon_days)
    x["Vol_Ratio"] = x["Vol_20"] / (x["Vol_H"] + EPS)
    
    # Add momentum indicators
    x["ROC_21"] = x["Close"].pct_change(21)  # 21-day rate of change
    x["Mom_Ratio"] = x["LogRet_5"] / (x["LogRet_21"].abs() + EPS)  # Recent vs longer momentum
    
    # Volume-price divergence (volume trend vs price trend)
    x["Vol_SMA_20"] = x["Volume"].rolling(20, min_periods=20).mean()
    x["Vol_Ratio_Price"] = (x["Volume"] / (x["Vol_SMA_20"] + EPS)) - 1.0

    x["Future_Close"] = x["Close"].shift(-horizon_days)
    x["Target_LogRet"] = np.log((x["Future_Close"] + EPS) / (x["Close"] + EPS))
    x["Target_Up"] = (x["Target_LogRet"] > 0).astype(int)

    return x


def fit_models(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[GradientBoostingClassifier, GradientBoostingRegressor, GradientBoostingRegressor, GradientBoostingRegressor, dict]:
    X = train[feature_cols]
    y_up = train["Target_Up"]
    y_ret = train["Target_LogRet"]

    # Tuned hyperparameters with stronger regularization
    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    reg_med = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    reg_p10 = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.10,
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    reg_p90 = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.90,
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )

    clf.fit(X, y_up)
    reg_med.fit(X, y_ret)
    reg_p10.fit(X, y_ret)
    reg_p90.fit(X, y_ret)
    
    # Feature importance from regression model
    feat_imp = dict(zip(feature_cols, reg_med.feature_importances_))

    return clf, reg_med, reg_p10, reg_p90, feat_imp


def oos_sanity_metrics(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict[str, float]:
    """
    Lightweight walk-forward sanity check:
    - Directional accuracy (sign)
    - MAE on log-return (regression)
    """
    usable = data.dropna(subset=feature_cols + ["Target_LogRet", "Target_Up"])
    if len(usable) < 300:
        return {}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    dir_hits: list[float] = []
    maes: list[float] = []

    X_all = usable[feature_cols].to_numpy()
    y_up_all = usable["Target_Up"].to_numpy()
    y_ret_all = usable["Target_LogRet"].to_numpy()

    for train_idx, test_idx in tscv.split(X_all):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_up_tr, y_up_te = y_up_all[train_idx], y_up_all[test_idx]
        y_ret_tr, y_ret_te = y_ret_all[train_idx], y_ret_all[test_idx]

        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        reg = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        clf.fit(X_tr, y_up_tr)
        reg.fit(X_tr, y_ret_tr)

        p_up = clf.predict_proba(X_te)[:, 1]
        pred_up = (p_up >= 0.5).astype(int)
        dir_hits.append(float((pred_up == y_up_te).mean()))

        pred_ret = reg.predict(X_te)
        maes.append(float(np.mean(np.abs(pred_ret - y_ret_te))))

    return {
        "oos_dir_acc": float(np.mean(dir_hits)),
        "oos_mae_logret": float(np.mean(maes)),
    }


def pct_from_logret(x: float) -> float:
    return (math.exp(x) - 1.0) * 100.0


def run_one_horizon(
    ticker: str,
    df: pd.DataFrame,
    horizon_months: int,
    cfg: Config,
    refresh: bool = False,
) -> dict | None:
    """
    Run prediction for a single horizon. Internal function used by multi-horizon ensemble.
    """
    horizon_days = horizon_months * cfg.trading_days_per_month
    
    # For short histories, fall back to a shorter horizon
    horizon_used = horizon_days
    if len(df) < cfg.min_rows_long:
        horizon_used = int(min(horizon_days, max(5, len(df) // 6)))

    data = engineer_features(df, horizon_days=horizon_used)
    feature_cols = get_feature_cols_for_horizon(data, horizon_days=horizon_used, data_length=len(df))

    # training rows must have both features and target
    train = data.dropna(subset=feature_cols + ["Target_LogRet", "Target_Up"])
    if len(train) < (80 if len(df) >= cfg.min_rows_long else 40):
        return None

    # latest row: features must exist; target may not.
    latest = data.dropna(subset=feature_cols).iloc[[-1]]

    # Fit models
    if len(df) >= cfg.min_rows_long:
        clf, reg_med, reg_p10, reg_p90, feat_imp = fit_models(train=train, feature_cols=feature_cols)
        p_up_clf = float(clf.predict_proba(latest[feature_cols])[:, 1][0])
    else:
        p_up_clf = float("nan")
        reg_med = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        reg_p10 = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.10,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        reg_p90 = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.90,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        X = train[feature_cols]
        y = train["Target_LogRet"]
        reg_med.fit(X, y)
        reg_p10.fit(X, y)
        reg_p90.fit(X, y)

    pred_med = float(reg_med.predict(latest[feature_cols])[0])
    pred_p10 = float(reg_p10.predict(latest[feature_cols])[0])
    pred_p90 = float(reg_p90.predict(latest[feature_cols])[0])

    # Scale back to target horizon if we trained on a shorter horizon.
    scale = float(horizon_days) / float(horizon_used)
    pred_med_scaled = pred_med * scale
    pred_p10_scaled = pred_p10 * scale
    pred_p90_scaled = pred_p90 * scale

    # Implied probability from forecast distribution
    iq = max(EPS, (pred_p90_scaled - pred_p10_scaled))
    sigma = iq / 2.5631031310892007
    p_up_imp = normal_cdf(pred_med_scaled / max(EPS, sigma))

    if not np.isnan(p_up_clf):
        p_up = float(np.clip(0.7 * p_up_clf + 0.3 * p_up_imp, 0.0, 1.0))
        model_regime = "full"
    else:
        p_up = float(np.clip(p_up_imp, 0.0, 1.0))
        model_regime = "short_history"

    # Realized risk
    if "Vol_H" in latest.columns and pd.notna(latest["Vol_H"].iloc[0]):
        vol_used = float(latest["Vol_H"].iloc[0])
        vol_target = vol_used * math.sqrt(scale)
        vol_h = float(vol_target * 100.0)
    else:
        vol_h = float("nan")

    # Uncertainty proxy
    band_width = pct_from_logret(pred_p90_scaled) - pct_from_logret(pred_p10_scaled)
    
    # Confidence score: inverse of uncertainty band (wider band = less confidence)
    confidence = 1.0 / (1.0 + band_width / 100.0) if band_width > 0 else 0.5

    return {
        "horizon_months": horizon_months,
        "horizon_days": horizon_days,
        "horizon_days_used": int(horizon_used),
        "model_regime": model_regime,
        "p_up": p_up,
        "pred_med_scaled": pred_med_scaled,
        "pred_p10_scaled": pred_p10_scaled,
        "pred_p90_scaled": pred_p90_scaled,
        "vol_h": vol_h,
        "band_width": band_width,
        "confidence": confidence,
    }


def run_one_multi_horizon(ticker: str, cfg: Config, refresh: bool = False) -> dict | None:
    """
    Multi-horizon ensemble: train separate models for 1M, 3M, 6M and ensemble predictions.
    Uses confidence-weighted averaging where confidence = 1 / (1 + uncertainty_band).
    Trains models in parallel for speed since they all use the same data.
    """
    df = load_cached_or_download(ticker, cfg=cfg, refresh=refresh, period="max")
    if df is None or len(df) < cfg.min_rows_short:
        return None

    # Determine which horizons to use based on available data
    horizons_to_use = []
    if len(df) >= 100:  # Need at least ~100 days for 1M
        horizons_to_use.append(1)
    if len(df) >= 200:  # Need at least ~200 days for 3M
        horizons_to_use.append(3)
    if len(df) >= 400:  # Need at least ~400 days for 6M
        horizons_to_use.append(6)
    
    # Fallback: if we can't use multiple horizons, use single horizon approach
    if not horizons_to_use:
        return None
    
    # If only one horizon is available, fall back to single-horizon mode
    if len(horizons_to_use) == 1:
        return run_one(ticker, cfg, refresh)
    
    # Train models in parallel since they all use the same dataframe
    horizon_results = []
    with ThreadPoolExecutor(max_workers=min(len(horizons_to_use), 3)) as executor:
        # Submit all horizon training tasks
        future_to_horizon = {
            executor.submit(run_one_horizon, ticker, df, h_months, cfg, refresh): h_months
            for h_months in horizons_to_use
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_horizon):
            h_months = future_to_horizon[future]
            try:
                result = future.result()
                if result is not None:
                    horizon_results.append(result)
            except Exception:
                # If one horizon fails, continue with others
                # The ensemble will work with whatever horizons succeeded
                continue
    
    if not horizon_results:
        return None
    
    # Ensemble: confidence-weighted average
    # Weight by confidence (inverse of uncertainty band width)
    total_weight = sum(r["confidence"] for r in horizon_results)
    if total_weight <= EPS:
        # Fallback to equal weights
        weights = [1.0 / len(horizon_results)] * len(horizon_results)
    else:
        weights = [r["confidence"] / total_weight for r in horizon_results]
    
    # Weighted average of predictions
    ens_pred_med = sum(r["pred_med_scaled"] * w for r, w in zip(horizon_results, weights))
    ens_pred_p10 = sum(r["pred_p10_scaled"] * w for r, w in zip(horizon_results, weights))
    ens_pred_p90 = sum(r["pred_p90_scaled"] * w for r, w in zip(horizon_results, weights))
    ens_p_up = sum(r["p_up"] * w for r, w in zip(horizon_results, weights))
    
    # For risk, use the longest horizon's vol (most stable)
    longest_horizon_result = max(horizon_results, key=lambda x: x["horizon_days"])
    ens_vol_h = longest_horizon_result["vol_h"]
    
    # Ensemble uncertainty band
    ens_band_width = pct_from_logret(ens_pred_p90) - pct_from_logret(ens_pred_p10)
    
    # Use the primary horizon (from cfg) for final output
    primary_horizon_days = cfg.horizon_days
    
    # Risk-aware edge and score
    exp_ret_pct = pct_from_logret(ens_pred_med)
    signed_prob_edge = (2.0 * ens_p_up - 1.0)
    edge_pct = exp_ret_pct * signed_prob_edge
    score = edge_pct / (ens_vol_h + EPS)
    
    # Get OOS metrics from the longest horizon model (most reliable)
    if len(df) >= cfg.min_rows_long:
        data = engineer_features(df, horizon_days=longest_horizon_result["horizon_days"])
        feature_cols = get_feature_cols_for_horizon(data, horizon_days=longest_horizon_result["horizon_days"], data_length=len(df))
        metrics = oos_sanity_metrics(data=data, feature_cols=feature_cols)
    else:
        metrics = {}
    
    return {
        "Stock": ticker,
        "Price": float(df["Close"].iloc[-1]),
        "Horizon_Days": primary_horizon_days,
        "Horizon_Days_Used": int(longest_horizon_result["horizon_days_used"]),
        "Model_Regime": "multi_horizon_ensemble",
        "Data_Rows": int(len(df)),
        "P_Up": ens_p_up,
        "Forecast_Med_%": exp_ret_pct,
        "Forecast_P10_%": pct_from_logret(ens_pred_p10),
        "Forecast_P90_%": pct_from_logret(ens_pred_p90),
        "Uncertainty_Band_%": float(ens_band_width),
        "Risk_Vol_%": ens_vol_h,
        "Edge_%": float(edge_pct),
        "Risk_Adj_Score": float(score),
        "Ensemble_Horizons": ",".join(str(r["horizon_months"]) for r in horizon_results),
        **metrics,
    }


def run_one(ticker: str, cfg: Config, refresh: bool = False) -> dict | None:
    df = load_cached_or_download(ticker, cfg=cfg, refresh=refresh, period="max")
    if df is None or len(df) < cfg.min_rows_short:
        return None

    # For short histories (recent IPOs), fall back to a shorter horizon
    # and scale log-return/vol back to the target horizon.
    horizon_used = cfg.horizon_days
    if len(df) < cfg.min_rows_long:
        # Recent IPOs: use a shorter training horizon to create enough labeled examples.
        # We'll scale the predicted log-return/vol back to the target horizon.
        horizon_used = int(min(cfg.horizon_days, max(5, len(df) // 6)))

    data = engineer_features(df, horizon_days=horizon_used)
    # Adaptive feature set: avoid long-window indicators for very short histories.
    feature_cols = get_feature_cols_for_horizon(data, horizon_days=horizon_used, data_length=len(df))

    # training rows must have both features and target
    train = data.dropna(subset=feature_cols + ["Target_LogRet", "Target_Up"])
    if len(train) < (80 if len(df) >= cfg.min_rows_long else 40):
        return None

    # latest row: features must exist; target may not.
    latest = data.dropna(subset=feature_cols).iloc[[-1]]

    # Always fit quantile regressors; classifier is only used in long-history regime.
    # (Classifier on short history tends to be unstable.)
    if len(df) >= cfg.min_rows_long:
        clf, reg_med, reg_p10, reg_p90, feat_imp = fit_models(train=train, feature_cols=feature_cols)
        p_up_clf = float(clf.predict_proba(latest[feature_cols])[:, 1][0])
    else:
        p_up_clf = float("nan")
        reg_med = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        reg_p10 = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.10,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        reg_p90 = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.90,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        X = train[feature_cols]
        y = train["Target_LogRet"]
        reg_med.fit(X, y)
        reg_p10.fit(X, y)
        reg_p90.fit(X, y)

    pred_med = float(reg_med.predict(latest[feature_cols])[0])
    pred_p10 = float(reg_p10.predict(latest[feature_cols])[0])
    pred_p90 = float(reg_p90.predict(latest[feature_cols])[0])

    # Scale back to target horizon if we trained on a shorter horizon.
    scale = float(cfg.horizon_days) / float(horizon_used)
    pred_med_scaled = pred_med * scale
    pred_p10_scaled = pred_p10 * scale
    pred_p90_scaled = pred_p90 * scale

    # Implied probability from forecast distribution (normal approx):
    # sigma estimated from inter-quantile range (p90-p10).
    iq = max(EPS, (pred_p90_scaled - pred_p10_scaled))
    sigma = iq / 2.5631031310892007  # 2 * z(0.90) where z(0.90)≈1.28155
    p_up_imp = normal_cdf(pred_med_scaled / max(EPS, sigma))

    if not np.isnan(p_up_clf):
        # Blend classifier (directional) with implied probability (calibration-ish)
        p_up = float(np.clip(0.7 * p_up_clf + 0.3 * p_up_imp, 0.0, 1.0))
        model_regime = "full"
    else:
        p_up = float(np.clip(p_up_imp, 0.0, 1.0))
        model_regime = "short_history"

    # Realized risk (horizon-scaled), using latest available Vol_H
    # Vol_H was computed on horizon_used; scale to target horizon by sqrt(time).
    if "Vol_H" in latest.columns and pd.notna(latest["Vol_H"].iloc[0]):
        vol_used = float(latest["Vol_H"].iloc[0])
        vol_target = vol_used * math.sqrt(scale)
        vol_h = float(vol_target * 100.0)
    else:
        vol_h = float("nan")

    # Uncertainty proxy: forecast inter-quantile range (in %)
    band_width = pct_from_logret(pred_p90_scaled) - pct_from_logret(pred_p10_scaled)

    # Risk-aware edge: expected return * (2p-1); then normalize by realized vol
    # For LONG-ONLY ranking: we want positive edge = P_Up>0.5 AND positive forecast.
    # For AVOID/HEDGE ranking: we want negative edge = P_Up<0.5 AND negative forecast.
    # Score = |edge| / vol to rank by conviction strength (sign determines direction).
    exp_ret_pct = pct_from_logret(pred_med_scaled)
    signed_prob_edge = (2.0 * p_up - 1.0)  # -1..+1
    edge_pct = exp_ret_pct * signed_prob_edge
    score = edge_pct / (vol_h + EPS)  # Retains sign for direction awareness

    metrics = oos_sanity_metrics(data=data, feature_cols=feature_cols) if model_regime == "full" else {}

    return {
        "Stock": ticker,
        "Price": float(latest["Close"].iloc[0]),
        "Horizon_Days": cfg.horizon_days,
        "Horizon_Days_Used": int(horizon_used),
        "Model_Regime": model_regime,
        "Data_Rows": int(len(df)),
        "P_Up": p_up,
        "Forecast_Med_%": exp_ret_pct,
        "Forecast_P10_%": pct_from_logret(pred_p10_scaled),
        "Forecast_P90_%": pct_from_logret(pred_p90_scaled),
        "Uncertainty_Band_%": float(band_width),
        "Risk_Vol_%": vol_h,
        "Edge_%": float(edge_pct),
        "Risk_Adj_Score": float(score),
        **metrics,
    }


def run_one_with_fallback(
    raw_ticker: str,
    cfg: Config,
    refresh: bool,
    fallback_suffixes: list[str],
) -> dict | None:
    """
    Attempt download/forecast for multiple symbol variants, returning the first success.
    - If raw_ticker has no suffix: tries primary cfg.default_suffix then fallback_suffixes.
    - If raw_ticker ends with cfg.default_suffix and fails: also tries base + each fallback suffix.
    - Uses multi-horizon ensemble if enabled in cfg.
    """
    attempts: list[str] = []

    if has_suffix(raw_ticker):
        attempts = [raw_ticker]
        # Optional: if it's .NS and fails, try .BO etc.
        if raw_ticker.endswith(cfg.default_suffix):
            base = raw_ticker[: -len(cfg.default_suffix)]
            for sfx in fallback_suffixes:
                attempts.append(f"{base}{sfx}")
    else:
        attempts = ticker_candidates(raw_ticker, cfg.default_suffix, fallback_suffixes)

    for t in attempts:
        # Use multi-horizon ensemble if enabled
        if cfg.use_multi_horizon:
            res = run_one_multi_horizon(t, cfg=cfg, refresh=refresh)
        else:
            res = run_one(t, cfg=cfg, refresh=refresh)
        if res is not None:
            return res
    return None


def compute_return_matrix(
    tickers: list[str],
    cfg: Config,
    refresh: bool,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Build aligned daily log return matrix for correlation/vol estimation.
    """
    rets = []
    names = []
    for t in tickers:
        df = load_cached_or_download(t, cfg=cfg, refresh=refresh)
        if df is None or df.empty:
            continue
        lr = log_return(df["Close"], periods=1).rename(t)
        lr = lr.dropna()
        if lookback_days and len(lr) > lookback_days:
            lr = lr.iloc[-lookback_days:]
        rets.append(lr)
        names.append(t)
    if not rets:
        return pd.DataFrame()
    m = pd.concat(rets, axis=1).dropna(how="any")
    return m


def cap_and_renormalize(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights.copy()
    w[w < 0] = 0.0
    if w.sum() <= EPS:
        return w * 0.0
    w = w / w.sum()
    if max_weight <= 0 or max_weight >= 1:
        return w

    # Iteratively cap overweight names and renormalize remaining mass.
    for _ in range(10):
        over = w > max_weight
        if not bool(over.any()):
            break
        capped_mass = float(w[over].sum() - max_weight * over.sum())
        w[over] = max_weight
        under = ~over
        if float(w[under].sum()) <= EPS:
            break
        w[under] = w[under] + (w[under] / float(w[under].sum())) * capped_mass
    w = w / (w.sum() + EPS)
    return w


def build_portfolio(
    ranked: pd.DataFrame,
    cfg: Config,
    refresh: bool,
    top_n: int,
    min_p_up: float,
    max_corr: float,
    max_weight: float,
    lookback_days: int,
    short_budget: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Portfolio heuristic (robust, dependency-free):
    - Filter for minimum trend probability.
    - Greedy select up to top_n while avoiding highly correlated names.
    - Weight by positive edge scaled by risk (edge / vol^2), then cap.

    Separation:
    - If `Model_Regime` exists, split into:
      - core: Model_Regime == "full"
      - new:  Model_Regime != "full" (recent IPO / short history)
    - Allocate sleeve budgets: (1-short_budget) to core, short_budget to new.
    """
    def _select_and_weight(univ: pd.DataFrame, budget: float, bucket: str) -> pd.DataFrame:
        if univ.empty or budget <= 0:
            return pd.DataFrame()

        # Correlation matrix on recent returns (within this sleeve)
        tickers_local = univ["Stock"].tolist()
        ret_mat_local = compute_return_matrix(tickers_local, cfg=cfg, refresh=refresh, lookback_days=lookback_days)
        corr_local = ret_mat_local.corr() if not ret_mat_local.empty else pd.DataFrame()

        selected_local: list[str] = []
        for t in tickers_local:
            if len(selected_local) >= top_n:
                break
            if corr_local.empty or not selected_local:
                selected_local.append(t)
                continue
            ok = True
            for s in selected_local:
                v = corr_local.get(t, pd.Series()).get(s, np.nan)
                if pd.notna(v) and abs(float(v)) > max_corr:
                    ok = False
                    break
            if ok:
                selected_local.append(t)

        port_local = univ[univ["Stock"].isin(selected_local)].copy()
        if port_local.empty:
            return pd.DataFrame()

        edge = port_local["Edge_%"].astype(float) / 100.0
        vol = (port_local["Risk_Vol_%"].astype(float) / 100.0).replace(0.0, np.nan)
        raw = (edge.clip(lower=0.0) / (vol**2)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw.index = port_local["Stock"].astype(str).to_list()
        if float(raw.sum()) <= EPS:
            raw = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            raw.index = port_local["Stock"].astype(str).to_list()

        # Enforce global max weight by capping within sleeve
        sleeve_max = 0.0
        if budget > 0:
            sleeve_max = min(1.0, max_weight / budget)
        w_local = cap_and_renormalize(raw, max_weight=sleeve_max)

        port_local["Bucket"] = bucket
        port_local["Sleeve_Weight"] = port_local["Stock"].map(w_local).fillna(0.0)
        port_local["Weight"] = port_local["Sleeve_Weight"] * budget
        return port_local

    df = ranked.copy()
    # LONG-ONLY filter: P_Up >= threshold AND positive median forecast AND positive edge
    df = df[
        (df["P_Up"] >= min_p_up)
        & (df["Forecast_Med_%"] > 0.0)
        & (df["Edge_%"] > 0.0)
    ].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "Model_Regime" in df.columns:
        core = df[df["Model_Regime"].astype(str) == "full"].copy()
        new = df[df["Model_Regime"].astype(str) != "full"].copy()
    else:
        core = df
        new = df.iloc[0:0].copy()

    short_budget = float(np.clip(short_budget, 0.0, 1.0))
    core_budget = 1.0 - short_budget
    if core.empty and not new.empty:
        # If no core names exist, allocate everything to new.
        core_budget = 0.0
        short_budget = 1.0

    port_core = _select_and_weight(core, budget=core_budget, bucket="core_full_history")
    port_new = _select_and_weight(new, budget=short_budget, bucket="new_short_history")
    port = pd.concat([port_core, port_new], axis=0, ignore_index=True)
    if port.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Portfolio-level stats across ALL selected names
    selected_all = port["Stock"].astype(str).tolist()
    ret_mat = compute_return_matrix(selected_all, cfg=cfg, refresh=refresh, lookback_days=lookback_days)
    if not ret_mat.empty and set(selected_all).issubset(set(ret_mat.columns)):
        aligned = ret_mat[selected_all].dropna(how="any")
        w_map = port.set_index("Stock")["Weight"].to_dict()
        w_vec = np.array([float(w_map.get(t, 0.0)) for t in selected_all], dtype=float)
        cov = aligned.cov().to_numpy()
        port_vol = float(math.sqrt(max(0.0, w_vec @ cov @ w_vec)) * math.sqrt(cfg.horizon_days) * 100.0)
    else:
        port_vol = float("nan")

    port_ret = float((port["Weight"] * port["Forecast_Med_%"].astype(float)).sum())
    port_edge = float((port["Weight"] * port["Edge_%"].astype(float)).sum())

    summary = pd.DataFrame(
        [
            {
                "Horizon_Months": cfg.horizon_months,
                "Names": int(len(selected_all)),
                "Core_Budget": float(core_budget),
                "New_Budget": float(short_budget),
                "Exp_Forecast_Med_%": port_ret,
                "Exp_Edge_%": port_edge,
                "Port_Risk_Vol_%": port_vol,
            }
        ]
    )

    port = port.sort_values(["Bucket", "Weight", "Risk_Adj_Score"], ascending=[True, False, False]).reset_index(drop=True)
    return port, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Quant-Lite forecaster (trend probability + vol-aware ranking).")
    ap.add_argument("--tickers-file", default="stocks.txt", help="Path to a newline-separated ticker list.")
    ap.add_argument("--suffix", default=".NS", help="Default exchange suffix appended when ticker has no dot.")
    ap.add_argument(
        "--fallback-suffixes",
        default=".BO",
        help="Comma-separated suffixes to try if the primary suffix fails (Yahoo: BSE is usually .BO).",
    )
    ap.add_argument("--horizon-months", type=int, default=3, help="Forecast horizon in months (approx 21 trading days/month). Primary horizon for output; multi-horizon ensemble uses 1M/3M/6M.")
    ap.add_argument("--no-multi-horizon", action="store_true", help="Disable multi-horizon ensemble (use single-horizon mode).")
    ap.add_argument("--refresh", action="store_true", help="Force refresh download (ignore cache).")
    ap.add_argument("--top", type=int, default=25, help="Show top N ranked results.")
    ap.add_argument("--export", default="", help="Optional path to export CSV results.")
    ap.add_argument("--portfolio", action="store_true", help="Also build a long-only portfolio from the ranking.")
    ap.add_argument("--portfolio-n", type=int, default=8, help="Target number of names in the portfolio.")
    ap.add_argument("--min-p-up", type=float, default=0.55, help="Minimum P_Up to include in portfolio candidate set.")
    ap.add_argument("--max-corr", type=float, default=0.85, help="Max absolute correlation allowed vs existing picks (greedy).")
    ap.add_argument("--max-weight", type=float, default=0.25, help="Max weight per name (0..1).")
    ap.add_argument("--short-budget", type=float, default=0.25, help="Portfolio budget allocated to short-history names (0..1).")
    ap.add_argument("--corr-lookback", type=int, default=252, help="Lookback days for correlation/portfolio risk estimate.")
    args = ap.parse_args()

    cfg = Config(
        horizon_months=args.horizon_months,
        default_suffix=args.suffix,
        use_multi_horizon=not args.no_multi_horizon,
    )
    fallback_suffixes = [s.strip() for s in str(args.fallback_suffixes).split(",") if s.strip()]

    raw_tickers = read_tickers(args.tickers_file)
    tickers = raw_tickers

    results: list[dict] = []
    skipped: list[str] = []

    mode_str = "Multi-horizon ensemble (1M/3M/6M)" if cfg.use_multi_horizon else f"Single-horizon ({cfg.horizon_months}M)"
    print(f"Quant-Lite | Mode: {mode_str} | Primary horizon: {cfg.horizon_months} months (~{cfg.horizon_days} trading days)")
    print(f"Tickers: {len(tickers)} | Cache: {cfg.cache_dir.resolve()}")
    print("-" * 110)

    for t in tickers:
        res = run_one_with_fallback(t, cfg=cfg, refresh=args.refresh, fallback_suffixes=fallback_suffixes)
        if res is None:
            skipped.append(t)
            continue
        results.append(res)

    if not results:
        print("No results. Likely causes: tickers invalid for chosen suffix, or insufficient history.")
        if skipped:
            print("Skipped:", ", ".join(skipped[:25]) + ("..." if len(skipped) > 25 else ""))
        return 2

    df = pd.DataFrame(results).sort_values(["Risk_Adj_Score"], ascending=False).reset_index(drop=True)

    cols = [
        "Stock",
        "Price",
        "P_Up",
        "Forecast_Med_%",
        "Forecast_P10_%",
        "Forecast_P90_%",
        "Uncertainty_Band_%",
        "Risk_Vol_%",
        "Edge_%",
        "Risk_Adj_Score",
        "oos_dir_acc",
        "oos_mae_logret",
    ]
    show = df[cols].head(max(1, args.top)).copy()

    # Friendly formatting
    show["P_Up"] = (show["P_Up"] * 100).map(lambda x: f"{x:5.1f}%")
    for c in ["Forecast_Med_%", "Forecast_P10_%", "Forecast_P90_%", "Uncertainty_Band_%", "Risk_Vol_%", "Edge_%"]:
        show[c] = show[c].map(lambda x: f"{x:8.2f}")
    show["Risk_Adj_Score"] = show["Risk_Adj_Score"].map(lambda x: f"{x:9.3f}")
    if "oos_dir_acc" in show.columns:
        show["oos_dir_acc"] = show["oos_dir_acc"].map(lambda x: "" if pd.isna(x) else f"{x:0.3f}")
    if "oos_mae_logret" in show.columns:
        show["oos_mae_logret"] = show["oos_mae_logret"].map(lambda x: "" if pd.isna(x) else f"{x:0.4f}")

    print("RANKING (trend probability + return + uncertainty + realized vol)")
    print("-" * 110)
    print(show.to_string(index=False))
    print("-" * 110)
    print("P_Up: probability forward return > 0 (classifier).")
    print("Forecast_*: return forecast (median + p10/p90 bands) over the horizon.")
    print("Risk_Vol_%: realized horizon-scaled volatility from recent history.")
    print("Edge_%: Forecast_Med_% scaled by (2*P_Up-1) to downweight low-conviction signals.")
    print("Risk_Adj_Score: Edge_% divided by Risk_Vol_% (higher is better).")
    if skipped:
        print(f"Skipped ({len(skipped)}): " + ", ".join(skipped[:25]) + ("..." if len(skipped) > 25 else ""))

    if args.export:
        outp = Path(args.export)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"Exported: {outp.resolve()}")

    if args.portfolio:
        port, summary = build_portfolio(
            ranked=df,
            cfg=cfg,
            refresh=args.refresh,
            top_n=args.portfolio_n,
            min_p_up=args.min_p_up,
            max_corr=args.max_corr,
            max_weight=args.max_weight,
            lookback_days=args.corr_lookback,
            short_budget=args.short_budget,
        )
        print("\nPORTFOLIO (long-only heuristic; correlation-aware selection; vol-aware weights)")
        print("-" * 110)
        if port.empty:
            print("No portfolio built (try lowering --min-p-up or increasing --portfolio-n).")
        else:
            view_cols = ["Bucket", "Stock", "Weight", "Sleeve_Weight", "P_Up", "Forecast_Med_%", "Forecast_P10_%", "Forecast_P90_%", "Risk_Vol_%", "Risk_Adj_Score"]
            pv = port[view_cols].copy()
            pv["Weight"] = pv["Weight"].map(lambda x: f"{x*100:6.2f}%")
            pv["Sleeve_Weight"] = pv["Sleeve_Weight"].map(lambda x: f"{x*100:6.2f}%")
            pv["P_Up"] = (pv["P_Up"].astype(float) * 100).map(lambda x: f"{x:5.1f}%")
            for c in ["Forecast_Med_%", "Forecast_P10_%", "Forecast_P90_%", "Risk_Vol_%", "Risk_Adj_Score"]:
                pv[c] = pv[c].map(lambda x: f"{float(x):8.2f}")
            print(pv.to_string(index=False))
            print("-" * 110)
            print(summary.to_string(index=False))

            if args.export:
                port_path = Path(args.export).with_name(Path(args.export).stem + "_portfolio.csv")
                port.to_csv(port_path, index=False)
                summ_path = Path(args.export).with_name(Path(args.export).stem + "_portfolio_summary.csv")
                summary.to_csv(summ_path, index=False)
                print(f"Exported: {port_path.resolve()}")
                print(f"Exported: {summ_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


