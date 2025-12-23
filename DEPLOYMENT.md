# Indian Stock 3M Forecaster — Web App Deployment

## 🚀 Free Hosting on Streamlit Cloud

### Prerequisites
- GitHub account
- This repository pushed to GitHub

### Steps

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Add Streamlit forecaster app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/Indian-stock.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select:
     - **Repository**: `YOUR_USERNAME/Indian-stock`
     - **Branch**: `main`
     - **Main file path**: `app.py`
   - Click **Deploy**

3. **Your app will be live at**:
   ```
   https://YOUR_USERNAME-indian-stock-app-XXXXX.streamlit.app
   ```

---

## 🖥️ Run Locally

### Option 1: Using the venv
```bash
cd D:\Projects\Indian-stock
.\.venv\Scripts\python -m pip install streamlit plotly
.\.venv\Scripts\streamlit run app.py
```

### Option 2: Fresh install
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 Usage

1. **Enter a ticker** (e.g., `RELIANCE`, `TCS.NS`, `INFY`)
2. **Click "Forecast"**
3. Get:
   - 3-month return forecast (median + p10/p90 bands)
   - Trend probability (P_Up)
   - Realized volatility
   - Risk-adjusted score

The model automatically detects:
- **Full history** (≥500 days): uses all technical indicators
- **Medium history** (180–500 days): simplified features
- **Short history** (<180 days): scaled horizon + wider bands

---

## ⚙️ Configuration

Edit `app.py` to customize:
- Default suffix (`.NS` vs `.BO`)
- Forecast horizon (1–6 months)
- Model hyperparameters (in `quant_lite.py`)

---

## 🎨 Features

- ✅ Interactive ticker lookup
- ✅ Automatic NSE/BSE fallback
- ✅ Visual forecast bands (Plotly chart)
- ✅ Model regime detection (full/medium/short history)
- ✅ Detailed metrics table
- ✅ Mobile-responsive UI

---

## 🔧 Troubleshooting

**"Data fetch failed"**:
- Check ticker symbol (try `.NS` or `.BO` suffix)
- Verify stock exists on Yahoo Finance
- Try refreshing (`--refresh` flag in CLI mode)

**"Insufficient data"**:
- Stock needs ≥40 trading days for short-history model
- Recent IPOs (<2 months) may fail

**Slow first load**:
- Streamlit Cloud cold-starts take ~30s
- Local runs are instant after first data cache

---

## 📊 Model Details

**Architecture**: Gradient Boosting (Classifier + 3 Quantile Regressors)  
**Features**: Stationary indicators (returns, RSI, MACD, ATR, momentum, volume divergence)  
**Training**: Time-series split, no look-ahead bias  
**Validation**: Walk-forward out-of-sample directional accuracy

---

## 📝 Disclaimer

This tool is for **educational purposes only**. Not financial advice. Past performance ≠ future results.

