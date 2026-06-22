# 📈 Fintech Stock Predictor

A multi-asset market intelligence dashboard built with Streamlit, covering exploration, ML-based prediction, and macro analysis across equities, forex, commodities, and bond yields.

## Features

**Market Data Explorer**
- Browse 20+ instruments across equities, forex, commodities, and bond yields
- OHLC candlestick charts with MA20/50, Bollinger Bands, RSI, MACD
- Preprocessing audit: raw stats, missing values, engineered features
- Returns distribution, drawdown analysis, Sharpe/VaR/CVaR metrics
- Macro overview: cross-asset correlation heatmap, indexed price trends, US yield curve

**ML Prediction Dashboard**
- Random Forest and Gradient Boosting classifiers
- 20+ engineered features: returns, volatility, RSI, MACD, Bollinger Bands, momentum
- Walk-forward time-series cross-validation (5 splits)
- Confusion matrix, feature importance, prediction probability distribution
- Live scenario testing via sidebar sliders

## Setup

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
├── Home.py                          # Landing page
├── pages/
│   ├── 1_Market_Explorer.py         # Exploration dashboard
│   └── 2_Prediction_Dashboard.py    # ML prediction dashboard
├── requirements.txt
└── .streamlit/
    └── config.toml                  # Dark theme config
```

## Assets Covered

| Class | Instruments |
|-------|------------|
| Equities | SPY, QQQ, AAPL, MSFT, NVDA, Nifty 50, FTSE 100 |
| Forex | EUR/USD, GBP/USD, USD/JPY, USD/INR, AUD/USD, USD/CHF |
| Commodities | Gold, Crude Oil, Silver, Natural Gas, Copper |
| Bond Yields | US 2Y, 10Y, 30Y Treasury |

Data sourced live from Yahoo Finance. For educational purposes only — not financial advice.
