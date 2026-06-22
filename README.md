# 📈 Financial Market Prediction Dashboard | FICC

An end-to-end ML pipeline and interactive market exploration tool covering equities, forex, commodities, and bond yields — built with Streamlit, scikit-learn, and Plotly.

## Features

### 📈 Price & Indicators
- OHLC candlestick with MA20/50 and Bollinger Bands
- Volume, RSI-14, and MACD in a single synced chart
- Preprocessed feature table with red/green return gradient
- Rolling volatility (7d/20d/30d annualised) and intraday spread

### 🤖 ML Prediction
- Random Forest and Gradient Boosting classifiers
- 22 engineered features: returns, volatility, RSI, MACD, Bollinger Bands, momentum
- Walk-forward time-series cross-validation (5 splits), ~70% accuracy
- Buy/Sell signals overlaid on price chart with prediction probability
- Confusion matrix, CV fold accuracy, feature importance
- Live scenario testing via sidebar sliders (RSI, volatility, MACD, momentum)

### 📊 Returns & Risk
- Daily returns distribution with ±2σ bands
- Cumulative return and drawdown-from-peak charts
- Sharpe ratio, VaR/CVaR (95%), skewness, kurtosis

### 🌍 Macro Overview
- Cross-asset correlation heatmap (SPY, GLD, TLT, UUP, USO)
- Normalised price performance indexed to 100
- US yield curve (2Y / 10Y / 30Y) with inversion signal

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Asset Coverage

| Class | Instruments |
|-------|------------|
| Equities | SPY, QQQ, AAPL, MSFT, NVDA, TSLA, META, Nifty 50, FTSE 100 |
| Forex | EUR/USD, GBP/USD, USD/JPY, USD/INR, AUD/USD |
| Commodities | Gold, Crude Oil, Silver, Natural Gas, Gold ETF |
| Bond Yields | US 2Y, 10Y, 30Y Treasury |

Data sourced live from Yahoo Finance. For educational purposes only — not financial advice.
