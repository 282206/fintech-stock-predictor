import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 18px 22px;
    border-left: 4px solid #00c9a7;
    margin-bottom: 10px;
  }
  .metric-card.red  { border-left-color: #ff4b6e; }
  .metric-card.blue { border-left-color: #4b9eff; }
  .metric-card.gold { border-left-color: #ffd700; }
  .metric-label { font-size: 12px; color: #8b8fa8; letter-spacing: 1px; text-transform: uppercase; }
  .metric-val   { font-size: 28px; font-weight: 700; color: #f0f0f5; margin-top: 4px; }
  .metric-delta { font-size: 13px; margin-top: 2px; }
  .positive { color: #00c9a7; }
  .negative { color: #ff4b6e; }
  div[data-testid="stSidebar"] { background: #10131c; }
  .section-header { font-size: 18px; font-weight: 600; color: #c8cde4;
    border-bottom: 1px solid #2a2d3e; padding-bottom: 8px; margin: 20px 0 14px 0; }
</style>
""", unsafe_allow_html=True)

TICKERS = {
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
    "Nvidia (NVDA)": "NVDA", "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN", "Meta (META)": "META",
    "S&P 500 ETF (SPY)": "SPY", "Gold ETF (GLD)": "GLD",
    "Bitcoin (BTC)": "BTC-USD",
}

# ─── Feature Engineering ─────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]

    # Returns
    d["ret_1d"]  = c.pct_change(1)
    d["ret_3d"]  = c.pct_change(3)
    d["ret_5d"]  = c.pct_change(5)
    d["ret_10d"] = c.pct_change(10)

    # Moving averages
    for w in [5, 10, 20, 50]:
        d[f"ma_{w}"] = c.rolling(w).mean()
    d["ma_ratio_5_20"]  = d["ma_5"]  / d["ma_20"]
    d["ma_ratio_10_50"] = d["ma_10"] / d["ma_50"]
    d["price_vs_ma20"]  = c / d["ma_20"]
    d["price_vs_ma50"]  = c / d["ma_50"]

    # Volatility
    for w in [5, 10, 20, 30]:
        d[f"vol_{w}d"] = d["ret_1d"].rolling(w).std() * np.sqrt(252)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # Bollinger bands
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / bb_mid
    d["bb_pos"]   = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])

    # Volume features
    if "Volume" in d.columns:
        d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
        d["vol_ret"]   = d["Volume"] * d["ret_1d"]
    else:
        d["vol_ratio"] = 1.0
        d["vol_ret"]   = 0.0

    # Intraday spread
    if {"High", "Low"}.issubset(d.columns):
        d["spread_pct"] = (d["High"] - d["Low"]) / c

    # Momentum lag
    d["momentum_5"]  = c / c.shift(5)  - 1
    d["momentum_20"] = c / c.shift(20) - 1

    # Target: next-day direction
    d["target"] = (c.shift(-1) > c).astype(int)
    d["fwd_ret"] = c.pct_change(1).shift(-1)

    # Cumulative return (for display only)
    d["cum_return"] = (1 + d["ret_1d"]).cumprod() - 1

    return d


FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    "ma_ratio_5_20", "ma_ratio_10_50", "price_vs_ma20", "price_vs_ma50",
    "vol_5d", "vol_10d", "vol_20d", "vol_30d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pos",
    "vol_ratio", "vol_ret",
    "momentum_5", "momentum_20",
]


# ─── ML Pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_train(ticker: str, period: str, model_name: str):
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw.empty:
        return None
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.dropna()

    df = compute_features(raw)
    df = df.dropna()
    df = df.iloc[:-1]  # drop last row (no target)

    feat = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat].values
    y = df["target"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    tscv   = TimeSeriesSplit(n_splits=5)
    accs   = []
    clf_map = {
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=8,
                                                       min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                           learning_rate=0.05, random_state=42),
    }
    clf = clf_map[model_name]

    all_y_true, all_y_pred = [], []
    for tr_idx, te_idx in tscv.split(X_s):
        clf.fit(X_s[tr_idx], y[tr_idx])
        preds = clf.predict(X_s[te_idx])
        accs.append(accuracy_score(y[te_idx], preds))
        all_y_true.extend(y[te_idx])
        all_y_pred.extend(preds)

    # Final fit on all data
    clf.fit(X_s, y)

    # Predict on full set for signal
    df["signal"] = clf.predict(X_s)
    df["signal_prob"] = clf.predict_proba(X_s)[:, 1]

    fi = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False)
    cr = classification_report(all_y_true, all_y_pred, output_dict=True)
    cm = confusion_matrix(all_y_true, all_y_pred)

    return {
        "df": df,
        "raw": raw,
        "feat": feat,
        "scaler": scaler,
        "clf": clf,
        "cv_accs": accs,
        "accuracy": np.mean(accs),
        "feature_importance": fi,
        "class_report": cr,
        "confusion_matrix": cm,
    }


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    ticker_label = st.selectbox("Asset", list(TICKERS.keys()), index=0)
    ticker       = TICKERS[ticker_label]
    period       = st.selectbox("History", ["1y", "2y", "5y"], index=1)
    model_name   = st.selectbox("Model", ["Random Forest", "Gradient Boosting"], index=0)
    st.markdown("---")
    st.markdown("## 🔬 Scenario Testing")
    st.caption("Shift the latest feature values to simulate market conditions")
    scenario_rsi      = st.slider("RSI adjustment",       -30, 30, 0)
    scenario_vol      = st.slider("Volatility multiplier", 0.5, 3.0, 1.0, 0.1)
    scenario_macd     = st.slider("MACD shift",           -5.0, 5.0, 0.0, 0.5)
    scenario_momentum = st.slider("Momentum shift (%)",   -10, 10, 0)

# ─── Load data ───────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} and training model…"):
    result = load_and_train(ticker, period, model_name)

if result is None:
    st.error("Could not download data. Check ticker or internet connection.")
    st.stop()

df  = result["df"]
raw = result["raw"]
acc = result["accuracy"]

# ─── Hero header ─────────────────────────────────────────────────────────────
st.markdown(f"# 📈 Market Prediction Dashboard")
st.markdown(f"**{ticker_label}** · {period} history · {model_name}")

latest     = df.iloc[-1]
prev       = df.iloc[-2]
price_chg  = latest["Close"] - prev["Close"]
price_chg_pct = price_chg / prev["Close"] * 100
signal_txt = "🟢 BUY" if latest["signal"] == 1 else "🔴 SELL"
prob       = latest["signal_prob"] * 100
cum_ret    = latest["cum_return"] * 100

c1, c2, c3, c4 = st.columns(4)

def mcard(col, label, val, delta=None, color=""):
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta >= 0 else "negative"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="metric-delta {cls}">{sign} {abs(delta):.2f}%</div>'
    col.markdown(f"""
    <div class="metric-card {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-val">{val}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

mcard(c1, "Latest Close",     f"${latest['Close']:.2f}", price_chg_pct)
mcard(c2, "ML Signal",        signal_txt,                (prob - 50) * 2,  "blue")
mcard(c3, "Model Accuracy",   f"{acc*100:.1f}%",         None,             "gold")
mcard(c4, "Cumulative Return",f"{cum_ret:.1f}%",         cum_ret,          "red" if cum_ret < 0 else "")

# ─── Price Chart + Signals ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Price Chart & Signals</div>', unsafe_allow_html=True)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.55, 0.25, 0.2],
                    vertical_spacing=0.04)

fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="OHLC", increasing_line_color="#00c9a7", decreasing_line_color="#ff4b6e"), row=1, col=1)

for w, color in [(20, "#4b9eff"), (50, "#ffd700")]:
    if f"ma_{w}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"ma_{w}"], name=f"MA{w}",
                                 line=dict(color=color, width=1.5)), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
    line=dict(color="rgba(100,100,200,0.4)", dash="dash"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
    fill="tonexty", fillcolor="rgba(100,100,200,0.07)",
    line=dict(color="rgba(100,100,200,0.4)", dash="dash"), showlegend=False), row=1, col=1)

# Buy/Sell signals
buys  = df[df["signal"] == 1]
sells = df[df["signal"] == 0]
fig.add_trace(go.Scatter(x=buys.index,  y=buys["Low"]  * 0.985,
    mode="markers", marker=dict(symbol="triangle-up",   size=7, color="#00c9a7"), name="Buy Signal"),  row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.015,
    mode="markers", marker=dict(symbol="triangle-down", size=7, color="#ff4b6e"), name="Sell Signal"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14",
    line=dict(color="#c47aff", width=1.5)), row=2, col=1)
fig.add_hline(y=70, line_color="rgba(255,75,110,0.4)", line_dash="dot", row=2, col=1)
fig.add_hline(y=30, line_color="rgba(0,201,167,0.4)", line_dash="dot", row=2, col=1)

# Volume
colors = ["#00c9a7" if r >= 0 else "#ff4b6e" for r in df["ret_1d"].fillna(0)]
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
    marker_color=colors, opacity=0.7), row=3, col=1)

fig.update_layout(
    height=620, template="plotly_dark", paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    xaxis_rangeslider_visible=False,
)
fig.update_yaxes(gridcolor="#1e2130")
st.plotly_chart(fig, use_container_width=True)

# ─── ML Performance ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    st.caption("Cross-Validation Accuracy (Time Series Splits)")
    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(result['cv_accs']))],
                           "Accuracy": result["cv_accs"]})
    fig_cv = px.bar(cv_df, x="Fold", y="Accuracy", text_auto=".1%",
                    color="Accuracy", color_continuous_scale=["#ff4b6e", "#ffd700", "#00c9a7"],
                    range_y=[0.4, 1.0])
    fig_cv.add_hline(y=0.8, line_color="#ffd700", line_dash="dash", annotation_text="80%")
    fig_cv.update_layout(height=280, template="plotly_dark", paper_bgcolor="#0e1117",
                         plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=10, b=0),
                         showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_cv, use_container_width=True)

with col_b:
    st.caption("Confusion Matrix")
    cm   = result["confusion_matrix"]
    labs = ["Down", "Up"]
    fig_cm = px.imshow(cm, x=labs, y=labs, text_auto=True,
                       color_continuous_scale=["#0e1117", "#4b9eff"],
                       labels=dict(x="Predicted", y="Actual"))
    fig_cm.update_layout(height=280, template="plotly_dark", paper_bgcolor="#0e1117",
                         plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=10, b=0),
                         coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_c:
    st.caption("Prediction Probability Distribution")
    fig_prob = px.histogram(df, x="signal_prob", nbins=30,
                            color_discrete_sequence=["#4b9eff"])
    fig_prob.add_vline(x=0.5, line_color="#ffd700", line_dash="dash")
    fig_prob.update_layout(height=280, template="plotly_dark", paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_prob, use_container_width=True)

# ─── Feature Importance ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Feature Importance & Engineered Signals</div>', unsafe_allow_html=True)

col_fi, col_vol = st.columns([1, 1])

with col_fi:
    fi = result["feature_importance"].head(12).reset_index()
    fi.columns = ["Feature", "Importance"]
    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#1e2130", "#4b9eff", "#00c9a7"])
    fig_fi.update_layout(height=360, template="plotly_dark", paper_bgcolor="#0e1117",
                         plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=10, b=0),
                         yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

with col_vol:
    fig_v = go.Figure()
    for w, c in [(5, "#ff4b6e"), (20, "#4b9eff"), (30, "#00c9a7")]:
        if f"vol_{w}d" in df.columns:
            fig_v.add_trace(go.Scatter(x=df.index, y=df[f"vol_{w}d"] * 100,
                                       name=f"{w}d Vol", line=dict(color=c, width=1.5)))
    fig_v.update_layout(height=360, template="plotly_dark", paper_bgcolor="#0e1117",
                        plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=10, b=0),
                        title="Annualised Rolling Volatility (%)")
    st.plotly_chart(fig_v, use_container_width=True)

# ─── MACD ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">MACD Indicator</div>', unsafe_allow_html=True)

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"],
    name="MACD", line=dict(color="#4b9eff", width=1.5)))
fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
    name="Signal", line=dict(color="#ffd700", width=1.5)))
hist_colors = ["#00c9a7" if v >= 0 else "#ff4b6e" for v in df["macd_hist"].fillna(0)]
fig_macd.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
    name="Histogram", marker_color=hist_colors, opacity=0.7))
fig_macd.update_layout(height=260, template="plotly_dark", paper_bgcolor="#0e1117",
                        plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_macd, use_container_width=True)

# ─── Scenario Testing ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Scenario Testing</div>', unsafe_allow_html=True)
st.caption("Adjust sidebar sliders to simulate different market conditions and see how the model reacts.")

feat_cols = result["feat"]
scaler    = result["scaler"]
clf       = result["clf"]

last_feat  = df[feat_cols].iloc[-1].copy()
orig_feat  = last_feat.copy()

if "rsi_14"      in last_feat.index: last_feat["rsi_14"]      = np.clip(last_feat["rsi_14"] + scenario_rsi, 0, 100)
for w in [5, 10, 20, 30]:
    k = f"vol_{w}d"
    if k in last_feat.index: last_feat[k] = last_feat[k] * scenario_vol
if "macd"        in last_feat.index: last_feat["macd"]        = last_feat["macd"] + scenario_macd
if "momentum_5"  in last_feat.index: last_feat["momentum_5"]  = last_feat["momentum_5"] + scenario_momentum / 100
if "momentum_20" in last_feat.index: last_feat["momentum_20"] = last_feat["momentum_20"] + scenario_momentum / 100

orig_arr = scaler.transform(orig_feat.values.reshape(1, -1))
scen_arr = scaler.transform(last_feat.values.reshape(1, -1))

orig_prob = clf.predict_proba(orig_arr)[0, 1]
scen_prob = clf.predict_proba(scen_arr)[0, 1]
scen_pred = clf.predict(scen_arr)[0]

sc1, sc2, sc3 = st.columns(3)

def prob_gauge(prob, title):
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]},
               "bar":  {"color": "#4b9eff"},
               "steps": [{"range": [0, 40],  "color": "#2a1520"},
                          {"range": [40, 60], "color": "#1e2130"},
                          {"range": [60, 100],"color": "#152a20"}],
               "threshold": {"line": {"color": "#ffd700", "width": 3},
                              "value": 50}},
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig_g.update_layout(height=230, paper_bgcolor="#0e1117", font_color="#c8cde4",
                        margin=dict(l=20, r=20, t=40, b=10))
    return fig_g

sc1.plotly_chart(prob_gauge(orig_prob, "Baseline Probability (Up)"), use_container_width=True)
sc2.plotly_chart(prob_gauge(scen_prob, "Scenario Probability (Up)"),  use_container_width=True)

delta_prob = (scen_prob - orig_prob) * 100
sc3.markdown(f"""
<div class="metric-card {'blue' if delta_prob >= 0 else 'red'}" style="margin-top:30px">
  <div class="metric-label">Scenario Signal</div>
  <div class="metric-val">{'🟢 BUY' if scen_pred == 1 else '🔴 SELL'}</div>
  <div class="metric-delta {'positive' if delta_prob >= 0 else 'negative'}">
    {'▲' if delta_prob >= 0 else '▼'} {abs(delta_prob):.1f}% vs baseline
  </div>
</div>
""", unsafe_allow_html=True)

sc3.markdown(f"""
<div class="metric-card gold">
  <div class="metric-label">Scenario Parameters</div>
  <div class="metric-delta" style="color:#c8cde4; font-size:13px; line-height:1.8">
    RSI shift: <b>{scenario_rsi:+d}</b><br>
    Vol mult: <b>×{scenario_vol:.1f}</b><br>
    MACD shift: <b>{scenario_macd:+.1f}</b><br>
    Momentum: <b>{scenario_momentum:+d}%</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Returns Distribution ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Returns Analysis</div>', unsafe_allow_html=True)

col_r1, col_r2 = st.columns(2)

with col_r1:
    fig_ret = px.histogram(df["ret_1d"].dropna() * 100, nbins=60,
                           title="Daily Returns Distribution (%)",
                           color_discrete_sequence=["#4b9eff"])
    mean_r = df["ret_1d"].mean() * 100
    fig_ret.add_vline(x=mean_r, line_color="#ffd700", line_dash="dash",
                      annotation_text=f"μ={mean_r:.2f}%")
    fig_ret.update_layout(height=300, template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig_ret, use_container_width=True)

with col_r2:
    fig_cum = px.line(x=df.index, y=df["cum_return"] * 100,
                      title="Cumulative Return (%)", color_discrete_sequence=["#00c9a7"])
    fig_cum.add_hline(y=0, line_color="#8b8fa8", line_dash="dot")
    fig_cum.update_layout(height=300, template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_cum, use_container_width=True)

# ─── Stats table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)

cr = result["class_report"]
stats = {
    "Metric": ["CV Accuracy", "Precision (Up)", "Recall (Up)", "F1 (Up)",
               "Precision (Down)", "Recall (Down)", "F1 (Down)", "Avg Daily Return", "Annualised Vol"],
    "Value": [
        f"{acc*100:.1f}%",
        f"{cr['1']['precision']*100:.1f}%",
        f"{cr['1']['recall']*100:.1f}%",
        f"{cr['1']['f1-score']*100:.1f}%",
        f"{cr['0']['precision']*100:.1f}%",
        f"{cr['0']['recall']*100:.1f}%",
        f"{cr['0']['f1-score']*100:.1f}%",
        f"{df['ret_1d'].mean()*100:.3f}%",
        f"{df['ret_1d'].std()*np.sqrt(252)*100:.1f}%",
    ]
}
st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("⚠️ For educational and demonstration purposes only. Not financial advice.")
