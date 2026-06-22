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

st.set_page_config(
    page_title="FICC Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0d16; }
  [data-testid="stSidebar"]          { background: #0d1017; }
  [data-testid="stHeader"]           { background: transparent; }
  .kpi {
    background: #131720; border-radius: 10px; padding: 14px 16px;
    border-top: 3px solid var(--a); margin-bottom: 8px;
  }
  .kpi-label { font-size: 10px; color: #606480; letter-spacing: 1.2px; text-transform: uppercase; }
  .kpi-val   { font-size: 22px; font-weight: 700; color: #e8eaf6; margin-top: 3px; }
  .kpi-sub   { font-size: 12px; margin-top: 2px; }
  .up { color: #00c9a7; } .dn { color: #ff4b6e; } .nu { color: #606480; }
  .sec {
    font-size: 11px; font-weight: 700; color: #606480; letter-spacing: 1.2px;
    text-transform: uppercase; border-bottom: 1px solid #1a1e2e;
    padding-bottom: 5px; margin: 20px 0 12px 0;
  }
  .pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 700; letter-spacing: .9px; margin-right: 6px;
  }
</style>
""", unsafe_allow_html=True)

ASSETS = {
    "Equities": {
        "S&P 500 (SPY)":    ("SPY",    "#00c9a7"),
        "Nasdaq 100 (QQQ)": ("QQQ",    "#00c9a7"),
        "Apple (AAPL)":     ("AAPL",   "#00c9a7"),
        "Microsoft (MSFT)": ("MSFT",   "#00c9a7"),
        "Nvidia (NVDA)":    ("NVDA",   "#00c9a7"),
        "Tesla (TSLA)":     ("TSLA",   "#00c9a7"),
        "Meta (META)":      ("META",   "#00c9a7"),
        "Nifty 50 (^NSEI)": ("^NSEI",  "#00c9a7"),
        "FTSE 100 (^FTSE)": ("^FTSE",  "#00c9a7"),
    },
    "Forex": {
        "EUR/USD": ("EURUSD=X", "#4b9eff"),
        "GBP/USD": ("GBPUSD=X", "#4b9eff"),
        "USD/JPY": ("JPY=X",    "#4b9eff"),
        "USD/INR": ("INR=X",    "#4b9eff"),
        "AUD/USD": ("AUDUSD=X", "#4b9eff"),
    },
    "Commodities": {
        "Gold (GC=F)":        ("GC=F",  "#ffd700"),
        "Crude Oil (CL=F)":   ("CL=F",  "#ffd700"),
        "Silver (SI=F)":      ("SI=F",  "#ffd700"),
        "Gold ETF (GLD)":     ("GLD",   "#ffd700"),
        "Natural Gas (NG=F)": ("NG=F",  "#ffd700"),
    },
    "Bond Yields": {
        "US 10Y (^TNX)": ("^TNX", "#ff4b6e"),
        "US 2Y (^IRX)":  ("^IRX", "#ff4b6e"),
        "US 30Y (^TYX)": ("^TYX", "#ff4b6e"),
    },
}

PERIODS = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}

MACRO = {
    "S&P 500 (SPY)": "SPY",
    "Gold (GLD)":    "GLD",
    "Bonds (TLT)":   "TLT",
    "USD (UUP)":     "UUP",
    "Oil (USO)":     "USO",
}

FEATURE_COLS = [
    "ret_1d","ret_3d","ret_5d","ret_10d",
    "ma_ratio_5_20","ma_ratio_10_50","price_vs_ma20","price_vs_ma50",
    "vol_5d","vol_10d","vol_20d","vol_30d",
    "rsi_14","macd","macd_signal","macd_hist",
    "bb_width","bb_pos","vol_ratio","vol_ret","momentum_5","momentum_20",
]

CLS_ACCENT = {
    "Equities": "#00c9a7", "Forex": "#4b9eff",
    "Commodities": "#ffd700", "Bond Yields": "#ff4b6e",
}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    return raw.dropna()


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    d["ret_1d"]  = c.pct_change(1)
    d["ret_3d"]  = c.pct_change(3)
    d["ret_5d"]  = c.pct_change(5)
    d["ret_10d"] = c.pct_change(10)
    d["cum_ret"] = (1 + d["ret_1d"]).cumprod() - 1
    for w in [5, 10, 20, 50]:
        d[f"ma_{w}"] = c.rolling(w).mean()
    d["ma_ratio_5_20"]  = d["ma_5"]  / d["ma_20"]
    d["ma_ratio_10_50"] = d["ma_10"] / d["ma_50"]
    d["price_vs_ma20"]  = c / d["ma_20"]
    d["price_vs_ma50"]  = c / d["ma_50"]
    for w in [5, 10, 20, 30]:
        d[f"vol_{w}d"] = d["ret_1d"].rolling(w).std() * np.sqrt(252)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"]      = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12            = c.ewm(span=12, adjust=False).mean()
    ema26            = c.ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]
    bb_mid           = c.rolling(20).mean()
    bb_std           = c.rolling(20).std()
    d["bb_upper"]    = bb_mid + 2 * bb_std
    d["bb_lower"]    = bb_mid - 2 * bb_std
    d["bb_width"]    = (d["bb_upper"] - d["bb_lower"]) / bb_mid
    d["bb_pos"]      = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])
    if "Volume" in d.columns:
        d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
        d["vol_ret"]   = d["Volume"] * d["ret_1d"]
    else:
        d["vol_ratio"] = 1.0
        d["vol_ret"]   = 0.0
    if {"High", "Low"}.issubset(d.columns):
        d["spread_pct"] = (d["High"] - d["Low"]) / c * 100
    d["momentum_5"]  = c / c.shift(5)  - 1
    d["momentum_20"] = c / c.shift(20) - 1
    d["target"]      = (c.shift(-1) > c).astype(int)
    return d


@st.cache_data(ttl=1800, show_spinner=False)
def train_model(ticker: str, period: str, model_name: str):
    raw = fetch(ticker, period)
    if raw.empty:
        return None
    df = engineer(raw).dropna(subset=["ret_1d", "rsi_14", "macd"]).iloc[:-1]
    feat = [c for c in FEATURE_COLS if c in df.columns]
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(df[feat].values)
    y      = df["target"].values
    clf_map = {
        "Random Forest":     RandomForestClassifier(n_estimators=300, max_depth=8,
                                                     min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                         learning_rate=0.05, random_state=42),
    }
    clf = clf_map[model_name]
    all_true, all_pred, accs = [], [], []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X_s):
        clf.fit(X_s[tr], y[tr])
        p = clf.predict(X_s[te])
        accs.append(accuracy_score(y[te], p))
        all_true.extend(y[te])
        all_pred.extend(p)
    clf.fit(X_s, y)
    df["signal"]      = clf.predict(X_s)
    df["signal_prob"] = clf.predict_proba(X_s)[:, 1]
    fi = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False)
    return {
        "df": df, "feat": feat, "scaler": scaler, "clf": clf,
        "cv_accs": accs, "accuracy": np.mean(accs),
        "feature_importance": fi,
        "class_report": classification_report(all_true, all_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(all_true, all_pred),
    }


def dlayout(h=420, title=None):
    kw = dict(
        height=h, template="plotly_dark",
        paper_bgcolor="#0a0d16", plot_bgcolor="#0a0d16",
        margin=dict(l=0, r=0, t=28 if title else 8, b=0),
        font=dict(color="#b0b4cc"),
    )
    if title:
        kw["title"] = dict(text=title, font=dict(size=12, color="#606480"), x=0)
    return kw


def kpi_card(col, label, val, sub=None, accent="#00c9a7"):
    sh = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi" style="--a:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val">{val}</div>{sh}</div>',
        unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 FICC Dashboard")
    st.markdown('<div class="sec">Asset</div>', unsafe_allow_html=True)
    asset_class = st.selectbox("Class",      list(ASSETS.keys()),          label_visibility="collapsed")
    asset_name  = st.selectbox("Instrument", list(ASSETS[asset_class].keys()), label_visibility="collapsed")
    period_lbl  = st.selectbox("Period",     list(PERIODS.keys()), index=1, label_visibility="collapsed")
    period      = PERIODS[period_lbl]

    st.markdown('<div class="sec">ML Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Model", ["Random Forest", "Gradient Boosting"],
                               label_visibility="collapsed")

    st.markdown('<div class="sec">Scenario Testing</div>', unsafe_allow_html=True)
    st.caption("Shift features to simulate market conditions")
    sc_rsi  = st.slider("RSI adjustment",        -30,  30,  0)
    sc_vol  = st.slider("Volatility multiplier",  0.5, 3.0, 1.0, 0.1)
    sc_macd = st.slider("MACD shift",            -5.0, 5.0, 0.0, 0.5)
    sc_mom  = st.slider("Momentum shift (%)",    -10,  10,  0)


ticker, accent = ASSETS[asset_class][asset_name]
ac = CLS_ACCENT[asset_class]

# ── Load & engineer ───────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker}…"):
    raw = fetch(ticker, period)

if raw.empty:
    st.error(f"No data for {ticker}. Check your connection.")
    st.stop()

df = engineer(raw).dropna(subset=["ret_1d", "rsi_14", "macd"])

if len(df) < 2:
    st.error(f"Not enough data for {ticker} after preprocessing. Try a longer time period or a different instrument.")
    st.stop()

latest  = df.iloc[-1]
prev    = df.iloc[-2]
chg_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
cum     = latest["cum_ret"] * 100
vol30   = latest["vol_30d"] * 100

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<h2 style="margin:0 0 6px 0">'
    f'<span class="pill" style="background:{ac}22;color:{ac};border:1px solid {ac}">'
    f'{asset_class.upper()}</span>'
    f'{asset_name}&nbsp;<span style="font-size:14px;color:#606480">· {period_lbl} · {ticker}</span></h2>',
    unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
sign = "▲" if chg_pct >= 0 else "▼"
cls  = "up"   if chg_pct >= 0 else "dn"
kpi_card(k1, "Price",         f"{latest['Close']:.4g}",
         f'<span class="{cls}">{sign} {abs(chg_pct):.2f}%</span>', ac)
kpi_card(k2, "30d Vol (ann)", f"{vol30:.1f}%", None, ac)
kpi_card(k3, "Cum Return",    f"{cum:+.1f}%",
         f'<span class="{"up" if cum>=0 else "dn"}">{"▲" if cum>=0 else "▼"} {abs(cum):.1f}%</span>', ac)
kpi_card(k4, "RSI 14",        f"{latest['rsi_14']:.1f}",
         '<span class="up">Oversold</span>'   if latest["rsi_14"] < 30
    else '<span class="dn">Overbought</span>' if latest["rsi_14"] > 70
    else '<span class="nu">Neutral</span>', ac)
kpi_card(k5, "Trend",
         "Bullish" if latest["ma_20"] > latest["ma_50"] else "Bearish",
         f'MA20 {latest["ma_20"]:.4g} · MA50 {latest["ma_50"]:.4g}', ac)
kpi_card(k6, "MACD Signal",
         "Buy" if latest["macd"] > latest["macd_signal"] else "Sell",
         f'MACD {latest["macd"]:.3f}', ac)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "📈  Price & Indicators",
    "🤖  ML Prediction",
    "📊  Returns & Risk",
    "🌍  Macro Overview",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price & Indicators
# ════════════════════════════════════════════════════════════════════════════
with t1:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.48, 0.18, 0.17, 0.17],
                        vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#00c9a7", decreasing_line_color="#ff4b6e"), row=1, col=1)

    for w, col in [(20, "#4b9eff"), (50, "#ffd700")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"ma_{w}"], name=f"MA{w}",
            line=dict(color=col, width=1.4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"],
        line=dict(color="rgba(130,130,220,0.35)", dash="dash"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"],
        fill="tonexty", fillcolor="rgba(100,100,200,0.07)",
        line=dict(color="rgba(130,130,220,0.35)", dash="dash"), name="Bollinger Bands"), row=1, col=1)

    if "Volume" in df.columns:
        vcols = ["#00c9a7" if r >= 0 else "#ff4b6e" for r in df["ret_1d"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
            marker_color=vcols, opacity=0.6), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14",
        line=dict(color="#c47aff", width=1.4)), row=3, col=1)
    fig.add_hline(y=70, line_color="rgba(255,75,110,0.4)",  line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_color="rgba(0,201,167,0.4)",   line_dash="dot", row=3, col=1)

    hcols = ["#00c9a7" if v >= 0 else "#ff4b6e" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
        name="MACD",   line=dict(color="#4b9eff", width=1.3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
        name="Signal", line=dict(color="#ffd700", width=1.3)), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
        name="Histogram", marker_color=hcols, opacity=0.65), row=4, col=1)

    fig.update_layout(**dlayout(720), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_yaxes(gridcolor="#141720")
    fig.update_yaxes(title_text="Price",  title_font_size=10, row=1, col=1)
    fig.update_yaxes(title_text="Volume", title_font_size=10, row=2, col=1)
    fig.update_yaxes(title_text="RSI",    title_font_size=10, row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   title_font_size=10, row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec">Preprocessed Feature Table — last 60 rows</div>',
                unsafe_allow_html=True)
    show = [c for c in ["ret_1d","cum_ret","ma_20","ma_50","vol_20d","vol_30d",
                         "rsi_14","macd","bb_pos","spread_pct"] if c in df.columns]
    st.dataframe(df[show].tail(60).style.format("{:.4f}")
                 .background_gradient(cmap="RdYlGn", subset=["ret_1d","cum_ret"]),
                 use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="sec">Rolling Volatility (annualised %)</div>', unsafe_allow_html=True)
        fv = go.Figure()
        for w, col, lbl in [(7,"#ff4b6e","7d"),(20,"#4b9eff","20d"),(30,"#00c9a7","30d")]:
            k = f"vol_{w}d"
            if k in df.columns:
                fv.add_trace(go.Scatter(x=df.index, y=df[k]*100,
                    name=f"Vol {lbl}", line=dict(color=col, width=1.4)))
        fv.update_layout(**dlayout(260))
        fv.update_yaxes(gridcolor="#141720")
        st.plotly_chart(fv, use_container_width=True)

    with cb:
        if "spread_pct" in df.columns:
            st.markdown('<div class="sec">Intraday Spread (High−Low / Close %)</div>',
                        unsafe_allow_html=True)
            fs = px.area(x=df.index, y=df["spread_pct"], color_discrete_sequence=[accent])
            fs.update_layout(**dlayout(260))
            fs.update_yaxes(gridcolor="#141720")
            st.plotly_chart(fs, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML Prediction
# ════════════════════════════════════════════════════════════════════════════
with t2:
    with st.spinner(f"Training {model_name}…"):
        res = train_model(ticker, period, model_name)

    if res is None:
        st.error("Model training failed.")
        st.stop()

    mdf = res["df"]
    acc = res["accuracy"]
    cr  = res["class_report"]

    ml1, ml2, ml3, ml4 = st.columns(4)
    sig     = mdf.iloc[-1]
    prob    = sig["signal_prob"] * 100
    kpi_card(ml1, "ML Signal",      "🟢 BUY" if sig["signal"]==1 else "🔴 SELL",
             f'Probability: {prob:.1f}%', "#4b9eff")
    kpi_card(ml2, "CV Accuracy",    f"{acc*100:.1f}%", "5-fold time-series CV", "#4b9eff")
    kpi_card(ml3, "Precision (Up)", f"{cr['1']['precision']*100:.1f}%", None, "#4b9eff")
    kpi_card(ml4, "Recall (Up)",    f"{cr['1']['recall']*100:.1f}%",    None, "#4b9eff")

    st.markdown('<div class="sec">Price Chart with Buy / Sell Signals & Prediction Probability</div>',
                unsafe_allow_html=True)
    fs = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       row_heights=[0.70, 0.30], vertical_spacing=0.04)
    fs.add_trace(go.Candlestick(
        x=mdf.index, open=mdf["Open"], high=mdf["High"],
        low=mdf["Low"], close=mdf["Close"], name="OHLC",
        increasing_line_color="#00c9a7", decreasing_line_color="#ff4b6e"), row=1, col=1)
    for w, col in [(20, "#4b9eff"), (50, "#ffd700")]:
        fs.add_trace(go.Scatter(x=mdf.index, y=mdf[f"ma_{w}"], name=f"MA{w}",
            line=dict(color=col, width=1.3)), row=1, col=1)
    buys  = mdf[mdf["signal"] == 1]
    sells = mdf[mdf["signal"] == 0]
    fs.add_trace(go.Scatter(x=buys.index,  y=buys["Low"]   * 0.985, mode="markers",
        marker=dict(symbol="triangle-up",   size=7, color="#00c9a7"), name="Buy"),  row=1, col=1)
    fs.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.015, mode="markers",
        marker=dict(symbol="triangle-down", size=7, color="#ff4b6e"), name="Sell"), row=1, col=1)
    fs.add_trace(go.Scatter(x=mdf.index, y=mdf["signal_prob"]*100, name="Up Prob %",
        fill="tozeroy", fillcolor="rgba(75,158,255,0.08)",
        line=dict(color="#4b9eff", width=1.4)), row=2, col=1)
    fs.add_hline(y=50, line_color="#ffd700", line_dash="dot", row=2, col=1)
    fs.update_layout(**dlayout(560), xaxis_rangeslider_visible=False,
                     legend=dict(orientation="h", y=1.02, x=0))
    fs.update_yaxes(gridcolor="#141720")
    fs.update_yaxes(title_text="Prob %", title_font_size=10, row=2, col=1)
    st.plotly_chart(fs, use_container_width=True)

    pa, pb, pc = st.columns(3)
    with pa:
        st.markdown('<div class="sec">CV Accuracy per Fold</div>', unsafe_allow_html=True)
        cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(res["cv_accs"]))],
                              "Accuracy": res["cv_accs"]})
        fcv = px.bar(cv_df, x="Fold", y="Accuracy", text_auto=".1%",
                     color="Accuracy", color_continuous_scale=["#ff4b6e","#ffd700","#00c9a7"],
                     range_y=[0.4, 1.0])
        fcv.add_hline(y=0.8, line_color="#ffd700", line_dash="dash", annotation_text="80%")
        fcv.update_layout(**dlayout(280), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fcv, use_container_width=True)

    with pb:
        st.markdown('<div class="sec">Confusion Matrix</div>', unsafe_allow_html=True)
        fcm = px.imshow(res["confusion_matrix"], x=["Down","Up"], y=["Down","Up"],
                        text_auto=True, color_continuous_scale=["#0a0d16","#4b9eff"],
                        labels=dict(x="Predicted", y="Actual"))
        fcm.update_layout(**dlayout(280), coloraxis_showscale=False)
        st.plotly_chart(fcm, use_container_width=True)

    with pc:
        st.markdown('<div class="sec">Feature Importance (Top 12)</div>', unsafe_allow_html=True)
        fi = res["feature_importance"].head(12).reset_index()
        fi.columns = ["Feature","Importance"]
        ffi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#131720","#4b9eff","#00c9a7"])
        ffi.update_layout(**dlayout(280), yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
        st.plotly_chart(ffi, use_container_width=True)

    st.markdown('<div class="sec">Scenario Testing</div>', unsafe_allow_html=True)
    st.caption("Use the sidebar sliders to shift feature values and see how the model reacts.")
    feat_cols = res["feat"]
    scaler    = res["scaler"]
    clf       = res["clf"]
    last_f    = mdf[feat_cols].iloc[-1].copy()
    orig_f    = last_f.copy()
    if "rsi_14"      in last_f.index: last_f["rsi_14"]      = np.clip(last_f["rsi_14"]+sc_rsi, 0, 100)
    for w in [5, 10, 20, 30]:
        k = f"vol_{w}d"
        if k in last_f.index: last_f[k] *= sc_vol
    if "macd"        in last_f.index: last_f["macd"]        += sc_macd
    if "momentum_5"  in last_f.index: last_f["momentum_5"]  += sc_mom / 100
    if "momentum_20" in last_f.index: last_f["momentum_20"] += sc_mom / 100
    orig_p = clf.predict_proba(scaler.transform(orig_f.values.reshape(1,-1)))[0, 1]
    scen_p = clf.predict_proba(scaler.transform(last_f.values.reshape(1,-1)))[0, 1]
    scen_s = clf.predict(scaler.transform(last_f.values.reshape(1,-1)))[0]
    delta  = (scen_p - orig_p) * 100

    def gauge(prob, title):
        return go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            title={"text": title, "font": {"size": 13}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4b9eff"},
                   "steps": [{"range":[0,40],"color":"#1a0d12"},
                              {"range":[40,60],"color":"#131720"},
                              {"range":[60,100],"color":"#0d1a12"}],
                   "threshold": {"line":{"color":"#ffd700","width":3},"value":50}},
            number={"suffix":"%","font":{"size":28}},
        )).update_layout(height=220, paper_bgcolor="#0a0d16",
                         font_color="#b0b4cc", margin=dict(l=20,r=20,t=40,b=10))

    sc1, sc2, sc3 = st.columns(3)
    sc1.plotly_chart(gauge(orig_p, "Baseline (Up %)"), use_container_width=True)
    sc2.plotly_chart(gauge(scen_p, "Scenario (Up %)"), use_container_width=True)
    da = "#4b9eff" if delta >= 0 else "#ff4b6e"
    sc3.markdown(f"""
    <div class="kpi" style="--a:{da};margin-top:24px">
      <div class="kpi-label">Scenario Signal</div>
      <div class="kpi-val">{'🟢 BUY' if scen_s==1 else '🔴 SELL'}</div>
      <div class="kpi-sub {'up' if delta>=0 else 'dn'}">
        {'▲' if delta>=0 else '▼'} {abs(delta):.1f}% vs baseline
      </div>
    </div>
    <div class="kpi" style="--a:#ffd700;margin-top:10px">
      <div class="kpi-label">Scenario Inputs</div>
      <div class="kpi-sub" style="color:#b0b4cc;font-size:13px;line-height:2">
        RSI shift <b>{sc_rsi:+d}</b> &nbsp;·&nbsp; Vol ×<b>{sc_vol:.1f}</b><br>
        MACD <b>{sc_macd:+.1f}</b> &nbsp;·&nbsp; Momentum <b>{sc_mom:+d}%</b>
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Returns & Risk
# ════════════════════════════════════════════════════════════════════════════
with t3:
    rets = df["ret_1d"].dropna()
    mu   = rets.mean() * 100
    sd   = rets.std()  * 100

    r1, r2 = st.columns(2)
    with r1:
        fh = px.histogram(rets*100, nbins=60, title="Daily Returns Distribution (%)",
                          color_discrete_sequence=[accent])
        fh.add_vline(x=mu, line_color="#ffd700", line_dash="dash",
                     annotation_text=f"μ={mu:.2f}%")
        fh.add_vline(x=mu-2*sd, line_color="#ff4b6e", line_dash="dot")
        fh.add_vline(x=mu+2*sd, line_color="#ff4b6e", line_dash="dot",
                     annotation_text="±2σ")
        fh.update_layout(**dlayout(320))
        st.plotly_chart(fh, use_container_width=True)

    with r2:
        fc = go.Figure()
        fc.add_trace(go.Scatter(x=df.index, y=df["cum_ret"]*100,
            fill="tozeroy",
            fillcolor=f"rgba({'0,201,167' if cum>=0 else '255,75,110'},0.1)",
            line=dict(color=accent, width=1.8), name="Cumulative Return (%)"))
        fc.add_hline(y=0, line_color="#3a3e52", line_dash="dot")
        fc.update_layout(**dlayout(320, "Cumulative Return (%)"))
        fc.update_yaxes(gridcolor="#141720")
        st.plotly_chart(fc, use_container_width=True)

    st.markdown('<div class="sec">Drawdown from Peak</div>', unsafe_allow_html=True)
    wealth = (1 + rets).cumprod()
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak * 100
    max_dd = dd.min()
    fd = go.Figure()
    fd.add_trace(go.Scatter(x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(255,75,110,0.12)",
        line=dict(color="#ff4b6e", width=1.3), name="Drawdown %"))
    fd.add_hline(y=max_dd, line_color="#ffd700", line_dash="dash",
                 annotation_text=f"Max DD: {max_dd:.1f}%")
    fd.update_layout(**dlayout(240))
    fd.update_yaxes(gridcolor="#141720")
    st.plotly_chart(fd, use_container_width=True)

    st.markdown('<div class="sec">Risk Statistics</div>', unsafe_allow_html=True)
    ann_vol = rets.std() * np.sqrt(252) * 100
    sharpe  = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    var95   = float(np.percentile(rets, 5) * 100)
    cvar95  = float(rets[rets <= np.percentile(rets, 5)].mean() * 100)
    st.dataframe(pd.DataFrame({
        "Metric": ["Ann. Volatility","Sharpe Ratio","Skewness","Excess Kurtosis",
                   "VaR 95%","CVaR 95%","Max Drawdown","Best Day","Worst Day"],
        "Value":  [f"{ann_vol:.2f}%", f"{sharpe:.2f}", f"{float(rets.skew()):.3f}",
                   f"{float(rets.kurtosis()):.3f}", f"{var95:.2f}%", f"{cvar95:.2f}%",
                   f"{max_dd:.2f}%", f"{rets.max()*100:.2f}%", f"{rets.min()*100:.2f}%"],
    }), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Macro Overview
# ════════════════════════════════════════════════════════════════════════════
with t4:
    with st.spinner("Fetching macro assets…"):
        macro_ret   = {}
        macro_price = {}
        for label, t in MACRO.items():
            d = fetch(t, period)
            if not d.empty:
                macro_ret[label]   = d["Close"].pct_change().rename(label)
                macro_price[label] = (d["Close"] / d["Close"].iloc[0] * 100).rename(label)

    ma1, ma2 = st.columns(2)
    with ma1:
        st.markdown('<div class="sec">Cross-Asset Correlation</div>', unsafe_allow_html=True)
        if macro_ret:
            corr = pd.concat(macro_ret.values(), axis=1).dropna().corr()
            fcr  = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                             color_continuous_scale=["#ff4b6e","#0a0d16","#00c9a7"])
            fcr.update_layout(**dlayout(360))
            st.plotly_chart(fcr, use_container_width=True)

    with ma2:
        st.markdown('<div class="sec">Normalised Performance (Base = 100)</div>',
                    unsafe_allow_html=True)
        if macro_price:
            fp = go.Figure()
            colours = ["#00c9a7","#ffd700","#ff4b6e","#4b9eff","#c47aff"]
            for i, (label, series) in enumerate(macro_price.items()):
                fp.add_trace(go.Scatter(x=series.index, y=series.values,
                    name=label, line=dict(color=colours[i%len(colours)], width=1.5)))
            fp.add_hline(y=100, line_color="#3a3e52", line_dash="dot")
            fp.update_layout(**dlayout(360))
            fp.update_yaxes(gridcolor="#141720")
            st.plotly_chart(fp, use_container_width=True)

    st.markdown('<div class="sec">US Yield Curve</div>', unsafe_allow_html=True)
    with st.spinner("Fetching yields…"):
        tnx = fetch("^TNX", period)
        irx = fetch("^IRX", period)

    if not tnx.empty and not irx.empty:
        yc1, yc2 = st.columns(2)
        with yc1:
            fy = go.Figure()
            fy.add_trace(go.Scatter(x=tnx.index, y=tnx["Close"],
                name="10Y", line=dict(color="#ff4b6e", width=1.5)))
            fy.add_trace(go.Scatter(x=irx.index, y=irx["Close"],
                name="2Y",  line=dict(color="#4b9eff", width=1.5)))
            fy.update_layout(**dlayout(280, "US Treasury Yields (%)"))
            fy.update_yaxes(gridcolor="#141720")
            st.plotly_chart(fy, use_container_width=True)

        with yc2:
            common = tnx.index.intersection(irx.index)
            slope  = tnx.loc[common, "Close"] - irx.loc[common, "Close"]
            pos    = slope.iloc[-1] >= 0
            fsl = go.Figure()
            fsl.add_trace(go.Scatter(x=slope.index, y=slope.values,
                fill="tozeroy",
                fillcolor=f"rgba({'0,201,167' if pos else '255,75,110'},0.1)",
                line=dict(color="#00c9a7" if pos else "#ff4b6e", width=1.5),
                name="10Y − 2Y"))
            fsl.add_hline(y=0, line_color="#ffd700", line_dash="dash",
                          annotation_text="Inversion threshold")
            fsl.update_layout(**dlayout(280, "Yield Curve Slope: 10Y − 2Y"))
            fsl.update_yaxes(gridcolor="#141720")
            st.plotly_chart(fsl, use_container_width=True)

        st.info(
            f"**Yield Curve {'🔴 INVERTED' if not pos else '🟢 NORMAL'}** — "
            f"Slope: {slope.iloc[-1]:.2f} · "
            f"10Y: {tnx['Close'].iloc[-1]:.2f}% · "
            f"2Y: {irx['Close'].iloc[-1]:.2f}%"
        )

st.markdown("---")
st.caption("Data via Yahoo Finance · Educational use only · Not financial advice.")
