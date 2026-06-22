import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


st.markdown("""
<style>
  body, [data-testid="stAppViewContainer"] { background: #0a0d16; }
  div[data-testid="stSidebar"] { background: #0e1117; }
  .asset-pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: .8px; margin-right: 6px;
  }
  .pill-eq  { background: #1a2a1a; color: #00c9a7; border: 1px solid #00c9a7; }
  .pill-fx  { background: #1a1a2a; color: #4b9eff; border: 1px solid #4b9eff; }
  .pill-cm  { background: #2a2a1a; color: #ffd700; border: 1px solid #ffd700; }
  .pill-bd  { background: #2a1a1a; color: #ff4b6e; border: 1px solid #ff4b6e; }
  .kpi-box {
    background: #141720; border-radius: 10px; padding: 14px 18px;
    border-top: 3px solid var(--accent); margin-bottom: 8px;
  }
  .kpi-label { font-size: 11px; color: #7a7e96; letter-spacing: 1px; text-transform: uppercase; }
  .kpi-val   { font-size: 24px; font-weight: 700; color: #e8eaf6; margin-top: 3px; }
  .kpi-sub   { font-size: 12px; margin-top: 2px; }
  .up   { color: #00c9a7; } .down { color: #ff4b6e; } .neu { color: #7a7e96; }
  .section { font-size: 16px; font-weight: 600; color: #b0b4cc;
    border-bottom: 1px solid #1e2130; padding-bottom: 6px; margin: 18px 0 12px 0; }
  .tab-desc { font-size: 13px; color: #7a7e96; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

ASSETS = {
    "Equities": {
        "S&P 500 (SPY)":    ("SPY",    "eq"),
        "Nasdaq 100 (QQQ)": ("QQQ",    "eq"),
        "Apple (AAPL)":     ("AAPL",   "eq"),
        "Microsoft (MSFT)": ("MSFT",   "eq"),
        "Nvidia (NVDA)":    ("NVDA",   "eq"),
        "Nifty 50 (^NSEI)": ("^NSEI",  "eq"),
        "FTSE 100 (^FTSE)": ("^FTSE",  "eq"),
    },
    "Forex": {
        "EUR/USD":  ("EURUSD=X", "fx"),
        "GBP/USD":  ("GBPUSD=X", "fx"),
        "USD/JPY":  ("JPY=X",    "fx"),
        "USD/INR":  ("INR=X",    "fx"),
        "AUD/USD":  ("AUDUSD=X", "fx"),
        "USD/CHF":  ("CHF=X",    "fx"),
    },
    "Commodities": {
        "Gold (GC=F)":       ("GC=F",  "cm"),
        "Crude Oil (CL=F)":  ("CL=F",  "cm"),
        "Silver (SI=F)":     ("SI=F",  "cm"),
        "Natural Gas (NG=F)":("NG=F",  "cm"),
        "Copper (HG=F)":     ("HG=F",  "cm"),
        "Gold ETF (GLD)":    ("GLD",   "cm"),
    },
    "Bond Yields": {
        "US 10Y Yield (^TNX)": ("^TNX", "bd"),
        "US 2Y Yield (^IRX)":  ("^IRX", "bd"),
        "US 30Y Yield (^TYX)": ("^TYX", "bd"),
    },
}

PERIODS = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}

PILL_CLASS = {"eq": "pill-eq", "fx": "pill-fx", "cm": "pill-cm", "bd": "pill-bd"}
PILL_LABEL = {"eq": "EQUITY", "fx": "FOREX", "cm": "COMMODITY", "bd": "BOND"}
ACCENT     = {"eq": "#00c9a7", "fx": "#4b9eff", "cm": "#ffd700", "bd": "#ff4b6e"}

MACRO_TICKERS = {
    "SPY (Equities)": "SPY",
    "GLD (Gold)":     "GLD",
    "TLT (Bonds)":    "TLT",
    "UUP (USD)":      "UUP",
    "USO (Oil)":      "USO",
}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    return raw.dropna()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    d["ret_1d"]       = c.pct_change()
    d["cum_ret"]      = (1 + d["ret_1d"]).cumprod() - 1
    d["ma_20"]        = c.rolling(20).mean()
    d["ma_50"]        = c.rolling(50).mean()
    d["vol_7d"]       = d["ret_1d"].rolling(7).std() * np.sqrt(252)
    d["vol_30d"]      = d["ret_1d"].rolling(30).std() * np.sqrt(252)
    d["vol_90d"]      = d["ret_1d"].rolling(90).std() * np.sqrt(252)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).replace(0, np.nan).rolling(14).mean()
    d["rsi_14"] = 100 - 100 / (1 + gain / loss)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_pct"]   = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])
    if "High" in d.columns and "Low" in d.columns:
        d["spread_pct"] = (d["High"] - d["Low"]) / c * 100
    return d


def dark_layout(height=420, title=None):
    kwargs = dict(
        height=height, template="plotly_dark",
        paper_bgcolor="#0a0d16", plot_bgcolor="#0a0d16",
        margin=dict(l=0, r=0, t=30 if title else 10, b=0),
        font=dict(color="#b0b4cc"),
    )
    if title:
        kwargs["title"] = dict(text=title, font=dict(size=13, color="#7a7e96"), x=0)
    return kwargs


def kpi(col, label, val, sub=None, accent="#00c9a7"):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi-box" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val">{val}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 Market Explorer")
    asset_class = st.selectbox("Asset Class", list(ASSETS.keys()))
    asset_name  = st.selectbox("Instrument",  list(ASSETS[asset_class].keys()))
    period_lbl  = st.selectbox("Time Period", list(PERIODS.keys()), index=1)
    period      = PERIODS[period_lbl]
    st.markdown("---")
    st.markdown("### Compare Assets")
    compare_on  = st.checkbox("Enable cross-asset comparison")
    compare_sel = []
    if compare_on:
        all_labels = [f"{cls} › {nm}" for cls, d in ASSETS.items() for nm in d]
        compare_sel = st.multiselect("Compare with", all_labels, max_selections=4)

ticker, atype = ASSETS[asset_class][asset_name]
accent = ACCENT[atype]

# ── Load ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker}…"):
    raw = fetch(ticker, period)

if raw.empty:
    st.error(f"No data for {ticker}. Check connection.")
    st.stop()

df = preprocess(raw)

# ── Header ───────────────────────────────────────────────────────────────────
pill = f'<span class="asset-pill {PILL_CLASS[atype]}">{PILL_LABEL[atype]}</span>'
st.markdown(
    f'<h2 style="margin-bottom:2px">{pill} {asset_name} &nbsp;'
    f'<span style="font-size:14px;color:#7a7e96">· {period_lbl} · {ticker}</span></h2>',
    unsafe_allow_html=True,
)

latest  = df.iloc[-1]
prev    = df.iloc[-2]
chg     = latest["Close"] - prev["Close"]
chg_pct = chg / prev["Close"] * 100
vol30   = latest["vol_30d"] * 100
cum     = latest["cum_ret"] * 100

c1, c2, c3, c4, c5 = st.columns(5)
sign  = "▲" if chg >= 0 else "▼"
cls   = "up" if chg >= 0 else "down"
kpi(c1, "Price",        f"{latest['Close']:.4g}",
    f'<span class="{cls}">{sign} {abs(chg_pct):.2f}%</span>', accent)
kpi(c2, "30d Vol (ann)", f"{vol30:.1f}%",        None, accent)
kpi(c3, "Cum Return",   f"{cum:.1f}%",
    f'<span class="{"up" if cum>=0 else "down"}">{"▲" if cum>=0 else "▼"} {abs(cum):.1f}%</span>', accent)
kpi(c4, "RSI (14)",     f"{latest['rsi_14']:.1f}",
    '<span class="up">Oversold</span>' if latest["rsi_14"] < 30
    else ('<span class="down">Overbought</span>' if latest["rsi_14"] > 70 else '<span class="neu">Neutral</span>'),
    accent)
kpi(c5, "MA20 vs MA50",
    "Bullish" if latest["ma_20"] > latest["ma_50"] else "Bearish",
    f'MA20: {latest["ma_20"]:.4g} / MA50: {latest["ma_50"]:.4g}', accent)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price & Trends", "📊 Preprocessing", "🔁 Returns Analysis",
    "🌍 Macro Overview", "📋 Data Table",
])

# ── Tab 1: Price & Trends ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="tab-desc">OHLC candlestick with moving averages, Bollinger Bands, RSI, and MACD.</div>',
                unsafe_allow_html=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.48, 0.18, 0.17, 0.17],
                        vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", increasing_line_color="#00c9a7", decreasing_line_color="#ff4b6e"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["ma_20"], name="MA 20",
        line=dict(color="#4b9eff", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_50"], name="MA 50",
        line=dict(color="#ffd700", width=1.4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
        line=dict(color=f"rgba(150,150,255,0.35)", dash="dash"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
        fill="tonexty", fillcolor="rgba(100,100,200,0.07)",
        line=dict(color="rgba(150,150,255,0.35)", dash="dash"), showlegend=False), row=1, col=1)

    if "Volume" in df.columns:
        vol_colors = ["#00c9a7" if r >= 0 else "#ff4b6e" for r in df["ret_1d"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, opacity=0.65), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14",
        line=dict(color="#c47aff", width=1.4)), row=3, col=1)
    fig.add_hline(y=70, line_color="rgba(255,75,110,0.4)", line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_color="rgba(0,201,167,0.4)", line_dash="dot", row=3, col=1)

    hist_colors = ["#00c9a7" if v >= 0 else "#ff4b6e" for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
        name="MACD", line=dict(color="#4b9eff", width=1.3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
        name="Signal", line=dict(color="#ffd700", width=1.3)), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
        name="Histogram", marker_color=hist_colors, opacity=0.65), row=4, col=1)

    fig.update_layout(**dark_layout(720), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.01, x=0))
    fig.update_yaxes(gridcolor="#141720")
    fig.update_yaxes(title_text="Price",  row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Volume", row=2, col=1, title_font_size=10)
    fig.update_yaxes(title_text="RSI",    row=3, col=1, title_font_size=10, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   row=4, col=1, title_font_size=10)
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Preprocessing ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="tab-desc">Data preprocessing steps: missing values, normalisation, engineered features.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section">Raw vs Preprocessed</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)

    with ca:
        st.caption("Raw OHLCV statistics")
        raw_stat = raw[["Open", "High", "Low", "Close", "Volume"]].describe().T
        raw_stat.columns = [c.title() for c in raw_stat.columns]
        st.dataframe(raw_stat.style.format("{:.4g}"), use_container_width=True)

    with cb:
        st.caption("Missing values & data quality")
        total = len(raw)
        miss  = raw.isnull().sum()
        qual  = pd.DataFrame({
            "Column": miss.index,
            "Missing": miss.values,
            "Missing %": (miss.values / total * 100).round(2),
            "dtype": [str(raw[c].dtype) for c in miss.index],
        })
        st.dataframe(qual, use_container_width=True, hide_index=True)

    st.markdown('<div class="section">Engineered Features</div>', unsafe_allow_html=True)

    feat_cols = ["ret_1d", "cum_ret", "ma_20", "ma_50", "vol_7d", "vol_30d",
                 "rsi_14", "macd", "macd_signal", "bb_pct"]
    existing  = [c for c in feat_cols if c in df.columns]
    st.dataframe(df[existing].tail(60).style.format("{:.4f}").background_gradient(
        cmap="RdYlGn", subset=["ret_1d", "cum_ret"]), use_container_width=True)

    st.markdown('<div class="section">Rolling Volatility Regimes</div>', unsafe_allow_html=True)

    fig_vol = go.Figure()
    for w, col, label in [(7, "#ff4b6e", "7d"), (30, "#4b9eff", "30d"), (90, "#00c9a7", "90d")]:
        col_name = f"vol_{w}d"
        if col_name in df.columns:
            fig_vol.add_trace(go.Scatter(x=df.index, y=df[col_name] * 100,
                name=f"Vol {label} (ann. %)", line=dict(color=col, width=1.5)))
    fig_vol.update_layout(**dark_layout(300, "Annualised Rolling Volatility (%)"))
    fig_vol.update_yaxes(gridcolor="#141720")
    st.plotly_chart(fig_vol, use_container_width=True)

    if "spread_pct" in df.columns:
        st.markdown('<div class="section">Intraday Spread (High−Low / Close)</div>', unsafe_allow_html=True)
        fig_sp = px.area(x=df.index, y=df["spread_pct"],
                         color_discrete_sequence=[accent])
        fig_sp.update_layout(**dark_layout(200))
        fig_sp.update_yaxes(gridcolor="#141720")
        st.plotly_chart(fig_sp, use_container_width=True)


# ── Tab 3: Returns Analysis ────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="tab-desc">Distribution, autocorrelation, and drawdown analysis of daily returns.</div>',
                unsafe_allow_html=True)

    rets = df["ret_1d"].dropna()

    r1, r2 = st.columns(2)

    with r1:
        fig_hist = px.histogram(rets * 100, nbins=60, title="Daily Returns Distribution (%)",
                                color_discrete_sequence=[accent])
        mean_r = rets.mean() * 100
        std_r  = rets.std()  * 100
        fig_hist.add_vline(x=mean_r, line_color="#ffd700", line_dash="dash",
                           annotation_text=f"μ={mean_r:.2f}%")
        fig_hist.add_vline(x=mean_r - 2*std_r, line_color="#ff4b6e", line_dash="dot")
        fig_hist.add_vline(x=mean_r + 2*std_r, line_color="#ff4b6e", line_dash="dot",
                           annotation_text="±2σ")
        fig_hist.update_layout(**dark_layout(320))
        st.plotly_chart(fig_hist, use_container_width=True)

    with r2:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=df.index, y=df["cum_ret"] * 100,
            fill="tozeroy",
            fillcolor=f"rgba({'0,201,167' if cum>=0 else '255,75,110'},0.1)",
            line=dict(color=accent, width=1.8), name="Cumulative Return (%)"))
        fig_cum.add_hline(y=0, line_color="#3a3e52", line_dash="dot")
        fig_cum.update_layout(**dark_layout(320, "Cumulative Return (%)"))
        fig_cum.update_yaxes(gridcolor="#141720")
        st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown('<div class="section">Drawdown Analysis</div>', unsafe_allow_html=True)

    wealth    = (1 + rets).cumprod()
    peak      = wealth.cummax()
    drawdown  = (wealth - peak) / peak * 100
    max_dd    = drawdown.min()

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(255,75,110,0.15)",
        line=dict(color="#ff4b6e", width=1.2), name="Drawdown (%)"))
    fig_dd.add_hline(y=max_dd, line_color="#ffd700", line_dash="dash",
                     annotation_text=f"Max DD: {max_dd:.1f}%")
    fig_dd.update_layout(**dark_layout(260, "Drawdown from Peak (%)"))
    fig_dd.update_yaxes(gridcolor="#141720")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown('<div class="section">Return Statistics</div>', unsafe_allow_html=True)
    ann_vol  = rets.std() * np.sqrt(252) * 100
    sharpe   = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    skew     = float(rets.skew())
    kurt     = float(rets.kurtosis())
    var_95   = float(np.percentile(rets, 5) * 100)
    cvar_95  = float(rets[rets <= np.percentile(rets, 5)].mean() * 100)

    stats_df = pd.DataFrame({
        "Metric": ["Ann. Volatility", "Sharpe Ratio", "Skewness", "Excess Kurtosis",
                   "VaR (95%)", "CVaR (95%)", "Max Drawdown", "Best Day", "Worst Day"],
        "Value":  [f"{ann_vol:.2f}%", f"{sharpe:.2f}", f"{skew:.3f}", f"{kurt:.3f}",
                   f"{var_95:.2f}%", f"{cvar_95:.2f}%", f"{max_dd:.2f}%",
                   f"{rets.max()*100:.2f}%", f"{rets.min()*100:.2f}%"],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ── Tab 4: Macro Overview ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="tab-desc">Cross-asset correlation, macro indicator trends, and yield curve analysis.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section">Cross-Asset Correlation Matrix</div>', unsafe_allow_html=True)

    with st.spinner("Fetching macro assets…"):
        macro_dfs = {}
        for label, t in MACRO_TICKERS.items():
            d = fetch(t, period)
            if not d.empty:
                macro_dfs[label] = d["Close"].pct_change().rename(label)

    if macro_dfs:
        merged = pd.concat(macro_dfs.values(), axis=1).dropna()
        corr   = merged.corr()

        fig_corr = px.imshow(
            corr, text_auto=".2f", zmin=-1, zmax=1,
            color_continuous_scale=["#ff4b6e", "#141720", "#00c9a7"],
            title="Cross-Asset Return Correlation",
        )
        fig_corr.update_layout(**dark_layout(400))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown('<div class="section">Normalised Price Trends (Indexed to 100)</div>',
                    unsafe_allow_html=True)

        price_frames = {}
        for label, t in MACRO_TICKERS.items():
            d = fetch(t, period)
            if not d.empty:
                price_frames[label] = (d["Close"] / d["Close"].iloc[0] * 100)

        if price_frames:
            fig_idx = go.Figure()
            colors  = ["#00c9a7", "#ffd700", "#ff4b6e", "#4b9eff", "#c47aff"]
            for i, (label, series) in enumerate(price_frames.items()):
                fig_idx.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    name=label, line=dict(color=colors[i % len(colors)], width=1.5),
                ))
            fig_idx.add_hline(y=100, line_color="#3a3e52", line_dash="dot")
            fig_idx.update_layout(**dark_layout(380, "Indexed Price Performance (Base = 100)"))
            fig_idx.update_yaxes(gridcolor="#141720")
            st.plotly_chart(fig_idx, use_container_width=True)

    st.markdown('<div class="section">US Yield Curve</div>', unsafe_allow_html=True)

    with st.spinner("Fetching yield data…"):
        tnx = fetch("^TNX", period)
        irx = fetch("^IRX", period)

    if not tnx.empty and not irx.empty:
        ya, yb = st.columns(2)

        with ya:
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(x=tnx.index, y=tnx["Close"],
                name="US 10Y", line=dict(color="#ff4b6e", width=1.5)))
            fig_yc.add_trace(go.Scatter(x=irx.index, y=irx["Close"],
                name="US 2Y",  line=dict(color="#4b9eff", width=1.5)))
            fig_yc.update_layout(**dark_layout(300, "US Treasury Yields (%)"))
            fig_yc.update_yaxes(gridcolor="#141720")
            st.plotly_chart(fig_yc, use_container_width=True)

        with yb:
            common = tnx.index.intersection(irx.index)
            slope  = tnx.loc[common, "Close"] - irx.loc[common, "Close"]
            fig_sl = go.Figure()
            fig_sl.add_trace(go.Scatter(
                x=slope.index, y=slope.values,
                fill="tozeroy",
                fillcolor="rgba(0,201,167,0.1)" if slope.iloc[-1] >= 0 else "rgba(255,75,110,0.1)",
                line=dict(color="#00c9a7" if slope.iloc[-1] >= 0 else "#ff4b6e", width=1.5),
                name="10Y − 2Y spread",
            ))
            fig_sl.add_hline(y=0, line_color="#ffd700", line_dash="dash",
                             annotation_text="Inversion threshold")
            fig_sl.update_layout(**dark_layout(300, "Yield Curve Slope: 10Y − 2Y (bps÷10)"))
            fig_sl.update_yaxes(gridcolor="#141720")
            st.plotly_chart(fig_sl, use_container_width=True)

        cur_slope = slope.iloc[-1]
        inverted  = cur_slope < 0
        st.info(
            f"**Yield Curve is {'🔴 INVERTED' if inverted else '🟢 NORMAL'}** "
            f"— Current slope: {cur_slope:.2f} (10Y: {tnx['Close'].iloc[-1]:.2f}% / "
            f"2Y: {irx['Close'].iloc[-1]:.2f}%)"
        )


# ── Tab 5: Data Table ──────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="tab-desc">Full preprocessed dataset with all engineered features.</div>',
                unsafe_allow_html=True)

    show_cols = [c for c in ["Open", "High", "Low", "Close", "Volume",
                              "ret_1d", "cum_ret", "ma_20", "ma_50",
                              "vol_30d", "rsi_14", "macd", "bb_pct"] if c in df.columns]
    n_rows = st.slider("Rows to display", 20, min(500, len(df)), 100)
    disp   = df[show_cols].tail(n_rows).sort_index(ascending=False)
    st.dataframe(disp.style.format("{:.4g}").background_gradient(
        cmap="RdYlGn", subset=["ret_1d"] if "ret_1d" in disp.columns else []),
        use_container_width=True)

    csv = df[show_cols].to_csv()
    st.download_button("Download CSV", csv, file_name=f"{ticker}_processed.csv", mime="text/csv")


# ── Cross-asset comparison ─────────────────────────────────────────────────────
if compare_on and compare_sel:
    st.markdown("---")
    st.markdown('<div class="section">Cross-Asset Comparison</div>', unsafe_allow_html=True)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=df.index, y=(df["Close"] / df["Close"].iloc[0] * 100),
        name=asset_name, line=dict(color=accent, width=2),
    ))

    colors = ["#4b9eff", "#ffd700", "#c47aff", "#ff4b6e"]
    for i, label in enumerate(compare_sel):
        parts = label.split(" › ")
        if len(parts) == 2:
            cls_name, inst_name = parts
            if cls_name in ASSETS and inst_name in ASSETS[cls_name]:
                cmp_ticker, cmp_type = ASSETS[cls_name][inst_name]
                with st.spinner(f"Fetching {cmp_ticker}…"):
                    cmp_df = fetch(cmp_ticker, period)
                if not cmp_df.empty:
                    idx = cmp_df["Close"] / cmp_df["Close"].iloc[0] * 100
                    fig_cmp.add_trace(go.Scatter(
                        x=cmp_df.index, y=idx,
                        name=inst_name, line=dict(color=colors[i % len(colors)], width=1.6),
                    ))

    fig_cmp.add_hline(y=100, line_color="#3a3e52", line_dash="dot")
    fig_cmp.update_layout(**dark_layout(380, f"Normalised Performance vs {asset_name} (Base = 100)"))
    fig_cmp.update_yaxes(gridcolor="#141720")
    st.plotly_chart(fig_cmp, use_container_width=True)


st.markdown("---")
st.caption("Data via Yahoo Finance · For educational and research purposes only · Not financial advice.")
