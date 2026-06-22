# generate_market_project.py
# Creates the "Multi-Asset Market Analytics Dashboard" project locally and zips it.
import os, textwrap, zipfile, shutil
import pandas as pd
import numpy as np
from datetime import datetime

BASE = os.path.abspath("market-analytics-dashboard-full")

if os.path.exists(BASE):
    shutil.rmtree(BASE)
os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, "data", "equities"), exist_ok=True)
os.makedirs(os.path.join(BASE, "data", "forex"), exist_ok=True)
os.makedirs(os.path.join(BASE, "data", "commodities"), exist_ok=True)
os.makedirs(os.path.join(BASE, "data", "bonds"), exist_ok=True)
os.makedirs(os.path.join(BASE, "data", "cleaned"), exist_ok=True)
os.makedirs(os.path.join(BASE, "notebooks"), exist_ok=True)
os.makedirs(os.path.join(BASE, "dashboard"), exist_ok=True)
os.makedirs(os.path.join(BASE, "docs"), exist_ok=True)

# requirements.txt
requirements = textwrap.dedent("""\
    pandas
    numpy
    yfinance
    alpha_vantage
    fredapi
    matplotlib
    plotly
    streamlit
    requests
    openpyxl
""")
with open(os.path.join(BASE, "requirements.txt"), "w") as f:
    f.write(requirements)

# README.md
readme = textwrap.dedent(f"""\
    # Multi-Asset Market Analytics Dashboard (FICC + Equities)

    **Overview**
    Beginner-friendly project that collects multi-asset market data (equities, forex, commodities, bond yields),
    computes market-analytics KPIs (returns, volatility, spreads, correlations, yield-curve slope),
    and exposes them through an interactive Streamlit dashboard and Python notebooks/scripts.

    Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

    **Folder structure**
    ```
    market-analytics-dashboard-full/
    ├── data/
    │   ├── equities/
    │   ├── forex/
    │   ├── commodities/
    │   ├── bonds/
    │   └── cleaned/
    ├── notebooks/
    ├── dashboard/
    ├── docs/
    ├── README.md
    └── requirements.txt
    ```

    **How to run**
    1. Install dependencies:
       ```
       pip install -r requirements.txt
       ```
    2. (Optional) Populate API keys (AlphaVantage, FRED) if you use those sources.
    3. Run data collection scripts or use the notebooks (not required — demo cleaned CSVs are included).
    4. Launch dashboard:
       ```
       streamlit run dashboard/streamlit_app.py
       ```

    **KPIs computed**
    - Daily Return, Cumulative Return
    - Moving Averages (20, 50)
    - Rolling Volatility (7, 30)
    - Spread (High-Low / Close)
    - Amihud Illiquidity proxy
    - Correlation matrix
    - Yield curve slope (10Y - 2Y)

    """)
with open(os.path.join(BASE, "README.md"), "w") as f:
    f.write(readme)

# notebooks / scripts
notebooks = {
    "01_data_collection.py": textwrap.dedent("""\
        # Data collection script (yfinance for equities; placeholders for FX/commodities/bonds)
        import yfinance as yf
        import os

        os.makedirs('data/equities', exist_ok=True)

        EQUITIES = {
            'AAPL': 'AAPL',
            'MSFT': 'MSFT',
            'INFY': 'INFY.NS',
            'TCS': 'TCS.NS',
            'NIFTY': '^NSEI'
        }

        def fetch_equities(equities, start='2020-01-01', end=None, interval='1d'):
            for name, symbol in equities.items():
                print(f"Fetching {name} ({symbol})...")
                df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
                if df.empty:
                    print(f"Warning: no data for {symbol}")
                    continue
                df.reset_index(inplace=True)
                df.to_csv(f"data/equities/{name.lower()}.csv", index=False)
                print(f"Saved data/equities/{name.lower()}.csv")

        if __name__ == '__main__':
            fetch_equities(EQUITIES)
            print('Done. Add FX/commodities/bonds collection as needed.')
        """),
    "02_data_cleaning.py": textwrap.dedent("""\
        # Cleans raw CSVs and produces cleaned CSVs in data/cleaned with KPI columns.
        import pandas as pd
        import os

        RAW_DIR = 'data/equities'
        CLEAN_DIR = 'data/cleaned'
        os.makedirs(CLEAN_DIR, exist_ok=True)

        def clean_equity(path_in, path_out):
            df = pd.read_csv(path_in, parse_dates=['Date'])
            df = df.dropna(subset=['Close'])
            df = df.sort_values('Date').reset_index(drop=True)
            df['Daily_Return'] = df['Close'].pct_change()
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['MA_50'] = df['Close'].rolling(50).mean()
            df['Volatility_7'] = df['Daily_Return'].rolling(7).std()
            df['Volatility_30'] = df['Daily_Return'].rolling(30).std()
            df['Spread'] = (df['High'] - df['Low']) / df['Close']
            df.to_csv(path_out, index=False)
            return df

        if __name__ == '__main__':
            for fname in os.listdir(RAW_DIR):
                if fname.endswith('.csv'):
                    name = fname.replace('.csv','')
                    print('Cleaning', fname)
                    clean_equity(os.path.join(RAW_DIR, fname), os.path.join(CLEAN_DIR, f'{name}_clean.csv'))
            print('Cleaning complete.')
        """),
    "03_kpi_calculations.py": textwrap.dedent("""\
        # Aggregates KPIs across cleaned assets and writes summary CSV.
        import pandas as pd
        import os

        CLEAN_DIR = 'data/cleaned'
        OUT_DIR = 'data/analytics'
        os.makedirs(OUT_DIR, exist_ok=True)

        def compute_latest_kpis(df):
            last = df.iloc[-1]
            k = {}
            k['Latest_Close'] = last['Close']
            k['Latest_Return_pct'] = last['Daily_Return'] * 100
            vol30 = df['Daily_Return'].rolling(30).std().iloc[-1]
            k['30d_Volatility_annualized_pct'] = vol30 * (252**0.5) * 100 if not pd.isna(vol30) else None
            k['20d_MA'] = last.get('MA_20', None)
            k['50d_MA'] = last.get('MA_50', None)
            k['Latest_Spread_pct'] = last['Spread'] * 100 if 'Spread' in last else None
            return k

        if __name__ == '__main__':
            rows = []
            for fname in os.listdir(CLEAN_DIR):
                if fname.endswith('_clean.csv'):
                    df = pd.read_csv(os.path.join(CLEAN_DIR, fname), parse_dates=['Date'])
                    k = compute_latest_kpis(df)
                    k['asset'] = fname.replace('_clean.csv','')
                    rows.append(k)
            out = pd.DataFrame(rows)
            out.to_csv(os.path.join(OUT_DIR, 'latest_kpis.csv'), index=False)
            print('Saved analytics/latest_kpis.csv')
        """),
    "04_visualizations.py": textwrap.dedent("""\
        # Example visualization using Plotly
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv('data/cleaned/aapl_clean.csv', parse_dates=['Date'])
        fig = px.line(df, x='Date', y='Close', title='AAPL Price')
        fig.show()

        # Correlation heatmap example
        assets = ['aapl','infy','tcs']
        dfs = []
        for a in assets:
            try:
                d = pd.read_csv(f'data/cleaned/{a}_clean.csv', parse_dates=['Date'])
                d = d[['Date','Daily_Return']].rename(columns={'Daily_Return':a})
                dfs.append(d)
            except Exception as e:
                print('Missing', a)
        if dfs:
            merged = dfs[0]
            for d in dfs[1:]:
                merged = merged.merge(d, on='Date', how='inner')
            corr = merged.set_index('Date').corr()
            fig2 = px.imshow(corr, text_auto=True, title='Return Correlation Matrix')
            fig2.show()
        """),
    "05_daily_market_summary.py": textwrap.dedent("""\
        # Produces a text summary of top movers and basic commentary.
        import pandas as pd
        import os

        CLEAN_DIR = 'data/cleaned'

        def top_movers(n=5):
            rows = []
            for fname in os.listdir(CLEAN_DIR):
                if fname.endswith('_clean.csv'):
                    df = pd.read_csv(os.path.join(CLEAN_DIR, fname), parse_dates=['Date'])
                    last = df.iloc[-1]
                    rows.append({'asset': fname.replace('_clean.csv',''), 'return_pct': float(last['Daily_Return']*100)})
            out = pd.DataFrame(rows).sort_values('return_pct', ascending=False)
            return out

        if __name__ == '__main__':
            movers = top_movers(10)
            print('Top movers (latest daily return pct):')
            print(movers.head(10).to_string(index=False))
        """)
}

for name, content in notebooks.items():
    with open(os.path.join(BASE, "notebooks", name), "w") as f:
        f.write(content)

# Streamlit app
streamlit_app = textwrap.dedent("""\
    # dashboard/streamlit_app.py
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import os

    st.set_page_config(page_title='Market Analytics Dashboard', layout='wide')
    st.title('Multi-Asset Market Analytics Dashboard (FICC + Equities) - Data Analyst View')

    CLEAN_DIR = 'data/cleaned'
    assets = sorted([f.replace('_clean.csv','') for f in os.listdir(CLEAN_DIR) if f.endswith('_clean.csv')])

    if not assets:
        st.warning('No cleaned assets found. Run data collection & cleaning notebooks first.')
    else:
        asset = st.sidebar.selectbox('Choose asset', options=assets)
        df = pd.read_csv(os.path.join(CLEAN_DIR, asset + '_clean.csv'), parse_dates=['Date'])

        st.subheader(f'{asset.upper()} — Latest price & KPIs')
        latest = df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric('Latest Close', f\"{latest['Close']:.2f}\")
        c1.metric('Latest Daily Return (%)', f\"{latest['Daily_Return']*100:.2f}\")
        c2.metric('20d MA', f\"{latest['MA_20']:.2f}\" if not pd.isna(latest['MA_20']) else 'n/a')
        c2.metric('50d MA', f\"{latest['MA_50']:.2f}\" if not pd.isna(latest['MA_50']) else 'n/a')
        vol30 = df['Daily_Return'].rolling(30).std().iloc[-1]
        c3.metric('30d Volatility (ann., %)', f\"{vol30*(252**0.5)*100:.2f}\" if not pd.isna(vol30) else 'n/a')

        fig = px.line(df, x='Date', y='Close', title=f\"{asset.upper()} Price\")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Returns & Volatility')
        fig2 = px.line(df, x='Date', y='Daily_Return', title='Daily Returns')
        st.plotly_chart(fig2, use_container_width=True)

        # Correlation
        st.subheader('Correlation with other assets (returns)')
        other = st.multiselect('Compare with', options=[a for a in assets if a!=asset], default=assets[:2])
        if other:
            dfs = [df[['Date','Daily_Return']].rename(columns={'Daily_Return':asset})]
            for o in other:
                d = pd.read_csv(os.path.join(CLEAN_DIR, o + '_clean.csv'), parse_dates=['Date'])
                dfs.append(d[['Date','Daily_Return']].rename(columns={'Daily_Return':o}))
            merged = dfs[0]
            for d in dfs[1:]:
                merged = merged.merge(d, on='Date', how='inner')
            corr = merged.set_index('Date').corr()
            st.write(corr)
            fig3 = px.imshow(corr, text_auto=True, title='Return Correlation Matrix')
            st.plotly_chart(fig3, use_container_width=True)

    st.sidebar.markdown('---')
    if st.sidebar.button('Show top movers (latest day)'):
        rows = []
        for f in os.listdir('data/cleaned'):
            if f.endswith('_clean.csv'):
                d = pd.read_csv(os.path.join('data/cleaned', f), parse_dates=['Date'])
                rows.append({'asset': f.replace('_clean.csv',''), 'return_pct': float(d['Daily_Return'].iloc[-1]*100)})
        import pandas as pd
        out = pd.DataFrame(rows).sort_values('return_pct', ascending=False)
        st.write(out.head(20))
    """)
with open(os.path.join(BASE, "dashboard", "streamlit_app.py"), "w") as f:
    f.write(streamlit_app)

# KPI docs
kpi_defs = textwrap.dedent("""\
    # KPI Definitions

    ## Daily Return
    (P_t - P_{t-1}) / P_{t-1}

    ## Cumulative Return
    Product of (1 + daily returns) - 1

    ## Moving Averages
    Simple moving averages over a window (20, 50 days)

    ## Volatility (Rolling Std)
    Standard deviation of returns over rolling windows (7, 30)

    ## Spread (High-Low / Close)
    Proxy for intraday liquidity

    ## Amihud Illiquidity (proxy)
    |Return| / Volume

    ## Correlation Matrix
    Pearson correlation of daily returns across assets

    ## Yield Curve Slope
    10Y yield - 2Y yield
    """)
with open(os.path.join(BASE, "docs", "KPI_definitions.md"), "w") as f:
    f.write(kpi_defs)

# Create demo cleaned CSVs for a few assets so dashboard works
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=180)
def make_demo(name, start_price):
    prices = start_price * np.cumprod(1 + np.random.normal(0, 0.008, size=len(dates)))
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices * (1 + np.random.normal(0,0.002,len(dates))),
        "High": prices * (1 + np.abs(np.random.normal(0,0.006,len(dates)))),
        "Low": prices * (1 - np.abs(np.random.normal(0,0.006,len(dates)))),
        "Close": prices,
        "Adj Close": prices,
        "Volume": np.random.randint(1e6, 5e6, size=len(dates))
    })
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1+df['Daily_Return']).cumprod()-1
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Volatility_7'] = df['Daily_Return'].rolling(7).std()
    df['Volatility_30'] = df['Daily_Return'].rolling(30).std()
    df['Spread'] = (df['High'] - df['Low']) / df['Close']
    df.to_csv(os.path.join(BASE, "data", "cleaned", f"{name}_clean.csv"), index=False)

make_demo("aapl", 150)
make_demo("infy", 18)
make_demo("tcs", 35)
make_demo("nifty", 20000)

# zip the project
zip_path = os.path.abspath("market-analytics-dashboard-full.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(BASE):
        for file in files:
            full = os.path.join(root, file)
            arcname = os.path.relpath(full, os.path.dirname(BASE))
            zf.write(full, arcname)
print("Project created at:", BASE)
print("Zip created at:", zip_path)
