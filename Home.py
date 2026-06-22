import streamlit as st

st.set_page_config(
    page_title="Fintech Stock Predictor",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
  body { background: #0a0d16; }
  .hero { text-align: center; padding: 60px 0 40px 0; }
  .hero h1 { font-size: 48px; font-weight: 800; color: #e8eaf6; margin-bottom: 8px; }
  .hero p  { font-size: 18px; color: #7a7e96; margin-bottom: 40px; }
  .card {
    background: #141720; border-radius: 14px; padding: 28px 24px;
    border-top: 3px solid var(--accent); text-align: center;
  }
  .card h3 { font-size: 20px; color: #e8eaf6; margin: 12px 0 8px 0; }
  .card p  { font-size: 14px; color: #7a7e96; line-height: 1.6; }
  .badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
  }
</style>

<div class="hero">
  <h1>📈 Fintech Stock Predictor</h1>
  <p>Multi-asset market intelligence — exploration, prediction, and macro analysis in one place.</p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("""
    <div class="card" style="--accent:#00c9a7">
      <span class="badge" style="background:#0d2b22;color:#00c9a7;border:1px solid #00c9a7">EXPLORE</span>
      <h3>🌐 Market Data Explorer</h3>
      <p>Browse equities, forex, commodities, and bond yields. Visualise trends,
      preprocessing steps, return distributions, drawdowns, and cross-asset correlations.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Market_Explorer.py", label="Open Market Explorer →", use_container_width=True)

with c2:
    st.markdown("""
    <div class="card" style="--accent:#4b9eff">
      <span class="badge" style="background:#0d1a2b;color:#4b9eff;border:1px solid #4b9eff">PREDICT</span>
      <h3>🤖 ML Prediction Dashboard</h3>
      <p>Train Random Forest or Gradient Boosting classifiers on 20+ engineered features.
      Walk-forward cross-validation, confusion matrix, feature importance, and scenario testing.</p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Prediction_Dashboard.py", label="Open Prediction Dashboard →", use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:8px">
  <div style="color:#7a7e96;font-size:13px">✅ &nbsp;Equities · Forex · Commodities · Bonds</div>
  <div style="color:#7a7e96;font-size:13px">✅ &nbsp;20+ technical indicators computed live</div>
  <div style="color:#7a7e96;font-size:13px">✅ &nbsp;Walk-forward ML with time-series CV</div>
  <div style="color:#7a7e96;font-size:13px">✅ &nbsp;Scenario testing & yield curve analysis</div>
</div>
<br>
<div style="color:#3a3e52;font-size:12px">Data via Yahoo Finance · Educational use only · Not financial advice.</div>
""", unsafe_allow_html=True)
