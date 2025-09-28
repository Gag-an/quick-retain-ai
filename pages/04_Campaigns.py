import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Page config
st.set_page_config(page_title="Campaigns", layout="wide", initial_sidebar_state="collapsed")

# ---------- NAVBAR (reuse existing style) ----------
def navbar(active: str):
    c_brand, c_spacer, c1, c2, c3, c4, c5 = st.columns([1.4, 5.8, 1.25, 1.9, 1.6, 1.25, 1.6])
    with c_brand:
        st.markdown('<div class="qr-brand"><span class="qr-cube"></span><span>QuickRetain</span></div>', unsafe_allow_html=True)
    with c1: st.page_link("app.py", label="Features")
    with c2: st.page_link("pages/01_Churn_SHAP.py", label="Churn + SHAP")
    with c3: st.page_link("pages/02_Retention_RL.py", label="Retention RL")
    with c4: st.page_link("pages/03_Logistics.py", label="Logistics")
    with c5: st.page_link("pages/04_Campaigns.py", label="🎯 Campaigns")

    st.markdown(f"""
    <style>
      div[data-testid="stHorizontalBlock"]:has(.qr-brand) {{
        position: sticky; top: 0; z-index: 9999;
        background: rgba(255,255,255,.92);
        backdrop-filter: saturate(170%) blur(12px);
        border-bottom: 1px solid rgba(20,16,50,.06);
        box-shadow: 0 8px 24px rgba(20,16,50,.06);
        padding: 14px 18px;
        margin-top: -8px;
      }}
      div[data-testid="stHorizontalBlock"]:has(.qr-brand) > div {{ padding-top: 0 !important; }}
      .qr-brand {{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.15rem; color:#121826; }}
      .qr-cube {{ width:18px; height:18px; border-radius:4px; background: linear-gradient(135deg,#7C4DFF,#6C3BE2); box-shadow: 0 2px 8px rgba(98,56,226,.35); display:inline-block; }}
      a[data-testid="stPageLink"] {{
        display:inline-flex; align-items:center; gap:.4rem; padding:10px 16px !important; border-radius:999px !important;
        text-decoration:none !important; border:1px solid rgba(20,16,50,.12) !important; background: rgba(255,255,255,.78) !important;
        box-shadow: 0 4px 12px rgba(20,16,50,.06) !important; color:#121826 !important; font-weight:600 !important;
        white-space: nowrap !important; transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease;
      }}
      a[data-testid="stPageLink"]:hover {{ transform: translateY(-1px); border-color: rgba(102,52,226,.45) !important; box-shadow: 0 8px 18px rgba(20,16,50,.10) !important; }}
      a[data-testid="stPageLink"][href$="{active}"] {{ background: linear-gradient(180deg,#fff,#F5F4FF) !important; border-color: rgba(102,52,226,.65) !important; box-shadow: 0 10px 22px rgba(102,52,226,.18) !important; }}
    </style>
    """, unsafe_allow_html=True)

navbar(active="pages/04_Campaigns.py")
st.title("🎯 Targeted Campaign Builder")
st.caption("Score uplift (offer vs no-offer), pick top customers under a budget, and export CSV.")

# ---------- Load processed data ----------
CAND_PATHS = [
    r"D:\\quick Retain Ai\\data\\processed\\retention_events.csv",
    r"D:\\data-logistics\\processed\\retention_events.csv",
    "data/processed/retention_events.csv",
]
PATH = next((p for p in CAND_PATHS if os.path.exists(p)), None)
if not PATH:
    st.error("retention_events.csv not found in data/processed")
    st.stop()

df = pd.read_csv(PATH, parse_dates=["timestamp"]).sort_values(["user_id","timestamp"]).copy()

# ---------- Scope & basic features ----------
SCOPE = st.radio("Scope", ["Both", "blinkit", "bigbasket"], horizontal=True)
if SCOPE != "Both":
    df = df[df["platform"] == SCOPE].copy()

# Feature engineering
df["recency_days"] = (df.groupby("user_id")["timestamp"].diff().dt.days).fillna(999).clip(0, 365)
df["orders_so_far"] = df.groupby("user_id").cumcount()
df["hour"] = df["timestamp"].dt.hour
df["dow"]  = df["timestamp"].dt.dayofweek

# Logged actions & rewards from historical data
df["action"] = (df["discount_given"] > 0).astype(int)
df["reward"] = np.where(df["repeat_purchase"].eq(1), df["basket_value"] - df["discount_given"], 0.0)

use_feats = ["basket_value","discount_given","recency_days","orders_so_far","hour","dow","lat","lon"]

# ---------- Train counterfactual reward models ----------
@st.cache_resource(show_spinner=False)
def train_reward_models(data: pd.DataFrame):
    m0 = RandomForestRegressor(n_estimators=300, random_state=42)
    m1 = RandomForestRegressor(n_estimators=300, random_state=42)
    mask0, mask1 = data["action"].eq(0), data["action"].eq(1)
    if mask0.sum() < 50 or mask1.sum() < 50:
        st.warning("Not enough historical variety to train both action models robustly. Results may be noisy.")
    X0, y0 = data.loc[mask0, use_feats], data.loc[mask0, "reward"]
    X1, y1 = data.loc[mask1, use_feats], data.loc[mask1, "reward"]
    if len(X0) == 0:
        X0, y0 = data[use_feats], data["reward"]
    if len(X1) == 0:
        X1, y1 = data[use_feats], data["reward"]
    m0.fit(X0, y0)
    m1.fit(X1, y1)
    return m0, m1

rf0, rf1 = train_reward_models(df)

# ---------- Build a scoring set (latest context per user) ----------
latest = df.sort_values("timestamp").groupby("user_id").tail(1).copy()

st.markdown("### ① Configure Campaign")
c1, c2, c3, c4 = st.columns(4)
with c1:
    offer_rupees = st.number_input("Offer amount (₹)", 0, 500, 100, 10)
with c2:
    max_coupons  = st.number_input("Max coupons to issue today", 1, 50000, 1000, 50)
with c3:
    min_uplift   = st.number_input("Min uplift (₹) to qualify", 0.0, 500.0, 5.0, 1.0)
with c4:
    min_basket   = st.number_input("Min basket value (₹)", 0.0, 10000.0, 200.0, 50.0)

# Optional cooldown: exclude users who very recently got an offer
cooldown_days = st.slider("Exclude users who purchased within last X days", 0, 60, 0)
if cooldown_days > 0:
    latest = latest[(latest["recency_days"] >= cooldown_days) | latest["recency_days"].isna()]

# ---------- Counterfactual prediction for each user ----------
def predict_rewards(latest_rows: pd.DataFrame, offer_amount: float) -> pd.DataFrame:
    X0 = latest_rows[use_feats].copy()
    X1 = latest_rows[use_feats].copy()
    X0["discount_given"] = 0
    X1["discount_given"] = offer_amount

    r0 = rf0.predict(X0)  # no offer
    r1 = rf1.predict(X1)  # offer
    out = latest_rows[["user_id","platform","timestamp","basket_value","discount_given","recency_days","orders_so_far","hour","dow","lat","lon"]].copy()
    out["r_no_offer"] = r0
    out["r_offer"]    = r1
    out["uplift"]     = out["r_offer"] - out["r_no_offer"]
    return out

scores = predict_rewards(latest, offer_rupees)

# ---------- Filter + pick under budget ----------
cand = scores.query("basket_value >= @min_basket and uplift >= @min_uplift").copy()
cand = cand.sort_values("uplift", ascending=False)
selected = cand.head(int(max_coupons)).copy()

uplift_sum = float(selected["uplift"].sum())
roi_per_rupee = (uplift_sum / (len(selected) * offer_rupees)) if offer_rupees > 0 and len(selected) > 0 else np.nan

st.markdown("### ② Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Candidates", f"{len(cand):,}")
m2.metric("Selected (budget)", f"{len(selected):,}")
m3.metric("Expected uplift (₹)", f"{uplift_sum:,.2f}")
m4.metric("ROI per ₹ coupon", f"{roi_per_rupee:,.2f}" if not np.isnan(roi_per_rupee) else "—")

st.dataframe(selected[["user_id","platform","timestamp","basket_value","r_no_offer","r_offer","uplift"]]
             .rename(columns={"r_no_offer":"Reward: no-offer","r_offer":"Reward: offer"})
             .round(2), use_container_width=True, hide_index=True)

# ---------- Download CSV ----------
csv = selected.sort_values("uplift", ascending=False).round(3).to_csv(index=False)
st.download_button("⬇️ Download campaign CSV", data=csv, file_name=f"campaign_{SCOPE.lower()}_{offer_rupees}Rs.csv", mime="text/csv")
st.caption("CSV includes expected rewards for both actions and the uplift per user.")
