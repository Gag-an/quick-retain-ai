# app.py
# QuickRetain — minimal white theme with glass navbar + full-bleed hero (no box)

import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="QuickRetain AI", page_icon="🚀", layout="wide")

# ---- Truck loader (CSS + helpers) ----
_TRUCK_LOADER_CSS = """
<style>
.qr-loader-wrap{ width:100%; height:60vh; display:flex; align-items:center; justify-content:center; background:transparent; }
.qr-loader{ width:fit-content; height:fit-content; display:flex; align-items:center; justify-content:center; }
.qr-truckWrapper{ width:200px; height:100px; display:flex; flex-direction:column; position:relative; align-items:center; justify-content:flex-end; overflow-x:hidden; }
.qr-truckBody{ width:130px; height:fit-content; margin-bottom:6px; animation:qr-motion 1s linear infinite; }
@keyframes qr-motion{ 0%{transform:translateY(0)} 50%{transform:translateY(3px)} 100%{transform:translateY(0)} }
.qr-truckTires{ width:130px; height:fit-content; display:flex; align-items:center; justify-content:space-between; padding:0 10px 0 15px; position:absolute; bottom:0; }
.qr-truckTires svg{ width:24px; }
.qr-road{ width:100%; height:1.5px; background-color:#282828; position:relative; bottom:0; align-self:flex-end; border-radius:3px; }
.qr-road::before{ content:""; position:absolute; width:20px; height:100%; background-color:#282828; right:-50%; border-radius:3px; animation:qr-roadAnim 1.4s linear infinite; border-left:10px solid #fff; }
.qr-road::after{ content:""; position:absolute; width:10px; height:100%; background-color:#282828; right:-65%; border-radius:3px; animation:qr-roadAnim 1.4s linear infinite; border-left:4px solid #fff; }
.qr-lampPost{ position:absolute; bottom:0; right:-90%; height:90px; animation:qr-roadAnim 1.4s linear infinite; }
@keyframes qr-roadAnim{ 0%{transform:translateX(0)} 100%{transform:translateX(-350px)} }
</style>
"""

_TRUCK_LOADER_HTML = """
<div class="qr-loader-wrap">
  <div class="qr-loader">
    <div class="qr-truckWrapper">
      <div class="qr-truckBody">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 198 93" class="trucksvg">
          <path stroke-width="3" stroke="#282828" fill="#F83D3D" d="M135 22.5H177.264C178.295 22.5 179.22 23.133 179.594 24.0939L192.33 56.8443C192.442 57.1332 192.5 57.4404 192.5 57.7504V89C192.5 90.3807 191.381 91.5 190 91.5H135C133.619 91.5 132.5 90.3807 132.5 89V25C132.5 23.6193 133.619 22.5 135 22.5Z"/>
          <path stroke-width="3" stroke="#282828" fill="#7D7C7C" d="M146 33.5H181.741C182.779 33.5 183.709 34.1415 184.078 35.112L190.538 52.112C191.16 53.748 189.951 55.5 188.201 55.5H146C144.619 55.5 143.5 54.3807 143.5 53V36C143.5 34.6193 144.619 33.5 146 33.5Z"/>
          <path stroke-width="2" stroke="#282828" fill="#282828" d="M150 65C150 65.39 149.763 65.8656 149.127 66.2893C148.499 66.7083 147.573 67 146.5 67C145.427 67 144.501 66.7083 143.873 66.2893C143.237 65.8656 143 65.39 143 65C143 64.61 143.237 64.1344 143.873 63.7107C144.501 63.2917 145.427 63 146.5 63C147.573 63 148.499 63.2917 149.127 63.7107C149.763 64.1344 150 64.61 150 65Z"/>
          <rect stroke-width="2" stroke="#282828" fill="#FFFCAB" rx="1" height="7" width="5" y="63" x="187"/>
          <rect stroke-width="2" stroke="#282828" fill="#282828" rx="1" height="11" width="4" y="81" x="193"/>
          <rect stroke-width="3" stroke="#282828" fill="#DFDFDF" rx="2.5" height="90" width="121" y="1.5" x="6.5"/>
          <rect stroke-width="2" stroke="#282828" fill="#282828" rx="2" height="4" width="6" y="84" x="1"/>
        </svg>
      </div>
      <div class="qr-truckTires">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 30 30" class="tiresvg">
          <circle stroke-width="3" stroke="#282828" fill="#282828" r="13.5" cy="15" cx="15"></circle>
          <circle fill="#DFDFDF" r="7" cy="15" cx="15"></circle>
        </svg>
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 30 30" class="tiresvg">
          <circle stroke-width="3" stroke="#282828" fill="#282828" r="13.5" cy="15" cx="15"></circle>
          <circle fill="#DFDFDF" r="7" cy="15" cx="15"></circle>
        </svg>
      </div>
      <div class="qr-road"></div>
      <svg viewBox="0 0 453.459 453.459" class="qr-lampPost" xmlns="http://www.w3.org/2000/svg">
        <path d="M252.882,0c-37.781,0-68.686,29.953-70.245,67.358h-6.917v8.954c-26.109,2.163-45.463,10.011-45.463,19.366h9.993 c-1.65,5.146-2.507,10.54-2.507,16.017c0,28.956,23.558,52.514,52.514,52.514c28.956,0,52.514-23.558,52.514-52.514 c0-5.478-0.856-10.872-2.506-16.017h9.992c0-9.354-19.352-17.204-45.463-19.366v-8.954h-6.149C200.189,38.779,223.924,16,252.882,16 c29.952,0,54.32,24.368,54.32,54.32c0,28.774-11.078,37.009-25.105,47.437c-17.444,12.968-37.216,27.667-37.216,78.884v113.914 h-0.797c-5.068,0-9.174,4.108-9.174,9.177c0,2.844,1.293,5.383,3.321,7.066c-3.432,27.933-26.851,95.744-8.226,115.459v11.202h45.75 v-11.202c18.625-19.715-4.794-87.527-8.227-115.459c2.029-1.683,3.322-4.223,3.322-7.066c0-5.068-4.107-9.177-9.176-9.177h-0.795 V196.641c0-43.174,14.942-54.283,30.762-66.043c14.793-10.997,31.559-23.461,31.559-60.277C323.202,31.545,291.656,0,252.882,0z"/>
      </svg>
    </div>
  </div>
</div>
"""

def inject_truck_loader_css_once():
    if not st.session_state.get("_truck_css_injected", False):
        st.markdown(_TRUCK_LOADER_CSS, unsafe_allow_html=True)
        st.session_state["_truck_css_injected"] = True

def show_truck_loader(placeholder: st.delta_generator.DeltaGenerator):
    inject_truck_loader_css_once()
    placeholder.markdown(_TRUCK_LOADER_HTML, unsafe_allow_html=True)

def hide_truck_loader(placeholder: st.delta_generator.DeltaGenerator):
    placeholder.empty()

# Show loader on first boot then rerun without it
if "boot_done" not in st.session_state:
    _ph_boot = st.empty()
    show_truck_loader(_ph_boot)
    # Preload/cold start work can go here
    time.sleep(1.2)
    hide_truck_loader(_ph_boot)
    st.session_state["boot_done"] = True
    st.rerun()

# --- NAVBAR COMPONENT ---
def navbar(active: str):
    # 1) Build the row first (brand | spacer | 5 links)
    c_brand, c_spacer, c1, c2, c3, c4, c5 = st.columns([1.4, 5.8, 1.25, 1.9, 1.6, 1.25, 1.6])

    with c_brand:
        st.markdown(
            '<div class="qr-brand"><span class="qr-cube"></span><span>QuickRetain</span></div>',
            unsafe_allow_html=True
        )
    with c1: st.page_link("app.py", label="Features")
    with c2: st.page_link("pages/01_Churn_SHAP.py", label="Churn + SHAP")
    with c3: st.page_link("pages/02_Retention_RL.py", label="Retention RL")
    with c4: st.page_link("pages/03_Logistics.py", label="Logistics")
    with c5: st.page_link("pages/04_Campaigns.py", label="🎯 Campaigns")

    # 2) Style the row that contains .qr-brand AS the sticky navbar
    st.markdown(f"""
    <style>
      /* make the columns row that contains .qr-brand the sticky glass navbar */
      div[data-testid="stHorizontalBlock"]:has(.qr-brand) {{
        position: sticky; top: 0; z-index: 9999;
        background: rgba(255,255,255,.92);
        backdrop-filter: saturate(170%) blur(12px);
        border-bottom: 1px solid rgba(20,16,50,.06);
        box-shadow: 0 8px 24px rgba(20,16,50,.06);
        padding: 14px 18px;
        margin-top: -8px; /* tight to top */
      }}
      /* tighten inner cols so the bar looks flush */
      div[data-testid="stHorizontalBlock"]:has(.qr-brand) > div {{
        padding-top: 0 !important;
      }}

      /* brand */
      .qr-brand {{ display:flex; align-items:center; gap:10px;
                  font-weight:800; font-size:1.15rem; color:#121826; }}
      .qr-cube {{ width:18px; height:18px; border-radius:4px;
                 background: linear-gradient(135deg,#7C4DFF,#6C3BE2);
                 box-shadow: 0 2px 8px rgba(98,56,226,.35); display:inline-block; }}

      /* pill styling for ALL page links */
      a[data-testid="stPageLink"] {{
        display:inline-flex; align-items:center; gap:.4rem;
        padding:10px 16px !important;
        border-radius:999px !important;
        text-decoration:none !important;
        border:1px solid rgba(20,16,50,.12) !important;
        background: rgba(255,255,255,.78) !important;
        box-shadow: 0 4px 12px rgba(20,16,50,.06) !important;
        color:#121826 !important; font-weight:600 !important;
        white-space: nowrap !important;
        transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease;
      }}
      a[data-testid="stPageLink"]:hover {{
        transform: translateY(-1px);
        border-color: rgba(102,52,226,.45) !important;
        box-shadow: 0 8px 18px rgba(20,16,50,.10) !important;
      }}

      /* active page glow */
      a[data-testid="stPageLink"][href$="{active}"] {{
        background: linear-gradient(180deg,#fff,#F5F4FF) !important;
        border-color: rgba(102,52,226,.65) !important;
        box-shadow: 0 10px 22px rgba(102,52,226,.18) !important;
      }}

      /* mobile wrap nicely */
      @media (max-width: 820px){{
        div[data-testid="stHorizontalBlock"]:has(.qr-brand) {{
          padding: 10px 12px;
        }}
      }}
    </style>
    """, unsafe_allow_html=True)

# Use the navbar on the home page
navbar(active="app.py")
# anchor at page top (for clicking the logo)
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# --- keep using the new query params API for router elsewhere in your app
params = st.query_params
selected_model = params.get("model", None)
if isinstance(selected_model, list):  # normalize if list
    selected_model = selected_model[0]

# --- Router param (reuse your st.query_params approach)
page = params.get("page", None)
if isinstance(page, list):
    page = page[0]

def show_login():
    st.markdown('<div id="login"></div>', unsafe_allow_html=True)
    st.markdown("### Welcome back")
    st.caption("Sign in to start QuickRetain AI")
    with st.form("login_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        email = col1.text_input("Email", placeholder="you@company.com")
        password = col2.text_input("Password", type="password", placeholder="••••••••")
        keep = st.checkbox("Keep me signed in", value=True)
        submitted = st.form_submit_button("Sign in")
    if submitted:
        # TODO: replace with your real auth
        if email and password:
            st.session_state["user"] = email
            st.success("Signed in ✔")
            # Redirect to models area after login
            st.markdown('<meta http-equiv="refresh" content="0; url=?model=churn#models">', unsafe_allow_html=True)
        else:
            st.error("Please enter email and password.")

# If /?page=login, draw login right away and stop the rest (so it looks like a page)
if page == "login":
    show_login()
    st.stop()

def render_churn_page():
    st.markdown("<h1>Churn + SHAP</h1>", unsafe_allow_html=True)
    st.caption("Explainable churn modeling with RandomForest/SMOTE and SHAP.")
    # TODO: render your churn inputs/plots here

def render_bandit_page():
    st.markdown("<h1>Retention Bandit</h1>", unsafe_allow_html=True)
    st.caption("Contextual bandit (ε-greedy) for incentive ROI.")
    # TODO: render your bandit demo/controls here

def render_logistics_page():
    st.markdown("<h1>Logistics Optimizer</h1>", unsafe_allow_html=True)
    st.caption("KMeans catchments + nearest-neighbor routing.")
    # TODO: render your logistics demo/maps here

# If a model page is requested, redirect to the proper page
if selected_model == "bandit":
    render_bandit_page(); st.stop()
elif selected_model == "logistics":
    render_logistics_page(); st.stop()


# THEME PRIMITIVES
ACCENT = "#6634E2"  # your purple


def inject_base_styles():
    import streamlit as st
    st.markdown(f"""
    <style>
      .glass {{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        box-shadow: 0 22px 60px rgba(15,23,42,0.06);
        padding: 18px 20px;
        transition: all 0.3s ease;
        transform: translateY(0);
      }}
      .glass:hover {{
        transform: translateY(-8px);
        box-shadow: 0 32px 80px rgba(15,23,42,0.12);
        border-color: rgba(102,52,226,0.2);
      }}
      .sec-title {{text-align:center; font-weight:900; letter-spacing:-0.5px;
        font-size:38px; line-height:1.08; margin: 8px 0 6px;}}
      .sec-title .accent {{color:{ACCENT};}}
      .pill {{width: 96%; margin:0 auto 12px auto; padding: 12px 18px;
        border-radius:999px; background:rgba(255,255,255,0.96);
        border:1px solid rgba(102,52,226,0.15);
        box-shadow:0 24px 60px rgba(102,52,226,0.08); text-align:center; font-weight:700;
        transition: all 0.3s ease; transform: translateY(0);}}
      .pill:hover {{transform: translateY(-2px); box-shadow:0 28px 70px rgba(102,52,226,0.12);}}
      .tag {{display:inline-block; font-size:12px; font-weight:700; color:#fff;
        background:{ACCENT}; padding:4px 10px; border-radius:999px}}
      .step-num {{width:32px; height:32px; border-radius:999px; display:inline-flex;
        align-items:center; justify-content:center; font-weight:800; color:#fff; background:{ACCENT};}}
      .muted {{color:#6b7280}}
      .feature-emoji {{font-size:22px; margin-right:8px}}
    </style>
    """, unsafe_allow_html=True)

def render_feature_grid():
    st.markdown('<div class="sec-title"><span class="accent">Features</span></div>', unsafe_allow_html=True)

    features = [
        ("🧠", "Explainable AI", "SHAP and clear diagnostics out-of-the-box."),
        ("🎯", "Smart Incentives", "Contextual bandits learn profitable offers."),
        ("🗺️", "Faster Routes", "K-Means zones + nearest-neighbor routing."),
        ("⚡", "Python Native", "Pandas, scikit-learn, PyTorch friendly."),
        ("🔒", "Data Safe", "Runs locally or in your VPC."),
        ("🚀", "Lightweight UI", "Web3-ish look; minimal, fast, elegant."),
        ("📊", "Real-time Analytics", "Live dashboards with instant insights."),
        ("🔄", "Auto ML Pipeline", "Automated model training and deployment."),
        ("🎨", "Custom Models", "Build and train your own ML models."),
        ("📈", "Performance Tracking", "Monitor model accuracy and drift."),
        ("🔧", "Easy Integration", "REST APIs and webhook support."),
        ("💡", "Smart Recommendations", "AI-powered business insights."),
    ]
    
    # Create 3 columns with proper spacing
    cols = st.columns(3, gap="large")
    
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="glass" style="margin:10px 0; text-align:left;">
              <div class="feature-emoji">{icon}</div>
              <strong>{title}</strong>
              <div class="muted" style="margin-top:6px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

def render_how_it_works():
    st.markdown('<div class="sec-title">How It <span class="accent">Works</span></div>', unsafe_allow_html=True)
    steps = [
        ("Ingest", "Upload CSV or connect your store/warehouse data."),
        ("Predict", "Train churn model (SMOTE optional) and score customers."),
        ("Explain", "See global & per-customer drivers via SHAP."),
        ("Decide", "Bandits test offers & timing; logistics groups orders."),
    ]
    
    # Create 4 columns with proper spacing
    cols = st.columns(4, gap="large")
    
    for i, (title, desc) in enumerate(steps, start=1):
        with cols[i-1]:
            st.markdown(f"""
            <div class="glass" style="text-align:center; margin:10px 0;">
              <span class="step-num">{i}</span>
              <div style="font-weight:800; margin-top:10px;">{title}</div>
              <div class="muted" style="margin-top:6px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

def render_use_cases():
    st.markdown('<div class="sec-title">Where It <span class="accent">Shines</span></div>', unsafe_allow_html=True)
    cases = [
        ("Retail & D2C", "Curb subscription churn, tailor coupons, boost LTV.", "↑ ROI 1.8×", "↓ Churn 25%"),
        ("Fintech", "Retain users with nudge timing & offers; explain decisions.", "↑ Active users 12%", "↑ AUC 0.94"),
        ("Quick-Commerce", "Cluster dark-store zones; cut route kilometers fast.", "↓ Route km 20–30%", "↑ On-time 10%"),
    ]
    cols = st.columns(3, gap="large")
    for i, (title, desc, m1, m2) in enumerate(cases):
        with cols[i]:
            st.markdown(f"""
            <div class="glass" style="margin:10px 0;">
              <span class="tag">{title}</span>
              <div style="font-weight:800; margin-top:8px;">Outcome</div>
              <div class="muted" style="margin-top:4px;">{desc}</div>
              <div style="display:flex; gap:10px; margin-top:12px;">
                <div class="pill" style="width:auto; padding:6px 12px;">{m1}</div>
                <div class="pill" style="width:auto; padding:6px 12px;">{m2}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

def render_faq():
    st.markdown('<div class="sec-title">Common <span class="accent">Questions</span></div>', unsafe_allow_html=True)
    faqs = [
        ("Do I need a GPU?", "No. The demo runs CPU-only. Your models can use GPU if available."),
        ("Where does data live?", "You can run locally or inside your own cloud/VPC."),
        ("Can I bring my model?", "Yes—swap the scorer to any sklearn-style or PyTorch model."),
        ("Is it explainable?", "Yes—SHAP global & per-customer views are built in."),
        ("How do incentives learn?", "We use contextual bandits (ε-greedy by default) to optimize ROI."),
    ]
    for q,a in faqs:
        with st.expander(q, expanded=False):
            st.write(a)

def render_centered_cta():
    st.markdown("""
    <style>
      /* CTA container */
      .qr-cta{
        max-width: 1100px;              /* keeps it aligned with the rest of the page */
        margin: 48px auto 120px auto;   /* centered */
        padding: 36px 24px;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(102,52,226,0.00), rgba(102,52,226,0.05));
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 30px 80px rgba(0,0,0,0.08), 0 12px 24px rgba(0,0,0,0.04);
      }

      /* Headline + sub */
      .qr-cta h2{
        margin: 0 0 8px 0;
        text-align: center;
        font-size: clamp(28px, 4vw, 40px);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: .2px;
      }
      .qr-cta h2 .accent{ color:#6634E2; }

      .qr-cta p{
        margin: 0 0 22px 0;
        text-align: center;
        color: #6B7280;                 /* neutral-500 */
        font-size: clamp(14px, 1.8vw, 18px);
      }

      /* Button wrapper = perfectly centered */
      .qr-cta .btn-wrap{
        display: flex;
        justify-content: center;
        align-items: center;
      }

      /* CTA button */
      .qr-cta a.btn{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        text-decoration: none;
        background: #111827;           /* deep slate */
        color: #fff;
        padding: 16px 22px;
        border-radius: 14px;
        box-shadow: 0 15px 40px rgba(0,0,0,.18);
        font-weight: 700;
        letter-spacing: .2px;
        width: min(420px, calc(100% - 32px));  /* not full width on desktop */
        justify-content: center;
      }
      .qr-cta a.btn:hover{
        transform: translateY(-1px);
        transition: transform .15s ease;
      }

      @media (max-width: 640px){
        .qr-cta{ margin: 28px auto 80px auto; padding: 26px 16px; }
      }
    </style>

    <div class="qr-cta">
      <h2>Ready to <span class="accent">retain smarter</span>?</h2>
      <p>Start with the sample dataset or connect your own.</p>
      <div class="btn-wrap">
        <a class="btn" href="#get-started">🚀 Get Started</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- THEME HEADING STYLES (put once) ---
st.markdown("""
<style>
.section-head{display:flex;align-items:center;gap:12px;margin:28px 0 12px;}
.sec-icon{
  width:36px;height:36px;border-radius:12px;display:grid;place-items:center;font-size:18px;color:#fff;
  background: radial-gradient(120% 120% at 10% 10%, #A78BFA 0%, #7C3AED 45%, #5B21B6 100%);
  box-shadow: 0 8px 18px rgba(124,58,237,.25), inset 0 1px 0 rgba(255,255,255,.15);
}
.sec-title{
  font-size:26px;font-weight:800;letter-spacing:-.01em;color:#0F172A;
}
.sec-subtitle{font-size:14px;color:#475467;margin-top:6px}
@media (max-width: 768px){ .sec-title{font-size:22px} }
</style>
""", unsafe_allow_html=True)

def section_header(title:str, subtitle:str="", icon:str="📊"):
    st.markdown(
        f"""
        <div class="section-head">
            <div class="sec-icon">{icon}</div>
            <div>
                <div class="sec-title">{title}</div>
                {f'<div class="sec-subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- GLOBAL CSS ----------
CSS = """
<style>
:root{
  --ink:#0f1222;
  --muted:#6b7280;
  --accent:#6634E2;
  --card:#ffffff;
  --border:rgba(255,255,255,0.45);
  --brand:#6634E2;
  --brand-2:#6E5AE6;
  --surface:#FFFFFF;
  --glass:rgba(255,255,255,.65);
  --glass-border:rgba(15,23,42,.08);
}

/* Page reset */
html, body, [data-testid="stAppViewContainer"]{
  background: white;
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Inter, Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
}
.main .block-container{ 
  padding-top: 0 !important; 
  background: white;
}

/* Clean white background */

/* ---- NAVBAR ---- */
.nav-wrap{
  position: sticky; top: 0; z-index: 999;
  padding: 16px 0 6px;
  backdrop-filter: blur(6px);
}
.nav{
  display:flex; align-items:center; justify-content:space-between;
  background: var(--glass);
  border:1px solid var(--glass-border);
  border-radius: 14px;
  padding: 14px clamp(18px, 2vw, 28px);
  box-shadow: 0 6px 26px rgba(3,7,18,.06);
}
.brand{
  display:flex; align-items:center; gap:10px; text-decoration:none;
  color: var(--ink);
}
.brand .logo{
  width:28px; height:28px; display:inline-block;
}
.brand .word{
  font-weight: 700; letter-spacing:.1px; font-size: 18px;
}

/* nav links */
.nav-links{
  display:flex; align-items:center; gap:28px;
}
.nav-links a{
  color: var(--ink); text-decoration:none; font-weight:600; opacity:.78;
}
.nav-links a:hover{ opacity:1 }
.nav-links a, .nav-cta{ text-decoration: none !important; }

/* CTA */
.nav-cta{
  background: #0B1220;
  color:#fff !important; text-decoration:none !important; font-weight:700;
  padding:10px 16px; border-radius: 12px; display:inline-flex; align-items:center; gap:8px;
  box-shadow: 0 8px 24px rgba(2,6,23,.18);
}
.nav-cta:hover{ transform: translateY(-1px); transition: .2s ease; }

/* small spacer under nav */
.spacer{ height: 8px; }

/* ---- HERO (Clean white Lumino style) ---- */
.hero {
    text-align: center;
    padding: 120px 20px 100px 20px;
}

.hero h1 {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 20px;
    color: #111;
}

.highlight {
    background: linear-gradient(90deg, #6634E2, #9a6bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}

.hero p {
    font-size: 1.2rem;
    color: #444;
    margin-bottom: 40px;
    max-width: 750px;
    margin-left: auto;
    margin-right: auto;
}

/* Clean hero styles */

/* CTA Buttons */
.cta-buttons {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 30px;
}

.btn {
    padding: 14px 32px;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.3s ease-in-out;
}

/* Primary Button */
.btn-primary {
    background: linear-gradient(90deg, #6634E2, #9a6bff);
    color: white !important;
    text-decoration: none !important;
    box-shadow: 0 8px 25px rgba(102, 52, 226, 0.4);
}
.btn-primary:hover { 
    transform: translateY(-3px);
    color: white !important;
    text-decoration: none !important;
}

/* Secondary Button */
.btn-secondary {
    background: #111111;
    color: white !important;
    text-decoration: none !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
}
.btn-secondary:hover { 
    transform: translateY(-3px);
    color: white !important;
    text-decoration: none !important;
}

/* Why QuickRetain AI Section */
.why-section {
    text-align: center;
    padding: 60px 20px 40px 20px;
}
.why-section h2 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #111;
    margin-bottom: 15px;
}
.why-section .highlight {
    background: linear-gradient(90deg, #6634E2, #9a6bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.why-section p {
  font-size: 1.1rem;
    color: #444;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

/* Chart Section Title */
.chart-section-title {
    text-align: center;
    margin: 40px 0 20px 0;
}

.chart-section-title h2 {
    font-size: 2rem;
    font-weight: 800;
    color: #111;
    margin-bottom: 8px;
}

.chart-section-title p {
    font-size: 1rem;
    color: #666;
    margin: 0;
}

/* Chart Wrapper */
.chart-wrapper {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}

/* Chart Card (matches KPI card styling) */
.chart-card {
    background: white;
    border: 1px solid rgba(15,18,34,0.07);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    border-radius: 18px;
    padding: 20px;
    max-width: 800px;
    width: 100%;
    transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
}

.chart-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
}

/* General section styling */
.section {
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}

/* anchors for smooth scroll spacing */
.section-anchor{ position: relative; top: -84px; visibility: hidden; }

/* ========== GLASS HERO (OPTIONAL) ========== */
.glass-hero{
  background:rgba(255,255,255,0.35);
  border:1px solid rgba(255,255,255,0.55);
  box-shadow:0 20px 50px rgba(15,18,34,0.10), 0 2px 8px rgba(15,18,34,0.06);
  backdrop-filter:blur(18px) saturate(140%);
  -webkit-backdrop-filter:blur(18px) saturate(140%);
  border-radius:22px;
}

/* ========== GLASS KPI CARDS (upgrade) ========== */
.kpi-wrap{
  display:grid;
  grid-template-columns:repeat(4,minmax(220px,1fr));
  gap:18px;
  margin:26px 0 10px;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}

.kpi{
  /* true glass */
  background:rgba(255,255,255,0.28);
  border:1px solid var(--border);
  box-shadow:0 12px 36px rgba(15,18,34,0.10), 0 2px 8px rgba(15,18,34,0.06);
  backdrop-filter:blur(16px) saturate(150%);
  -webkit-backdrop-filter:blur(16px) saturate(150%);
  border-radius:18px;
  padding:18px 20px;
  transition: all 0.3s ease;
  transform: translateY(0);
  cursor: pointer;
}
.kpi:hover{
  transform: translateY(-6px);
  box-shadow:0 20px 50px rgba(15,18,34,0.15), 0 8px 16px rgba(15,18,34,0.08);
  background:rgba(255,255,255,0.35);
  border-color: rgba(102,52,226,0.3);
}

.kpi-top{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:8px;
}

.kpi h4{
  margin:0;
  font-size:.95rem;
  color:#384152;
  letter-spacing:.2px;
  font-weight:600;
}

.kpi-icon{
  color:var(--accent);
  background:linear-gradient(145deg, rgba(255,255,255,0.75), rgba(255,255,255,0.55));
  border:1px solid rgba(255,255,255,0.65);
  width:34px;height:34px;
  border-radius:10px;
  display:grid;place-items:center;
  box-shadow:0 6px 20px rgba(102,52,226,0.18);
}

.kpi-value{
  font-size:1.75rem;
  font-weight:800;
  letter-spacing:.3px;
  color:var(--ink);
  margin:6px 0 6px;
}

.delta{
  font-size:.9rem;
  font-weight:600;
  display:flex;
  align-items:center;
  gap:6px;
}
.delta.up{ color:#16a34a; }   /* green gain */
.delta.down{ color:#ef4444; } /* red drop */

.kpi:hover{
  transform:translateY(-2px);
  background:rgba(255,255,255,0.34);
  box-shadow:0 16px 42px rgba(15,18,34,0.14), 0 3px 10px rgba(15,18,34,0.07);
}

/* Responsive */
@media (max-width:1100px){
  .kpi-wrap{ grid-template-columns:repeat(2,minmax(220px,1fr)); }
}
@media (max-width:600px){
  .kpi-wrap{ grid-template-columns:1fr; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- SKY BACKGROUND WRAPPER ----------
# st.markdown('<div class="sky-bg">', unsafe_allow_html=True)

# Hero section
st.markdown("""
<section class="hero">
  <h1>
    Retain Smarter. <span>Deliver Faster with </span>
    <span class="highlight">QuickRetain AI</span>
  </h1>
  <p>
    Predict churn with <b>explainable AI</b>, optimize incentives with <b>reinforcement learning</b>, 
    and streamline delivery routes — all in one intelligent platform.
  </p>
  <div class="cta-buttons">
    <a href="?page=login#login" class="btn btn-primary">🚀 Get Started</a>
    <a href="#how-it-works" class="btn btn-secondary">ℹ️ Learn More</a>
  </div>
</section>
""", unsafe_allow_html=True)

# ---------- WHY QUICKRETAIN AI SECTION ----------
WHY_SECTION_HTML = """
<section class="why-section">
  <h2>Why <span class="highlight">QuickRetain AI?</span></h2>
  <p>
    One platform, three wins: churn prevention, incentive optimization, and logistics efficiency.  
    Explainable by design. Python-native. Light on the browser. Heavy on results. 🚀
  </p>
        </section>
"""
st.markdown(WHY_SECTION_HTML, unsafe_allow_html=True)

# ---------- HIGHLIGHTS SECTION ----------
HIGHLIGHTS_HTML = """
<section id="highlights" class="kpi-wrap">
  <!-- Card 1 -->
  <article class="kpi">
    <div class="kpi-top">
      <h4>Customers Retained (30d)</h4>
      <span class="kpi-icon users" aria-hidden="true">
        <!-- users icon -->
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M16 14c2.761 0 5 2.239 5 5v1H3v-1c0-2.761 2.239-5 5-5h8Z" stroke="currentColor" stroke-width="1.5"/>
          <circle cx="12" cy="7" r="4" stroke="currentColor" stroke-width="1.5"/>
        </svg>
      </span>
    </div>
    <div class="kpi-value">12,408</div>
    <div class="delta up">▲ 8.5% vs last month</div>
  </article>

  <!-- Card 2 -->
  <article class="kpi">
    <div class="kpi-top">
      <h4>Churn Model AUC</h4>
      <span class="kpi-icon auc" aria-hidden="true">
        <!-- shield/metric icon -->
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M12 3l7 3v6c0 4.418-3.134 8.418-7 9-3.866-.582-7-4.582-7-9V6l7-3Z" stroke="currentColor" stroke-width="1.5"/>
          <path d="M8 12l2.5 2.5L16 9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </span>
    </div>
    <div class="kpi-value">0.94</div>
    <div class="delta up">▲ +0.02 vs last run</div>
  </article>

  <!-- Card 3 -->
  <article class="kpi">
    <div class="kpi-top">
      <h4>Offer ROI (Bandit)</h4>
      <span class="kpi-icon roi" aria-hidden="true">
        <!-- coin icon -->
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.5"/>
          <path d="M8.5 12h7M12 8.5v7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </span>
    </div>
    <div class="kpi-value">1.8×</div>
    <div class="delta up">▲ +0.3× vs control</div>
  </article>

  <!-- Card 4 -->
  <article class="kpi">
    <div class="kpi-top">
      <h4>Route km Saved</h4>
      <span class="kpi-icon route" aria-hidden="true">
        <!-- route/refresh icon -->
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M20 11a8 8 0 10.002 2H17" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M17 7v4h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </span>
    </div>
    <div class="kpi-value">72%</div>
    <div class="delta up">▲ +5.0% vs naïve routing</div>
  </article>
</section>
"""
st.markdown(HIGHLIGHTS_HTML, unsafe_allow_html=True)

# --- Churn Model Performance: ROC + PR example section ------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import streamlit as st

def render_model_performance_examples():
    # ---------- Styling (heading, pills, glass cards) ----------
    st.markdown("""
<style>
      .sec-title {text-align:center; font-weight:900; letter-spacing:-0.5px;
                  font-size:44px; line-height:1.08; margin: 60px 0 6px;}
      .sec-title .accent {color:#6634E2;} /* theme purple */
      .sec-caption {text-align:center; color:#6b7280; margin:4px 0 18px; font-size:16px;}

      .pill {
        width: 96%;
        margin: 0 auto 12px auto;
        padding: 14px 22px;
        border-radius: 999px;
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(102,52,226,0.15);
        box-shadow: 0 24px 60px rgba(102,52,226,0.08);
        text-align: center;
        font-weight: 700;
        color: #0f172a;
      }
      .glass-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        padding: 14px;
        box-shadow: 0 22px 60px rgba(15,23,42,0.06);
        transition: all 0.3s ease;
        transform: translateY(0);
      }
      .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 28px 70px rgba(15,23,42,0.12);
        border-color: rgba(102,52,226,0.2);
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Churn Model <span class="accent">Performance</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-caption">Examples of ROC and Precision-Recall curves.</div>', unsafe_allow_html=True)

    # ---------- Make deterministic synthetic data (not real-time) ----------
    rng = np.random.default_rng(7)      # fixed seed -> reproducible
    n = 1800
    y_true = np.r_[np.ones(n//2), np.zeros(n//2)]  # balanced 0/1 labels

    # calibrated scores: positives pushed higher; add noise
    raw = 2.2*y_true + 0.9*rng.normal(size=n)
    # squashing to (0,1) to look like probabilities
    y_scores = 1/(1+np.exp(-raw))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    # Precision–Recall
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ap = float(average_precision_score(y_true, y_scores))

    # ---------- Layout ----------
    col1, col2 = st.columns(2, gap="large")

    # === Left: ROC ===
    with col1:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            # Matplotlib figure (compact; fits container)
            fig1, ax1 = plt.subplots(figsize=(5.6, 4.1), dpi=175)
            # theme colors
            purple = "#6634E2"
            grid = "#E5E7EB"
            diag  = "#9CA3AF"

            ax1.plot(fpr, tpr, color=purple, lw=2.5, label=f"AUC = {roc_auc:0.3f}")
            ax1.fill_between(fpr, tpr, 0, color=purple, alpha=0.08)
            ax1.plot([0, 1], [0, 1], linestyle="--", color=diag, lw=1.5)

            ax1.set_xlim([-0.005, 1.005])
            ax1.set_ylim([-0.005, 1.005])
            ax1.set_xlabel("False Positive Rate", fontsize=10)
            ax1.set_ylabel("True Positive Rate", fontsize=10)
            ax1.set_title("ROC Curve", fontsize=12, fontweight="bold", pad=10)
            ax1.grid(color=grid, linestyle="-", linewidth=0.7, alpha=0.6)
            ax1.legend(loc="lower right", frameon=False)
            
            fig1.tight_layout(pad=1.2)

            st.pyplot(fig1, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

    # === Right: Precision–Recall ===
    with col2:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(5.6, 4.1), dpi=175)
            purple = "#6634E2"; grid = "#E5E7EB"
            ax2.plot(rec, prec, color=purple, lw=2.5, label=f"AP = {ap:0.3f}")
            ax2.fill_between(rec, prec, 0, color=purple, alpha=0.08)

            ax2.set_xlim([-0.005, 1.005])
            ax2.set_ylim([-0.005, 1.005])
            ax2.set_xlabel("Recall", fontsize=10)
            ax2.set_ylabel("Precision", fontsize=10)
            ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold", pad=10)
            ax2.grid(color=grid, linestyle="-", linewidth=0.7, alpha=0.6)
            ax2.legend(loc="lower left", frameon=False)
            
            fig2.tight_layout(pad=1.2)

            st.pyplot(fig2, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

# --- call it where you want the section to appear ---
st.markdown('<div id="models"></div>', unsafe_allow_html=True)
render_model_performance_examples()

# Inject base styles
inject_base_styles()

# --------- HOME CONTENT ----------
st.markdown('<div id="features"></div>', unsafe_allow_html=True)
render_feature_grid()

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
render_how_it_works()

# Add more spacing
st.markdown("<br><br>", unsafe_allow_html=True)

render_use_cases()

# Add more spacing
st.markdown("<br><br>", unsafe_allow_html=True)

render_faq()

# Add more spacing
st.markdown("<br><br>", unsafe_allow_html=True)

render_centered_cta()

# kill the CTA box (keep centered text + button)
st.markdown("""
<style>
  .qr-cta{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;        /* tighten space if you like */
    margin: 32px auto 96px auto;  /* keep it centered nicely */
    max-width: 1100px;            /* align with the rest of the layout */
  }
</style>
""", unsafe_allow_html=True)

# ---------- ANCHORS / DUMMY SECTIONS (so links don't 404) ----------
st.markdown('<div id="get-started" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div id="features" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div id="pricing" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div id="changelog" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div id="contact" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div id="docs" class="section-anchor"></div>', unsafe_allow_html=True)

# Close sky background wrapper
# st.markdown('</div>', unsafe_allow_html=True)

# Show lightweight placeholder content (you can replace these with your modules)
st.write("")


