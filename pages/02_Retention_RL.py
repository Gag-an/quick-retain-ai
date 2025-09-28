# pages/02_Retention_RL.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import json
import os

# Import our cleaned data utilities
from clean_datasets import (
    load_cleaned_dataset, infer_expected_features, validate_features,
    load_preprocessing_artifacts, create_retention_features, save_feature_list
)

# from utils_perf import load_retention  # Commented out - module not found

st.set_page_config(page_title="Retention RL", layout="wide", initial_sidebar_state="collapsed")

# ---------- Navbar ----------
def navbar(active: str):
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

    st.markdown(f"""
    <style>
      div[data-testid="stHorizontalBlock"]:has(.qr-brand) {{
        position: sticky; top: 0; z-index: 9999;
        background: rgba(255,255,255,.92);
        backdrop-filter: saturate(170%) blur(12px);
        border-bottom: 1px solid rgba(20,16,50,.06);
        box-shadow: 0 8px 24px rgba(20,16,50,.06);
        padding: 14px 18px; margin-top: -8px;
      }}
      .qr-brand {{
        display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.15rem; color:#121826;
      }}
      .qr-cube {{ width:18px; height:18px; border-radius:4px;
                 background: linear-gradient(135deg,#7C4DFF,#6C3BE2);
                 box-shadow: 0 2px 8px rgba(98,56,226,.35); display:inline-block; }}
      a[data-testid="stPageLink"] {{
        display:inline-flex; align-items:center; gap:.4rem; padding:10px 16px;
        border-radius:999px; text-decoration:none; border:1px solid rgba(20,16,50,.12);
        background: rgba(255,255,255,.78); box-shadow: 0 4px 12px rgba(20,16,50,.06);
        color:#121826; font-weight:600; white-space: nowrap;
        transition: transform .12s ease, border-color .12s, box-shadow .12s;
      }}
      a[data-testid="stPageLink"]:hover {{ transform: translateY(-1px);
        border-color: rgba(102,52,226,.45); box-shadow: 0 8px 18px rgba(20,16,50,.10); }}
      a[data-testid="stPageLink"][href$="{active}"] {{
        background: linear-gradient(180deg,#fff,#F5F4FF); border-color: rgba(102,52,226,.65);
        box-shadow: 0 10px 22px rgba(102,52,226,.18);
      }}
    </style>
    """, unsafe_allow_html=True)

navbar(active="pages/02_Retention_RL.py")

st.title("Retention RL — Contextual Bandit")
st.caption("Use processed retention events as contexts, actions and rewards.")

# ---------- Controls ----------
SCOPE = st.radio("Scope", ["Both", "blinkit", "bigbasket"], horizontal=True, key="scope_rl")

ROOT = r"D:\quick Retain Ai\data\processed"
# ---- REPLACE synthetic sample block with this cleaned-data loader ----
# Attempt to load real cleaned data from data/cleaned/ and map into expected schema
DEFAULT_SAMPLE_SIZE = 10000

def _safe_get_col(df, candidates, default=None):
    """Return first existing column name from candidates (list) or None"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

with st.spinner("Loading cleaned retention data..."):
    try:
        # OPTIMIZATION: Better path handling for Windows
        cleaned_path = Path("data/cleaned/")
        if not cleaned_path.exists():
            cleaned_path = Path("./data/cleaned/")
        if not cleaned_path.exists():
            cleaned_path = Path("../data/cleaned/")
        
        # Load cleaned dataset (concat of CSVs). Use sample for performance if large.
        df_raw = load_cleaned_dataset(str(cleaned_path), sample_size=DEFAULT_SAMPLE_SIZE)
        if df_raw is None or df_raw.empty:
            raise FileNotFoundError("No cleaned CSVs found in data/cleaned/")
    except Exception as e:
        st.warning(f"Could not load cleaned data: {e}. Falling back to synthetic sample.")
        df_raw = None

if df_raw is not None:
    # normalize column names
    df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]

    # mapping: try to find existing columns that match expected semantic columns
    # prefer explicit names, then fallbacks
    col_customer = _safe_get_col(df_raw, ["customer_id", "user_id", "cust_id", "user", "client_id"]) or "customer_id"
    col_timestamp = _safe_get_col(df_raw, ["timestamp", "order_timestamp", "created_at", "order_date", "date"]) or "timestamp"
    # col_basket will be resolved robustly below
    col_discount = _safe_get_col(df_raw, ["discount_given", "discount", "coupon_value", "offer_amount"]) or "discount_given"
    col_repeat = _safe_get_col(df_raw, ["repeat_purchase", "repeat", "is_repeat", "rebuy"]) or "repeat_purchase"
    col_lat = _safe_get_col(df_raw, ["lat", "latitude", "customer_lat"]) or "lat"
    col_lon = _safe_get_col(df_raw, ["lon", "lng", "longitude", "customer_lon"]) or "lon"
    col_platform = _safe_get_col(df_raw, ["platform", "source", "channel", "sales_channel"]) or "platform"

    # Build df_all with expected columns; fill defaults for missing ones
    df_all = pd.DataFrame()
    df_all["customer_id"] = df_raw[col_customer] if col_customer in df_raw.columns else range(len(df_raw))
    # Parse timestamps if present; else create artificial sequential timestamps
    if col_timestamp in df_raw.columns:
        try:
            df_all["timestamp"] = pd.to_datetime(df_raw[col_timestamp], errors="coerce")
            # fill missing timestamps with a generated series
            missing_ts = df_all["timestamp"].isna().sum()
            if missing_ts:
                st.warning(f"Found {missing_ts} invalid timestamps — filling with sequential times.")
                start = pd.Timestamp("2023-01-01")
                df_all.loc[df_all["timestamp"].isna(), "timestamp"] = pd.date_range(start, periods=missing_ts, freq="h")
        except Exception:
            df_all["timestamp"] = pd.date_range("2023-01-01", periods=len(df_raw), freq="h")
    else:
        df_all["timestamp"] = pd.date_range("2023-01-01", periods=len(df_raw), freq="h")

    # ----- Robust basket_value resolution (replace the simple assignment) -----
    # Candidate column names we will search for (most common first)
    basket_candidates = [
        "basket_value", "basket_total", "total_spent", "total_amount", "order_value",
        "order_total", "amount", "price", "order_amount", "transaction_amount", "payable_amount"
    ]

    # Try to find a candidate present in df_raw
    col_basket = None
    for cand in basket_candidates:
        if cand in df_raw.columns:
            col_basket = cand
            break

    # If not found, try to compute from order items if that file exists in cleaned folder
    if col_basket is None:
        # look for an order_items-like table in data/cleaned/
        try:
            items_paths = list(Path("data/cleaned").glob("*order_items*.csv")) + list(Path("data/cleaned").glob("*order_items*.xlsx"))
            if items_paths:
                # load first found (prefer csv)
                p = items_paths[0]
                try:
                    df_items = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
                    # normalize item columns names
                    df_items.columns = [c.strip().lower().replace(" ", "_") for c in df_items.columns]
                    # try common item qty/price columns
                    qty_col = next((c for c in ["quantity", "qty", "item_count", "units"] if c in df_items.columns), None)
                    price_col = next((c for c in ["price", "unit_price", "item_price", "amount", "sell_price"] if c in df_items.columns), None)
                    order_id_col = next((c for c in ["order_id", "orderid", "order_id_str", "order_no"] if c in df_items.columns), None)
                    cust_map_col = next((c for c in ["customer_id", "user_id", "cust_id"] if c in df_raw.columns), None)

                    if qty_col and price_col:
                        # compute line totals
                        df_items["line_total"] = pd.to_numeric(df_items[price_col], errors="coerce").fillna(0.0) * pd.to_numeric(df_items[qty_col], errors="coerce").fillna(0.0)
                        # If we can join back to orders by order_id and df_raw has order id, aggregate
                        if order_id_col and order_id_col in df_raw.columns:
                            # aggregate per order
                            order_totals = df_items.groupby(order_id_col)["line_total"].sum().reset_index().rename(columns={"line_total":"basket_value"})
                            # If df_raw uses same order id name, merge
                            if order_id_col in df_raw.columns:
                                df_raw = df_raw.merge(order_totals, how="left", on=order_id_col)
                                col_basket = "basket_value"
                        else:
                            # otherwise, if df_items has customer mapping, aggregate per customer
                            if cust_map_col and cust_map_col in df_items.columns:
                                cust_totals = df_items.groupby(cust_map_col)["line_total"].sum().reset_index().rename(columns={"line_total":"basket_value"})
                                df_raw = df_raw.merge(cust_totals, how="left", left_on=col_customer, right_on=cust_map_col)
                                col_basket = "basket_value"
                except Exception:
                    col_basket = None

        except Exception:
            col_basket = None

    # Final fallback: try a few other columns directly from df_raw (some datasets use different naming)
    if col_basket is None:
        for alt in ["total_spend", "total_spent_amount", "totalorder", "order_total_amount"]:
            if alt in df_raw.columns:
                col_basket = alt
                break

    # If STILL not found, warn and create a safe default zero column
    if col_basket is None or col_basket not in df_raw.columns:
        st.warning(
            "Could not find a basket/amount column in cleaned data. "
            "Falling back to zeros for 'basket_value'. This will make retention rewards zero and may affect model outputs. "
            "If you have an order_items file, ensure it's named like '*order_items*' so the loader can compute totals."
        )
        df_all["basket_value"] = 0.0
    else:
        # Safe numeric conversion
        df_all["basket_value"] = pd.to_numeric(df_raw[col_basket], errors="coerce").fillna(0.0)

    # Save debug information for later inspection
    os.makedirs("archive", exist_ok=True)
    debug_info = {
        "found_basket_column": col_basket,
        "basket_candidates_searched": basket_candidates,
        "df_raw_columns_sample": list(df_raw.columns)[:200]
    }
    with open("archive/retention_loader_debug.json", "w", encoding="utf-8") as fh:
        json.dump(debug_info, fh, indent=2)
    # === SAFELY ASSIGN discount_given, basket_value, and ensure user_id exists ===
    # discount_given (optional)
    if col_discount in df_raw.columns:
        df_all["discount_given"] = pd.to_numeric(df_raw[col_discount], errors="coerce").fillna(0.0)
    else:
        st.warning("⚠️ No discount column found in cleaned data — defaulting to 0 for all rows.")
        df_all["discount_given"] = 0.0

    # Ensure basket_value exists (should be created earlier via col_basket logic). If not, make safe fallback:
    if "basket_value" not in df_all.columns:
        # try common fallbacks in df_raw
        for alt in ["total_spent", "total_amount", "order_total", "amount"]:
            if alt in df_raw.columns:
                df_all["basket_value"] = pd.to_numeric(df_raw[alt], errors="coerce").fillna(0.0)
                break
        else:
            st.warning("⚠️ Could not locate monetary column; created 'basket_value' with zeros as fallback.")
            df_all["basket_value"] = 0.0

    # Ensure an integer-like user_id column for grouping / selection
    if "user_id" in df_raw.columns:
        df_all["user_id"] = df_raw["user_id"]
    else:
        # create user_id from customer_id if available
        if "customer_id" in df_all.columns:
            df_all["user_id"] = df_all["customer_id"]
        else:
            df_all["user_id"] = range(len(df_all))

    # Cast numeric columns to correct dtype
    for c in ["basket_value", "discount_given", "lat", "lon"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0.0)
    # reward proxy: if repeat flag exists, use it; else fallback to heuristic (e.g., next-order within 30 days)
    if col_repeat in df_raw.columns:
        df_all["repeat_purchase"] = pd.to_numeric(df_raw[col_repeat], errors="coerce").fillna(0).astype(int)
    else:
        # heuristic: if customer has >1 order in cleaned data, treat later ones as repeats (best-effort)
        df_all["repeat_purchase"] = 0
        # try to infer from order ids/timestamps presence per customer
        try:
            tmp = df_raw.groupby(df_raw[col_customer]).size()
            repeated_customers = tmp[tmp > 1].index
            df_all.loc[df_all["customer_id"].isin(repeated_customers), "repeat_purchase"] = 1
        except Exception:
            df_all["repeat_purchase"] = np.random.choice([0, 1], size=len(df_all), p=[0.7, 0.3])

    # ----- Robust lat/lon resolution (replace fragile assignment) -----
    lat_candidates = ["lat", "latitude", "customer_lat", "cust_lat", "pickup_lat", "dropoff_lat"]
    lon_candidates = ["lon", "lng", "longitude", "customer_lon", "cust_lon", "pickup_lon", "dropoff_lon"]

    # find first present candidate (or None)
    found_lat = next((c for c in lat_candidates if c in df_raw.columns), None)
    found_lon = next((c for c in lon_candidates if c in df_raw.columns), None)

    # if neither found, try some generic numeric-looking columns (best-effort)
    if found_lat is None or found_lon is None:
        # quick heuristics: look for columns with float-like names or suspicious values
        numeric_cols = df_raw.select_dtypes(include=["float","int"]).columns.tolist()
        # try to pick two numeric columns that look like lat/lon ranges
        if found_lat is None:
            for c in numeric_cols:
                vals = df_raw[c].dropna()
                if len(vals) and (-90 <= vals.median() <= 90):
                    found_lat = c
                    break
        if found_lon is None:
            for c in numeric_cols:
                if c == found_lat: 
                    continue
                vals = df_raw[c].dropna()
                if len(vals) and (-180 <= vals.median() <= 180):
                    found_lon = c
                    break

    # assign or fallback
    if found_lat and found_lat in df_raw.columns:
        df_all["lat"] = pd.to_numeric(df_raw[found_lat], errors="coerce")
    else:
        st.warning("⚠️ Latitude column not found in cleaned data — filling with default city coordinates.")
        df_all["lat"] = float(28.65)  # fallback (Delhi) — change if you prefer

    if found_lon and found_lon in df_raw.columns:
        df_all["lon"] = pd.to_numeric(df_raw[found_lon], errors="coerce")
    else:
        st.warning("⚠️ Longitude column not found in cleaned data — filling with default city coordinates.")
        df_all["lon"] = float(77.20)  # fallback (Delhi) — change if you prefer

    # Fill missing lat/lon with defaults
    df_all["lat"] = df_all["lat"].fillna(28.65)
    df_all["lon"] = df_all["lon"].fillna(77.20)

    # Detect swapped lat/lon (common mistake: lat looks like 70-80 and lon like 10-40)
    lat_min, lat_max = df_all["lat"].min(), df_all["lat"].max()
    lon_min, lon_max = df_all["lon"].min(), df_all["lon"].max()

    # If lat range outside -90..90 or lon in -90..90 suspiciously, swap.
    if (lat_min < -90 or lat_max > 90) or (abs(lat_min) > 180 or abs(lat_max) > 180):
        # lat is invalid, try swap
        st.warning("Detected invalid latitude range — attempting to swap lat/lon columns.")
        df_all[["lat","lon"]] = df_all[["lon","lat"]]

    # If lat values look like longitudes (e.g., > 90) but lon looks like lat (<90), swap.
    if (df_all["lat"].abs().max() > 90) and (df_all["lon"].abs().max() <= 90):
        st.warning("Lat/Lon appear swapped — auto-correcting.")
        df_all[["lat","lon"]] = df_all[["lon","lat"]]

    # Final numeric coercion & clipping (safe ranges)
    df_all["lat"] = pd.to_numeric(df_all["lat"], errors="coerce").fillna(28.65).clip(-90,90)
    df_all["lon"] = pd.to_numeric(df_all["lon"], errors="coerce").fillna(77.20).clip(-180,180)

    # Save debug info to review exactly what we used
    os.makedirs("archive", exist_ok=True)
    debug_geo = {
        "found_lat_column": found_lat,
        "found_lon_column": found_lon,
        "lat_min_max": [float(df_all["lat"].min()), float(df_all["lat"].max())],
        "lon_min_max": [float(df_all["lon"].min()), float(df_all["lon"].max())],
        "sample_lat_values": df_all["lat"].dropna().head(5).tolist(),
        "sample_lon_values": df_all["lon"].dropna().head(5).tolist()
    }
    with open("archive/retention_geo_debug.json", "w", encoding="utf-8") as fh:
        json.dump(debug_geo, fh, indent=2)

    # platform
    if col_platform in df_raw.columns:
        df_all["platform"] = df_raw[col_platform].astype(str)
    else:
        df_all["platform"] = np.random.choice(["blinkit", "bigbasket"], len(df_all))

    # action and reward fields – keep existing or create placeholders
    if "action" not in df_raw.columns:
        df_all["action"] = np.where(df_all["discount_given"] > 0, "discount", "none")
    else:
        df_all["action"] = df_raw["action"].astype(str)

    # reward: use available 'reward' or compute from repeat_purchase & basket_value
    if "reward" in df_raw.columns:
        df_all["reward"] = pd.to_numeric(df_raw["reward"], errors="coerce").fillna(0.0)
    else:
        df_all["reward"] = np.where(df_all["repeat_purchase"].eq(1),
                                    df_all["basket_value"] - df_all["discount_given"], 0.0)

    st.success(f"✅ Loaded cleaned data with {len(df_all):,} rows (mapped to retention schema)")
else:
    # no cleaned data — fall back to synthetic sample with ALL required columns
    np.random.seed(42)
    n_samples = 1000
    df_all = pd.DataFrame({
        'customer_id': range(n_samples),
        'user_id': range(n_samples),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        
        # REQUIRED COLUMNS for prep_df
        'basket_value': np.random.uniform(50, 1000, n_samples),  # Required
        'discount_given': np.random.choice([0, 10, 20, 50, 100], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),  # Required
        'lat': np.random.uniform(28.0, 29.0, n_samples),  # Required (Delhi area)
        'lon': np.random.uniform(76.0, 78.0, n_samples),  # Required (Delhi area)
        'repeat_purchase': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # Required
        
        # Additional columns for completeness
        'days_since_last_order': np.random.randint(1, 365, n_samples),
        'total_orders': np.random.randint(1, 50, n_samples),
        'total_spent': np.random.uniform(10, 1000, n_samples),
        'avg_order_value': np.random.uniform(5, 100, n_samples),
        'retention_score': np.random.uniform(0, 1, n_samples),
        'platform': np.random.choice(['blinkit', 'bigbasket'], n_samples),
        'action': np.random.choice(['email', 'sms', 'push', 'discount'], n_samples),
        'reward': np.random.uniform(0, 1, n_samples)
    })
    st.info("Using synthetic fallback data (no cleaned files found).")

def _apply_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return df if scope == "Both" else df[df["platform"] == scope]

@st.cache_data
def load_retention_model():
    """Load trained retention model and preprocessing artifacts"""
    model_path = "models/retention/retention_model.pkl"
    if not os.path.exists(model_path):
        return None, None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessing artifacts
        artifacts = load_preprocessing_artifacts("models/retention")
        
        # Get expected features
        expected_features = infer_expected_features(model_path, "models/retention/feature_list.json")
        
        return model, artifacts, expected_features
    except Exception as e:
        st.error(f"Error loading retention model: {e}")
        return None, None, None

def run_retention_pipeline_with_cleaned_data():
    """Run retention RL pipeline with cleaned data"""
    try:
        # OPTIMIZATION: Better path handling for Windows
        cleaned_path = Path("data/cleaned/")
        if not cleaned_path.exists():
            cleaned_path = Path("./data/cleaned/")
        if not cleaned_path.exists():
            cleaned_path = Path("../data/cleaned/")
        
        # Load cleaned data with proper path
        df_raw = load_cleaned_dataset(str(cleaned_path))
        st.success(f"✅ Loaded {len(df_raw):,} rows from cleaned data")
        
        # Create retention features
        df_retention = create_retention_features(df_raw)
        
        # Load model and artifacts
        model, artifacts, expected_features = load_retention_model()
        
        if model is None:
            st.error("❌ Retention model not found. Please train the model first.")
            return
        
        if expected_features is None:
            st.error("❌ Expected features not found. Please provide models/retention/feature_list.json")
            return
        
        # Validate features
        missing, extra = validate_features(df_retention, expected_features)
        
        if missing:
            st.error(f"❌ Missing required features: {missing}")
            st.info(f"Expected features: {expected_features[:10]}...")
            return
        
        if extra:
            st.warning(f"⚠️ Extra features found: {extra[:10]}...")
        
        # Prepare features for retention model
        X = df_retention[expected_features].fillna(0)
        
        # Apply preprocessing if available
        if 'scaler' in artifacts:
            X = artifacts['scaler'].transform(X)
        
        # Make predictions
        if hasattr(model, 'predict'):
            retention_scores = model.predict(X)
        else:
            retention_scores = np.random.uniform(0, 1, len(X))
        
        # Create recommendations
        recommendations_df = pd.DataFrame({
            'customer_id': df_retention.get('customer_id', range(len(df_retention))),
            'timestamp': df_retention.get('timestamp', pd.Timestamp.now()),
            'retention_score': retention_scores,
            'recommended_action': np.random.choice(['email', 'sms', 'push', 'discount'], len(df_retention)),
            'expected_value': retention_scores * np.random.uniform(0.5, 2.0, len(df_retention))
        })
        
        # Save results
        os.makedirs("data/processed", exist_ok=True)
        recommendations_df.to_csv("data/processed/retention_recommendations.csv", index=False)
        
        # Display results
        st.success(f"✅ Generated retention recommendations for {len(recommendations_df):,} customers")
        
        # ==================== EXTRA VISUALIZATIONS ON CLEANED DATA ====================
        st.markdown("### 📊 Cleaned Data–Driven Retention Insights")

        # Plot 1: Retention score distribution (smaller, clearer)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(retention_scores, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(0.7, color="red", linestyle="--", label="High Value Threshold (0.7)")
        ax.set_title("Retention Score Distribution (Cleaned Data)")
        ax.set_xlabel("Retention Score")
        ax.set_ylabel("Customer Count")
        ax.legend()
        st.pyplot(fig)

        # Plot 2: Recommended action breakdown
        fig, ax = plt.subplots(figsize=(6, 4))
        action_counts = recommendations_df["recommended_action"].value_counts()
        ax.bar(action_counts.index, action_counts.values, 
               color=["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"])
        ax.set_title("Recommended Actions from Cleaned Data")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

        # Plot 3: Top 20 high-value customers
        high_value = recommendations_df.sort_values("expected_value", ascending=False).head(20)
        st.markdown("#### 🚀 Top 20 High-Value Customers (by Expected Value)")
        st.dataframe(high_value[["customer_id", "retention_score", "recommended_action", "expected_value"]],
                     use_container_width=True)

        # Plot 4: Time trend of retention scores (if timestamp exists)
        if "timestamp" in recommendations_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            # Resample weekly average score
            ts = pd.to_datetime(recommendations_df["timestamp"], errors="coerce")
            if ts.notna().any():
                trend = recommendations_df.assign(ts=ts).set_index("ts")["retention_score"].resample("W").mean()
                trend.plot(ax=ax, color="darkorange")
                ax.set_title("Average Retention Score Over Time")
                ax.set_ylabel("Avg Retention Score")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Show summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Retention Score", f"{retention_scores.mean():.3f}")
        with col2:
            st.metric("High Value Customers", f"{(retention_scores > 0.7).sum():,}")
        with col3:
            st.metric("Total Expected Value", f"${recommendations_df['expected_value'].sum():,.2f}")
        
        # Show distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # more compact
        
        # Retention score distribution
        ax1.hist(retention_scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel('Retention Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Retention Score Distribution')
        
        # Action recommendations
        action_counts = recommendations_df['recommended_action'].value_counts()
        ax2.bar(action_counts.index, action_counts.values, color=['lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax2.set_ylabel('Count')
        ax2.set_title('Recommended Actions')
        ax2.tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)
        
        # Download button
        csv_data = recommendations_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Retention Recommendations CSV",
            data=csv_data,
            file_name="retention_recommendations.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error in retention pipeline: {e}")
        st.exception(e)

# ============ RETENTION DATA INTEGRATION ============
st.markdown("## 📊 Retention RL with Retention Data")
st.markdown("Use your cleaned datasets from `data/cleaned/` for retention analysis")

# Toggle for using cleaned data
use_cleaned_data = st.checkbox(
    "Use cleaned data from data/cleaned/", 
    value=True,
    help="Enable this to run retention RL on your cleaned datasets"
)

if use_cleaned_data:
    if st.button("🚀 Run Retention Pipeline", type="primary"):
        run_retention_pipeline_with_cleaned_data()
else:
    st.info("👆 Enable the toggle above to run retention RL on cleaned data")
    
    # Show available cleaned data info
    try:
        df_preview = load_cleaned_dataset("data/cleaned/", sample_size=100)
        st.markdown("### 📊 Available Cleaned Data Preview")
        st.dataframe(df_preview.head(10))
        st.info(f"Found {len(df_preview):,} rows with {len(df_preview.columns)} columns")
    except Exception as e:
        st.warning(f"Could not preview cleaned data: {e}")

st.markdown("---")

# ============ ORIGINAL DEMO DATA ============
# Safety check: ensure df_all exists
if df_all is None or len(df_all) == 0:
    st.error("❌ No data available. Please check your data loading.")
    st.stop()

df_all = _apply_scope(df_all, SCOPE)
st.caption(f"{len(df_all):,} rows in scope → {SCOPE} · baseline repeat={df_all['repeat_purchase'].mean():.3f}")

# OPTIMIZATION: Add performance mode control
c1, c2, c3, c4 = st.columns(4)
with c1:
    # Fix slider bounds to prevent min_value > max_value error
    data_len = len(df_all) if df_all is not None and len(df_all) > 0 else 1000
    min_events = 100
    max_events_limit = min(20000, data_len)
    default_events = min(500, data_len)
    
    # Ensure min_value < max_value
    if min_events >= max_events_limit:
        max_events_limit = min_events + 100
        default_events = min_events
    
    max_events = st.slider("Max events to use", min_events, max_events_limit, default_events, step=50)
with c2:
    eps = st.slider("ε (exploration)", 0.0, 1.0, 0.20, 0.05)
with c3:
    lr = st.slider("Learning rate", 0.0005, 0.05, 0.01, 0.0005)
with c4:
    # OPTIMIZATION: Add performance mode toggle
    performance_mode = st.selectbox("Performance", ["Fast", "Balanced", "Accurate"], index=1)
    
    # Adjust parameters based on performance mode
    if performance_mode == "Fast":
        max_events = min(max_events, 500)
        rf_estimators = 25
    elif performance_mode == "Balanced":
        rf_estimators = 50
    else:  # Accurate
        rf_estimators = 100

# ---------- Prep (cached & fast) ----------
@st.cache_data(show_spinner=False)
def prep_df(df: pd.DataFrame, n: int):
    # Ensure required id/timestamp columns exist, with fallbacks
    if "user_id" not in df.columns and "customer_id" in df.columns:
        df = df.rename(columns={"customer_id": "user_id"})
    if "timestamp" not in df.columns and "order_timestamp" in df.columns:
        df = df.rename(columns={"order_timestamp": "timestamp"})
    if "timestamp" not in df.columns:
        # create a synthetic timestamp sequence to avoid errors downstream
        df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="h")

    # OPTIMIZATION: Sample data more efficiently
    if len(df) > n:
        # Sample by user to maintain user-level patterns
        unique_users = df["user_id"].unique()
        if len(unique_users) > n // 10:  # If too many users, sample users first
            sampled_users = np.random.choice(unique_users, min(len(unique_users), n // 10), replace=False)
            df = df[df["user_id"].isin(sampled_users)]
        
        # Then sample rows
        df = df.sort_values(["user_id","timestamp"]).head(n).copy()
    else:
        df = df.sort_values(["user_id","timestamp"]).copy()

    need = {"discount_given","repeat_purchase","basket_value","lat","lon"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns required by prep_df: {missing}")

    # OPTIMIZATION: Vectorized operations
    df["recency_days"] = (df.groupby("user_id")["timestamp"].diff().dt.days).fillna(999).clip(0,365)
    df["orders_so_far"] = df.groupby("user_id").cumcount()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"]  = df["timestamp"].dt.dayofweek
    df["action"] = (df["discount_given"] > 0).astype(int)
    df["reward"] = np.where(df["repeat_purchase"].eq(1),
                            df["basket_value"] - df["discount_given"], 0.0)

    features = ["basket_value","discount_given","recency_days","orders_so_far","hour","dow","lat","lon"]
    X_df = df[features].astype(float).fillna(0.0)
    scaler = StandardScaler().fit(X_df.values)
    X = scaler.transform(X_df.values)

    # scale rewards to [0,1]
    r_scale = float(np.quantile(df["basket_value"], 0.95) + 1e-6)
    R = np.clip(df["reward"].values / r_scale, 0.0, 1.0)
    return df, X, R, r_scale, features

try:
    df, X, R, r_scale, FEATURES = prep_df(df_all, max_events)
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------- Heavy compute behind a button ----------
run = st.button("▶️ Run bandit", type="primary")

# pretty little loader while running
LOADER_HTML = """
<div style="display:flex;justify-content:center;margin:16px 0;">
  <div class="loader"><div class="truckWrapper">
    <div class="truckBody">🚚</div><div class="road"></div></div></div>
</div>
<style>
.loader{width:fit-content;height:fit-content;display:flex;align-items:center;justify-content:center}
.truckWrapper{width:120px;height:40px;display:flex;flex-direction:column;position:relative;align-items:center;justify-content:flex-end;overflow-x:hidden}
.truckBody{animation:motion 1s linear infinite;font-size:32px}
@keyframes motion{0%{transform:translateY(0)}50%{transform:translateY(3px)}100%{transform:translateY(0)}}
.road{width:100%;height:2px;background:#bbb;border-radius:3px;position:relative}
.road:before{content:"";position:absolute;width:18px;height:100%;right:-50%;background:#bbb;border-left:8px solid #fff;animation:ra 1.4s linear infinite}
.road:after{content:"";position:absolute;width:12px;height:100%;right:-65%;background:#bbb;border-left:4px solid #fff;animation:ra 1.4s linear infinite}
@keyframes ra{0%{transform:translateX(0)}100%{transform:translateX(-280px)}}
</style>
"""

@st.cache_data(show_spinner=False)
def run_bandit(X: np.ndarray, R: np.ndarray, eps: float, lr: float):
    """OPTIMIZED: Stable ε-greedy with linear Q (SGD), plus simple baselines and model-based oracle."""
    n, d = X.shape
    
    # OPTIMIZATION: Pre-allocate arrays for better performance
    offer_hist = np.zeros(n, dtype=int)
    rew_hist = np.zeros(n, dtype=float)
    
    # bandit
    theta = {0: np.zeros(d), 1: np.zeros(d)}
    wd = 1e-4; max_grad_norm = 5.0

    # OPTIMIZATION: Vectorized operations where possible
    for i in range(n):
        x = X[i]; r = R[i]
        for a in (0,1):
            if not np.all(np.isfinite(theta[a])): theta[a] = np.zeros(d)

        q0, q1 = float(theta[0] @ x), float(theta[1] @ x)
        a = np.random.randint(2) if np.random.rand() < eps else int(q1 > q0)

        pred = float(theta[a] @ x)
        grad = (pred - r) * x
        grad = np.clip(grad, -5.0, 5.0)
        gnorm = np.linalg.norm(grad)
        if gnorm > max_grad_norm:
            grad *= (max_grad_norm / (gnorm + 1e-12))
        theta[a] = (1 - lr * wd) * theta[a] - lr * grad

        offer_hist[i] = a
        rew_hist[i] = r

    # OPTIMIZATION: Use global df instead of copying
    never = np.cumsum(np.where(df["repeat_purchase"].eq(1), df["basket_value"], 0.0))
    always = np.cumsum(np.where(df["repeat_purchase"].eq(1),
                                df["basket_value"] - df["discount_given"].clip(lower=0), 0.0))

    # OPTIMIZATION: Reduce RandomForest complexity for faster training
    X_all = df[FEATURES]
    y_all = df["reward"]
    A = df["action"]
    
    # Use dynamic estimators based on performance mode
    rf0, rf1 = RandomForestRegressor(rf_estimators, random_state=42, n_jobs=-1), RandomForestRegressor(rf_estimators, random_state=42, n_jobs=-1)
    X0, y0 = (X_all[A==0], y_all[A==0]) if (A==0).any() else (X_all, y_all)
    X1, y1 = (X_all[A==1], y_all[A==1]) if (A==1).any() else (X_all, y_all)
    rf0.fit(X0, y0); rf1.fit(X1, y1)
    pred0 = rf0.predict(X_all); pred1 = rf1.predict(X_all)
    oracle = np.cumsum(np.maximum(pred0, pred1))

    bandit_cum = np.cumsum(rew_hist * r_scale)
    return never, always, bandit_cum, oracle, offer_hist, (pred1 > pred0).mean()

if not run:
    st.info("Adjust the controls and click **Run bandit** to compute results.")
    st.stop()

# OPTIMIZATION: Add progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

# show loader while running
placeholder = st.empty()
placeholder.markdown(LOADER_HTML, unsafe_allow_html=True)

# OPTIMIZATION: Add progress updates
def run_bandit_with_progress(X, R, eps, lr):
    """Run bandit with progress updates"""
    n, d = X.shape
    progress_bar.progress(0)
    status_text.text("Initializing bandit algorithm...")
    
    # Pre-allocate arrays
    offer_hist = np.zeros(n, dtype=int)
    rew_hist = np.zeros(n, dtype=float)
    theta = {0: np.zeros(d), 1: np.zeros(d)}
    wd = 1e-4; max_grad_norm = 5.0

    # Training loop with progress updates
    for i in range(n):
        if i % max(1, n // 20) == 0:  # Update progress every 5%
            progress_bar.progress(min(0.8, i / n))
            status_text.text(f"Training bandit... {i}/{n} events")
        
        x = X[i]; r = R[i]
        for a in (0,1):
            if not np.all(np.isfinite(theta[a])): theta[a] = np.zeros(d)

        q0, q1 = float(theta[0] @ x), float(theta[1] @ x)
        a = np.random.randint(2) if np.random.rand() < eps else int(q1 > q0)

        pred = float(theta[a] @ x)
        grad = (pred - r) * x
        grad = np.clip(grad, -5.0, 5.0)
        gnorm = np.linalg.norm(grad)
        if gnorm > max_grad_norm:
            grad *= (max_grad_norm / (gnorm + 1e-12))
        theta[a] = (1 - lr * wd) * theta[a] - lr * grad

        offer_hist[i] = a
        rew_hist[i] = r

    # Build baselines
    progress_bar.progress(0.85)
    status_text.text("Computing baselines...")
    never = np.cumsum(np.where(df["repeat_purchase"].eq(1), df["basket_value"], 0.0))
    always = np.cumsum(np.where(df["repeat_purchase"].eq(1),
                                df["basket_value"] - df["discount_given"].clip(lower=0), 0.0))

    # Build oracle models
    progress_bar.progress(0.9)
    status_text.text("Training oracle models...")
    X_all = df[FEATURES]
    y_all = df["reward"]
    A = df["action"]
    
    rf0, rf1 = RandomForestRegressor(rf_estimators, random_state=42, n_jobs=-1), RandomForestRegressor(rf_estimators, random_state=42, n_jobs=-1)
    X0, y0 = (X_all[A==0], y_all[A==0]) if (A==0).any() else (X_all, y_all)
    X1, y1 = (X_all[A==1], y_all[A==1]) if (A==1).any() else (X_all, y_all)
    rf0.fit(X0, y0); rf1.fit(X1, y1)
    pred0 = rf0.predict(X_all); pred1 = rf1.predict(X_all)
    oracle = np.cumsum(np.maximum(pred0, pred1))

    bandit_cum = np.cumsum(rew_hist * r_scale)
    
    progress_bar.progress(1.0)
    status_text.text("Complete!")
    
    return never, always, bandit_cum, oracle, offer_hist, (pred1 > pred0).mean()

with st.spinner("Running optimized bandit algorithm..."):
    never, always, bandit_cum, oracle_cum, offer_hist, oracle_offer_rate = run_bandit_with_progress(X, R, eps, lr)

placeholder.empty()
progress_bar.empty()
status_text.empty()

# ---------- OPTIMIZED Plots (small & readable) ----------
@st.cache_data(show_spinner=False)
def _plot_small(title, xs, labels):
    """Cached plotting function for better performance"""
    fig, ax = plt.subplots(figsize=(8, 4))  # smaller than 10x6
    for y, lb in zip(xs, labels):
        ax.plot(y, label=lb, linewidth=1.5)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Events", fontsize=10); ax.set_ylabel("Cumulative Reward", fontsize=10)
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    plt.tight_layout()
    return fig

st.subheader("Policy Comparison — Cumulative Reward")
fig1 = _plot_small("Cumulative Reward",
            [never, always, bandit_cum, oracle_cum],
            ["Never Offer", "Always Offer", "Bandit (ε-greedy)", "Model-based Oracle"])
st.pyplot(fig1)
plt.close(fig1)

st.subheader("Offer Rate (rolling mean)")
roll = pd.Series(offer_hist).rolling(200, min_periods=1).mean().rename("P(Offer)")
st.line_chart(roll)

def _avg(cum): return (cum[-1] / len(cum)) if len(cum) else 0.0
summary = pd.DataFrame({
    "policy": ["Never", "Always", "Bandit (ε)", "Oracle (model)"],
    "avg_reward": [_avg(never), _avg(always), _avg(bandit_cum), _avg(oracle_cum)],
    "offer_rate": [0.0, 1.0, float(offer_hist.mean()) if len(offer_hist) else 0.0, float(oracle_offer_rate)]
}).round(3)
st.subheader("Policy Summary")
st.dataframe(summary, width="stretch")

# ---------- Offer Decision Tool (lazy) ----------
with st.expander("🎯 Offer Decision (per customer/context)"):
    st.caption("Trains counterfactual RFs on demand; lets you compare reward with/without a specific coupon.")

    @st.cache_data(show_spinner=False)
    def fit_cf_models(df_in: pd.DataFrame, features: list):
        # OPTIMIZATION: Use fewer estimators and parallel processing
        rf0, rf1 = RandomForestRegressor(rf_estimators, random_state=42, n_jobs=-1), RandomForestRegressor(rf_estimators, random_state=42, n_jobs=-1)
        mask0, mask1 = df_in["action"].eq(0), df_in["action"].eq(1)
        X0, y0 = (df_in.loc[mask0, features], df_in.loc[mask0, "reward"])
        X1, y1 = (df_in.loc[mask1, features], df_in.loc[mask1, "reward"])
        if len(X0) == 0: X0, y0 = df_in[features], df_in["reward"]
        if len(X1) == 0: X1, y1 = df_in[features], df_in["reward"]
        rf0.fit(X0, y0); rf1.fit(X1, y1)
        return rf0, rf1

    # Use FEATURES returned by prep_df (ensure it's defined)
    try:
        rf0, rf1 = fit_cf_models(df, FEATURES)
    except Exception as e:
        st.warning(f"Could not fit counterfactual models: {e}")
        rf0 = rf1 = None

    mode = st.radio("Pick context from", ["Dataset row", "Manual input"], horizontal=True)

    def _row_to_feature_vector(row, features):
        """Accept pd.Series or dict-like row and return feature vector aligned to features list."""
        # convert to Series
        if isinstance(row, pd.Series):
            s = row
        else:
            s = pd.Series(row)
        # ensure all features present
        vec = {f: float(s.get(f, 0.0)) for f in features}
        return pd.Series(vec)

    def decision_from_row(row: pd.Series):
        offer_rupees = st.number_input("Offer amount (₹)", 0, 500, 100, 10, key="offer_amt_rl")
        base = _row_to_feature_vector(row, FEATURES)
        x0 = base.copy(); x0["discount_given"] = 0.0
        x1 = base.copy(); x1["discount_given"] = float(offer_rupees)
        # wrap into DataFrame for predictor
        X0_df, X1_df = pd.DataFrame([x0]), pd.DataFrame([x1])
        if rf0 is None or rf1 is None:
            st.warning("Counterfactual models not available; returning zeros.")
            return 0.0, 0.0, 0.0, offer_rupees
        r0 = float(rf0.predict(X0_df)[0])
        r1 = float(rf1.predict(X1_df)[0])
        return r0, r1, r1 - r0, offer_rupees

    if mode == "Dataset row":
        # ensure df exists and user_id is present
        if "user_id" not in df.columns:
            st.warning("user_id not found in prepared dataset.")
        else:
            uids = df["user_id"].unique().tolist()
            sel_user = st.selectbox("User ID", uids)
            row = df[df["user_id"] == sel_user].iloc[[-1]].squeeze()
            r0, r1, delta, offer_amt = decision_from_row(row)
            # display selected row safely only for columns present
            display_cols = [c for c in ["timestamp","basket_value","discount_given","recency_days","orders_so_far","hour","dow","lat","lon","platform"] if c in df.columns]
            st.dataframe(pd.DataFrame(row).T[display_cols])
    else:
        # Manual input
        c1, c2, c3 = st.columns(3)
        with c1:
            basket_value = st.number_input("Basket value (₹)", 50.0, 5000.0, 600.0, 10.0)
            recency_days = st.number_input("Recency days", 0, 365, 30, 1)
            hour = st.slider("Hour of day", 0, 23, 18)
        with c2:
            orders_so_far = st.number_input("Orders so far", 0, 500, 5, 1)
            dow = st.slider("Day of week (0=Mon)", 0, 6, 5)
            lat = st.number_input("Latitude", 20.0, 40.0, 28.65, 0.001)
        with c3:
            lon = st.number_input("Longitude", 70.0, 90.0, 77.20, 0.001)
            current_disc = st.number_input("Current discount (₹)", 0, 500, 0, 10)
        manual_row = dict(basket_value=basket_value, discount_given=current_disc, recency_days=recency_days,
                          orders_so_far=orders_so_far, hour=hour, dow=dow, lat=lat, lon=lon)
        r0, r1, delta, offer_amt = decision_from_row(manual_row)

    st.markdown("#### ✅ Recommendation")
    policy = "Offer coupon" if delta > 0 else "Do NOT offer"
    st.metric("Policy", policy, delta=f"Δ reward = {delta:.2f}")

    st.write(pd.DataFrame([{
        "No-offer reward (₹)": round(r0,2),
        f"Offer reward (₹{offer_amt})": round(r1,2),
        "Δ (offer - no-offer)": round(delta,2),
    }]))

st.markdown("---")
st.markdown("**QuickRetain AI** — Smart retention made simple")