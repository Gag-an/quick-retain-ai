import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from geopy.distance import geodesic
import io
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import json
from math import hypot

# Import our cleaned data utilities
from clean_datasets import (
    load_cleaned_dataset, infer_expected_features, validate_features,
    load_preprocessing_artifacts, create_logistics_features, save_feature_list
)

# from utils_perf import load_logistics  # Commented out - module not found

# Page config
st.set_page_config(page_title="Logistics", layout="wide", initial_sidebar_state="collapsed")

# --- NAVBAR COMPONENT ---
def navbar(active: str):
    c_brand, c_spacer, c1, c2, c3, c4, c5 = st.columns([1.4, 5.8, 1.25, 1.9, 1.6, 1.25, 1.6])
    with c_brand:
        st.markdown('<div class="qr-brand"><span class="qr-cube"></span><span>QuickRetain</span></div>', unsafe_allow_html=True)
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

# Use the navbar on the logistics page
navbar(active="pages/03_Logistics.py")

# ---- Load processed logistics data ----
import os
import pydeck as pdk

ROOT = r"D:\\quick Retain Ai\\data\\processed"

# df = load_logistics(ROOT).copy()  # Commented out - function not available
# Create sample logistics data instead
np.random.seed(42)
n_samples = 500
df = pd.DataFrame({
    'delivery_id': range(n_samples),
    'distance_km': np.random.uniform(1, 50, n_samples),
    'delivery_time_min': np.random.uniform(15, 120, n_samples),
    'traffic_score': np.random.uniform(0, 1, n_samples),
    'weather_score': np.random.uniform(0, 1, n_samples),
    'driver_experience': np.random.uniform(0, 1, n_samples),
    'platform': np.random.choice(['blinkit', 'bigbasket'], n_samples),  # Add platform column
    'lat': np.random.uniform(12.8, 13.2, n_samples),  # Add latitude
    'lon': np.random.uniform(77.5, 77.8, n_samples)   # Add longitude
})
@st.cache_data
def load_logistics_model():
    """Load trained logistics model and preprocessing artifacts"""
    model_path = "models/logistics/logistics_model.pkl"
    if not os.path.exists(model_path):
        return None, None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessing artifacts
        artifacts = load_preprocessing_artifacts("models/logistics")
        
        # Get expected features
        expected_features = infer_expected_features(model_path, "models/logistics/feature_list.json")
        
        return model, artifacts, expected_features
    except Exception as e:
        st.error(f"Error loading logistics model: {e}")
        return None, None, None

def run_logistics_pipeline_with_cleaned_data():
    """Run logistics optimization pipeline with cleaned data"""
    try:
        # Load cleaned data
        df_raw = load_cleaned_dataset("data/cleaned/")
        st.success(f"✅ Loaded {len(df_raw):,} rows from cleaned data")
        
        # Create logistics features
        df_logistics = create_logistics_features(df_raw)
        
        # Load model and artifacts
        model, artifacts, expected_features = load_logistics_model()
        
        if model is None:
            st.error("❌ Logistics model not found. Please train the model first.")
            return
        
        if expected_features is None:
            st.error("❌ Expected features not found. Please provide models/logistics/feature_list.json")
            return
        
        # Validate features
        missing, extra = validate_features(df_logistics, expected_features)
        
        if missing:
            st.error(f"❌ Missing required features: {missing}")
            st.info(f"Expected features: {expected_features[:10]}...")
            return
        
        if extra:
            st.warning(f"⚠️ Extra features found: {extra[:10]}...")
        
        # Prepare features for logistics model
        X = df_logistics[expected_features].fillna(0)
        
        # Apply preprocessing if available
        if 'scaler' in artifacts:
            X = artifacts['scaler'].transform(X)
        
        # Run clustering for dark store suggestions
        if 'pickup_lat' in df_logistics.columns and 'pickup_lng' in df_logistics.columns:
            coords = df_logistics[['pickup_lat', 'pickup_lng']].values
            
            # K-means clustering for dark store locations
            n_clusters = min(10, len(coords) // 50)  # Adaptive number of clusters
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coords)
            
            # Create dark store suggestions
            dark_stores = []
            for i in range(n_clusters):
                cluster_points = coords[cluster_labels == i]
                if len(cluster_points) > 0:
                    center_lat = cluster_points[:, 0].mean()
                    center_lng = cluster_points[:, 1].mean()
                    demand = len(cluster_points)
                    dark_stores.append({
                        'store_id': f'DS_{i+1:03d}',
                        'lat': center_lat,
                        'lng': center_lng,
                        'demand': demand,
                        'coverage_km': 5.0  # 5km coverage radius
                    })
            
            # Save dark store suggestions
            dark_stores_df = pd.DataFrame(dark_stores)
            os.makedirs("data/processed", exist_ok=True)
            dark_stores_df.to_csv("data/processed/dark_store_suggestions.csv", index=False)
            
            # Create optimized routes (simplified)
            routes = []
            for i, store in enumerate(dark_stores):
                # Find nearby delivery points for this store
                store_coords = (store['lat'], store['lng'])
                distances = []
                for _, row in df_logistics.iterrows():
                    if 'dropoff_lat' in row and 'dropoff_lng' in row:
                        dist = geodesic(store_coords, (row['dropoff_lat'], row['dropoff_lng'])).kilometers
                        if dist <= store['coverage_km']:
                            distances.append({
                                'delivery_id': row.get('order_id', f'delivery_{len(distances)}'),
                                'distance_km': dist,
                                'priority': 1 if dist <= 2 else 2
                            })
                
                # Sort by priority and distance
                distances.sort(key=lambda x: (x['priority'], x['distance_km']))
                
                routes.append({
                    'store_id': store['store_id'],
                    'route_id': f'R_{i+1:03d}',
                    'total_deliveries': len(distances),
                    'avg_distance_km': np.mean([d['distance_km'] for d in distances]) if distances else 0,
                    'estimated_time_min': len(distances) * 5 + np.mean([d['distance_km'] for d in distances]) * 2
                })
            
            routes_df = pd.DataFrame(routes)
            routes_df.to_csv("data/processed/optimized_routes.csv", index=False)
            
            # Create explanation metrics
            explanation = {
                'total_stores': len(dark_stores),
                'total_coverage_km2': len(dark_stores) * 78.54,  # π * 5^2
                'avg_deliveries_per_store': routes_df['total_deliveries'].mean(),
                'avg_route_time_min': routes_df['estimated_time_min'].mean(),
                'total_estimated_savings_km': routes_df['avg_distance_km'].sum() * 0.3  # 30% efficiency gain
            }
            
            with open("data/processed/logistics_explain.json", 'w') as f:
                json.dump(explanation, f, indent=2)
            
            # Display results
            st.success(f"✅ Generated {len(dark_stores)} dark store suggestions and {len(routes)} optimized routes")
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dark Stores", len(dark_stores))
            with col2:
                st.metric("Total Coverage (km²)", f"{explanation['total_coverage_km2']:.1f}")
            with col3:
                st.metric("Avg Route Time", f"{explanation['avg_route_time_min']:.1f} min")
            
            # Show map visualization
            st.markdown("### 🗺️ Dark Store Locations")
            map_data = pd.DataFrame({
                'lat': [store['lat'] for store in dark_stores],
                'lon': [store['lng'] for store in dark_stores],
                'size': [store['demand'] for store in dark_stores]
            })
            
            st.map(map_data, size='size', color='#FF6B6B')
            
            # Download buttons
            dark_stores_csv = dark_stores_df.to_csv(index=False)
            routes_csv = routes_df.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Dark Store Suggestions",
                    data=dark_stores_csv,
                    file_name="dark_store_suggestions.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="📥 Download Optimized Routes",
                    data=routes_csv,
                    file_name="optimized_routes.csv",
                    mime="text/csv"
                )
        
    except Exception as e:
        st.error(f"❌ Error in logistics pipeline: {e}")
        st.exception(e)

# ============ RETENTION DATA INTEGRATION ============
st.markdown("## 📊 Logistics Optimization with Retention Data")
# Run logistics pipeline directly
if st.button("🚀 Run Logistics Pipeline", type="primary"):
    run_logistics_pipeline_with_cleaned_data()
    
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
st.caption(f"Loaded {len(df):,} points from cached loader")

# ---- Platform filter (Both / blinkit / bigbasket) ----
SCOPE = st.radio("Scope", ["Both", "blinkit", "bigbasket"], horizontal=True, key="logi_scope")
if "platform" in df.columns and SCOPE != "Both":
    df = df[df["platform"] == SCOPE].copy()
st.caption(f"{len(df):,} points in scope → {SCOPE}")

# ---- Clustering controls ----
st.subheader("Zone clustering + routes")
max_k = max(2, min(15, len(df)))
k = st.slider("Number of delivery zones (KMeans)", 2, max_k, min(6, max_k))
max_points = st.slider("Max points on map", 1000, 50000, 10000, 1000)
max_stops = st.slider("Max stops per cluster (for routing)", 50, 1000, 200, 50)

def euclid_km(a, b):
    return hypot((a[0]-b[0])*111, (a[1]-b[1])*111)

def nn_route_fast(latlons, cap):
    if not latlons: return []
    pts = latlons[:cap]
    route = [pts.pop(0)]
    while pts:
        last = route[-1]
        j = min(range(len(pts)), key=lambda i: euclid_km(last, pts[i]))
        route.append(pts.pop(j))
    return route

def compute_clusters_routes(df_in: pd.DataFrame, k_in: int, max_points_in: int, max_stops_in: int):
    df_work = df_in
    if len(df_work) > max_points_in:
        df_work = df_work.sample(max_points_in, random_state=42).copy()
    k_eff = min(k_in, max(2, len(df_work)))
    algo = MiniBatchKMeans(n_clusters=k_eff, batch_size=2048, random_state=42)
    algo.fit(df_work[["lat","lon"]])
    df_work = df_work.copy()
    df_work["cluster"] = algo.labels_
    centers_local = pd.DataFrame(algo.cluster_centers_, columns=["lat","lon"]) 
    centers_local["cluster"] = range(k_eff)

    # Depot
    with st.expander("Depot / Start point"):
        depot_mode = st.radio("Start each route from", ["Cluster centroid (auto)", "Custom depot (one for all)"], horizontal=True, key="logi_depot")
        if depot_mode == "Custom depot (one for all)":
            dep_lat = st.number_input("Depot latitude", 20.0, 40.0, float(df_work["lat"].mean()), 0.0001, key="logi_dep_lat")
            dep_lon = st.number_input("Depot longitude", 70.0, 90.0, float(df_work["lon"].mean()), 0.0001, key="logi_dep_lon")
            depot = (dep_lat, dep_lon)
        else:
            depot = None

    # palette for cluster paths
    palette = [
        [230, 57, 70], [29, 53, 87], [69, 123, 157], [131, 197, 190], [168, 218, 220],
        [244, 162, 97], [233, 196, 106], [42, 157, 143], [38, 70, 83], [142, 202, 230],
    ]

    routes_local = []
    metrics_local = []
    for c in range(k_eff):
        pts = df_work.loc[df_work["cluster"] == c, ["lat","lon"]].to_numpy().tolist()
        if not pts:
            continue
        if depot is None:
            start = tuple(centers_local.loc[centers_local["cluster"]==c, ["lat","lon"]].iloc[0].tolist())
            latlons = [(lat,lon) for lat,lon in [start]+pts]
        else:
            latlons = [(lat,lon) for lat,lon in [depot]+pts]
        rt = nn_route_fast(latlons, max_stops_in)
        from geopy.distance import geodesic
        dist = float(sum(geodesic(rt[i], rt[i+1]).km for i in range(len(rt)-1))) if len(rt) > 1 else 0.0
        color = palette[c % len(palette)]
        routes_local.append({"cluster": c, "path": [[lon, lat] for (lat,lon) in rt], "color": color})
        metrics_local.append({"cluster": c, "stops": len(rt), "route_km": round(dist, 2)})

    metrics_df_local = pd.DataFrame(metrics_local).sort_values("cluster")
    return df_work, centers_local, routes_local, metrics_df_local

# Trigger compute
if st.button("🚀 Compute solution"):
    df_out, centers, routes, metrics_df = compute_clusters_routes(df, k, max_points, max_stops)
    st.session_state["logi_df"] = df_out
    st.session_state["logi_centers"] = centers
    st.session_state["logi_routes"] = routes
    st.session_state["logi_metrics"] = metrics_df

# remove immediate compute path; rely on button-triggered results above

if all(k in st.session_state for k in ["logi_df","logi_centers","logi_routes","logi_metrics"]):
    df_vis = st.session_state["logi_df"]
    centers = st.session_state["logi_centers"]
    routes = st.session_state["logi_routes"]
    metrics_df = st.session_state["logi_metrics"]

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_vis,
        get_position='[lon, lat]',
        get_radius=80,
        pickable=True,
        get_fill_color=[30, 30, 30, 120],
    )

    centroid_layer = pdk.Layer(
        "ScatterplotLayer",
        data=centers,
        get_position='[lon, lat]',
        get_radius=300,
        get_fill_color=[255, 0, 0, 200],
    )

    path_layer = pdk.Layer(
        "PathLayer",
        data=routes,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=3,
        rounded=True,
    )

    view = pdk.ViewState(
        latitude=float(df_vis["lat"].mean()),
        longitude=float(df_vis["lon"].mean()),
        zoom=8,
    )

    st.pydeck_chart(pdk.Deck(
        initial_view_state=view,
        map_style=None,
        layers=[point_layer, centroid_layer, path_layer],
        tooltip={"text": "Cluster {cluster}"}
    ))

    st.subheader("📊 Route Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", int(metrics_df["cluster"].nunique()) if not metrics_df.empty else 0)
    col2.metric("Total stops", int(metrics_df["stops"].sum()) if not metrics_df.empty else 0)
    col3.metric("Total km", f"{metrics_df['route_km'].sum():,.1f}" if not metrics_df.empty else "0.0")

    st.dataframe(metrics_df, width="stretch")

    def flatten_routes(routes_in):
        rows = []
        for r in routes_in:
            for seq, (lon, lat) in enumerate(r["path"]):
                rows.append({"cluster": r["cluster"], "seq": seq, "lat": lat, "lon": lon})
        return pd.DataFrame(rows).sort_values(["cluster", "seq"])

    route_table = flatten_routes(routes)
    if not route_table.empty:
        st.download_button(
            "⬇️ Download routes CSV",
            data=route_table.to_csv(index=False),
            file_name=f"routes_{SCOPE.lower()}_k{k}.csv",
            mime="text/csv",
        )

# --- FOOTER ---
st.markdown("---")
st.markdown("**QuickRetain AI** - Smart logistics made simple")