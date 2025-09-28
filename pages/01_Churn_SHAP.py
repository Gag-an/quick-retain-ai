# pages/01_Churn_SHAP.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
import os
import warnings
from typing import List, Tuple, Dict, Any

# Utilities you already have in repo
from clean_datasets import (
    load_cleaned_dataset,
    infer_expected_features,
    validate_features,
    load_preprocessing_artifacts,
    create_churn_features,
    save_feature_list
)

st.set_page_config(page_title="Churn + SHAP", layout="wide", initial_sidebar_state="collapsed")

# ---------- Helpers for alignment / safety ----------
def get_expected_columns_from_preprocessor(preprocessor) -> List[str]:
    """
    Extract expected column names from a fitted preprocessor if possible.
    Falls back to returning [].
    """
    if preprocessor is None:
        return []
    # sklearn objects sometimes have feature_names_in_
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    # ColumnTransformer: aggregate columns referenced by transformers_
    cols = []
    if hasattr(preprocessor, "transformers_"):
        for name, trans, col_sel in preprocessor.transformers_:
            try:
                if isinstance(col_sel, (list, tuple, pd.Index, np.ndarray)):
                    cols.extend(list(col_sel))
                elif isinstance(col_sel, slice):
                    # can't resolve slice here
                    continue
                elif isinstance(col_sel, str):
                    # sometimes a single column name
                    cols.append(col_sel)
            except Exception:
                continue
    # dedupe while preserving order
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

def load_expected_features_from_artifacts(artifacts_path: str, preprocessor=None) -> List[str]:
    """
    Try multiple ways to get expected columns:
     1) from preprocessor.feature_names_in_
     2) from preprocessor.transformers_
     3) from models/churn/feature_list.json
    """
    expected = []
    if preprocessor is not None:
        expected = get_expected_columns_from_preprocessor(preprocessor)
        if expected:
            return expected

    # fallback to feature_list.json
    fpath = Path(artifacts_path) / "feature_list.json"
    if fpath.exists():
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                expected = json.load(f)
                if isinstance(expected, list):
                    return expected
        except Exception:
            pass
    return expected

def align_dataframe_for_preprocessor(expected_cols: List[str], df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Ensure df contains every column in expected_cols. Add missing cols with safe defaults.
    Returns (df_aligned, missing_added, extra_cols).
    """
    # normalize input column names similarly to how cleaned loader does
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]

    # heuristics for default values
    for c in missing:
        # text fields
        if "feedback_text" in c or c.endswith("_text") or c.startswith("comment") or "review" in c:
            df[c] = ""
        # date fields
        elif c.endswith("_date") or "date" in c:
            df[c] = pd.NaT
        # binary flags
        elif c.startswith("is_") or c.startswith("has_") or c.startswith("sentiment_") or c in ("repeat_purchase", "churn"):
            df[c] = 0
        # ids
        elif any(tok in c for tok in ("id", "identifier", "code")):
            df[c] = -1
        # numeric-ish hints
        elif any(tok in c for tok in ("total", "count", "num_", "age", "year", "amount", "price", "charges", "spent")):
            df[c] = 0
        else:
            # last resort, NaN so imputer can handle
            df[c] = np.nan

    # Reorder to expected_cols if expected_cols not empty
    if expected_cols:
        df_aligned = df.reindex(columns=expected_cols)
    else:
        df_aligned = df

    return df_aligned, missing, extra

# ---------- Cached model & artifact loader ----------
@st.cache_resource(show_spinner=False)
def load_churn_artifacts() -> Tuple[Any, Dict[str, Any], List[str]]:
    """
    Loads model, preprocessing artifacts and expected features.
    Returns (model, artifacts_dict, expected_features).
    """
    model_path = Path("models/churn/churn_model.pkl")
    artifacts_dir = Path("models/churn")
    model = None
    artifacts = {}
    expected_features = []

    if not model_path.exists():
        return None, artifacts, expected_features

    # Load model (joblib preferred)
    try:
        model = joblib.load(str(model_path))
    except Exception:
        # fallback to pickle
        try:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Could not load churn model: {e}")

    # Load preprocessor/scaler/encoder if present
    artifacts = load_preprocessing_artifacts(str(artifacts_dir)) if 'load_preprocessing_artifacts' in globals() else {}
    # attempt to get preprocessor if key exists
    preprocessor = artifacts.get("preprocessor") if isinstance(artifacts, dict) else None

    # infer expected features
    expected_features = load_expected_features_from_artifacts(str(artifacts_dir), preprocessor=preprocessor)

    return model, artifacts, expected_features

# ---------- Main pipeline ----------
def run_churn_pipeline_with_cleaned_data(sample_size: int = 10000):
    import os  # Move import to top of function
    import numpy as np  # Move numpy import to top as well
    try:
        # Load cleaned data with robust error handling
        with st.spinner("Loading cleaned data (may sample for performance)..."):
            try:
                # Try with relative path first
                df_raw = load_cleaned_dataset("data/cleaned", sample_size=sample_size)
            except Exception as e:
                st.warning(f"Could not load cleaned data: {e}")
                # Try with absolute path as fallback
                current_dir = os.getcwd()
                cleaned_path = os.path.join(current_dir, "data", "cleaned")
                if os.path.exists(cleaned_path):
                    df_raw = load_cleaned_dataset(cleaned_path, sample_size=sample_size)
                else:
                    st.error(f"Cleaned data directory not found at {cleaned_path}")
                    return
            
        if df_raw is None or len(df_raw) == 0:
            st.warning("No cleaned data found. Creating synthetic data for demonstration.")
            # Create synthetic data as fallback
            np.random.seed(42)
            n_samples = min(sample_size, 5000)  # Limit synthetic data size
            df_raw = pd.DataFrame({
                'customer_id': range(n_samples),
                'total_orders': np.random.randint(1, 50, n_samples),
                'total_spent': np.random.uniform(10, 1000, n_samples),
                'avg_order_value': np.random.uniform(5, 100, n_samples),
                'days_since_last_order': np.random.randint(1, 365, n_samples),
                'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'platform': np.random.choice(['blinkit', 'bigbasket'], n_samples),
                'user_id': range(n_samples),
                'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='h')
            })
            st.info(f"Using synthetic data with {len(df_raw):,} rows for demonstration")

        st.success(f"✅ Loaded {len(df_raw):,} rows from cleaned data (sample used for performance)")

        # Create features required by churn model (your helper)
        df_churn = create_churn_features(df_raw)

        # Load artifacts & model
        model, artifacts, expected_features = load_churn_artifacts()

        if model is None:
            st.error("❌ Churn model not found. Please train the model first (scripts/train_churn.py).")
            return

        if not expected_features:
            st.warning("⚠️ Expected feature list not found in preprocessor or models/churn/feature_list.json. Attempting to infer from available cleaned data.")
            # fallback: use df_churn.columns (best-effort) but warn strongly
            expected_features = list(df_churn.columns)
            st.info("Using cleaned data columns as expected features (best-effort). Consider creating models/churn/feature_list.json")

        # --- STRICT FEATURE ALIGNMENT ---
        # Only keep expected raw features (ignore all extras)
        df_churn = df_churn.copy()
        df_churn.columns = [c.strip().lower().replace(" ", "_") for c in df_churn.columns]

        # Show what we're working with
        missing_before = [c for c in expected_features if c not in df_churn.columns]
        extra_before = [c for c in df_churn.columns if c not in expected_features]

        if missing_before:
            st.warning(f"⚠️ {len(missing_before)} expected features are missing from cleaned data. They will be auto-created with safe defaults. Missing example: {missing_before[:10]}")
        if extra_before:
            st.info(f"ℹ️ Found {len(extra_before)} extra columns in cleaned data (these will be DROPPED to prevent bloat): {extra_before[:10]}")

        # Slice down to expected features only (strict alignment)
        X_aligned = df_churn.reindex(columns=expected_features, fill_value=0)
        
        # Fill missing columns with safe defaults
        for col in expected_features:
            if col not in df_churn.columns:
                if col in ['customer_id', 'user_id'] or 'id' in col:
                    X_aligned[col] = range(len(X_aligned))
                elif 'date' in col or 'time' in col:
                    X_aligned[col] = pd.Timestamp.now()
                elif 'text' in col or 'comment' in col or 'review' in col or 'feedback' in col:
                    X_aligned[col] = "default_text"
                elif 'category' in col or 'type' in col or 'status' in col:
                    X_aligned[col] = "unknown"
                else:
                    X_aligned[col] = 0

        st.success(f"✅ Strict feature alignment complete: {len(X_aligned.columns)} features (dropped {len(extra_before)} extra columns)")

        # --- DIRECT PREDICTION WITH STRICT FEATURES ---
        # Now pass raw features into the pipeline (it has the preprocessor inside)
        st.info("Passing ONLY expected raw features into model. Dropping all extras.")
        
        try:
            # Ensure we have the right feature order
            X_input = X_aligned[expected_features].copy()
            
            # Detect if model is a pipeline with preprocessing
            model_is_pipeline = hasattr(model, "named_steps") and isinstance(model.named_steps, dict)
            pipeline_has_pre = False
            if model_is_pipeline:
                pnames = [k.lower() for k in model.named_steps.keys()]
                pipeline_has_pre = any(name in pnames for name in ("pre", "preprocessor", "preproc", "transformer", "col_transform"))
            
            if pipeline_has_pre:
                st.info("Model pipeline contains preprocessing → passing RAW features directly.")
                probabilities = model.predict_proba(X_input)[:, 1]
            else:
                # Try with separate preprocessor if available
                preprocessor = artifacts.get("preprocessor", None)
                if preprocessor is not None:
                    st.info("Using separate preprocessor to transform data before calling model.")
                    X_transformed = preprocessor.transform(X_input)
                    probabilities = model.predict_proba(X_transformed)[:, 1]
                else:
                    st.info("No preprocessor found → passing RAW features to model.")
                    probabilities = model.predict_proba(X_input)[:, 1]
            
            predictions = (probabilities >= 0.5).astype(int)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
            # Save debug info
            os.makedirs("archive/preprocess_debug", exist_ok=True)
            X_input.head(50).to_csv("archive/preprocess_debug/X_input_head.csv", index=False)
            with open("archive/preprocess_debug/expected_features.json", "w") as f:
                json.dump(expected_features, f, indent=2)
            return

        # Build results df and save - ensure proper alignment
        out_ids = df_churn.get("customer_id")
        if out_ids is None:
            out_ids = np.arange(len(df_churn))
        
        # Reset indices to ensure proper alignment
        out_ids = out_ids.reset_index(drop=True) if hasattr(out_ids, 'reset_index') else pd.Series(out_ids).reset_index(drop=True)
        probabilities = pd.Series(probabilities).reset_index(drop=True)
        predictions = pd.Series(predictions).reset_index(drop=True)
        
        # Ensure all arrays have the same length
        min_length = min(len(out_ids), len(probabilities), len(predictions))
        out_ids = out_ids.iloc[:min_length]
        probabilities = probabilities.iloc[:min_length]
        predictions = predictions.iloc[:min_length]

        results_df = pd.DataFrame({
            "customer_id": out_ids.values,
            "probability": probabilities.values,
            "prediction": predictions.values
        })

        os.makedirs("data/processed", exist_ok=True)
        csv_out = Path("data/processed/churn_predictions.csv")
        results_df.to_csv(csv_out, index=False)
        st.success(f"✅ Generated & saved predictions to {csv_out}")

        # ==================== EXTRA VISUALIZATIONS ON CLEANED DATA ====================
        st.markdown("### 📊 Cleaned Data–Driven Churn Insights")

        # Plot 1: Churn probability distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(probabilities, bins=30, alpha=0.7, color="coral", edgecolor="black")
        ax.axvline(0.5, color="red", linestyle="--", label="Churn Threshold (0.5)")
        ax.axvline(0.7, color="orange", linestyle="--", label="High Risk Threshold (0.7)")
        ax.set_title("Churn Probability Distribution (Cleaned Data)")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Customer Count")
        ax.legend()
        st.pyplot(fig)

        # Plot 2: Risk level breakdown
        fig, ax = plt.subplots(figsize=(6, 4))
        risk_df = results_df.copy()
        risk_df["risk_level"] = pd.cut(risk_df["probability"], bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"])
        risk_counts = risk_df["risk_level"].value_counts()
        ax.bar(risk_counts.index, risk_counts.values, 
               color=["#4CAF50", "#FF9800", "#F44336"])
        ax.set_title("Churn Risk Level Distribution")
        ax.set_ylabel("Customer Count")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

        # Plot 3: Top 20 high-risk customers
        # Filter out any rows with None/NaN values to ensure clean display
        clean_results = results_df.dropna(subset=["customer_id", "probability", "prediction"])
        high_risk = clean_results.sort_values("probability", ascending=False).head(20)
        st.markdown("#### ⚠️ Top 20 High-Risk Customers (by Churn Probability)")
        st.dataframe(high_risk[["customer_id", "probability", "prediction"]],
                     use_container_width=True)

        # Plot 4: Time trend of churn probabilities (if timestamp exists)
        if "timestamp" in results_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ts = pd.to_datetime(results_df["timestamp"], errors="coerce")
            if ts.notna().any():
                trend = results_df.assign(ts=ts).set_index("ts")["probability"].resample("W").mean()
                trend.plot(ax=ax, color="darkred")
                ax.set_title("Average Churn Probability Over Time")
                ax.set_ylabel("Avg Churn Probability")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        # Display high-level metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Churn Rate", f"{results_df['prediction'].mean():.1%}")
        with col2:
            st.metric("High Risk (p>0.7)", f"{(results_df['probability']>0.7).sum():,}")
        with col3:
            st.metric("Avg Probability", f"{results_df['probability'].mean():.3f}")
        with col4:
            st.metric("Total Customers", f"{len(results_df):,}")
        
        # Add AUC and Confusion Matrix
        st.markdown("#### 📊 Model Performance Metrics")
        
        # Since we don't have true labels, we'll show what we can
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confusion Matrix (Predicted vs Threshold)**")
            # Create a confusion matrix based on threshold
            threshold = 0.5
            predicted_positive = (results_df['probability'] >= threshold).sum()
            predicted_negative = (results_df['probability'] < threshold).sum()
            
            # Create a simple confusion matrix display
            cm_data = {
                'Predicted Churn': [predicted_positive, predicted_negative],
                'Predicted Retain': [0, 0]  # We don't have true labels
            }
            cm_df = pd.DataFrame(cm_data, index=['Actual Churn', 'Actual Retain'])
            st.dataframe(cm_df, use_container_width=True)
            
            st.info("Note: True labels not available - showing predicted distribution only")
        
        with col2:
            st.markdown("**Probability Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(results_df['probability'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
            ax.set_xlabel('Churn Probability')
            ax.set_ylabel('Count')
            ax.set_title('Churn Probability Distribution')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        
        # Try to calculate AUC if we have any ground truth indicators
        st.markdown("#### 📈 Model Performance Analysis")
        
        # Check if we can derive some performance metrics
        try:
            from sklearn.metrics import roc_auc_score, classification_report
            
            # Try to find any churn indicators in the original data
            churn_indicators = []
            if 'churn' in df_churn.columns:
                churn_indicators = df_churn['churn'].values
            elif 'is_churn' in df_churn.columns:
                churn_indicators = df_churn['is_churn'].values
            elif 'churned' in df_churn.columns:
                churn_indicators = df_churn['churned'].values
            
            if len(churn_indicators) > 0 and len(churn_indicators) == len(results_df):
                # We have ground truth - calculate real metrics
                auc_score = roc_auc_score(churn_indicators, results_df['probability'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AUC Score", f"{auc_score:.3f}")
                with col2:
                    st.metric("Model Quality", "Excellent" if auc_score > 0.8 else "Good" if auc_score > 0.7 else "Fair")
                
                # Show classification report
                st.markdown("**Classification Report**")
                report = classification_report(churn_indicators, results_df['prediction'], 
                                            target_names=['Retain', 'Churn'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3), use_container_width=True)
            else:
                st.info("No ground truth labels available for AUC calculation. Showing prediction distribution only.")
                
        except Exception as e:
            st.warning(f"Could not calculate performance metrics: {e}")
            st.info("Showing prediction distribution only.")

        # ---------- Visualizations ----------
        st.markdown("### 📊 Churn Analysis Visualizations")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📈 Probability Distribution",
            "🎯 Risk Segmentation",
            "🔍 Feature Analysis",
            "📋 Customer Insights"
        ])

        # PROBABILITY DISTRIBUTION
        with viz_tab1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # more compact
            ax1, ax2, ax3, ax4 = axes.flatten()

            ax1.hist(results_df["probability"], bins=30, alpha=0.8, edgecolor="k")
            ax1.axvline(0.5, color="red", linestyle="--", label="0.5")
            ax1.axvline(0.7, color="orange", linestyle="--", label="0.7")
            ax1.set_title("Churn Probability Distribution")
            ax1.set_xlabel("Probability")
            ax1.legend()
            ax1.grid(alpha=0.2)

            pred_counts = results_df["prediction"].value_counts().reindex([0,1]).fillna(0)
            bars = ax2.bar(["No Churn", "Churn"], pred_counts.values, color=["#8fd3a4", "#f28b82"])
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*pred_counts.values.max(),
                         f"{int(bar.get_height()):,}", ha="center", fontweight="bold")
            ax2.set_title("Prediction Counts")
            ax2.grid(alpha=0.2, axis="y")

            risk_segments = pd.cut(results_df["probability"], bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"])
            risk_counts = risk_segments.value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            ax3.bar(risk_counts.index, risk_counts.values, color=["#8fd3a4", "#ffd27f", "#f28b82"])
            ax3.set_title("Risk Segmentation")
            for idx, val in enumerate(risk_counts.values):
                ax3.text(idx, val + 0.01*max(risk_counts.values), f"{int(val):,}", ha="center", fontweight="bold")

            # scatter sample
            sample_n = min(1000, len(results_df))
            sample_idx = np.random.choice(len(results_df), sample_n, replace=False)
            ax4.scatter(range(sample_n), results_df["probability"].values[sample_idx], s=10, alpha=0.6)
            ax4.set_title(f"Probability Scatter (sample {sample_n})")
            ax4.set_xlabel("Sample Index")
            ax4.set_ylabel("Prob")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # RISK SEGMENTATION (detailed)
        with viz_tab2:
            st.markdown("#### Risk segmentation details")
            risk_df = results_df.copy()
            risk_df["risk_level"] = pd.cut(risk_df["probability"], bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"])
            risk_summary = risk_df.groupby("risk_level")["probability"].agg(["count", "mean", "min", "max"]).rename(columns={"count":"Count","mean":"Avg","min":"Min","max":"Max"})
            st.dataframe(risk_summary, use_container_width=True)

            fig, ax = plt.subplots(figsize=(4, 4))  # smaller square
            risk_counts = risk_df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)

            colors = ["#8fd3a4", "#ffd27f", "#f28b82"]
            wedges, texts, autotexts = ax.pie(
                risk_counts.values,
                labels=risk_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                textprops={"fontsize": 10}
            )
            ax.set_title("Risk Distribution", fontsize=12, weight="bold")
            ax.axis("equal")  # keep circular
            st.pyplot(fig)
            plt.close(fig)

        # FEATURE ANALYSIS
        with viz_tab3:
            st.markdown("#### Feature importance / proxy analysis")
            try:
                # attempt to show feature_importances_ if present
                if hasattr(model, "feature_importances_"):
                    feat_names = expected_features if expected_features else [f"f{i}" for i in range(len(model.feature_importances_))]
                    fi = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
                    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
                    st.dataframe(fi.head(30), use_container_width=True)

                    fig, ax = plt.subplots(figsize=(6, 4))  # compact size to match pie chart
                    top = fi.head(10)[::-1]
                    ax.barh(top["feature"], top["importance"], color="#6fa8dc")
                    ax.set_title("Top 10 Feature Importances", fontsize=12, weight="bold")
                    ax.tick_params(axis='both', which='major', labelsize=9)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Feature importance not available for this model. Use SHAP for detailed explanations.")
            except Exception as e:
                st.warning(f"Could not show feature importances: {e}")

        # CUSTOMER INSIGHTS
        with viz_tab4:
            st.markdown("#### High risk customer list & recommendations")
            
            # Create a more informative customer table - filter out None values first
            clean_results = results_df.dropna(subset=["customer_id", "probability", "prediction"])
            high_risk_df = clean_results[clean_results["probability"] > 0.7].sort_values("probability", ascending=False)
            
            if len(high_risk_df) > 0:
                # Add risk level categorization
                high_risk_df = high_risk_df.copy()
                high_risk_df['risk_level'] = high_risk_df['probability'].apply(
                    lambda x: 'Critical' if x > 0.9 else 'High' if x > 0.8 else 'Medium'
                )
                
                # Format the display
                display_df = high_risk_df[['customer_id', 'probability', 'prediction', 'risk_level']].copy()
                display_df['probability'] = display_df['probability'].round(3)
                display_df['customer_id'] = display_df['customer_id'].astype(str)
                
                st.dataframe(display_df.head(50), use_container_width=True)
                
                # Add summary stats
                st.markdown(f"**Summary:** {len(high_risk_df)} high-risk customers found")
                st.markdown(f"- Critical Risk (>0.9): {(high_risk_df['probability'] > 0.9).sum()}")
                st.markdown(f"- High Risk (0.8-0.9): {((high_risk_df['probability'] > 0.8) & (high_risk_df['probability'] <= 0.9)).sum()}")
                st.markdown(f"- Medium Risk (0.7-0.8): {((high_risk_df['probability'] >= 0.7) & (high_risk_df['probability'] <= 0.8)).sum()}")
            else:
                st.info("No customers with probability > 0.7 in this sample.")

            st.markdown("""
            **Suggested next steps for high-risk customers**
            - **Critical Risk (>0.9)**: Immediate phone call + personalized retention offer
            - **High Risk (0.8-0.9)**: Email + SMS campaign with discount
            - **Medium Risk (0.7-0.8)**: Targeted email with product recommendations
            - Check recent support tickets or complaints for all risk levels
            """)

        # ---------- SHAP (optional) ----------
        st.markdown("### 🔍 SHAP explainability (optional)")
        run_shap = st.checkbox("Run SHAP analysis (may be slow)", value=False)
        if run_shap:
            try:
                import shap
                st.info("Running SHAP on a sampled subset (to limit memory/time)...")
                # sample a small subset
                sample_n = min(200, len(df_churn))
                sample_idx = np.random.choice(len(df_churn), sample_n, replace=False)
                # Use strict feature alignment for SHAP sample too
                X_sample = X_aligned.iloc[sample_idx][expected_features].values

                # Build/Load explainer
                shap_path = Path("models/churn/shap_explainer.pkl")
                explainer = None
                if shap_path.exists():
                    try:
                        explainer = joblib.load(shap_path)
                    except Exception:
                        explainer = None

                if explainer is None:
                    # Build explainer (best-effort)
                    if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
                        explainer = shap.TreeExplainer(model)
                    else:
                        # KernelExplainer requires a background; take small sample
                        bg = X_sample if hasattr(X_sample, "__array__") else X_sample
                        explainer = shap.KernelExplainer(model.predict_proba, bg)
                    # cache it
                    try:
                        joblib.dump(explainer, shap_path)
                    except Exception:
                        pass

                # Compute SHAP values and plot summary
                with st.spinner("Computing SHAP values..."):
                    sv = explainer.shap_values(X_sample)
                    fig = plt.figure(figsize=(10,6))
                    # for tree models, shap_values may be list of arrays; take class 1
                    if isinstance(sv, list) and len(sv) > 1:
                        shap.summary_plot(sv[1], X_aligned.iloc[sample_idx], feature_names=expected_features, show=False)
                    else:
                        shap.summary_plot(sv, X_aligned.iloc[sample_idx], feature_names=expected_features, show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                st.success("✅ SHAP complete")
            except ImportError:
                st.warning("SHAP not installed. Install with: pip install shap")
            except Exception as e:
                st.warning(f"SHAP failed: {e}")

        # ---------- Save full SHAP details optionally ----------
        if st.checkbox("Save predictions + SHAP to disk for UI / download", value=False):
            # Save full pred CSV already done; optionally save SHAP per-row if available
            try:
                # write JSON with full per-row predictions (no heavy SHAP included unless computed earlier)
                out_json = Path("data/processed/churn_predictions_with_shap.json")
                # minimal content: predictions + index mapping
                payload = []
                for idx, row in results_df.head(1000).iterrows():  # limit size for safety
                    payload.append({
                        "customer_id": int(row["customer_id"]) if pd.notna(row["customer_id"]) else None,
                        "probability": float(row["probability"]),
                        "prediction": int(row["prediction"])
                    })
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                st.success(f"Saved sample predictions to {out_json}")
            except Exception as e:
                st.warning(f"Could not save predictions JSON: {e}")

        # Download button for preds
        st.download_button("📥 Download predictions CSV", data=results_df.to_csv(index=False), file_name="churn_predictions.csv")

    except Exception as e:
        st.error(f"❌ Unhandled error in churn pipeline: {e}")
        st.exception(e)


# ---------- UI ----------
st.title("📊 Churn Prediction ")
st.markdown("Use cleaned datasets from `data/cleaned/` for churn prediction.")

use_cleaned_data = st.checkbox("Use cleaned data from data/cleaned/", value=True,
                               help="Enable this to run churn prediction on your cleaned datasets (concat of CSVs).")

if use_cleaned_data:
    col_run = st.columns([1, 4])[0]
    if col_run.button("🚀 Run Churn Pipeline"):
        run_churn_pipeline_with_cleaned_data(sample_size=10000)
else:
    st.info("Toggle 'Use cleaned data' to enable predictions on your cleaned datasets.")
    # preview cleaned data
    try:
        preview = load_cleaned_dataset("data/cleaned/", sample_size=100)
        if preview is None or preview.empty:
            st.warning("No cleaned CSVs found in data/cleaned/")
        else:
            st.markdown("### Preview of cleaned data")
            st.dataframe(preview.head(10))
            st.info(f"Found {len(preview):,} rows and {len(preview.columns)} columns (sample preview).")
    except Exception as e:
        st.warning(f"Could not preview cleaned data: {e}")
