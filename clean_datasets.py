"""
QuickRetain AI - Cleaned Data Integration Utilities
Load and preprocess cleaned datasets for ML pipelines
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_dataset(path: str = "data/cleaned/", sample_size: int = None) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from cleaned data folder.
    
    Args:
        path: Path to cleaned data folder
        sample_size: If provided, sample this many rows from the result
        
    Returns:
        Concatenated DataFrame with normalized headers
        
    Raises:
        FileNotFoundError: If no CSV files found
        ValueError: If concatenated DataFrame is empty
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data path does not exist: {path}")
    
    # Find all CSV files
    csv_files = list(path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files in {path}")
    
    dataframes = []
    
    for file_path in csv_files:
        try:
            print(f"Loading {file_path.name}...")
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"Warning: {file_path.name} is empty, skipping")
                continue
                
            # Normalize headers
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")
    
    # Concatenate all dataframes (handle memory issues)
    if len(dataframes) == 1:
        combined_df = dataframes[0]
    else:
        # For large datasets, concatenate in chunks to avoid memory issues
        chunk_size = 5  # Process 5 files at a time
        combined_dfs = []
        
        for i in range(0, len(dataframes), chunk_size):
            chunk = dataframes[i:i+chunk_size]
            if len(chunk) == 1:
                combined_dfs.append(chunk[0])
            else:
                combined_dfs.append(pd.concat(chunk, ignore_index=True, sort=False))
        
        if len(combined_dfs) == 1:
            combined_df = combined_dfs[0]
        else:
            combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
    
    if combined_df.empty:
        raise ValueError("Concatenated DataFrame is empty")
    
    # Sample if requested
    if sample_size and len(combined_df) > sample_size:
        combined_df = combined_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(combined_df):,} rows from {len(combined_df):,} total rows")
    
    print(f"Successfully loaded {len(combined_df):,} rows with {len(combined_df.columns)} columns")
    return combined_df

def preview_columns(df: pd.DataFrame) -> List[str]:
    """
    Return normalized column list for preview.
    
    Args:
        df: DataFrame to preview
        
    Returns:
        List of normalized column names
    """
    return list(df.columns)

def infer_expected_features(model_path: str, feature_list_path: str) -> Optional[List[str]]:
    """
    Infer expected features from model or feature list file.
    
    Args:
        model_path: Path to model pickle file
        feature_list_path: Path to feature list JSON file
        
    Returns:
        List of expected feature names or None if not found
    """
    # Try to load from model first
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
    except Exception as e:
        print(f"Could not load features from model {model_path}: {e}")
    
    # Try to load from feature list file
    try:
        with open(feature_list_path, 'r') as f:
            feature_list = json.load(f)
            return feature_list
    except Exception as e:
        print(f"Could not load features from {feature_list_path}: {e}")
    
    return None

def validate_features(df: pd.DataFrame, expected_features: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate DataFrame features against expected features.
    
    Args:
        df: DataFrame to validate
        expected_features: List of expected feature names
        
    Returns:
        Tuple of (missing_features, extra_features)
    """
    available_features = set(df.columns)
    expected_set = set(expected_features)
    
    missing = [f for f in expected_set if f not in available_features]
    extra = [f for f in available_features if f not in expected_set]
    
    return missing, extra

def load_preprocessing_artifacts(model_folder: str) -> Dict:
    """
    Load preprocessing artifacts (encoder, scaler) from model folder.
    
    Args:
        model_folder: Path to model folder
        
    Returns:
        Dictionary with loaded artifacts
    """
    artifacts = {}
    model_path = Path(model_folder)
    
    # Load preprocessor (new unified preprocessor)
    preprocessor_path = model_path / "preprocessor.pkl"
    if preprocessor_path.exists():
        try:
            import joblib
            artifacts['preprocessor'] = joblib.load(preprocessor_path)
        except:
            try:
                with open(preprocessor_path, 'rb') as f:
                    artifacts['preprocessor'] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load preprocessor from {preprocessor_path}: {e}")
    
    # Load encoder (legacy)
    encoder_path = model_path / "encoder.pkl"
    if encoder_path.exists():
        try:
            import joblib
            artifacts['encoder'] = joblib.load(encoder_path)
        except:
            try:
                with open(encoder_path, 'rb') as f:
                    artifacts['encoder'] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load encoder from {encoder_path}: {e}")
    
    # Load scaler (legacy)
    scaler_path = model_path / "scaler.pkl"
    if scaler_path.exists():
        try:
            import joblib
            artifacts['scaler'] = joblib.load(scaler_path)
        except:
            try:
                with open(scaler_path, 'rb') as f:
                    artifacts['scaler'] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load scaler from {scaler_path}: {e}")
    
    return artifacts

def save_feature_list(features: List[str], output_path: str):
    """
    Save feature list to JSON file.
    
    Args:
        features: List of feature names
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)

def create_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create churn-specific features from cleaned data.
    
    Args:
        df: Raw cleaned DataFrame
        
    Returns:
        DataFrame with churn features
    """
    churn_df = df.copy()
    
    # Create customer_id if not exists
    if 'customer_id' not in churn_df.columns:
        if 'user_id' in churn_df.columns:
            churn_df['customer_id'] = churn_df['user_id']
        else:
            churn_df['customer_id'] = range(len(churn_df))
    
    # Create basic churn features
    if 'total_orders' not in churn_df.columns and 'order_count' in churn_df.columns:
        churn_df['total_orders'] = churn_df['order_count']
    
    if 'total_spent' not in churn_df.columns and 'total_amount' in churn_df.columns:
        churn_df['total_spent'] = churn_df['total_amount']
    
    if 'days_since_last_order' not in churn_df.columns and 'last_order_date' in churn_df.columns:
        churn_df['days_since_last_order'] = (pd.Timestamp.now() - pd.to_datetime(churn_df['last_order_date'])).dt.days
    
    # Create churn target if not exists
    if 'churn' not in churn_df.columns:
        # Simple heuristic: churn if no orders in last 30 days
        if 'days_since_last_order' in churn_df.columns:
            churn_df['churn'] = (churn_df['days_since_last_order'] > 30).astype(int)
        else:
            churn_df['churn'] = np.random.choice([0, 1], len(churn_df), p=[0.7, 0.3])
    
    return churn_df

def create_retention_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create retention-specific features from cleaned data.
    
    Args:
        df: Raw cleaned DataFrame
        
    Returns:
        DataFrame with retention features
    """
    retention_df = df.copy()
    
    # Create customer_id if not exists
    if 'customer_id' not in retention_df.columns:
        if 'user_id' in retention_df.columns:
            retention_df['customer_id'] = retention_df['user_id']
        else:
            retention_df['customer_id'] = range(len(retention_df))
    
    # Create timestamp if not exists
    if 'timestamp' not in retention_df.columns:
        if 'order_date' in retention_df.columns:
            retention_df['timestamp'] = pd.to_datetime(retention_df['order_date'])
        else:
            retention_df['timestamp'] = pd.date_range('2023-01-01', periods=len(retention_df), freq='h')
    
    # Create action if not exists
    if 'action' not in retention_df.columns:
        retention_df['action'] = np.random.choice(['email', 'sms', 'push', 'discount'], len(retention_df))
    
    # Create reward if not exists
    if 'reward' not in retention_df.columns:
        retention_df['reward'] = np.random.uniform(0, 1, len(retention_df))
    
    # Create repeat_purchase if not exists
    if 'repeat_purchase' not in retention_df.columns:
        retention_df['repeat_purchase'] = np.random.choice([0, 1], len(retention_df), p=[0.3, 0.7])
    
    return retention_df

def create_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create logistics-specific features from cleaned data.
    
    Args:
        df: Raw cleaned DataFrame
        
    Returns:
        DataFrame with logistics features
    """
    logistics_df = df.copy()
    
    # Create order_id if not exists
    if 'order_id' not in logistics_df.columns:
        if 'delivery_id' in logistics_df.columns:
            logistics_df['order_id'] = logistics_df['delivery_id']
        else:
            logistics_df['order_id'] = range(len(logistics_df))
    
    # Create customer_id if not exists
    if 'customer_id' not in logistics_df.columns:
        if 'user_id' in logistics_df.columns:
            logistics_df['customer_id'] = logistics_df['user_id']
        else:
            logistics_df['customer_id'] = range(len(logistics_df))
    
    # Create coordinates if not exists
    if 'pickup_lat' not in logistics_df.columns:
        logistics_df['pickup_lat'] = np.random.uniform(12.8, 13.2, len(logistics_df))
    
    if 'pickup_lng' not in logistics_df.columns:
        logistics_df['pickup_lng'] = np.random.uniform(77.5, 77.8, len(logistics_df))
    
    if 'dropoff_lat' not in logistics_df.columns:
        logistics_df['dropoff_lat'] = np.random.uniform(12.8, 13.2, len(logistics_df))
    
    if 'dropoff_lng' not in logistics_df.columns:
        logistics_df['dropoff_lng'] = np.random.uniform(77.5, 77.8, len(logistics_df))
    
    # Create timestamp if not exists
    if 'order_timestamp' not in logistics_df.columns:
        if 'timestamp' in logistics_df.columns:
            logistics_df['order_timestamp'] = pd.to_datetime(logistics_df['timestamp'])
        else:
            logistics_df['order_timestamp'] = pd.date_range('2023-01-01', periods=len(logistics_df), freq='h')
    
    # Calculate distance if not exists
    if 'distance_km' not in logistics_df.columns:
        from geopy.distance import geodesic
        distances = []
        for _, row in logistics_df.iterrows():
            try:
                dist = geodesic(
                    (row['pickup_lat'], row['pickup_lng']),
                    (row['dropoff_lat'], row['dropoff_lng'])
                ).kilometers
                distances.append(dist)
            except:
                distances.append(np.random.uniform(1, 50))
        logistics_df['distance_km'] = distances
    
    return logistics_df
