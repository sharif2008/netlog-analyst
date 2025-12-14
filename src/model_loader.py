"""
Model loader utility for NIDS predictions.
Uses the RF CICIDS2017 model from the models folder.
"""
import os
import joblib
import numpy as np
import pandas as pd

# Model paths (relative to project root, not src folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_cicids2017_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'rf_feature_metadata.pkl')

# Cache for model and metadata
_model_cache = None
_metadata_cache = None

def load_model():
    """
    Load the trained model and metadata from files.
    
    Returns:
        tuple: (model, metadata) - Loaded model object and metadata dict
    """
    global _model_cache, _metadata_cache
    
    if _model_cache is not None and _metadata_cache is not None:
        return _model_cache, _metadata_cache
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please ensure the model file exists in the models folder."
        )
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. "
            "Please ensure the metadata file exists in the models folder."
        )
    
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH)
        _model_cache = model
        _metadata_cache = metadata
        return model, metadata
    except Exception as e:
        raise Exception(f"Error loading model or metadata: {str(e)}")

def predict(model, df, metadata=None):
    """
    Make predictions on the dataframe using the RF CICIDS2017 model.
    
    Args:
        model: Loaded model object
        df: pandas DataFrame with features
        metadata: Metadata dict with feature information (optional, will load if not provided)
    
    Returns:
        tuple: (predictions, probabilities, valid_indices) - List of predictions, probabilities, and valid row indices
    """
    if metadata is None:
        _, metadata = load_model()
    
    # Validate metadata structure
    if not isinstance(metadata, dict):
        raise TypeError(f"Metadata must be a dictionary, got {type(metadata)}")
    
    # Get feature information from metadata with error handling
    try:
        selected_features = metadata.get("selected_features")
        selected_mask = metadata.get("selected_mask")
        base_features_raw = metadata.get("base_features")
        
        if base_features_raw is None:
            raise KeyError("'base_features' key not found in metadata")
        if selected_mask is None:
            raise KeyError("'selected_mask' key not found in metadata")
            
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid metadata structure: {str(e)}. Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}")
    
    # Normalize column names (strip spaces) but KEEP ORDER
    if isinstance(base_features_raw, list):
        base_features = [str(c).strip() for c in base_features_raw]
    else:
        raise TypeError(f"base_features must be a list, got {type(base_features_raw)}")
    
    # Normalize column names in the CSV (strip leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Check that all required base features exist in the CSV
    missing = [col for col in base_features if col not in df.columns]
    if missing:
        raise ValueError(
            f"The following required base features are missing from your CSV: {', '.join(missing[:10])}"
            + (f" and {len(missing) - 10} more..." if len(missing) > 10 else "")
        )
    
    # Keep ONLY the base features (drops Label and any other extra columns)
    df_base = df[base_features].copy()
    
    # Clean infinities / NaNs
    df_base = df_base.replace([np.inf, -np.inf], np.nan)
    
    # Store original index to track which rows were dropped
    original_indices = df_base.index.tolist()
    df_base = df_base.dropna()
    
    if df_base.empty:
        raise ValueError("After cleaning NaN/inf, no rows remain to predict on.")
    
    # Convert to numpy and apply same feature-selection mask used at training
    X_base = df_base.values              # shape: (n_samples, len(base_features))
    
    # Handle selected_mask - it could be boolean array, integer indices, or feature names
    if isinstance(selected_mask, (list, np.ndarray)):
        # Convert to numpy array if it's a list
        selected_mask = np.array(selected_mask)
        
        # Check if it's boolean mask or integer indices
        if selected_mask.dtype == bool:
            # Boolean mask - use directly
            X_sel = X_base[:, selected_mask]
        elif selected_mask.dtype in [np.int32, np.int64, int]:
            # Integer indices - use directly
            X_sel = X_base[:, selected_mask]
        else:
            # Try to convert to boolean if it contains 0/1
            try:
                selected_mask = selected_mask.astype(bool)
                X_sel = X_base[:, selected_mask]
            except:
                raise ValueError(f"selected_mask must be boolean array or integer indices, got dtype: {selected_mask.dtype}")
    else:
        raise TypeError(f"selected_mask must be list or numpy array, got {type(selected_mask)}")
    
    # Make predictions
    try:
        preds = model.predict(X_sel)
        proba = model.predict_proba(X_sel)
    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")
    
    # Convert predictions to 'normal' or 'attack'
    # Model outputs: 0 = BENIGN, 1 = ATTACK
    predictions = ['attack' if pred == 1 else 'normal' for pred in preds]
    
    # Get probabilities for attack (class 1)
    attack_probabilities = proba[:, 1].tolist() if proba.shape[1] > 1 else [0.0] * len(predictions)
    
    return predictions, attack_probabilities, df_base.index.tolist()

