# scale_data.py
"""
scale_data.py
Version: 2025-03-17

Functions to create, save, load, and apply scalers so that
your training and daily pipelines use the exact same transformations.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def create_and_save_scalers(df, ticker, target_col='Close', exclude_cols=None):
    """
    Fits and saves two scalers:
      1) A features scaler for all numeric columns (except exclude_cols & target_col)
      2) A target scaler for the target_col (if you want to scale it separately)

    The scalers are saved in the "scalers" folder as:
      - {ticker}_features_scaler.pkl
      - {ticker}_target_scaler.pkl

    Parameters:
    -----------
    df : pd.DataFrame
        Your processed data (e.g., after adding technical indicators).
    ticker : str
        The stock ticker (e.g., "AAPL").
    target_col : str
        The column name for the prediction target (usually "Close").
    exclude_cols : list of str, optional
        Columns to exclude from scaling (e.g., sentiment columns).
    """
    if exclude_cols is None:
        # By default, exclude sentiment if you don't want to scale them
        exclude_cols = ['sentiment_polarity', 'sentiment_subjectivity']
    
    # Ensure we have a scalers folder
    os.makedirs("scalers", exist_ok=True)

    # Identify numeric columns to scale for features
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols + [target_col]]

    # 1) Fit a scaler for features
    features_scaler = MinMaxScaler()
    features_scaler.fit(df[feature_cols])

    # Save the features scaler
    features_scaler_path = os.path.join("scalers", f"{ticker}_features_scaler.pkl")
    joblib.dump(features_scaler, features_scaler_path)
    print(f"✅ Features scaler saved to {features_scaler_path}")

    # 2) Fit a separate scaler for the target (if you want to scale your target)
    #    If you do NOT want to scale the target, skip this.
    if target_col in df.columns:
        target_scaler = MinMaxScaler()
        target_scaler.fit(df[[target_col]])

        # Save the target scaler
        target_scaler_path = os.path.join("scalers", f"{ticker}_target_scaler.pkl")
        joblib.dump(target_scaler, target_scaler_path)
        print(f"✅ Target scaler saved to {target_scaler_path}")

def load_scalers(ticker):
    """
    Loads the feature scaler and target scaler (if it exists) for the given ticker
    from the "scalers" folder.

    Returns (features_scaler, target_scaler).
    If the target scaler does not exist, returns None for target_scaler.
    """
    features_scaler_path = os.path.join("scalers", f"{ticker}_features_scaler.pkl")
    target_scaler_path   = os.path.join("scalers", f"{ticker}_target_scaler.pkl")

    if not os.path.exists(features_scaler_path):
        print(f"⚠️ Features scaler not found for {ticker} at {features_scaler_path}.")
        return None, None
    
    features_scaler = joblib.load(features_scaler_path)
    print(f"✅ Loaded features scaler from {features_scaler_path}")

    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)
        print(f"✅ Loaded target scaler from {target_scaler_path}")
    else:
        target_scaler = None
        print(f"⚠️ Target scaler not found for {ticker}; continuing without it.")

    return features_scaler, target_scaler

def load_scaler_and_scale_data(df, ticker, exclude_cols=None):
    """
    Loads the pre-trained features scaler for the given ticker and applies it
    to the numeric columns in df (except those in exclude_cols).

    Returns the scaled DataFrame.
    (Does NOT handle the target column here—this is for features only.)

    Usage:
      scaled_df = load_scaler_and_scale_data(processed_df, "AAPL")

    If you have a separate target scaler, you can apply it to your target column
    in your training or inference code as needed.
    """
    if exclude_cols is None:
        exclude_cols = ['sentiment_polarity', 'sentiment_subjectivity']

    # Load the scalers
    features_scaler, _ = load_scalers(ticker)
    if features_scaler is None:
        return None

    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]
    
    df_scaled = df.copy()
    df_scaled[numeric_cols] = features_scaler.transform(df_scaled[numeric_cols])
    return df_scaled

def invert_target_scaling(predictions, ticker):
    """
    If you scaled your target column, you can invert-scale the model's predictions
    using the loaded target scaler.

    Example:
      preds_rescaled = invert_target_scaling(model_preds, "AAPL")
    """
    _, target_scaler = load_scalers(ticker)
    if target_scaler is None:
        print("⚠️ No target scaler found, returning predictions as-is.")
        return predictions
    preds_2d = predictions.reshape(-1, 1)
    return target_scaler.inverse_transform(preds_2d).flatten()
