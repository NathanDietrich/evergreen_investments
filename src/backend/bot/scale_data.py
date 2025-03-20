# src/backend/bot/scale_data.py
"""
scale_data.py
Version: 2025-03-17

Contains functions to load and apply pre-trained scalers for features and target.
Scalers are expected to be saved as:
  {ticker}_features_scaler.pkl and {ticker}_target_scaler.pkl
in the local "scalers" directory.
The function apply_existing_scalers() applies these to a given DataFrame.
"""

import os
import joblib
import numpy as np
import pandas as pd

def apply_existing_scalers(df, ticker, target_col='Close', exclude_cols=['sentiment_polarity', 'sentiment_subjectivity']):
    """
    Loads pre-trained scalers from the local "scalers" folder and applies them to df.
    Assumes scalers are named:
      {ticker}_features_scaler.pkl and {ticker}_target_scaler.pkl
    Returns:
      (df_scaled, scaler_target)
    """
    scaler_features_path = os.path.join("src", "backend", "scalers", f"{ticker}_features_scaler.pkl")
    scaler_target_path = os.path.join("src", "backend", "scalers", f"{ticker}_target_scaler.pkl")

    
    if not os.path.exists(scaler_features_path):
        print(f"⚠️ Feature scaler not found for {ticker} at {scaler_features_path}.")
        return None, None
    if not os.path.exists(scaler_target_path):
        print(f"⚠️ Target scaler not found for {ticker} at {scaler_target_path}.")
        return None, None
        
    scaler_features = joblib.load(scaler_features_path)
    scaler_target = joblib.load(scaler_target_path)
    
    numeric_cols = [col for col in df.select_dtypes(include=['float64','int64']).columns if col not in exclude_cols]
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler_features.transform(df_scaled[numeric_cols])
    return df_scaled, scaler_target

def invert_target_scaling(predictions, ticker):
    """
    Loads the pre-trained target scaler and inversely transforms predictions.
    """
    scaler_target_path = os.path.join("scalers", f"{ticker}_target_scaler.pkl")
    if not os.path.exists(scaler_target_path):
        print(f"⚠️ Target scaler not found for {ticker} at {scaler_target_path}. Returning predictions as-is.")
        return predictions.flatten()
    scaler_target = joblib.load(scaler_target_path)
    predictions_2d = np.array(predictions).reshape(-1, 1)
    inverted = scaler_target.inverse_transform(predictions_2d)
    return inverted.flatten()
