# src/backend/bot/scale_data.py
"""
scale_data.py
Version: 2025-03-17

This module contains functions to:
  1. Fit new scalers on processed data.
  2. Save the fitted scalers as .pkl files locally (in the 'scalers' folder).
  3. Apply existing scalers to new data.
  4. Invert the scaling on predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_and_save_scalers(df, ticker, target_col='CloseTomorrow', exclude_cols=['sentiment_polarity', 'sentiment_subjectivity'], scaler_dir=None):
    """
    Fits MinMaxScalers for the features and for the target on the given DataFrame,
    then saves them as {ticker}_features_scaler.pkl and {ticker}_target_scaler.pkl
    in the provided scaler_dir (or in the local "scalers" folder if not provided).
    
    Returns:
      df_scaled: DataFrame with scaled features and target.
      scaler_features, scaler_target: the fitted scaler objects.
      
    Note: Exclude target_col from the features scaler so that it is only scaled by the target scaler.
    """
    if scaler_dir is None:
        scaler_dir = os.path.join(os.getcwd(), "scalers")
    os.makedirs(scaler_dir, exist_ok=True)
    
    # Feature columns: all numeric columns except exclude_cols and target_col
    numeric_cols = [col for col in df.select_dtypes(include=['float64','int64']).columns if col not in exclude_cols + [target_col]]
    
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    df_scaled = df.copy()
    # Scale features (excluding target_col)
    df_scaled[numeric_cols] = scaler_features.fit_transform(df_scaled[numeric_cols])
    # Scale target_col only with target scaler
    df_scaled[target_col] = scaler_target.fit_transform(df_scaled[[target_col]])
    
    features_scaler_path = os.path.join(scaler_dir, f"{ticker}_features_scaler.pkl")
    target_scaler_path = os.path.join(scaler_dir, f"{ticker}_target_scaler.pkl")
    
    joblib.dump(scaler_features, features_scaler_path)
    joblib.dump(scaler_target, target_scaler_path)
    
    print(f"✅ Feature scaler saved to: {features_scaler_path}")
    print(f"✅ Target scaler saved to: {target_scaler_path}")
    
    return df_scaled, scaler_features, scaler_target

def apply_existing_scalers(df, ticker, target_col='CloseTomorrow', exclude_cols=['sentiment_polarity', 'sentiment_subjectivity'], scaler_dir=None):
    """
    Loads pre-fitted scalers from the provided scaler_dir (or the local "scalers" folder by default)
    and applies them to df.
    Assumes scalers are saved as:
      {ticker}_features_scaler.pkl and {ticker}_target_scaler.pkl.
    Returns:
      (df_scaled, scaler_target)
      
    Note: The features scaler is applied to all numeric columns except those in exclude_cols and the target column.
    """
    if scaler_dir is None:
        scaler_dir = os.path.join(os.getcwd(), "scalers")
    
    features_scaler_path = os.path.join(scaler_dir, f"{ticker}_features_scaler.pkl")
    target_scaler_path = os.path.join(scaler_dir, f"{ticker}_target_scaler.pkl")
    
    if not os.path.exists(features_scaler_path):
        print(f"⚠️ Feature scaler not found for {ticker} at {features_scaler_path}.")
        return None, None
    if not os.path.exists(target_scaler_path):
        print(f"⚠️ Target scaler not found for {ticker} at {target_scaler_path}.")
        return None, None
    
    scaler_features = joblib.load(features_scaler_path)
    scaler_target = joblib.load(target_scaler_path)
    
    # Use the same feature columns as during scaler fitting: all numeric columns except exclude_cols and target_col.
    numeric_cols = [col for col in df.select_dtypes(include=['float64','int64']).columns if col not in exclude_cols + [target_col]]
    
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler_features.transform(df_scaled[numeric_cols])
    
    return df_scaled, scaler_target

def invert_target_scaling(predictions, ticker, scaler_dir=None):
    """
    Loads the pre-fitted target scaler from the provided scaler_dir (or the local "scalers" folder by default)
    and inversely transforms predictions.
    Returns a flattened array of predictions in the original scale.
    """
    if scaler_dir is None:
        scaler_dir = os.path.join(os.getcwd(), "scalers")
    
    scaler_target_path = os.path.join(scaler_dir, f"{ticker}_target_scaler.pkl")
    if not os.path.exists(scaler_target_path):
        print(f"⚠️ Target scaler not found for {ticker} at {scaler_target_path}. Returning predictions as-is.")
        return predictions.flatten()
    scaler_target = joblib.load(scaler_target_path)
    predictions_2d = np.array(predictions).reshape(-1, 1)
    inverted = scaler_target.inverse_transform(predictions_2d)
    return inverted.flatten()
