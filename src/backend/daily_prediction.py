# src/backend/daily_prediction.py
"""
daily_prediction.py
Version: 2025-03-17 (Updated for pure prediction)

This script fetches raw stock and sentiment data for a given ticker,
processes and scales the data (saving the scaled CSV locally in src/backend/data),
builds an input sequence from the latest 120 days of raw data (ensuring enough data remains after processing),
loads the pre-trained model for that ticker, and outputs a prediction
for tomorrow's Close using today's features.
"""

import os
import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()  # Loads .env from current directory

# Import modules using relative imports
from .bot.data_fetcher import (
    fetch_stock_data_polygon,
    fetch_sentiment_data_polygon,
    analyze_sentiment,
    merge_stock_and_sentiment
)
from .bot.process_data import process_data  # returns (X, y)
from .bot.scale_data import apply_existing_scalers, invert_target_scaling
from .bot.input_sequence import build_inference_sequence
from .bot.model_loading import load_model_for_ticker

# ----------------------------------------------------------------------------
# Compute the project root, so we can reliably locate "scalers/" at the root.
# This assumes your project layout is something like:
# evergreen_investments/
#   ├─ scalers/
#   ├─ src/
#       └─ backend/
#           └─ daily_prediction.py
# ----------------------------------------------------------------------------
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..', '..'))
SCALER_DIR = os.path.join(PROJECT_ROOT, 'scalers')

def update_and_save_data(ticker, days=120):
    """
    Fetches raw data for the past `days` (ending yesterday),
    processes it using process_data, and saves the processed CSV locally.
    Returns the processed DataFrame.
    Version: 2025-03-17
    """
    api_key = os.getenv("Polygon_Key")
    if not api_key:
        print("Polygon API key not set!")
        return None

    # Use a longer period: 120 days to ensure enough data after processing.
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Fetching stock data for {ticker} from {start_str} to {end_str}...")
    df_stock = fetch_stock_data_polygon(ticker, start_str, end_str, api_key)
    if df_stock is None or df_stock.empty:
        print("No stock data fetched.")
        return None

    print("Fetching sentiment data...")
    sentiment_raw = fetch_sentiment_data_polygon(ticker, start_str, end_str, api_key)
    sentiment_data = analyze_sentiment(sentiment_raw)

    print("Merging stock and sentiment data...")
    merged_df = merge_stock_and_sentiment(df_stock, sentiment_data)
    if merged_df is None or merged_df.empty:
        print("Merged data is empty.")
        return None

    print("Processing data (calculating technical indicators)...")
    X, y = process_data(merged_df)
    if X is None or X.empty:
        print("Processed features DataFrame is empty.")
        return None

    # Save processed CSV in src/backend/data (relative to this file)
    data_folder = os.path.join(THIS_FILE_DIR, 'data')
    os.makedirs(data_folder, exist_ok=True)
    processed_filepath = os.path.join(data_folder, f"{ticker}_processed.csv")
    X.to_csv(processed_filepath, index=False)
    print(f"Processed data saved to {processed_filepath}")
    return X

def predict_next_close(ticker, sequence_length=60):
    """
    Updates the data (fetch, process, scale), builds an inference sequence from the latest data,
    loads the saved model, and predicts tomorrow's Close.
    Version: 2025-03-17
    """
    # Use a longer fetch period (120 days) to ensure there is enough data for a 60-day sequence.
    X_processed = update_and_save_data(ticker, days=120)
    if X_processed is None:
        print("Failed to update data.")
        return None

    # Load existing scalers (features + target) and apply them to the processed data
    df_scaled, target_scaler = apply_existing_scalers(
        X_processed, 
        ticker, 
        scaler_dir=SCALER_DIR
    )
    if df_scaled is None:
        print("Scaling failed.")
        return None

    # Feature columns (must match training, including 'Close')
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI',
        'MACD', 'MACD_Signal', 'sentiment_polarity', 'sentiment_subjectivity'
    ]
    missing = [col for col in feature_cols if col not in df_scaled.columns]
    if missing:
        print("Feature mismatch! Missing:", missing)
        return None

    # Build inference sequence using the last 'sequence_length' days
    input_seq = build_inference_sequence(df_scaled, feature_cols, sequence_length)
    print(f"Input shape for model: {input_seq.shape}")  # Expected: (1, 60, 14)

    # Load the saved model
    model = load_model_for_ticker(ticker)
    if model is None:
        print("Model could not be loaded.")
        return None

    # Predict tomorrow's Close (currently in scaled space)
    scaled_prediction = model.predict(input_seq)

    # Now invert the scaling of that prediction using the target scaler at SCALER_DIR
    predicted_close = invert_target_scaling(
        scaled_prediction,
        ticker,
        scaler_dir=SCALER_DIR
    )

    # In case predicted_close is a numpy array, cast to float for printing
    predicted_close_value = float(predicted_close[0])
    print(f"Predicted tomorrow's Close for {ticker}: {predicted_close_value:.4f}")
    return predicted_close_value

def main():
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("No ticker provided.")
        return
    predict_next_close(ticker)

if __name__ == "__main__":
    main()
