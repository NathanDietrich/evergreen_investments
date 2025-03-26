"""
daily_prediction.py
Version: 2025-03-17 (Updated for more efficient simulations)

This script fetches raw stock and sentiment data for a given ticker,
processes/scales data, outputs a prediction for tomorrow's Close,
and logs the result to a CSV. It also includes functions for
simulating multiple days using a single pre-fetched DataFrame.
"""

import os
import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()  # Loads .env from current directory

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

# ---------------------------------------------------------------------
# 1) PRE-EXISTING FUNCTIONS (unchanged except docstrings)
# ---------------------------------------------------------------------
def update_and_save_data(ticker, days=120, override_end_date=None):
    """
    Fetches raw data for the past `days`, processes it, and saves to CSV.
    (Used for normal single-day usage.)
    """
    api_key = os.getenv("Polygon_Key")
    if not api_key:
        print("Polygon API key not set!")
        return None

    # Determine end_date
    if override_end_date:
        if isinstance(override_end_date, str):
            end_date = datetime.datetime.strptime(override_end_date, "%Y-%m-%d").date()
        else:
            end_date = override_end_date
    else:
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

    # Save processed CSV
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_folder, exist_ok=True)
    processed_filepath = os.path.join(data_folder, f"{ticker}_processed.csv")
    X.to_csv(processed_filepath, index=False)
    print(f"Processed data saved to {processed_filepath}")
    return X

def log_prediction(ticker, predicted_close, processed_df, timestamp_override=None):
    """
    Logs prediction details to daily_predictions_log.csv.
    If timestamp_override is provided (e.g., simulated date),
    it uses that as the 'timestamp' instead of now().
    """
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_folder, exist_ok=True)
    log_filepath = os.path.join(data_folder, "daily_predictions_log.csv")
    
    try:
        log_df = pd.read_csv(log_filepath)
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=[
            "timestamp", "ticker", "predicted_close", "direction",
            "sentiment_polarity", "sentiment_subjectivity", "historical_close"
        ])
    
    previous_pred = None
    ticker_logs = log_df[log_df["ticker"] == ticker]
    if not ticker_logs.empty:
        previous_pred = ticker_logs.iloc[-1]["predicted_close"]

    if previous_pred is None:
        direction = "N/A"
    else:
        if predicted_close > previous_pred:
            direction = "up"
        elif predicted_close < previous_pred:
            direction = "down"
        else:
            direction = "no change"
    
    last_row = processed_df.iloc[-1]
    sentiment_polarity = last_row.get("sentiment_polarity", None)
    sentiment_subjectivity = last_row.get("sentiment_subjectivity", None)
    historical_close = last_row.get("Close", None)
    
    if timestamp_override:
        if isinstance(timestamp_override, datetime.date):
            ts_str = timestamp_override.isoformat()
        elif isinstance(timestamp_override, str):
            ts_str = timestamp_override
        else:
            ts_str = datetime.datetime.now().isoformat()
    else:
        ts_str = datetime.datetime.now().isoformat()

    new_entry = {
        "timestamp": ts_str,
        "ticker": ticker,
        "predicted_close": predicted_close,
        "direction": direction,
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
        "historical_close": historical_close
    }
    
    new_entry_df = pd.DataFrame([new_entry])
    log_df = pd.concat([log_df, new_entry_df], ignore_index=True)
    log_df.to_csv(log_filepath, index=False)
    print(f"Logged prediction for {ticker} to {log_filepath}")

def predict_next_close(ticker, sequence_length=60, override_end_date=None):
    """
    Normal usage: fetch for 'days=120' ending at override_end_date or yesterday,
    process/scale, run inference, log with current timestamp.
    """
    X_processed = update_and_save_data(ticker, days=120, override_end_date=override_end_date)
    if X_processed is None:
        print("Failed to update data.")
        return None

    df_scaled, target_scaler = apply_existing_scalers(X_processed, ticker)
    if df_scaled is None:
        print("Scaling failed.")
        return None

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI',
        'MACD', 'MACD_Signal', 'sentiment_polarity', 'sentiment_subjectivity'
    ]
    missing = [col for col in feature_cols if col not in df_scaled.columns]
    if missing:
        print("Feature mismatch! Missing:", missing)
        return None

    input_seq = build_inference_sequence(df_scaled, feature_cols, sequence_length)
    print(f"Input shape for model: {input_seq.shape}")

    model = load_model_for_ticker(ticker)
    if model is None:
        print("Model could not be loaded.")
        return None

    scaled_prediction = model.predict(input_seq)
    predicted_close = invert_target_scaling(scaled_prediction, ticker)
    predicted_close_value = predicted_close[0]
    print(f"Predicted tomorrow's Close for {ticker}: {predicted_close_value:.4f}")

    # Log prediction using override_end_date as timestamp if provided
    log_prediction(ticker, predicted_close_value, X_processed, timestamp_override=override_end_date)
    return predicted_close_value

# ---------------------------------------------------------------------
# 2) NEW FUNCTIONS FOR MORE EFFICIENT SIMULATIONS
# ---------------------------------------------------------------------
def prepare_data_once(ticker, total_days=150):
    """
    Fetches raw data (stock + sentiment) for total_days, merges,
    returns the merged raw DataFrame (unprocessed). This is done ONCE
    for more efficient batch simulations.
    """
    api_key = os.getenv("Polygon_Key")
    if not api_key:
        print("Polygon API key not set!")
        return None

    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=total_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"[prepare_data_once] Fetching stock data for {ticker}, {start_str} to {end_str}")
    df_stock = fetch_stock_data_polygon(ticker, start_str, end_str, api_key)
    if df_stock is None or df_stock.empty:
        print("No stock data fetched.")
        return None

    print("[prepare_data_once] Fetching sentiment data...")
    sentiment_raw = fetch_sentiment_data_polygon(ticker, start_str, end_str, api_key)
    sentiment_data = analyze_sentiment(sentiment_raw)

    print("[prepare_data_once] Merging stock & sentiment...")
    merged_df = merge_stock_and_sentiment(df_stock, sentiment_data)
    if merged_df is None or merged_df.empty:
        print("Merged data is empty.")
        return None

    return merged_df

def predict_next_close_with_prefetch(ticker, full_df, simulate_date, sequence_length=60):
    """
    Efficient simulation:
      - Takes a full pre-fetched raw DataFrame (covering 150 days).
      - Slices it up to simulate_date.
      - Processes/scales the data, runs inference, logs with that date as timestamp.
    """
    # 1. Slice the data up to simulate_date
    if isinstance(simulate_date, datetime.date):
        cutoff_dt = simulate_date
    else:
        cutoff_dt = datetime.datetime.strptime(simulate_date, "%Y-%m-%d").date()
    
    if "Date" in full_df.columns and not pd.api.types.is_datetime64_any_dtype(full_df["Date"]):
        full_df["Date"] = pd.to_datetime(full_df["Date"]).dt.date

    if "Date" in full_df.columns:
        sim_df = full_df[full_df["Date"] <= cutoff_dt].copy()
    else:
        print("No 'Date' column found, cannot slice data for simulation.")
        return None

    if sim_df.empty:
        print(f"No data available up to {simulate_date} for {ticker}.")
        return None

    # 2. Process data
    X, y = process_data(sim_df)
    if X is None or X.empty:
        print("Processed DataFrame is empty after slicing.")
        return None

    # 3. Scale
    df_scaled, target_scaler = apply_existing_scalers(X, ticker)
    if df_scaled is None:
        print("Scaling failed.")
        return None

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI',
        'MACD', 'MACD_Signal', 'sentiment_polarity', 'sentiment_subjectivity'
    ]
    missing = [col for col in feature_cols if col not in df_scaled.columns]
    if missing:
        print("Feature mismatch! Missing:", missing)
        return None

    # 4. Build inference sequence
    if len(df_scaled) < sequence_length:
        print(f"Not enough rows ({len(df_scaled)}) to build a {sequence_length}-day sequence.")
        return None

    input_seq = build_inference_sequence(df_scaled, feature_cols, sequence_length)
    print(f"[predict_next_close_with_prefetch] Input shape: {input_seq.shape}")

    # 5. Load model & predict
    model = load_model_for_ticker(ticker)
    if model is None:
        print("Model could not be loaded.")
        return None

    scaled_prediction = model.predict(input_seq)
    predicted_close = invert_target_scaling(scaled_prediction, ticker)
    predicted_close_value = predicted_close[0]
    print(f"[simulate] Predicted tomorrow's Close for {ticker} on {simulate_date}: {predicted_close_value:.4f}")

    # 6. Log result with simulate_date as timestamp
    log_prediction(ticker, predicted_close_value, X, timestamp_override=simulate_date)
    return predicted_close_value

def main():
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("No ticker provided.")
        return
    predict_next_close(ticker)

if __name__ == "__main__":
    main()
