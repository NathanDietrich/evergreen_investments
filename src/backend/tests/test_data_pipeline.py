"""
test_data_pipeline.py
Version: 2025-03-XX

This script tests the full data pipeline:
  1. Fetches raw stock & sentiment data for a given ticker for the last 120 days.
  2. Merges and processes data (calculates technical indicators and drops the Date column).
  3. Applies pre-trained scalers to the processed data.
  4. Saves the processed and scaled CSV files to src/backend/data.
  5. Prints out the shape and column names of the scaled DataFrame for verification.
  
Ensure your environment variable "Polygon_Key" is set.
"""

import os
import sys
import glob
import pandas as pd
import datetime
from dotenv import load_dotenv

# Add the parent directory (src/backend) to sys.path so modules in 'bot' can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables (assumes .env is in the project root)
load_dotenv()

# Import functions from your modules
from bot.data_fetcher import (
    fetch_stock_data_polygon,
    fetch_sentiment_data_polygon,
    analyze_sentiment,
    merge_stock_and_sentiment
)
from bot.process_data import calculate_technical_indicators
from bot.scale_data import apply_existing_scalers, invert_target_scaling

def test_data_pipeline(ticker="AAPL"):
    # Calculate the date range: last 120 days (ending yesterday)
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=120)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Fetching stock data for {ticker} from {start_str} to {end_str}...")
    
    api_key = os.getenv("Polygon_Key")
    if not api_key:
        print("Polygon API key not set!")
        return

    # 1. Fetch raw stock data
    stock_df = fetch_stock_data_polygon(ticker, start_str, end_str, api_key)
    if stock_df is None or stock_df.empty:
        print("❌ Stock data fetch failed.")
        return

    # 2. Fetch sentiment data
    print("Fetching sentiment data...")
    sentiment_raw = fetch_sentiment_data_polygon(ticker, start_str, end_str, api_key)
    sentiment_data = analyze_sentiment(sentiment_raw)

    # 3. Merge the stock and sentiment data
    print("Merging stock and sentiment data...")
    merged_df = merge_stock_and_sentiment(stock_df, sentiment_data)
    if merged_df is None or merged_df.empty:
        print("❌ Merged data is empty.")
        return

    # 4. Process data: calculate technical indicators and drop the Date column.
    print("Processing data (calculating technical indicators)...")
    processed_df = calculate_technical_indicators(merged_df.copy())
    processed_df.drop(columns=["Date"], inplace=True, errors="ignore")

    # Debug: Print number of rows after processing
    print(f"Processed data contains {len(processed_df)} rows.")

    # 5. Save the processed data in src/backend/data
    data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_folder, exist_ok=True)
    processed_filepath = os.path.join(data_folder, f"{ticker}_processed_test.csv")
    processed_df.to_csv(processed_filepath, index=False)
    print(f"✅ Processed data saved to {processed_filepath}")
    
    # 6. Apply existing scalers to the processed data
    df_scaled, scaler_target = apply_existing_scalers(processed_df, ticker)
    if df_scaled is None:
        print("Scaling failed.")
        return

    # Save the scaled data (update filename to indicate it's scaled)
    scaled_filepath = processed_filepath.replace("_processed_test", "_scaled_test")
    df_scaled.to_csv(scaled_filepath, index=False)
    print(f"✅ Scaled data saved to {scaled_filepath}")
    
    # 7. Load the scaled data and print shape and column information
    df_scaled_loaded = pd.read_csv(scaled_filepath)
    print("Scaled DataFrame shape:", df_scaled_loaded.shape)
    print("Scaled DataFrame columns:", df_scaled_loaded.columns.tolist())

if __name__ == "__main__":
    test_data_pipeline("AAPL")
