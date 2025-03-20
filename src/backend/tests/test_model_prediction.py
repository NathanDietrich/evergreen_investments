"""
test_model_prediction.py
Version: 2025-03-XX

This script tests the prediction pipeline:
  1. Loads a scaled CSV from src/backend/data (e.g. AAPL_scaled_test.csv).
  2. Uses build_inference_sequence from bot/input_sequence.py to build an input sequence.
  3. Loads the pre-trained model using load_model_for_ticker from bot/model_loading.py.
  4. Makes a prediction and prints the scaled prediction.

Ensure that your scaled CSV file exists and that your .env is loaded if needed.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add the parent directory (src/backend) to sys.path so that modules in 'bot' can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables (if any)
load_dotenv()

from bot.input_sequence import build_inference_sequence
from bot.model_loading import load_model_for_ticker

def test_prediction(ticker="AAPL", sequence_length=60):
    # Path to the scaled CSV file (make sure the file is named correctly)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{ticker}_scaled_test.csv")
    if not os.path.exists(data_path):
        print(f"Scaled data file not found: {data_path}")
        return
    
    df_scaled = pd.read_csv(data_path)
    print(f"Loaded scaled data from {data_path}, shape: {df_scaled.shape}")

    # Define feature columns (must match training)
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI',
        'MACD', 'MACD_Signal', 'sentiment_polarity', 'sentiment_subjectivity'
    ]
    
    # Build inference sequence using the last 'sequence_length' days of data
    try:
        input_seq = build_inference_sequence(df_scaled, feature_cols, sequence_length)
    except ValueError as e:
        print("Error building inference sequence:", e)
        return

    print("Input sequence shape:", input_seq.shape)  # Expected: (1, sequence_length, 14)

    # Load the pre-trained model for the ticker
    model = load_model_for_ticker(ticker)
    if model is None:
        print(f"Failed to load model for {ticker}.")
        return

    # Use the model to make a prediction
    prediction = model.predict(input_seq)
    print("Scaled prediction:", prediction)
    return prediction

if __name__ == "__main__":
    test_prediction("AAPL", sequence_length=60)
