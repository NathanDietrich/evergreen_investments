# process_data.py
"""
process_data.py
Version: 2025-03-16

Contains functions for computing technical indicators and
forward-filling/back-filling missing values (matching the training pipeline).
"""

import pandas as pd

def calculate_technical_indicators(df):
    """
    Calculates the technical indicators that match the training pipeline:
      - SMA_10, SMA_20
      - EMA_10, EMA_20
      - RSI
      - MACD, MACD_Signal
    
    Leaves 'Open', 'High', 'Low', 'Close', 'Volume' intact.
    Also preserves any sentiment columns that might be present
    ('sentiment_polarity', 'sentiment_subjectivity').

    Instead of dropping rows with NaN, we forward-fill and back-fill
    so the pipeline can handle data outages.

    Final columns should include:
      'Open', 'High', 'Low', 'Close', 'Volume',
      'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI',
      'MACD', 'MACD_Signal', 'sentiment_polarity', 'sentiment_subjectivity'
    """
    if df is None or df.empty:
        return df

    # Calculate SMAs
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()

    # Calculate EMAs
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)  # avoid div-by-zero
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Forward fill and back fill to handle missing data
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df
