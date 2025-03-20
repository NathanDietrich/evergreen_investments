# src/backend/bot/process_data.py
import pandas as pd

def calculate_technical_indicators(df):
    """
    Adds technical indicators: SMA_10, SMA_20, EMA_10, EMA_20, RSI, MACD, and MACD_Signal.
    Drops rows with NaN values.
    Version: 2025-03-17
    """
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def process_data(raw_df):
    """
    Processes raw data by calculating technical indicators, dropping the Date column,
    and splitting into features and target.
    Version: 2025-03-17
    Returns:
      X: DataFrame with features (includes today's values, including 'Close')
      y: Series with target (tomorrow's Close)
    """
    df = raw_df.copy()
    df = calculate_technical_indicators(df)
    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)
    # X includes all features (including today's Close)
    y = df['Close'].copy()
    X = df.copy()
    return X, y
