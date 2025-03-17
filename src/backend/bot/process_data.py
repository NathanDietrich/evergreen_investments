# src/backend/services/process_data.py

import pandas as pd

def process_data(raw_data: dict):
    """
    Process raw data by merging historical stock data with sentiment data,
    then applying a scaling function.
    
    Args:
        raw_data (dict): A dictionary containing:
            - "historical_data": Expected to be a dict with a "results" key.
            - "sentiment_data": Expected to be a dict with a "results" key.
            - "market_open": A flag (0 or 1).
            
    Returns:
        dict: A dictionary containing the processed data and market flag.
    """
    # Extract historical data from raw_data
    historical_results = raw_data.get("historical_data", {}).get("results", [])
    if historical_results:
        hist_df = pd.DataFrame(historical_results)
        # Convert the timestamp 't' (in ms) to a datetime and extract date only
        hist_df['date'] = pd.to_datetime(hist_df['t'], unit='ms')
        hist_df['date_only'] = hist_df['date'].dt.date
    else:
        hist_df = pd.DataFrame()
    
    # Extract sentiment data from raw_data
    sentiment_results = raw_data.get("sentiment_data", {}).get("results", [])
    if sentiment_results:
        sent_df = pd.DataFrame(sentiment_results)
        # Assuming sentiment data includes a published date, e.g., 'published_utc'
        if 'published_utc' in sent_df.columns:
            sent_df['date'] = pd.to_datetime(sent_df['published_utc'])
            sent_df['date_only'] = sent_df['date'].dt.date
    else:
        sent_df = pd.DataFrame()
    
    # Merge the historical and sentiment data on the date (if both exist)
    if not hist_df.empty and not sent_df.empty:
        merged_df = pd.merge(hist_df, sent_df, on='date_only', how='left', suffixes=('_hist', '_sent'))
    else:
        # If sentiment data is empty, just use historical data
        merged_df = hist_df.copy()

    # Now, pass the merged data to a scaling function (dummy for now)
    scaled_data = scale_data(merged_df)
    
    return {
        "processed_data": scaled_data.to_dict(orient="records"),
        "market_open": raw_data.get("market_open", 0)
    }

def scale_data(df: pd.DataFrame):
    """
    Dummy scaling function.
    In the future, you'll load saved scalers and apply them.
    For now, this function just returns the original DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to scale.
        
    Returns:
        pd.DataFrame: The (unscaled) DataFrame.
    """
    # Here you can eventually load and apply a scaler, e.g.:
    # from joblib import load
    # scaler = load("path/to/scaler.joblib")
    # df[["o", "c", "h", "l"]] = scaler.transform(df[["o", "c", "h", "l"]])
    return df
