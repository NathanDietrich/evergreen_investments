# backfill.py
import os
import datetime
import pandas as pd
from .daily_prediction import prepare_data_once, predict_next_close_with_prefetch

def get_last_logged_date(ticker, log_filepath):
    """
    Returns the latest logged date (as a datetime.date) for the given ticker.
    If no entry exists, returns None.
    """
    try:
        log_df = pd.read_csv(log_filepath)
    except FileNotFoundError:
        return None
    
    ticker_logs = log_df[log_df["ticker"] == ticker]
    if ticker_logs.empty:
        return None
    
    # Convert the 'timestamp' column to dates and return the max date.
    ticker_logs["timestamp"] = pd.to_datetime(ticker_logs["timestamp"]).dt.date
    return ticker_logs["timestamp"].max()

def backfill_predictions_for_ticker(ticker, target_end_date, default_start_date=None):
    """
    Backfill predictions for a given ticker starting from the day after the last logged date 
    (or default_start_date if no log exists) up to target_end_date (inclusive).
    """
    # Use the same data folder as daily_prediction
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_folder, exist_ok=True)
    log_filepath = os.path.join(data_folder, "daily_predictions_log.csv")
    
    last_date = get_last_logged_date(ticker, log_filepath)
    if last_date:
        start_date = last_date + datetime.timedelta(days=1)
        print(f"[{ticker}] Last logged date found: {last_date}. Backfilling from {start_date}...")
    else:
        if default_start_date is None:
            default_start_date = target_end_date - datetime.timedelta(days=30)
        start_date = default_start_date
        print(f"[{ticker}] No previous log found. Starting backfill from default start date: {start_date}...")
    
    if start_date > target_end_date:
        print(f"[{ticker}] No backfill needed. Last logged date {last_date} is on or after target end date {target_end_date}.")
        return
    
    # Pre-fetch full data (e.g., for 150 days) once for the ticker.
    full_df = prepare_data_once(ticker, total_days=150)
    if full_df is None:
        print(f"[{ticker}] Unable to prefetch data. Skipping backfill.")
        return
    
    # Loop through each date from start_date to target_end_date (inclusive)
    current_date = start_date
    while current_date <= target_end_date:
        date_str = current_date.isoformat()
        print(f"[{ticker}] Backfilling prediction for date: {date_str}")
        try:
            predict_next_close_with_prefetch(ticker, full_df, simulate_date=date_str)
        except Exception as e:
            print(f"[{ticker}] Error backfilling for {date_str}: {e}")
        current_date += datetime.timedelta(days=1)

def sort_log_by_date():
    """
    Sorts the daily_predictions_log.csv file by the timestamp column in ascending order.
    """
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    log_filepath = os.path.join(data_folder, "daily_predictions_log.csv")
    try:
        log_df = pd.read_csv(log_filepath)
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        log_df = log_df.sort_values("timestamp")
        log_df.to_csv(log_filepath, index=False)
        print(f"Sorted the log file at {log_filepath}.")
    except FileNotFoundError:
        print("Log file not found. Nothing to sort.")

if __name__ == "__main__":
    # Define the tickers to backfill
    tickers = ["AAPL", "AMZN", "MSFT", "QQQ", "SPY"]
    
    # Define the target end date for backfilling.
    # For example, set the target as yesterday’s date:
    target_end_date = datetime.date.today() - datetime.timedelta(days=1)
    
    # Optionally, define a default start date if no log exists (e.g., 30 days before target_end_date)
    default_start_date = target_end_date - datetime.timedelta(days=30)
    
    # Backfill each ticker
    for ticker in tickers:
        print(f"\nStarting backfill for {ticker}...")
        backfill_predictions_for_ticker(ticker, target_end_date, default_start_date=default_start_date)
    
    # Finally, sort the log file by date.
    sort_log_by_date()
