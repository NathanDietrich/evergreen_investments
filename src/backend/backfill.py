"""
backfill.py

This module handles the backfilling of daily prediction logs for a set of stock tickers.
It reads the existing prediction log, determines the last logged date for each ticker,
and then generates predictions for any missing dates up to a specified target end date.
The module also provides a utility to sort the log file by date.
"""

import os
import datetime
import pandas as pd
from .daily_prediction import prepare_data_once, predict_next_close_with_prefetch


def get_last_logged_date(ticker, log_filepath):
    """
    Retrieve the most recent logged date for the specified ticker from the prediction log.

    Parameters:
        ticker (str): The stock symbol to search in the log.
        log_filepath (str): The file path to the CSV prediction log.

    Returns:
        datetime.date or None: The latest date found for the ticker, or None if no entry exists.
    """
    try:
        log_df = pd.read_csv(log_filepath)
    except FileNotFoundError:
        # If the log file doesn't exist, return None.
        return None

    # Filter the log for entries matching the given ticker.
    ticker_logs = log_df[log_df["ticker"] == ticker]
    if ticker_logs.empty:
        return None

    # Convert the 'timestamp' column to date objects and return the most recent date.
    ticker_logs["timestamp"] = pd.to_datetime(ticker_logs["timestamp"]).dt.date
    return ticker_logs["timestamp"].max()


def backfill_predictions_for_ticker(ticker, target_end_date, default_start_date=None):
    """
    Backfill prediction logs for a given ticker starting from the day after the last logged date
    (or a specified default start date if no log exists) up to the target end date (inclusive).

    Parameters:
        ticker (str): The stock symbol to backfill predictions for.
        target_end_date (datetime.date): The final date for which predictions should be generated.
        default_start_date (datetime.date, optional): The start date to use if no previous log exists.
                                                      Defaults to 30 days before target_end_date.
    """
    # Determine the data folder location (same folder used by daily_prediction) and ensure it exists.
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_folder, exist_ok=True)
    log_filepath = os.path.join(data_folder, "daily_predictions_log.csv")

    # Retrieve the most recent logged date for this ticker.
    last_date = get_last_logged_date(ticker, log_filepath)
    if last_date:
        # If a log exists, begin backfilling from the day after the last logged date.
        start_date = last_date + datetime.timedelta(days=1)
        print(f"[{ticker}] Last logged date found: {last_date}. Backfilling from {start_date}...")
    else:
        # If no log exists, use the default start date, or set it to 30 days before the target if not provided.
        if default_start_date is None:
            default_start_date = target_end_date - datetime.timedelta(days=30)
        start_date = default_start_date
        print(f"[{ticker}] No previous log found. Starting backfill from default start date: {start_date}...")

    # If the calculated start date is later than the target end date, no backfilling is required.
    if start_date > target_end_date:
        print(f"[{ticker}] No backfill needed. Last logged date {last_date} is on or after target end date {target_end_date}.")
        return

    # Pre-fetch historical data (e.g., for 150 days) for the ticker to use during backfilling.
    full_df = prepare_data_once(ticker, total_days=150)
    if full_df is None:
        print(f"[{ticker}] Unable to prefetch data. Skipping backfill.")
        return

    # Loop through each day from start_date up to and including target_end_date.
    current_date = start_date
    while current_date <= target_end_date:
        date_str = current_date.isoformat()
        print(f"[{ticker}] Backfilling prediction for date: {date_str}")
        try:
            # Generate and log the prediction for the current date using the pre-fetched data.
            predict_next_close_with_prefetch(ticker, full_df, simulate_date=date_str)
        except Exception as e:
            print(f"[{ticker}] Error backfilling for {date_str}: {e}")
        # Move to the next day.
        current_date += datetime.timedelta(days=1)


def sort_log_by_date():
    """
    Sort the prediction log file by the timestamp column in ascending order and save the updated log.
    """
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    log_filepath = os.path.join(data_folder, "daily_predictions_log.csv")
    try:
        log_df = pd.read_csv(log_filepath)
        # Convert the timestamp column to datetime objects.
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        # Sort the DataFrame by timestamp.
        log_df = log_df.sort_values("timestamp")
        # Save the sorted log back to the CSV file.
        log_df.to_csv(log_filepath, index=False)
        print(f"Sorted the log file at {log_filepath}.")
    except FileNotFoundError:
        print("Log file not found. Nothing to sort.")


if __name__ == "__main__":
    # Define the list of tickers for which predictions should be backfilled.
    tickers = ["AAPL", "AMZN", "MSFT", "QQQ", "SPY"]

    # Set the target end date for backfilling (e.g., yesterday's date).
    target_end_date = datetime.date.today() - datetime.timedelta(days=1)

    # Optionally, specify a default start date if no log exists (e.g., 30 days before the target end date).
    default_start_date = target_end_date - datetime.timedelta(days=30)

    # Iterate over each ticker and perform the backfill.
    for ticker in tickers:
        print(f"\nStarting backfill for {ticker}...")
        backfill_predictions_for_ticker(ticker, target_end_date, default_start_date=default_start_date)

    # Finally, sort the prediction log to ensure it is in chronological order.
    sort_log_by_date()
