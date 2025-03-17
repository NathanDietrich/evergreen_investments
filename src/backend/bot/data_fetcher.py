# data_fetcher.py
"""
data_fetcher.py
Version: 2025-03-16

Contains functions for fetching stock & sentiment data from Polygon,
analyzing sentiment, and merging the results.
"""

import os
import time
import datetime
import requests
import pandas as pd
from textblob import TextBlob

def fetch_stock_data_polygon(ticker, start_date, end_date, api_key):
    """
    Fetches historical stock data from Polygon.io for the given ticker
    and date range (start_date to end_date in YYYY-MM-DD).
    
    Returns a DataFrame with columns:
      [Date, Open, High, Low, Close, Volume].
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching stock data for {ticker}: {response.text}")
        return None
    data = response.json()
    if "results" not in data:
        print(f"No results found for {ticker}.")
        return None
    
    df = pd.DataFrame(data["results"])
    df["Date"] = pd.to_datetime(df["t"], unit="ms").dt.date
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df

def fetch_sentiment_data_polygon(ticker, start_date, end_date, api_key, limit=1000):
    """
    Fetches sentiment data (news) from Polygon.io in chunks and
    returns the raw JSON results.
    
    start_date and end_date in YYYY-MM-DD.
    limit is the max items per request.
    """
    url = "https://api.polygon.io/v2/reference/news"
    all_results = []
    current_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    final_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    while current_start_date < final_end_date:
        chunk_end_date = current_start_date + datetime.timedelta(days=30)
        if chunk_end_date > final_end_date:
            chunk_end_date = final_end_date
        chunk_start_str = current_start_date.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end_date.strftime("%Y-%m-%d")

        params = {
            "ticker": ticker,
            "published_utc.gte": chunk_start_str,
            "published_utc.lte": chunk_end_str,
            "apiKey": api_key,
            "limit": limit
        }

        while True:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                all_results.extend(results)

                next_cursor = data.get("next_cursor")
                if not next_cursor:
                    break
                params["cursor"] = next_cursor
            else:
                print(f"Error fetching sentiment data for {ticker}: {response.text}")
                break

        current_start_date = chunk_end_date
        time.sleep(14)  # avoid hitting rate limits

    return all_results

def analyze_sentiment(news_data):
    """
    Uses TextBlob to compute sentiment polarity & subjectivity
    for each news article in the raw Polygon news data.
    
    Returns a list of dicts with "sentiment_polarity" and "sentiment_subjectivity".
    """
    analyzed_data = []
    for article in news_data:
        title = article.get("title", "")
        description = article.get("description", "")
        full_text = f"{title} {description}"
        sentiment = TextBlob(full_text).sentiment
        
        analyzed_data.append({
            "sentiment_polarity": sentiment.polarity,
            "sentiment_subjectivity": sentiment.subjectivity
        })
    return analyzed_data

def merge_stock_and_sentiment(stock_df, sentiment_data):
    """
    Merges stock data with sentiment data by assigning average sentiment
    for the entire period. If no sentiment data is available, sets neutral (0).
    """
    if stock_df is None or stock_df.empty:
        return stock_df

    if not sentiment_data:
        # No sentiment available, set neutral
        stock_df["sentiment_polarity"] = 0
        stock_df["sentiment_subjectivity"] = 0
        return stock_df

    sentiment_df = pd.DataFrame(sentiment_data)
    avg_polarity = sentiment_df["sentiment_polarity"].mean()
    avg_subjectivity = sentiment_df["sentiment_subjectivity"].mean()

    stock_df["sentiment_polarity"] = avg_polarity
    stock_df["sentiment_subjectivity"] = avg_subjectivity

    return stock_df
