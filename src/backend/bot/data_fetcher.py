# src/backend/bot/data_fetcher.py
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
from dotenv import load_dotenv
load_dotenv()

def fetch_stock_data_polygon(ticker, start_date, end_date, api_key):
    """
    Fetches historical stock data from Polygon.io.
    Returns a DataFrame with columns: [Date, Open, High, Low, Close, Volume].
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
    Fetches sentiment data from Polygon.io in chunks.
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
        print(f"📡 Fetching sentiment data for {ticker} from {chunk_start_str} to {chunk_end_str}...")
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
                print(f"⚠️ Error fetching sentiment data for {ticker}: {response.text}")
                break
        current_start_date = chunk_end_date
        time.sleep(14)
    return all_results

def analyze_sentiment(news_data):
    """
    Uses TextBlob to compute sentiment polarity and subjectivity for each news article.
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
    Merges stock data with sentiment data by date.
    """
    if stock_df is None or stock_df.empty:
        return stock_df
    if not sentiment_data:
        stock_df["sentiment_polarity"] = 0
        stock_df["sentiment_subjectivity"] = 0
        return stock_df
    sentiment_df = pd.DataFrame(sentiment_data)
    avg_polarity = sentiment_df["sentiment_polarity"].mean()
    avg_subjectivity = sentiment_df["sentiment_subjectivity"].mean()
    stock_df["sentiment_polarity"] = avg_polarity
    stock_df["sentiment_subjectivity"] = avg_subjectivity
    return stock_df

def collect_raw_data():
    # For predictive purposes, we only need the last 80 days.
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=80)).strftime("%Y-%m-%d")
    # Use a single stock for now
    ticker = "AAPL"
    api_key = os.getenv("Polygon_Key")
    if not api_key or api_key == "YOUR_POLYGON_API_KEY":
        print("Please set your Polygon API key in the environment variable Polygon_Key")
        return
    print(f"\n================== Processing {ticker} ==================")
    print(f"📊 Fetching stock data for {ticker} from {start_date} to {end_date}...")
    stock_df = fetch_stock_data_polygon(ticker, start_date, end_date, api_key)
    if stock_df is None:
        print(f"❌ No stock data found for {ticker}.")
        return
    print(f"📰 Fetching sentiment data for {ticker} from {start_date} to {end_date}...")
    news_data = fetch_sentiment_data_polygon(ticker, start_date, end_date, api_key, limit=1000)
    if not news_data:
        print(f"⚠️ No news data found for {ticker}. Proceeding without sentiment data.")
    print("💡 Performing sentiment analysis...")
    sentiment_data = analyze_sentiment(news_data)
    print("🔗 Merging stock and sentiment data...")
    merged_df = merge_stock_and_sentiment(stock_df, sentiment_data)
    save_dir = "data/StockData"  # local directory for raw data
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{ticker}_{start_date}_to_{end_date}_raw.csv")
    merged_df.to_csv(filename, index=False)
    print(f"✅ Raw data for {ticker} saved to: {filename}")

# Run raw data collection only when executing this module directly.
if __name__ == "__main__":
    collect_raw_data()
