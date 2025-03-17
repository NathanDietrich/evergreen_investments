import os
import requests
import datetime

def fetch_stock_data(stock_symbol: str):
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise Exception("Polygon API key is not set in the environment.")

    # Calculate date range: from two months ago to today
    today = datetime.date.today()
    two_months_ago = today - datetime.timedelta(days=60)  # approx. 2 months
    start_date_str = two_months_ago.strftime("%Y-%m-%d")
    end_date_str = today.strftime("%Y-%m-%d")
    
    # Determine if the market is open (simplified check)
    now = datetime.datetime.now()
    market_open_flag = 0
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now.weekday() < 5 and market_open_time <= now <= market_close_time:
        market_open_flag = 1

    # Fetch historical stock data from Polygon
    historical_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/day/"
        f"{start_date_str}/{end_date_str}?apiKey={api_key}"
    )
    historical_response = requests.get(historical_url)
    historical_data = historical_response.json()

    # Fetch sentiment data (if applicable)
    sentiment_url = (
        f"https://api.polygon.io/v2/reference/news?ticker={stock_symbol}"
        f"&published_utc.gte={start_date_str}"
        f"&published_utc.lte={end_date_str}&apiKey={api_key}"
    )
    sentiment_response = requests.get(sentiment_url)
    sentiment_data = sentiment_response.json()

    # Return the data
    return {
        "historical_data": historical_data,
        "sentiment_data": sentiment_data,
        "market_open": market_open_flag
    }
