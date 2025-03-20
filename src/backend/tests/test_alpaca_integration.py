# src/backend/tests/test_alpaca_integration.py
"""
test_alpaca_integration.py
Version: 2025-03-19

A simple test script to place a paper trade via Alpaca's paper trading API.
Expects environment variables:
  ALPACA_API_KEY
  ALPACA_SECRET_KEY
  ALPACA_BASE_URL (default: https://paper-api.alpaca.markets)

Usage:
  - Put this file in your tests folder.
  - Ensure you have the .env or environment variables set.
  - Run: python -m src.backend.tests.test_alpaca_integration
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv()  # This loads environment variables from .env file in the current working directory
print(os.getenv("ALPACA_API_KEY"))  # Check that the key is loaded

def place_paper_trade(ticker: str, side: str = "buy", quantity: int = 1):
    """
    Places a market order for `quantity` shares of `ticker` on the Alpaca paper endpoint.
    
    :param ticker: Stock symbol (e.g. "AAPL")
    :param side: "buy" or "sell"
    :param quantity: number of shares
    :return: JSON response from Alpaca if successful, otherwise None
    """
    # Load Alpaca credentials from environment
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("❌ Alpaca credentials not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        return None

    # Build the request headers
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json"
    }

    # Build the order payload
    order_data = {
        "symbol": ticker,
        "qty": quantity,
        "side": side,
        "type": "market",
        "time_in_force": "gtc"
    }

    # Make the request
    url = f"{ALPACA_BASE_URL}/v2/orders"
    response = requests.post(url, headers=headers, data=json.dumps(order_data))

    if response.status_code in [200, 201]:
        resp_json = response.json()
        print(f"✅ Order placed successfully for {ticker}:")
        print(json.dumps(resp_json, indent=2))
        return resp_json
    else:
        print(f"❌ Error placing order. Status: {response.status_code}, Body: {response.text}")
        return None

def main():
    # Example usage: Place a market BUY order for 1 share of AAPL
    ticker = input("Enter a stock ticker (e.g., AAPL): ").strip().upper() or "AAPL"
    side = input("Enter side (buy/sell): ").strip().lower() or "buy"
    quantity = input("Enter quantity: ").strip()
    quantity = int(quantity) if quantity.isdigit() else 1

    place_paper_trade(ticker, side, quantity)

if __name__ == "__main__":
    main()
