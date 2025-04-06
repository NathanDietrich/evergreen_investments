import streamlit as st
import os
import requests
import json

def place_paper_trade(ticker: str, side: str = "buy", quantity: int = 1):
    """
    Places a market order for `quantity` shares of `ticker` on the Alpaca paper endpoint.
    """
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"error": "Alpaca credentials not set."}

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json"
    }

    order_data = {
        "symbol": ticker,
        "qty": quantity,
        "side": side,
        "type": "market",
        "time_in_force": "gtc"
    }

    url = f"{ALPACA_BASE_URL}/v2/orders"
    response = requests.post(url, headers=headers, data=json.dumps(order_data))
    if response.status_code in [200, 201]:
        return response.json()
    else:
        return {"error": f"Error placing order. Status: {response.status_code}, {response.text}"}

def trading_dashboard():
    st.title("Trading Dashboard")
    st.write("Place a paper trade via Alpaca.")

    with st.form(key="trade_form"):
        ticker = st.text_input("Ticker", value="AAPL")
        side = st.selectbox("Side", options=["buy", "sell"])
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        submit_button = st.form_submit_button("Place Order")
    
    if submit_button:
        result = place_paper_trade(ticker, side, quantity)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Your {side} order for {quantity} shares of {ticker.upper()} has been placed successfully!")
            order_id = result.get("id", "N/A")
            st.info(f"Order ID: {order_id}")

if __name__ == "__main__":
    trading_dashboard()
