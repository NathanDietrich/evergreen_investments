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
        return "Alpaca credentials not set."

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
        return f"Error placing order. Status: {response.status_code}, {response.text}"

def trading_dashboard():
    st.subheader("Trading Dashboard")
    st.markdown("Place a paper trade via Alpaca:")
    
    with st.form(key="trade_form"):
        trade_ticker = st.text_input("Ticker", value="AAPL")
        trade_side = st.selectbox("Side", ["buy", "sell"])
        trade_quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        submit_trade = st.form_submit_button(label="Place Order")
    
    if submit_trade:
        trade_result = place_paper_trade(trade_ticker, trade_side, trade_quantity)
        if isinstance(trade_result, dict):
            st.success("Order placed successfully!")
            st.json(trade_result)
        else:
            st.error(f"Order error: {trade_result}")
