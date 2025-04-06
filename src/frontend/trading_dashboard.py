import streamlit as st
import requests
import json

def place_paper_trade(ticker: str, side: str = "buy", quantity: int = 1):
    """
    Places a market order for `quantity` shares of `ticker` on the Alpaca paper endpoint.
    Uses credentials from Streamlit secrets.
    """
    try:
        ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
        ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
        ALPACA_BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    except Exception:
        return {"error": "Alpaca credentials not set in Streamlit secrets."}

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json"
    }

    order_data = {
        "symbol": ticker.upper(),  # Ensure ticker is uppercase
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

if __name__ == "__main__":
    trading_dashboard()
