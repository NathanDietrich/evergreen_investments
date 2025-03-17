import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Evergreen Investments - Stock Predictions")

# Let the user choose a stock
selected_stock = st.selectbox("Choose a Stock:", ["AAPL", "GOOGL", "AMZN", "MSFT"])

if st.button("Get Data"):
    st.write(f"Fetching data for {selected_stock}...")
    # Construct the URL to call the backend endpoint
    backend_url = f"http://localhost:8000/predict?stock={selected_stock}"
    
    try:
        # Make the request to the backend
        response = requests.get(backend_url)
        response.raise_for_status()  # raise exception for bad responses
        data = response.json()

        # Display historical data from the backend
        st.subheader("Historical Data")
        st.write(data["historical_data"])
        
        # Optionally, if historical_data contains a "results" key for charting:
        if "results" in data["historical_data"]:
            df = pd.DataFrame(data["historical_data"]["results"])
            # Assuming "c" is the closing price from Polygon's API response
            st.line_chart(df["c"])
        
        # Display sentiment data
        st.subheader("Sentiment Data")
        st.write(data["sentiment_data"])
        
        # Display market open status
        st.write("Market Open:", data["market_open"])
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
