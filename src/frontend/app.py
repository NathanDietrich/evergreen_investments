import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as mticker  # renamed to avoid conflict with 'ticker' variable

def load_data():
    # Path to the CSV log file created by daily_prediction.py
    csv_path = os.path.join("src", "backend", "data", "daily_predictions_log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Parse timestamps with inferred format
        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, errors='coerce')
        return df
    else:
        return pd.DataFrame()

def plot_pred_vs_actual_with_direction(dates, actual, predicted, ticker="Ticker"):
    """
    Plots the actual price (blue line) and predicted price segments.
    Segments are green if the predicted direction is correct,
    red if not. Uses actual dates on the x-axis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the actual price
    ax.plot(dates, actual, label='Actual Price', color='blue')
    
    # Plot predicted segments with color based on direction correctness
    for i in range(len(dates) - 1):
        seg_x = [dates[i], dates[i+1]]
        seg_y = [predicted[i], predicted[i+1]]
        correct_direction = (actual[i+1] - actual[i]) * (predicted[i+1] - predicted[i]) >= 0
        color = 'green' if correct_direction else 'red'
        ax.plot(seg_x, seg_y, color=color)
    
    # Format the x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Create a custom legend
    blue_line = mlines.Line2D([], [], color='blue', label='Actual Price')
    green_line = mlines.Line2D([], [], color='green', label='Predicted (Correct Dir)')
    red_line = mlines.Line2D([], [], color='red', label='Predicted (Wrong Dir)')
    ax.legend(handles=[blue_line, green_line, red_line])

    # Add grid lines
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    
    # Increase the number of major ticks on the Y-axis for better price granularity
    ax.yaxis.set_major_locator(mticker.MaxNLocator(10))

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f"{ticker} - Predicted vs Actual")
    return fig

def app():
    st.set_page_config(page_title="Evergreen Investments Dashboard", layout="wide")
    st.title("Evergreen Investments - Daily Stock Predictions Dashboard")

    # Disclaimer section
    st.markdown("""
    **Disclaimer:**  
    Past returns do not indicate future performance. Data is provided "as is" and may be subject to delays.  
    The API (Polygon) typically updates after market close, so predictions may be slightly behind depending on the time of day.  
    The logged timestamp indicates the date for which the prediction applies.
    """)

    df = load_data()
    if df.empty:
        st.warning("No prediction data available yet. Run the backend to generate predictions.")
        return

    # Display a note with the latest prediction data date
    latest_date = df['timestamp'].max().date()
    st.markdown(f"**Latest prediction data is from: {latest_date}**")

    # Sidebar: Select stock
    tickers = sorted(df["ticker"].unique())
    selected_stock = st.sidebar.selectbox("Select Stock", tickers)

    # Filter and sort data for the selected stock
    df_stock = df[df["ticker"] == selected_stock].copy().sort_values("timestamp")
    
    # Sidebar custom up/down indicator with enlarged display
    if not df_stock.empty:
        latest_entry = df_stock.iloc[-1]
        price_pred = f"{latest_entry['predicted_close']:.2f}"
        direction = latest_entry["direction"].lower()
        if direction == "up":
            arrow = "⬆️"
            color = "green"
            direction_text = "Up"
        elif direction == "down":
            arrow = "⬇️"
            color = "red"
            direction_text = "Down"
        else:
            arrow = ""
            color = "black"
            direction_text = "No change"
        st.sidebar.markdown(f"""
        <div style="text-align:center;">
            <h1 style="font-size:3rem;">Price Prediction: {price_pred}</h1>
            <h2 style="font-size:2rem; color:{color};">Direction: {arrow} {direction_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Rename columns for readability in charts/tables
    df_stock.rename(columns={
        "predicted_close": "Predicted",
        "historical_close": "Actual",
        "sentiment_polarity": "Polarity",
        "sentiment_subjectivity": "Subjectivity"
    }, inplace=True)

    # Create tabs for organizing the dashboard
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predicted vs Actual",
        "Sentiment Over Time",
        "Direction Frequency",
        "Data Table"
    ])

    # Tab 1: Custom Matplotlib Chart with actual dates on x-axis
    with tab1:
        st.subheader("Predicted vs Actual")
        if len(df_stock) < 2:
            st.info("Not enough data points for the chart.")
        else:
            # Use actual dates from the timestamp column
            dates = df_stock['timestamp'].dt.date.values
            actual_prices = df_stock["Actual"].values
            predicted_prices = df_stock["Predicted"].values
            fig = plot_pred_vs_actual_with_direction(dates, actual_prices, predicted_prices, ticker=selected_stock)
            st.pyplot(fig)

    # Tab 2: Sentiment Over Time
    with tab2:
        st.subheader("Sentiment Over Time")
        sentiment_df = df_stock.set_index("timestamp")[["Polarity", "Subjectivity"]]
        st.line_chart(sentiment_df)

    # Tab 3: Direction Frequency (using Altair for better styling)
    with tab3:
        st.subheader("Prediction Direction Frequency")
        direction_counts = df_stock["direction"].value_counts().reset_index()
        direction_counts.columns = ["Direction", "Count"]
        chart = (
            alt.Chart(direction_counts)
            .mark_bar()
            .encode(
                x=alt.X("Direction:N", sort=None),
                y="Count:Q",
                color="Direction:N",
                tooltip=["Direction", "Count"]
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    # Tab 4: Data Table
    with tab4:
        st.subheader("Latest Predictions Data")
        st.dataframe(df_stock.sort_values("timestamp", ascending=False).head(20))

if __name__ == "__main__":
    app()
