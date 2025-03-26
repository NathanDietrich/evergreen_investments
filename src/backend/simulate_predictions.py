import datetime
from .daily_prediction import (
    prepare_data_once,
    predict_next_close_with_prefetch
)

def simulate_last_30_days():
    tickers = ["AAPL", "AMZN", "MSFT", "QQQ", "SPY"]
    
    end_sim_date = datetime.date.today() - datetime.timedelta(days=1)
    start_sim_date = end_sim_date - datetime.timedelta(days=29)

    for ticker in tickers:
        print(f"\n=== Fetching a single dataset (150 days) for {ticker} ===")
        # Fetch and merge stock+sentiment once for 150 days
        df_prefetch = prepare_data_once(ticker, total_days=150)
        if df_prefetch is None or df_prefetch.empty:
            print(f"No data fetched for {ticker}. Skipping...")
            continue
        
        # Now simulate each day using the pre-fetched data
        current_date = start_sim_date
        while current_date <= end_sim_date:
            print(f"\nSimulating {ticker} on {current_date} ...")
            try:
                predict_next_close_with_prefetch(
                    ticker=ticker,
                    full_df=df_prefetch,
                    simulate_date=current_date
                )
            except Exception as e:
                print(f"Error simulating {ticker} on {current_date}: {e}")
            current_date += datetime.timedelta(days=1)

def main():
    simulate_last_30_days()

if __name__ == "__main__":
    main()
