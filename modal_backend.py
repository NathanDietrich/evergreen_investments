# modal_backend.py
import sys
import os
import modal
import boto3
import pandas as pd
import datetime


# Create your Modal secrets from AWS credentials:
# modal secret create evergreen-secrets AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy AWS_DEFAULT_REGION=us-east-1

# We'll attach that secret to our app
secret = modal.Secret.from_name("evergreen-secrets")

# Create your Modal App with the secret
app = modal.App("evergreen-fastapi-backend", secrets=[secret])

# Build your image with dependencies (including boto3) and mount local code
image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.2",
        "requests==2.32.3",
        "joblib==1.3.2",
        "textblob==0.17.1",
        "python-dotenv==1.0.1",
        "tensorflow==2.18.0",
        "ml-dtypes>=0.4.0,<0.5.0",
        "tensorboard>=2.18,<2.19",
        "scikit-learn==1.4.1.post1",
        "fastapi==0.110.0",
        "uvicorn==0.29.0",
        "alpaca-trade-api",
        "websockets>=13.0,<15.0",
        "keras-tuner==1.4.7",
        "matplotlib==3.8.3",
        "boto3"
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scalers", remote_path="/root/scalers")
    .add_local_dir("models", remote_path="/root/models")
)

# We can add /root/src to sys.path so your daily_prediction module is discoverable
sys.path.append("/root/src")

# A helper function to upload your daily_predictions_log.csv to S3
def update_predictions_log_on_s3(bucket_name: str):
    """
    Downloads the newest daily_predictions_log CSV from S3,
    checks if today's prediction exists,
    if not, it appends today's prediction (from the local log) and uploads the updated file.
    """
    # Set up S3 client using environment variables.
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-west-2")
    )
    
    prefix = "daily_predictions_log"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    objects = response.get("Contents", [])
    
    # Define local temporary file path
    tmp_local_path = "/tmp/daily_predictions_log.csv"
    
    if objects:
        # Find the newest file by LastModified
        latest_object = sorted(objects, key=lambda obj: obj["LastModified"], reverse=True)[0]
        latest_key = latest_object["Key"]
        print(f"Found latest S3 log file: {latest_key}")
        # Download the file from S3
        s3.download_file(bucket_name, latest_key, tmp_local_path)
        df_s3 = pd.read_csv(tmp_local_path)
    else:
        # If no file exists in S3, create an empty DataFrame with expected columns.
        print("No existing log file found in S3. Creating a new log DataFrame.")
        df_s3 = pd.DataFrame(columns=[
            "timestamp", "ticker", "predicted_close", "direction",
            "sentiment_polarity", "sentiment_subjectivity", "historical_close"
        ])
    
    # Get today's date string (ISO format) using only the date part.
    today_date_str = datetime.datetime.now().date().isoformat()
    
    # Check if today's prediction is already present.
    if not df_s3[df_s3["timestamp"] == today_date_str].empty:
        print("Today's prediction already exists in the S3 log. No update needed.")
        return
    
    # Otherwise, run the daily prediction pipeline for today.
    print("Today's prediction not found. Running daily prediction pipeline...")
    from backend.daily_prediction import predict_next_close
    tickers = sorted(["AAPL", "AMZN", "MSFT", "SPY", "QQQ"])
    for ticker in tickers:
        result = predict_next_close(ticker, override_end_date=today_date_str)
        print(f"Prediction for {ticker}: {result}")
    
    # Read the local log file where today's predictions were appended.
    local_log_path = "/root/src/backend/data/daily_predictions_log.csv"
    if not os.path.exists(local_log_path):
        print("Local daily_predictions_log.csv not found. Aborting update.")
        return
    
    df_local = pd.read_csv(local_log_path)
    # Filter for rows with today's timestamp.
    df_today = df_local[df_local["timestamp"] == today_date_str]
    if df_today.empty:
        print("No new predictions logged locally for today. Aborting update.")
        return
    
    # Append today's predictions to the S3 DataFrame.
    updated_df = pd.concat([df_s3, df_today], ignore_index=True)
    # Optional: sort by timestamp.
    updated_df["timestamp"] = pd.to_datetime(updated_df["timestamp"])
    updated_df.sort_values("timestamp", inplace=True)
    
    # Save the updated log to a temporary file.
    updated_local_path = "/tmp/daily_predictions_log_updated.csv"
    updated_df.to_csv(updated_local_path, index=False)
    
    # Upload the updated file to S3 with a new timestamped key.
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_s3_key = f"daily_predictions_log_{now_str}.csv"
    s3.upload_file(updated_local_path, bucket_name, new_s3_key)
    print(f"Uploaded updated log to s3://{bucket_name}/{new_s3_key}")
    
# This function runs your daily prediction pipeline once.
@app.function(image=image, timeout=1600)
def run_daily_prediction():
    """
    1) Runs predictions for each ticker
    2) After predictions are logged, calls upload_predictions_log_to_s3
    """
    from backend.daily_prediction import predict_next_close
    tickers = sorted(["AAPL", "AMZN", "MSFT", "SPY", "QQQ"])
    for ticker in tickers:
        result = predict_next_close(ticker)
        print(f"Prediction for {ticker}: {result}")
    
    # After finishing predictions, upload the log to S3
    # Replace with your actual bucket name
    update_predictions_log_on_s3(bucket_name="evergreen-investments-daily-predictions-log")

# Schedule the daily prediction pipeline to run every day at 9:00 AM UTC.
@app.function(image=image, timeout=900, schedule=modal.Cron("5 4 * * *"))
def scheduled_daily_prediction():
    """
    Same as run_daily_prediction, but scheduled.
    """
    run_daily_prediction.remote()

# Local entrypoint to trigger the function manually
@app.local_entrypoint()
def main():
    print("Running daily prediction pipeline once, then uploading to S3...")
    run_daily_prediction.remote()
