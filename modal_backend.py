# modal_backend.py
import sys
import os
import modal
import boto3
import pandas as pd
import datetime

# Initialize a Modal secret using your AWS credentials.
# Run the following command in your terminal to create the secret:
# modal secret create evergreen-secrets AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy AWS_DEFAULT_REGION=us-east-1

# Retrieve the previously created secret for secure access.
secret = modal.Secret.from_name("evergreen-secrets")

# Initialize the Modal App and attach the retrieved secret.
app = modal.App("evergreen-fastapi-backend", secrets=[secret])

# Build a container image that includes all necessary dependencies,
# and mount local directories to the specified remote paths.
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

# Append the remote source directory to sys.path so that the daily_prediction module can be imported.
sys.path.append("/root/src")

# Function to update the daily predictions log on S3.
def update_predictions_log_on_s3(bucket_name: str):
    """
    Downloads the most recent daily_predictions_log CSV from S3,
    checks whether today's predictions are already recorded,
    and if not, appends today's predictions from the local log, then uploads the new file.
    """
    # Create an S3 client using credentials from the environment.
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-west-2")
    )
    
    prefix = "daily_predictions_log"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    objects = response.get("Contents", [])
    
    # Temporary local file to store the downloaded CSV.
    tmp_local_path = "/tmp/daily_predictions_log.csv"
    
    if objects:
        # Sort files by LastModified and select the most recent one.
        latest_object = sorted(objects, key=lambda obj: obj["LastModified"], reverse=True)[0]
        latest_key = latest_object["Key"]
        print(f"Latest S3 log file detected: {latest_key}")
        # Download the most recent log file from S3.
        s3.download_file(bucket_name, latest_key, tmp_local_path)
        df_s3 = pd.read_csv(tmp_local_path)
    else:
        # If no log file exists, initialize a new DataFrame with the expected columns.
        print("No existing S3 log file found. Initializing a new DataFrame for logs.")
        df_s3 = pd.DataFrame(columns=[
            "timestamp", "ticker", "predicted_close", "direction",
            "sentiment_polarity", "sentiment_subjectivity", "historical_close"
        ])
    
    # Get today's date in ISO format (YYYY-MM-DD).
    today_date_str = datetime.datetime.now().date().isoformat()
    
    # If today's prediction is already logged, no update is necessary.
    if not df_s3[df_s3["timestamp"] == today_date_str].empty:
        print("Today's predictions are already present in the S3 log. Skipping update.")
        return
    
    # Run the daily prediction pipeline because today's data is missing.
    print("No entry for today's predictions found. Executing the daily prediction pipeline...")
    from backend.daily_prediction import predict_next_close
    tickers = sorted(["AAPL", "AMZN", "MSFT", "SPY", "QQQ"])
    for ticker in tickers:
        result = predict_next_close(ticker, override_end_date=today_date_str)
        print(f"Prediction for {ticker}: {result}")
    
    # Read the local log that contains today's predictions.
    local_log_path = "/root/src/backend/data/daily_predictions_log.csv"
    if not os.path.exists(local_log_path):
        print("Local daily_predictions_log.csv not found. Aborting update.")
        return
    
    df_local = pd.read_csv(local_log_path)
    # Select only the entries corresponding to today's date.
    df_today = df_local[df_local["timestamp"] == today_date_str]
    if df_today.empty:
        print("No new predictions found locally for today. Aborting update.")
        return
    
    # Merge today's predictions with the existing S3 log.
    updated_df = pd.concat([df_s3, df_today], ignore_index=True)
    # Convert the timestamp column to datetime and sort the DataFrame.
    updated_df["timestamp"] = pd.to_datetime(updated_df["timestamp"])
    updated_df.sort_values("timestamp", inplace=True)
    
    # Save the updated DataFrame to a temporary file.
    updated_local_path = "/tmp/daily_predictions_log_updated.csv"
    updated_df.to_csv(updated_local_path, index=False)
    
    # Create a new S3 key using the current timestamp and upload the updated log.
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_s3_key = f"daily_predictions_log_{now_str}.csv"
    s3.upload_file(updated_local_path, bucket_name, new_s3_key)
    print(f"Updated log successfully uploaded to s3://{bucket_name}/{new_s3_key}")
    
# Function that executes the daily prediction process.
@app.function(image=image, timeout=1600)
def run_daily_prediction():
    """
    1) Executes predictions for a predefined list of tickers.
    2) Once predictions are recorded locally, updates the S3 log.
    """
    from backend.daily_prediction import predict_next_close
    tickers = sorted(["AAPL", "AMZN", "MSFT", "SPY", "QQQ"])
    for ticker in tickers:
        result = predict_next_close(ticker)
        print(f"Prediction for {ticker}: {result}")
    
    # After completing predictions, update the log on S3.
    update_predictions_log_on_s3(bucket_name="evergreen-investments-daily-predictions-log")

# Schedule the prediction pipeline to run daily at 9:00 AM UTC.
@app.function(image=image, timeout=900, schedule=modal.Cron("5 4 * * *"))
def scheduled_daily_prediction():
    """
    Triggers the daily prediction process automatically on a set schedule.
    """
    run_daily_prediction.remote()

# Define a local entry point to manually trigger the prediction pipeline.
@app.local_entrypoint()
def main():
    print("Manually triggering the daily prediction pipeline and subsequent S3 update...")
    run_daily_prediction.remote()
