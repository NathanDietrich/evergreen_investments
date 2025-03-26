from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import datetime
import logging

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS so your front end can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import the new daily prediction function
from .daily_prediction import predict_next_close

def run_daily_prediction():
    # Define the list of stocks (alphabetically sorted)
    stocks = sorted(["AAPL", "AMZN", "MSFT", "SPY", "QQQ"])
    results = {}
    logger.info("Starting daily prediction for stocks: %s", stocks)
    
    for stock in stocks:
        try:
            # Call the new predict_next_close function which logs the prediction
            predicted_value = predict_next_close(stock)
            logger.info(f"Predicted tomorrow's Close for {stock}: {predicted_value:.4f}")
            results[stock] = predicted_value
        except Exception as e:
            logger.error(f"Error predicting for {stock}: {e}")
            results[stock] = None
    
    # Since the system uses yesterday's data, the prediction is for tomorrow's close.
    prediction_for = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    
    return {
        "message": "Daily predictions complete",
        "predictions": results,
        "prediction_for": prediction_for,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/daily_prediction")
async def daily_prediction_endpoint():
    try:
        result = run_daily_prediction()
        return result
    except Exception as e:
        logger.error(f"Error in daily_prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
