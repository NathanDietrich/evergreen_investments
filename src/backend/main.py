# src/backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bot.data_fetcher import fetch_stock_data
from bot.process_data import process_data  # Import the processing function
from dotenv import load_dotenv
import os

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

@app.get("/predict")
async def predict(stock: str):
    if not stock:
        raise HTTPException(status_code=400, detail="Stock parameter is required")
    
    try:
        # Fetch raw data from Polygon
        raw_data = fetch_stock_data(stock)
        # Process the raw data (merge historical & sentiment, apply scaling, etc.)
        processed_data = process_data(raw_data)
        return processed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
