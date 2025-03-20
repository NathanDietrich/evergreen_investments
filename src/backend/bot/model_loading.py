# src/backend/bot/model_loading.py
"""
model_loading.py
Version: 2025-03-17

Contains a function to load the saved model for a given ticker.
This version attempts to bypass Lambda layer output shape inference issues
by enabling unsafe deserialization.
"""

import os
from tensorflow.keras.models import load_model
from tensorflow.keras import config

def load_model_for_ticker(ticker):
    """
    Loads the saved model for the given ticker.
    
    Enables unsafe deserialization (if available) so that Lambda layers defined with
    Python lambdas (without an explicit output_shape) can be loaded.
    
    :param ticker: Stock ticker (e.g., "AAPL")
    :return: The loaded Keras model, or None if not found.
    """
    # Try to enable unsafe deserialization; if not available, warn the user.
    try:
        config.enable_unsafe_deserialization()
    except AttributeError:
        print("Warning: unsafe deserialization is not available in this TensorFlow version.")

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f"{ticker}_best_model.keras")
    if not os.path.exists(model_path):
        print(f"Model file not found for {ticker} at {model_path}.")
        return None

    try:
        # Load the model without safe_mode (since it's no longer supported)
        model = load_model(model_path)
    except Exception as e:
        print("Error loading model:", e)
        return None

    return model
