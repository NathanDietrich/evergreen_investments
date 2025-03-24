# src/backend/bot/model_loading.py
"""
model_loading.py
Version: 2025-03-17

Contains a function to load the saved model for a given ticker.
This version attempts to bypass Lambda layer output shape inference issues
by enabling unsafe deserialization and by passing the custom layer in custom_objects.
"""

import os
from tensorflow.keras.models import load_model
from tensorflow.keras import config
from .custom_layers import ExtractWeight  # Import the custom layer

def load_model_for_ticker(ticker):
    """
    Loads the saved model for the given ticker.
    
    Enables unsafe deserialization (if available) so that custom layers such as ExtractWeight
    can be loaded. It passes ExtractWeight via custom_objects.
    
    This function expects your models to be saved with the following folder structure:
    
      <project_root>/models/BestEnsembleModel_{ticker}/{ticker}_best_model.keras
    
    :param ticker: Stock ticker (e.g., "AAPL")
    :return: The loaded Keras model, or None if not found.
    """
    # Try to enable unsafe deserialization; if not available, warn the user.
    try:
        config.enable_unsafe_deserialization()
    except AttributeError:
        print("Warning: unsafe deserialization is not available in this TensorFlow version.")
    
    # Use the current working directory as the project root
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, "models", f"BestEnsembleModel_{ticker}", f"{ticker}_best_model.keras")
    
    if not os.path.exists(model_path):
        print(f"Model file not found for {ticker} at {model_path}.")
        return None
    
    try:
        # Load the model with custom_objects for ExtractWeight
        model = load_model(model_path, custom_objects={"ExtractWeight": ExtractWeight})
    except Exception as e:
        print("Error loading model:", e)
        return None
    
    return model
