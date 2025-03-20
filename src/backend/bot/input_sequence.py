# src/backend/bot/input_sequence.py
import numpy as np

def create_sequences(df, feature_cols, label_col='Close', sequence_length=60):
    """
    Creates sequences from the DataFrame.
    Each sample: features from day T and label is day T+1's Close.
    Version: 2025-03-16
    Returns:
      X: array of shape (num_samples, sequence_length, num_features)
      y: array of shape (num_samples,)
    """
    data_array = df[feature_cols].values
    labels = df[label_col].values
    X, y = [], []
    for i in range(len(df) - sequence_length):
        seq_x = data_array[i : i + sequence_length]
        seq_y = labels[i + sequence_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_inference_sequence(df, feature_cols, sequence_length=60):
    """
    Builds an input sequence for inference using the last window of data.
    Returns an array of shape (1, sequence_length, num_features).
    Version: 2025-03-17
    """
    data_array = df[feature_cols].values
    if len(data_array) < sequence_length:
        raise ValueError("Not enough data to build the input sequence.")
    last_window = data_array[-sequence_length:]
    return last_window.reshape(1, sequence_length, len(feature_cols))
