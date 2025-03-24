import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    SimpleRNN, LSTM, Concatenate, Multiply, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from matplotlib.lines import Line2D

# Import custom modules using relative imports
from .bot.process_data import process_data
from .bot.scale_data import create_and_save_scalers, invert_target_scaling
from .bot.input_sequence import create_sequences

# ------------------------------
# Custom Layer: ExtractWeight
# ------------------------------
# Define inline since no separate custom_layers module exists
class ExtractWeight(tf.keras.layers.Layer):
    """
    ExtractWeight is a custom layer that takes branch weights (shape: (batch_size, 3))
    and returns a tensor of shape (batch_size, 1) corresponding to the specified index.
    This replaces inline Lambda functions for better serialization.
    """
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return tf.reshape(inputs[:, self.index], (-1, 1))

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

# ------------------------------
# Local Paths Setup
# ------------------------------
BASE_DIR = os.getcwd()  # Project root (should be your backend folder)
DATA_DIR = os.path.join(BASE_DIR, "data")             # Folder where raw CSV files are stored
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")            # Raw data folder
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")     # Folder to save processed CSV files
MODEL_DIR = os.path.join(BASE_DIR, "models")            # Folder to save trained models
SCALER_DIR = os.path.join(BASE_DIR, "scalers")          # Folder to save scaler objects

# ------------------------------
# Build the Ensemble Model (using ExtractWeight)
# ------------------------------
def build_ensemble_model(hp, input_shape):
    """
    Builds an ensemble model combining CNN, RNN, and LSTM branches with adaptive fusion.
    Uses the custom ExtractWeight layer to extract branch-specific weights.
    Version: 2025-03-18
    """
    inputs = Input(shape=input_shape)

    # --- CNN Branch ---
    cnn = Conv1D(
        filters=hp.Choice('cnn_filters', [64, 128, 256]),
        kernel_size=hp.Choice('cnn_kernel_size', [3, 5, 7]),
        activation='relu',
        padding='same'
    )(inputs)
    if input_shape[0] > 1:
        cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(50, activation='relu')(cnn)

    # --- RNN Branch ---
    rnn = SimpleRNN(
        units=hp.Choice('rnn_units', [75, 100, 125, 150]),
        return_sequences=True
    )(inputs)
    rnn = Dropout(hp.Choice('dropout_rate', [0.05, 0.1, 0.2]))(rnn)
    rnn = SimpleRNN(units=hp.Choice('rnn_units_2', [75, 100, 125, 150]))(rnn)
    rnn = Dropout(hp.Choice('dropout_rate_2', [0.05, 0.1, 0.2]))(rnn)
    rnn = Dense(50, activation='relu')(rnn)

    # --- LSTM Branch ---
    lstm = LSTM(
        units=hp.Choice('lstm_units', [50, 75, 100]),
        return_sequences=True
    )(inputs)
    lstm = LSTM(units=hp.Choice('lstm_units_2', [50, 75, 100]))(lstm)
    lstm = Dense(50, activation='relu')(lstm)
    lstm = Dropout(hp.Choice('dropout_rate_lstm', [0.1, 0.2, 0.3]))(lstm)

    # --- Adaptive Fusion ---
    combined = Concatenate()([cnn, rnn, lstm])
    weight_logits = Dense(3)(combined)
    branch_weights = Activation('softmax')(weight_logits)

    # Extract branch weights using the custom ExtractWeight layer
    cnn_weight = ExtractWeight(index=0)(branch_weights)
    rnn_weight = ExtractWeight(index=1)(branch_weights)
    lstm_weight = ExtractWeight(index=2)(branch_weights)

    cnn_scaled = Multiply()([cnn, cnn_weight])
    rnn_scaled = Multiply()([rnn, rnn_weight])
    lstm_scaled = Multiply()([lstm, lstm_weight])

    merged = Concatenate()([cnn_scaled, rnn_scaled, lstm_scaled])
    merged = Dense(
        units=hp.Choice('dense_units', [50, 100, 150]),
        activation="relu",
        kernel_regularizer=l2(0.001)
    )(merged)
    merged = Dropout(hp.Choice('dropout_rate_dense', [0.1, 0.2, 0.3]))(merged)
    output = Dense(1)(merged)

    model = tf.keras.models.Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ------------------------------
# Main Training Pipeline
# ------------------------------
def main():
    # Look for raw CSV files ending with _raw.csv in the RAW_DATA_DIR.
    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*_raw.csv"))
    if not raw_files:
        print("No raw CSV files found in the data/raw folder.")
        return

    for file in raw_files:
        # Expect file name like "AAPL_2021-01-01_to_2025-03-20_raw.csv"
        filename = os.path.basename(file)
        parts = filename.split("_")
        ticker = parts[0]
        print(f"\n=== Processing raw data for {ticker} from file: {filename} ===")
        
        # Load raw data
        raw_df = pd.read_csv(file, parse_dates=["Date"])
        raw_df.sort_values(by="Date", inplace=True)
        
        # Process data (add technical indicators, drop Date column, etc.)
        from .bot.process_data import process_data
        X, y = process_data(raw_df)
        
        # Save processed data to the PROCESSED_DIR
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        processed_filepath = os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv")
        X.to_csv(processed_filepath, index=False)
        print(f"Processed data saved to {processed_filepath}")
        
        # Scale data and save scalers locally
        from .bot.scale_data import create_and_save_scalers
        df_scaled, scaler_features, scaler_target = create_and_save_scalers(X, ticker, scaler_dir=SCALER_DIR)
        
        # Create training sequences
        from .bot.input_sequence import create_sequences
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'RSI',
            'MACD', 'MACD_Signal', 'sentiment_polarity', 'sentiment_subjectivity'
        ]
        sequence_length = 60
        X_seq, y_seq = create_sequences(df_scaled, feature_cols, label_col='Close', sequence_length=sequence_length)
        print(f"Sequences created for {ticker}: X shape {X_seq.shape}, y shape {y_seq.shape}")
        
        # Split sequences into train, validation, test (70/15/15 split)
        total = len(X_seq)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)
        X_train, y_train = X_seq[:train_end], y_seq[:train_end]
        X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
        X_test, y_test = X_seq[val_end:], y_seq[val_end:]
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"Model input shape: {input_shape}")
        
        # Create folder for saving model and tuning results
        model_folder = os.path.join(MODEL_DIR, f"BestEnsembleModel_{ticker}")
        os.makedirs(model_folder, exist_ok=True)
        tuning_dir = os.path.join(model_folder, "tuning")
        if os.path.exists(tuning_dir):
            import shutil
            shutil.rmtree(tuning_dir)
        best_hps_file = os.path.join(model_folder, "best_hyperparameters.json")
        if os.path.exists(best_hps_file):
            os.remove(best_hps_file)
        tuning_flag_file = os.path.join(model_folder, "hp_tuning_complete.flag")
        if os.path.exists(tuning_flag_file):
            os.remove(tuning_flag_file)
        
        print(f"🔍 Running hyperparameter tuning for {ticker} from scratch...")
        def model_builder(hp):
            return build_ensemble_model(hp, input_shape=input_shape)
        
        tuner = kt.BayesianOptimization(
            model_builder,
            objective="val_loss",
            max_trials=15,
            executions_per_trial=2,
            directory=tuning_dir,
            project_name=f"ensemble_{ticker}"
        )
        start_tuning = time.time()
        tuner.search(X_train, y_train, epochs=30, validation_data=(X_val, y_val), verbose=1)
        tuning_time = time.time() - start_tuning
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_dict = {param: best_hps.get(param) for param in best_hps.values.keys()}
        with open(best_hps_file, "w") as f:
            json.dump(best_hps_dict, f)
        with open(tuning_flag_file, "w") as f:
            f.write("tuning complete")
        
        print("Best hyperparameters found:", best_hps_dict)
        print(f"Tuning time: {tuning_time:.2f} seconds")
        model = tuner.hypermodel.build(best_hps)
        
        # Train final model and measure training time
        BATCH_SIZE = 32
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        start_train = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        training_time = time.time() - start_train
        
        # Save the model with .keras extension
        best_model_path = os.path.join(model_folder, f"{ticker}_best_model.keras")
        model.save(best_model_path)
        print(f"✅ Best Ensemble Model for {ticker} saved to {best_model_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Train Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ticker} - Training & Validation Loss")
        plt.legend()
        history_plot_path = os.path.join(model_folder, "training_history.png")
        plt.savefig(history_plot_path)
        plt.close()
        print(f"✅ Training history graph saved to {history_plot_path}")
        
        # Evaluate model on test set
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
        
        predictions = model.predict(X_test)
        predictions_rescaled = invert_target_scaling(predictions, ticker)
        y_test_rescaled = invert_target_scaling(y_test, ticker)
        
        # Calculate directional accuracy
        correct_direction = 0
        for i in range(len(y_test_rescaled) - 1):
            if (y_test_rescaled[i+1] - y_test_rescaled[i]) * (predictions_rescaled[i+1] - predictions_rescaled[i]) >= 0:
                correct_direction += 1
        directional_accuracy = (correct_direction / (len(y_test_rescaled) - 1)) * 100 if len(y_test_rescaled) > 1 else 0
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Plot predicted vs actual with directional colors
        x_vals = np.arange(len(y_test_rescaled))
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, y_test_rescaled, label="Actual Price", color='blue')
        for i in range(len(x_vals) - 1):
            seg_color = 'green' if (y_test_rescaled[i+1] - y_test_rescaled[i]) * (predictions_rescaled[i+1] - predictions_rescaled[i]) >= 0 else 'red'
            plt.plot(x_vals[i:i+2], predictions_rescaled[i:i+2], color=seg_color)
        blue_line = Line2D([0], [0], color='blue', label='Actual Price')
        green_line = Line2D([0], [0], color='green', label='Predicted (Correct Dir)')
        red_line = Line2D([0], [0], color='red', label='Predicted (Wrong Dir)')
        plt.legend(handles=[blue_line, green_line, red_line])
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(f"{ticker} - Predicted vs Actual (Inverse-Scaled)")
        plot_path = os.path.join(model_folder, "pred_vs_actual_inverscaled.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Inverse-scaled prediction vs actual plot saved to {plot_path}")
        
        # Save additional performance stats to file
        stats_path = os.path.join(model_folder, "model_performance.txt")
        with open(stats_path, "w") as f:
            f.write(f"Test Loss: {loss:.4f}\n")
            f.write(f"Test MAE: {mae:.4f}\n")
            f.write(f"Directional Accuracy: {directional_accuracy:.2f}%\n")
            f.write(f"Tuning Time (sec): {tuning_time:.2f}\n")
            f.write(f"Training Time (sec): {training_time:.2f}\n")
        print(f"✅ Performance stats saved to {stats_path}")
        print(f"🎉 Re-training and hyperparameter search complete for {ticker}!\n")

if __name__ == "__main__":
    main()
