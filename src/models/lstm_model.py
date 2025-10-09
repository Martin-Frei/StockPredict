import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class LSTMModel:
    """LSTM Neural Network for stock price prediction."""
    
    def __init__(self, sequence_length=60, units=50, dropout_rate=0.2):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """Build the LSTM architecture."""
        self.model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            
            LSTM(units=self.units, return_sequences=True),
            Dropout(self.dropout_rate),
            
            LSTM(units=self.units, return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(units=25),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("LSTM Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of data to use for validation
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        print(f"Training LSTM model for {epochs} epochs...")
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        print("LSTM model trained successfully")
        return self.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or train() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        rmse = np.sqrt(mse)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        
        print("\nLSTM Model Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
