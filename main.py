import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.data_preprocessing import DataPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.prophet_model import ProphetModel
from src.models.ensemble_model import EnsembleModel
from src.utils.visualization import Visualizer


class StockPredictor:
    """Main orchestrator for stock prediction system."""
    
    def __init__(self, ticker='AAPL', start_date='2020-01-01', end_date=None):
        """
        Initialize stock predictor.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data (default: today)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize components
        self.preprocessor = DataPreprocessor(ticker, start_date, end_date)
        self.visualizer = Visualizer()
        
        # Models
        self.arima_model = None
        self.lstm_model = None
        self.prophet_model = None
        self.ensemble_model = None
        
        # Data
        self.train_data = None
        self.test_data = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for modeling."""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("=" * 80)
        
        # Fetch data
        self.preprocessor.fetch_data()
        
        # Clean data
        self.preprocessor.clean_data()
        
        # Add features
        self.preprocessor.add_features()
        
        # Split data
        self.train_data, self.test_data = self.preprocessor.split_data(train_ratio=0.8)
        
        print(f"\nTrain data: {len(self.train_data)} records")
        print(f"Test data: {len(self.test_data)} records")
        
        # Visualize data
        self.visualizer.plot_stock_data(
            self.preprocessor.data,
            title=f'{self.ticker} Stock Price History',
            save_as='stock_history.png'
        )
        
        return self.train_data, self.test_data
    
    def train_arima(self):
        """Train ARIMA model."""
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING ARIMA MODEL")
        print("=" * 80)
        
        self.arima_model = ARIMAModel(order=(5, 1, 0))
        self.arima_model.train(self.train_data['Close'].values)
        
        return self.arima_model
    
    def train_lstm(self):
        """Train LSTM model."""
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING LSTM MODEL")
        print("=" * 80)
        
        # Prepare sequences for LSTM
        self.preprocessor.scale_data('Close')
        X, y = self.preprocessor.prepare_sequences(sequence_length=60)
        
        # Split into train and test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.lstm_model = LSTMModel(sequence_length=60, units=50)
        history = self.lstm_model.train(X_train, y_train, epochs=50, batch_size=32)
        
        # Visualize training history
        self.visualizer.plot_training_history(history, save_as='lstm_training_history.png')
        
        return self.lstm_model
    
    def train_prophet(self):
        """Train Prophet model."""
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING PROPHET MODEL")
        print("=" * 80)
        
        self.prophet_model = ProphetModel()
        self.prophet_model.train(self.train_data, date_column='Date', value_column='Close')
        
        return self.prophet_model
    
    def make_predictions(self, forecast_days=30):
        """Make predictions using all models."""
        print("\n" + "=" * 80)
        print("STEP 5: MAKING PREDICTIONS")
        print("=" * 80)
        
        predictions = {}
        
        # ARIMA predictions
        print("\nARIMA Predictions:")
        arima_pred = self.arima_model.predict(steps=len(self.test_data))
        predictions['ARIMA'] = arima_pred
        
        # LSTM predictions
        print("\nLSTM Predictions:")
        # Prepare test sequences
        X_test, y_test = self.preprocessor.prepare_sequences(sequence_length=60)
        split_idx = int(len(X_test) * 0.8)
        X_test = X_test[split_idx:]
        
        lstm_pred_scaled = self.lstm_model.predict(X_test)
        lstm_pred = self.preprocessor.inverse_scale(lstm_pred_scaled).flatten()
        predictions['LSTM'] = lstm_pred
        
        # Prophet predictions
        print("\nProphet Predictions:")
        prophet_forecast = self.prophet_model.predict(periods=len(self.test_data))
        # Get predictions for test period
        prophet_pred = prophet_forecast['yhat'].values[-len(self.test_data):]
        predictions['Prophet'] = prophet_pred
        
        return predictions
    
    def evaluate_models(self, predictions):
        """Evaluate all models."""
        print("\n" + "=" * 80)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 80)
        
        actual_values = self.test_data['Close'].values
        metrics = {}
        
        # Evaluate ARIMA
        arima_metrics = self.arima_model.evaluate(actual_values)
        metrics['ARIMA'] = arima_metrics
        
        # Evaluate LSTM (need to align with test data)
        lstm_pred = predictions['LSTM']
        min_len = min(len(lstm_pred), len(actual_values))
        
        mae = np.mean(np.abs(lstm_pred[:min_len] - actual_values[:min_len]))
        mse = np.mean((lstm_pred[:min_len] - actual_values[:min_len]) ** 2)
        rmse = np.sqrt(mse)
        
        metrics['LSTM'] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
        print("\nLSTM Model Evaluation:")
        for metric, value in metrics['LSTM'].items():
            print(f"{metric}: {value:.4f}")
        
        # Evaluate Prophet
        prophet_metrics, _ = self.prophet_model.evaluate(self.test_data, 
                                                          date_column='Date', 
                                                          value_column='Close')
        metrics['Prophet'] = prophet_metrics
        
        # Visualize metrics comparison
        self.visualizer.plot_metrics_comparison(metrics, save_as='metrics_comparison.png')
        
        return metrics
    
    def create_ensemble(self, predictions, weights=None):
        """Create ensemble predictions."""
        print("\n" + "=" * 80)
        print("STEP 7: ENSEMBLE PREDICTIONS")
        print("=" * 80)
        
        self.ensemble_model = EnsembleModel(weights=weights)
        
        # Add predictions from each model
        for model_name, pred in predictions.items():
            self.ensemble_model.add_prediction(model_name, pred)
        
        # Evaluate ensemble with different methods
        actual_values = self.test_data['Close'].values
        best_method, results = self.ensemble_model.optimize_weights(actual_values)
        
        # Get best ensemble predictions
        ensemble_pred = self.ensemble_model.combine_predictions(method=best_method)
        
        return ensemble_pred, best_method
    
    def visualize_results(self, predictions, ensemble_pred):
        """Visualize all predictions."""
        print("\n" + "=" * 80)
        print("STEP 8: VISUALIZATION")
        print("=" * 80)
        
        # Prepare data for visualization
        test_dates = self.test_data['Date'].values
        actual_values = self.test_data['Close'].values
        
        # Align all predictions to the same length
        min_len = min(len(test_dates), len(actual_values), 
                     len(predictions['ARIMA']), len(predictions['LSTM']), 
                     len(predictions['Prophet']), len(ensemble_pred))
        
        aligned_predictions = {
            'ARIMA': predictions['ARIMA'][:min_len],
            'LSTM': predictions['LSTM'][:min_len],
            'Prophet': predictions['Prophet'][:min_len],
            'Ensemble': ensemble_pred[:min_len]
        }
        
        # Plot all predictions
        self.visualizer.plot_predictions(
            test_dates[:min_len],
            actual_values[:min_len],
            aligned_predictions,
            title=f'{self.ticker} Stock Price Predictions',
            save_as='all_predictions.png'
        )
        
        print("\nVisualization complete!")
    
    def run(self, forecast_days=30, ensemble_weights=None):
        """Run the complete prediction pipeline."""
        print("\n" + "=" * 80)
        print(f"STOCK PREDICTION SYSTEM FOR {self.ticker}")
        print("=" * 80)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2-4: Train all models
        self.train_arima()
        self.train_lstm()
        self.train_prophet()
        
        # Step 5: Make predictions
        predictions = self.make_predictions(forecast_days)
        
        # Step 6: Evaluate models
        metrics = self.evaluate_models(predictions)
        
        # Step 7: Create ensemble
        ensemble_pred, best_method = self.create_ensemble(predictions, ensemble_weights)
        
        # Step 8: Visualize results
        self.visualize_results(predictions, ensemble_pred)
        
        print("\n" + "=" * 80)
        print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults saved in 'outputs/' directory")
        print(f"Best ensemble method: {best_method}")
        
        return predictions, ensemble_pred, metrics


if __name__ == "__main__":
    # Example usage
    predictor = StockPredictor(ticker='AAPL', start_date='2020-01-01')
    predictions, ensemble_pred, metrics = predictor.run(forecast_days=30)
