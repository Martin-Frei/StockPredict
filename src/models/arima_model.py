import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA model for time series forecasting."""
    
    def __init__(self, order=(5, 1, 0)):
        """
        Initialize ARIMA model.
        
        Args:
            order: Tuple of (p, d, q) where:
                   p = number of autoregressive terms
                   d = number of differences
                   q = number of moving average terms
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def check_stationarity(self, data):
        """Check if time series is stationary using Augmented Dickey-Fuller test."""
        result = adfuller(data)
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        
        if result[1] <= 0.05:
            print("Data is stationary")
            return True
        else:
            print("Data is non-stationary")
            return False
    
    def train(self, train_data):
        """
        Train ARIMA model.
        
        Args:
            train_data: Training time series data
        """
        print(f"Training ARIMA model with order {self.order}")
        
        # Check stationarity
        self.check_stationarity(train_data)
        
        # Fit the model
        self.model = ARIMA(train_data, order=self.order)
        self.fitted_model = self.model.fit()
        
        print("ARIMA model trained successfully")
        print(self.fitted_model.summary())
        
        return self.fitted_model
    
    def predict(self, steps=30):
        """
        Make predictions for future steps.
        
        Args:
            steps: Number of future steps to predict
            
        Returns:
            Predicted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_fitted_values(self):
        """Get fitted values on training data."""
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.fitted_model.fittedvalues
    
    def evaluate(self, test_data):
        """
        Evaluate model performance.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.predict(steps=len(test_data))
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - test_data))
        mse = np.mean((predictions - test_data) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print("\nARIMA Model Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
