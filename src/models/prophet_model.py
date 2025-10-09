import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class ProphetModel:
    """Facebook Prophet model for time series forecasting."""
    
    def __init__(self, seasonality_mode='multiplicative', yearly_seasonality=True,
                 weekly_seasonality=True, daily_seasonality=False):
        """
        Initialize Prophet model.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
        """
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        self.fitted = False
        
    def prepare_data(self, data, date_column='Date', value_column='Close'):
        """
        Prepare data in Prophet format.
        
        Args:
            data: DataFrame with date and value columns
            date_column: Name of the date column
            value_column: Name of the value column
            
        Returns:
            DataFrame in Prophet format (ds, y)
        """
        df = pd.DataFrame()
        df['ds'] = pd.to_datetime(data[date_column])
        df['y'] = data[value_column].values
        
        return df
    
    def train(self, train_data, date_column='Date', value_column='Close'):
        """
        Train Prophet model.
        
        Args:
            train_data: Training data DataFrame
            date_column: Name of the date column
            value_column: Name of the value column
        """
        print("Training Prophet model...")
        
        # Prepare data
        df = self.prepare_data(train_data, date_column, value_column)
        
        # Fit the model
        self.model.fit(df)
        self.fitted = True
        
        print("Prophet model trained successfully")
        return self.model
    
    def predict(self, periods=30, freq='D'):
        """
        Make future predictions.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            DataFrame with predictions
        """
        if not self.fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast
    
    def evaluate(self, test_data, date_column='Date', value_column='Close'):
        """
        Evaluate model performance.
        
        Args:
            test_data: Test data for evaluation
            date_column: Name of the date column
            value_column: Name of the value column
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare test data
        df_test = self.prepare_data(test_data, date_column, value_column)
        
        # Get predictions for test period
        forecast = self.model.predict(df_test)
        
        # Extract predictions
        predictions = forecast['yhat'].values
        actual = df_test['y'].values
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actual))
        mse = np.mean((predictions - actual) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print("\nProphet Model Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, forecast
    
    def get_components(self):
        """Get trend and seasonality components."""
        if not self.fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model
