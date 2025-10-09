import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """Handles data fetching, cleaning, and preprocessing for stock prediction."""
    
    def __init__(self, ticker='AAPL', start_date='2020-01-01', end_date=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or pd.Timestamp.now().strftime('%Y-%m-%d')
        self.scaler = MinMaxScaler()
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(start=self.start_date, end=self.end_date)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
            
        print(f"Fetched {len(self.data)} records")
        return self.data
    
    def clean_data(self):
        """Clean the data by handling missing values and outliers."""
        if self.data is None:
            raise ValueError("No data to clean. Call fetch_data() first.")
        
        # Remove any rows with missing values
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        
        if len(self.data) < initial_rows:
            print(f"Removed {initial_rows - len(self.data)} rows with missing values")
        
        # Reset index
        self.data.reset_index(inplace=True)
        
        return self.data
    
    def add_features(self):
        """Add technical indicators and features for modeling."""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() and clean_data() first.")
        
        # Moving averages
        self.data['MA7'] = self.data['Close'].rolling(window=7).mean()
        self.data['MA21'] = self.data['Close'].rolling(window=21).mean()
        
        # Exponential moving average
        self.data['EMA'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        
        # Volatility
        self.data['Volatility'] = self.data['Close'].rolling(window=7).std()
        
        # Price momentum
        self.data['Momentum'] = self.data['Close'].diff(5)
        
        # Drop NaN values created by feature engineering
        self.data = self.data.dropna()
        
        print(f"Added technical indicators. Data shape: {self.data.shape}")
        return self.data
    
    def scale_data(self, column='Close'):
        """Scale data for neural network training."""
        if self.data is None:
            raise ValueError("No data available for scaling.")
        
        values = self.data[column].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(values)
        
        return self.scaled_data
    
    def inverse_scale(self, scaled_values):
        """Transform scaled predictions back to original scale."""
        return self.scaler.inverse_transform(scaled_values)
    
    def prepare_sequences(self, sequence_length=60):
        """Prepare sequences for LSTM training."""
        if self.scaled_data is None:
            self.scale_data()
        
        X, y = [], []
        for i in range(sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-sequence_length:i, 0])
            y.append(self.scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def split_data(self, train_ratio=0.8):
        """Split data into train and test sets."""
        if self.data is None:
            raise ValueError("No data available for splitting.")
        
        split_idx = int(len(self.data) * train_ratio)
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        return train_data, test_data
