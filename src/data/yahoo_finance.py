import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceLoader:
    """
    Modern Yahoo Finance Loader
    - Unlimited API calls
    - Hourly data available for free
    - Fast bulk downloads
    """
    
    def __init__(self):
        self.save_path = config.data_raw
        logger.info("Yahoo Finance Loader initialized")
    
    def fetch_stock_data(self, symbol, hours_back=2160, interval='1h'):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g. 'JPM')
            hours_back: Hours to fetch (default: 2160 = ~3 months)
            interval: '1h', '1d', '1wk', '1mo'
        """
        logger.info(f"Fetching {symbol} from Yahoo Finance ({interval} interval)...")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                prepost=False
            )
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Format conversion (Alpha Vantage compatible)
            df = data.reset_index()
            
            # Rename columns
            column_mapping = {
                'Datetime': 'DateTime',
                'Date': 'DateTime',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Add Adj Close if not exists
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # Ensure DateTime
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Only trading days (Mo-Fr)
            df = df[df['DateTime'].dt.weekday < 5]
            
            # Sort and clean
            df = df.sort_values('DateTime')
            df = df.dropna()
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            logger.info(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def save_to_csv(self, data, symbol):
        """Save DataFrame to CSV"""
        if data.empty:
            logger.warning(f"No data to save for {symbol}")
            return None
        
        csv_file = self.save_path / f"{symbol}.csv"
        data.to_csv(csv_file, index=False)
        logger.info(f"Saved: {csv_file}")
        return csv_file
    
    def load_from_csv(self, symbol):
        """Load existing CSV file"""
        csv_file = self.save_path / f"{symbol}.csv"
        
        if not csv_file.exists():
            return None
        
        try:
            data = pd.read_csv(csv_file)
            data['DateTime'] = pd.to_datetime(data['DateTime'])
            logger.info(f"Loaded existing CSV: {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None
    
    def update_stock_data(self, symbol, hours_back=2160, interval='1h'):
        """Update stock data (incremental)"""
        existing_data = self.load_from_csv(symbol)
        
        if existing_data is not None and not existing_data.empty:
            last_date = existing_data['DateTime'].max()
            logger.info(f"Last date in CSV: {last_date}")
            
            # Calculate hours since last data
            hours_since = (datetime.now() - pd.Timestamp(last_date).tz_localize(None)).total_seconds() / 3600
            hours_to_fetch = min(int(hours_since) + 24, hours_back)  # +24 for overlap
            
            new_data = self.fetch_stock_data(symbol, hours_to_fetch, interval)
            
            if new_data.empty:
                logger.warning(f"No new data for {symbol}")
                return existing_data
            
            # Filter only new data
            new_data = new_data[new_data['DateTime'] > last_date]
            
            if new_data.empty:
                logger.info(f"{symbol} already up to date")
                return existing_data
            
            # Combine
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.sort_values('DateTime')
            combined_data = combined_data.drop_duplicates(subset=['DateTime'], keep='last')
            
            logger.info(f"Added {len(new_data)} new data points")
        else:
            logger.info(f"First load for {symbol}")
            combined_data = self.fetch_stock_data(symbol, hours_back, interval)
        
        if not combined_data.empty:
            self.save_to_csv(combined_data, symbol)
        
        return combined_data
    
    def load_multiple_stocks(self, symbols, hours_back=2160, interval='1h'):
        """Load multiple stocks at once"""
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
            data = self.update_stock_data(symbol, hours_back, interval)
            if not data.empty:
                results[symbol] = data
        
        logger.info(f"Successfully loaded {len(results)}/{len(symbols)} stocks")
        return results