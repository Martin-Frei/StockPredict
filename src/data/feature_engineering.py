import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Modern Feature Engineering Pipeline
    - Technical Indicators
    - Macro Features Integration
    - Target Variables (7h predictions)
    """
    
    def __init__(self):
        self.data_raw = config.data_raw
        self.data_processed = config.data_processed
        self.bank_symbols = config.bank_symbols
        self.macro_symbols = config.macro_symbols
        
        logger.info("Feature Engineer initialized")
        logger.info(f"Banks: {len(self.bank_symbols)}")
        logger.info(f"Macro: {len(self.macro_symbols)}")
    
    def load_stock_data(self, symbol):
        """Load raw stock data"""
        csv_file = self.data_raw / f"{symbol}.csv"
        
        if not csv_file.exists():
            logger.warning(f"{symbol}: CSV not found")
            return None
        
        try:
            df = pd.read_csv(csv_file)
            
            # Handle both 'Datetime' and 'DateTime' column names
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'DateTime'})
            
            df['DateTime'] = pd.to_datetime(df['DateTime'], utc=True)
            df = df.sort_values('DateTime')
            logger.info(f"{symbol}: Loaded {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"{symbol}: Error loading - {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        
        # Price Ratios
        df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
        df['Price_MA10_Ratio'] = df['Close'] / df['MA_10']
        df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
        
        # Volatility
        df['Vol_5'] = df['Returns'].rolling(5).std()
        df['Vol_10'] = df['Returns'].rolling(10).std()
        df['Vol_20'] = df['Returns'].rolling(20).std()
        
        # Volume Features
        df['Volume_MA'] = df['Volume'].rolling(10).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1)
        
        # High-Low Spread
        df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
        df['HL_Spread_MA'] = df['HL_Spread'].rolling(5).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Time Features
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
        df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
        
        # ========================================
        # DIVERGENCE FEATURES (NEU!)
        # ========================================
        
        window = 10
        
        # MACD Divergence
        if len(df) >= window:
            # Price Trend (last 10 hours)
            price_slope = (df['Close'] - df['Close'].shift(window)) / (df['Close'].shift(window) + 1e-10)
            
            # MACD Trend (last 10 hours)
            macd_slope = (df['MACD'] - df['MACD'].shift(window)) / (df['MACD'].shift(window).abs() + 1e-10)
            
            # Divergence: Price up + MACD down = Bearish Divergence
            df['MACD_Divergence'] = price_slope - macd_slope
        else:
            df['MACD_Divergence'] = 0
        
        # RSI Divergence
        if len(df) >= window:
            # RSI Trend
            rsi_slope = (df['RSI'] - df['RSI'].shift(window)) / (df['RSI'].shift(window) + 1e-10)
            
            # Divergence
            df['RSI_Divergence'] = price_slope - rsi_slope
        else:
            df['RSI_Divergence'] = 0
        
        # Volume Divergence
        if len(df) >= window:
            # Volume Trend
            volume_slope = (df['Volume'] - df['Volume'].shift(window)) / (df['Volume'].shift(window) + 1e-10)
            
            # Divergence: Price up + Volume down = Weak trend
            df['Volume_Divergence'] = price_slope - volume_slope
        else:
            df['Volume_Divergence'] = 0
        
        # Classic Bearish/Bullish Divergence Detection
        if len(df) >= 20:
            # Price near 20-period high?
            price_high_20 = df['Close'].rolling(20).max()
            df['Price_Near_High'] = df['Close'] / (price_high_20 + 1e-10)
            
            # MACD near 20-period high?
            macd_high_20 = df['MACD'].rolling(20).max()
            df['MACD_Near_High'] = df['MACD'] / (macd_high_20.abs() + 1e-10)
            
            # Classic Divergence: Price makes new highs, MACD doesn't
            df['Classic_Divergence'] = df['Price_Near_High'] - df['MACD_Near_High']
        else:
            df['Price_Near_High'] = 1.0
            df['MACD_Near_High'] = 1.0
            df['Classic_Divergence'] = 0.0
        
        logger.info(f"   Technical indicators calculated (incl. 6 divergence features)")
        
        # ========================================
        # LAG FEATURES (Momentum & Trend Memory)
        # ========================================
        
        # Returns Lags (Momentum)
        df['Returns_lag1'] = df['Returns'].shift(1)  # 1h ago
        df['Returns_lag2'] = df['Returns'].shift(2)  # 2h ago
        df['Returns_lag3'] = df['Returns'].shift(3)  # 3h ago
        
        # Returns Momentum (sum of last 3h)
        df['Returns_momentum_3h'] = (
            df['Returns_lag1'].fillna(0) + 
            df['Returns_lag2'].fillna(0) + 
            df['Returns_lag3'].fillna(0)
        )
        
        # MACD Lag (yesterday's momentum)
        df['MACD_lag1'] = df['MACD'].shift(1)
        
        # RSI Lag
        df['RSI_lag1'] = df['RSI'].shift(1)
        
        # Volume Lag & Trend
        df['Volume_lag1'] = df['Volume'].shift(1)
        df['Volume_trend'] = df['Volume'] - df['Volume_lag1']
        
        # Price Trend Direction (last 5h)
        if len(df) >= 5:
            df['Price_trend_5h'] = (df['Close'] > df['Close'].shift(5)).astype(int)
        else:
            df['Price_trend_5h'] = 0
        
        # Volatility Change
        df['Vol_change'] = df['Vol_10'] - df['Vol_10'].shift(1)
        
        # Consecutive Up/Down Hours
        df['Up_streak'] = (df['Returns'] > 0).astype(int)
        df['Up_streak'] = df['Up_streak'].groupby(
            (df['Up_streak'] != df['Up_streak'].shift()).cumsum()
        ).cumsum()
        
        logger.info(f"   Lag features calculated (11 new features)")
        
        return df
    
    def add_macro_features(self, bank_df, symbol):
        """Add macro features to bank data"""
        
        bank_df = bank_df.set_index('DateTime')
        macro_added = 0
        
        # Load and merge macro features
        for macro in self.macro_symbols:
            macro_data = self.load_stock_data(macro)
            if macro_data is None:
                continue
            
            macro_data = macro_data.set_index('DateTime')
            
            # Calculate returns if not exists
            if 'Returns' not in macro_data.columns:
                macro_data['Returns'] = macro_data['Close'].pct_change()
            
            # WICHTIG: Reindex mit nearest-neighbor + forward fill
            try:
                # Methode 1: Nearest neighbor (max 2h tolerance)
                macro_aligned = macro_data[['Returns', 'Close']].reindex(
                    bank_df.index,
                    method='nearest',
                    tolerance=pd.Timedelta('2 hours')
                )
                
                # Falls immer noch NaN: Forward-fill
                macro_aligned = macro_aligned.ffill()
                
                # Add to bank dataframe
                bank_df[f'{macro}_Return'] = macro_aligned['Returns']
                bank_df[f'{macro}_Level'] = macro_aligned['Close']
                
                # Count non-zero values
                non_zero = (bank_df[f'{macro}_Level'] != 0).sum()
                logger.info(f"   {macro}: {non_zero}/{len(bank_df)} non-zero values")
                
                macro_added += 1
                
            except Exception as e:
                logger.warning(f"   {macro}: Error aligning - {e}")
                continue
        
        logger.info(f"   {symbol}: Added {macro_added} macro features")
        
        # Fill remaining NaN with 0 (but should be minimal now)
        macro_cols = [col for col in bank_df.columns if any(m in col for m in self.macro_symbols)]
        for col in macro_cols:
            remaining_nan = bank_df[col].isna().sum()
            if remaining_nan > 0:
                logger.warning(f"   {col}: {remaining_nan} NaNs filled with 0")
                bank_df[col] = bank_df[col].fillna(0)
        
        bank_df = bank_df.reset_index()
        return bank_df
    
    def create_target_variables(self, df):
        """Create 7h prediction targets"""
        
        # 7h return target
        df['Target_7h'] = df['Returns'].rolling(7).sum().shift(-7)
        df['Target_Direction_7h'] = (df['Target_7h'] > 0).astype(int)
        
        # Also create 1h and 4h for comparison
        df['Target_1h'] = df['Returns'].shift(-1)
        df['Target_4h'] = df['Returns'].rolling(4).sum().shift(-4)
        
        logger.info(f"   Target variables created (7h focus)")
        return df
    
    def clean_data(self, df):
        """Clean data - remove NaN"""
        
        initial_rows = len(df)
        
        # Critical columns that cannot be NaN
        critical_cols = ['Close', 'Returns', 'Target_7h']
        df = df.dropna(subset=critical_cols)
        
        # Remove rows with too many NaN (60% threshold statt 80%)
        threshold = len(df.columns) * 0.6
        df = df.dropna(thresh=threshold)
        
        # Fill remaining NaNs in non-critical columns with 0
        non_critical_cols = [col for col in df.columns if col not in critical_cols + ['DateTime']]
        for col in non_critical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        logger.info(f"   Cleaned: {initial_rows} → {final_rows} rows ({removed} removed)")
        return df
    
    def process_bank(self, symbol):
        """Complete feature engineering pipeline for one bank"""
        
        logger.info(f"\nProcessing {symbol}...")
        
        # Load data
        df = self.load_stock_data(symbol)
        if df is None:
            return None
        
        # Technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Macro features
        df = self.add_macro_features(df, symbol)
        
        # Target variables
        df = self.create_target_variables(df)
        
        # Clean
        df = self.clean_data(df)
        
        # Save
        if len(df) >= 100:
            output_file = self.data_processed / f"{symbol}_enhanced.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"✅ {symbol}: Saved {len(df)} rows to {output_file.name}")
            return df
        else:
            logger.warning(f"❌ {symbol}: Only {len(df)} rows - too few!")
            return None
    
    def process_all_banks(self):
        """Process all bank stocks"""
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("="*60)
        
        results = {}
        
        for i, symbol in enumerate(self.bank_symbols, 1):
            logger.info(f"\n[{i}/{len(self.bank_symbols)}] {symbol}")
            df = self.process_bank(symbol)
            if df is not None:
                results[symbol] = df
        
        logger.info("\n" + "="*60)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Successfully processed: {len(results)}/{len(self.bank_symbols)}")
        logger.info(f"Output: data/processed/")
        logger.info("="*60 + "\n")
        
        return results

def run_feature_engineering():
    """Run feature engineering pipeline"""
    engineer = FeatureEngineer()
    return engineer.process_all_banks()

if __name__ == "__main__":
    run_feature_engineering()