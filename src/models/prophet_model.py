import pandas as pd
import numpy as np
from pathlib import Path
import logging
from prophet import Prophet
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetPredictor:
    """
    Prophet Model für Trend-Following
    - Erkennt langfristige Trends
    - Saisonalität (Wochentag, Stunde)
    - Changepoints (Trendwechsel)
    """
    
    def __init__(self):
        self.data_processed = config.data_processed
        self.predictions_path = config.predictions_path / 'prophet_predictions'
        self.predictions_path.mkdir(parents=True, exist_ok=True)
        
        self.bank_symbols = config.bank_symbols
        self.models = {}
        self.predictions = {}
        
        # Prophet params from config
        self.prophet_params = {
            'growth': 'linear',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': True
        }
        
        logger.info("Prophet Predictor initialized")
    
    def load_data(self, symbol):
        """Load processed data"""
        csv_file = self.data_processed / f"{symbol}_enhanced.csv"
        
        if not csv_file.exists():
            return None
        
        df = pd.read_csv(csv_file)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet (needs 'ds' and 'y')"""
        
        # Remove timezone (Prophet doesn't support it)
        df_copy = df.copy()
        df_copy['DateTime'] = pd.to_datetime(df_copy['DateTime']).dt.tz_localize(None)
        
        prophet_df = pd.DataFrame({
            'ds': df_copy['DateTime'],
            'y': df_copy['Close']
        })
        
        return prophet_df
    
    def train_model(self, symbol):
        """Train Prophet model"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING PROPHET: {symbol}")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_data(symbol)
        if df is None:
            return None
        
        # Prepare for Prophet
        prophet_df = self.prepare_prophet_data(df)
        
        logger.info(f"{symbol}: {len(prophet_df)} samples")
        
        # Create and train Prophet
        model = Prophet(**self.prophet_params)
        
        logger.info("Training Prophet...")
        model.fit(prophet_df)
        
        logger.info(f"✅ {symbol}: Prophet trained")
        
        self.models[symbol] = model
        return model
    
    def predict_7h(self, symbol):
        """Predict next 7 hours"""
        
        if symbol not in self.models:
            logger.warning(f"{symbol}: No model trained")
            return None
        
        logger.info(f"\nPredicting 7h for {symbol}...")
        
        model = self.models[symbol]
        
        # Create future dataframe for 7 hours
        future = model.make_future_dataframe(periods=7, freq='h')
        
        # Predict
        forecast = model.predict(future)
        
        # Get last 7 predictions
        predictions = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # Calculate returns
        last_price = forecast.iloc[-8]['yhat']  # Last historical
        predictions['predicted_return'] = (predictions['yhat'] - last_price) / last_price
        predictions['cumulative_return'] = predictions['predicted_return'].cumsum()
        
        total_return = predictions['predicted_return'].sum()
        
        logger.info(f"✅ {symbol} Prophet Prediction:")
        logger.info(f"   7h Return: {total_return:.4f} ({total_return*100:.2f}%)")
        
        self.predictions[symbol] = predictions
        return predictions
    
    def train_all_banks(self):
        """Train Prophet for all banks"""
        
        logger.info("\n" + "="*60)
        logger.info("PROPHET TRAINING - ALL BANKS")
        logger.info("="*60)
        
        for i, symbol in enumerate(self.bank_symbols, 1):
            logger.info(f"\n[{i}/{len(self.bank_symbols)}] {symbol}")
            self.train_model(symbol)
        
        logger.info("\n✅ All Prophet models trained!")
    
    def predict_all_banks(self):
        """Generate predictions for all banks"""
        
        logger.info("\n" + "="*60)
        logger.info("PROPHET PREDICTIONS - ALL BANKS")
        logger.info("="*60)
        
        for symbol in self.models.keys():
            self.predict_7h(symbol)
    
    def save_predictions(self):
        """Save predictions"""
        
        logger.info("\nSaving Prophet predictions...")
        
        for symbol, pred_df in self.predictions.items():
            output_file = self.predictions_path / f"{symbol}_prophet_7h.csv"
            pred_df.to_csv(output_file, index=False)
            logger.info(f"✅ {symbol}: {output_file.name}")
        
        # Summary
        summary_data = []
        for symbol, pred_df in self.predictions.items():
            total_return = pred_df['predicted_return'].sum()
            
            summary_data.append({
                'Symbol': symbol,
                'Total_7h_Return': total_return,
                'Total_7h_Return_Pct': total_return * 100,
                'Direction': 'Bullish' if total_return > 0 else 'Bearish'
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('Total_7h_Return', ascending=False)
        
        summary_file = self.predictions_path / 'prophet_trading_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"\n✅ Summary: {summary_file}")
        
        # Top 5
        logger.info("\n" + "="*60)
        logger.info("TOP 5 PROPHET PREDICTIONS:")
        logger.info("="*60)
        for idx, row in summary_df.head(5).iterrows():
            logger.info(f"{row['Symbol']:5s} | Return: {row['Total_7h_Return_Pct']:+6.2f}% | {row['Direction']}")

def run_prophet_pipeline():
    """Run Prophet pipeline"""
    predictor = ProphetPredictor()
    
    predictor.train_all_banks()
    predictor.predict_all_banks()
    predictor.save_predictions()
    
    logger.info("\n🎉 PROPHET PIPELINE COMPLETE!")

if __name__ == "__main__":
    run_prophet_pipeline()