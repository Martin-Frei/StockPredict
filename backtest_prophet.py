import pandas as pd
import numpy as np
from pathlib import Path
import logging
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetWalkForward:
    """
    Walk-Forward Backtesting für Prophet
    Täglich sliding: 6 Monate Train → 1 Tag (7h) Test
    """
    
    def __init__(self):
        self.data_processed = config.data_processed
        self.results_path = Path('data/backtest_results')
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.prophet_params = {
            'growth': 'linear',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': True
        }
        
        logger.info("Prophet Walk-Forward Backtester initialized")
    
    def load_data(self, symbol):
        """Load processed data"""
        csv_file = self.data_processed / f"{symbol}_enhanced.csv"
        
        if not csv_file.exists():
            return None
        
        df = pd.read_csv(csv_file)
        df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
        df = df.sort_values('DateTime')
        
        return df
    
    def walk_forward_test(self, symbol, train_hours=780, test_hours=7, step_hours=7):
        """
        Walk-Forward Backtesting für Prophet
        
        Args:
            symbol: Bank symbol
            train_hours: 780h (6 Monate)
            test_hours: 7h (1 Tag)
            step_hours: 7h (täglich)
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROPHET WALK-FORWARD: {symbol}")
        logger.info(f"Training: {train_hours}h | Test: {test_hours}h | Step: {step_hours}h")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_data(symbol)
        if df is None:
            return None
        
        results = []
        predictions_all = []
        
        start_idx = 0
        window_num = 0
        
        # Sliding Window
        while start_idx + train_hours + test_hours <= len(df):
            window_num += 1
            
            # Define windows
            train_start = start_idx
            train_end = start_idx + train_hours
            test_start = train_end
            test_end = test_start + test_hours
            
            # Split data
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            if len(test_data) == 0:
                break
            
            # Prepare Prophet format
            prophet_train = pd.DataFrame({
                'ds': train_data['DateTime'],
                'y': train_data['Close']
            })
            
            # Train Prophet
            try:
                model = Prophet(**self.prophet_params)
                model.fit(prophet_train, algorithm='Newton')  # Schneller!
                
                # Create future dataframe for test period
                future = pd.DataFrame({
                    'ds': test_data['DateTime']
                })
                
                # Predict
                forecast = model.predict(future)
                
                # Calculate returns
                last_train_price = train_data['Close'].iloc[-1]
                
                predicted_prices = forecast['yhat'].values
                actual_prices = test_data['Close'].values
                
                # Returns
                predicted_returns = (predicted_prices - last_train_price) / last_train_price
                actual_returns = (actual_prices - last_train_price) / last_train_price
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(actual_returns, predicted_returns))
                mae = mean_absolute_error(actual_returns, predicted_returns)
                r2 = r2_score(actual_returns, predicted_returns)
                
                # Direction accuracy
                correct_direction = sum(
                    (predicted_returns > 0) == (actual_returns > 0)
                ) / len(actual_returns)
                
                # Store
                results.append({
                    'window': window_num,
                    'test_date': test_data['DateTime'].iloc[0],
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'direction_accuracy': correct_direction,
                    'actual_return': actual_returns.mean(),
                    'predicted_return': predicted_returns.mean(),
                    'prediction_error': abs(predicted_returns.mean() - actual_returns.mean())
                })
                
                # Store predictions
                for time, actual, pred in zip(test_data['DateTime'], actual_returns, predicted_returns):
                    predictions_all.append({
                        'DateTime': time,
                        'Window': window_num,
                        'Actual_Return': actual,
                        'Predicted_Return': pred,
                        'Error': actual - pred,
                        'Direction_Correct': (pred > 0) == (actual > 0)
                    })
                
                # Log every 50 days
                if window_num % 50 == 0:
                    logger.info(f"Window {window_num}: "
                               f"{test_data['DateTime'].iloc[0].strftime('%Y-%m-%d')} | "
                               f"R²={r2:.3f} | "
                               f"Dir={correct_direction:.0%}")
            
            except Exception as e:
                logger.warning(f"Window {window_num}: Prophet failed - {e}")
            
            # Slide window
            start_idx += step_hours
        
        # Results DataFrame
        results_df = pd.DataFrame(results)
        predictions_df = pd.DataFrame(predictions_all)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"PROPHET BACKTEST SUMMARY: {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Days Tested: {len(results_df)}")
        logger.info(f"Avg R²: {results_df['r2'].mean():.3f}")
        logger.info(f"Median R²: {results_df['r2'].median():.3f}")
        logger.info(f"Avg RMSE: {results_df['rmse'].mean():.6f}")
        logger.info(f"Avg Direction Accuracy: {results_df['direction_accuracy'].mean():.1%}")
        logger.info(f"Avg Prediction Error: {results_df['prediction_error'].mean():.4f}")
        logger.info(f"Best R²: {results_df['r2'].max():.3f}")
        logger.info(f"Worst R²: {results_df['r2'].min():.3f}")
        logger.info(f"Days with R² > 0: {(results_df['r2'] > 0).sum()} ({(results_df['r2'] > 0).sum() / len(results_df):.1%})")
        
        # Save
        results_file = self.results_path / f'{symbol}_prophet_walkforward.csv'
        results_df.to_csv(results_file, index=False)
        
        predictions_file = self.results_path / f'{symbol}_prophet_predictions_daily.csv'
        predictions_df.to_csv(predictions_file, index=False)
        
        logger.info(f"\n✅ Results saved:")
        logger.info(f"   {results_file}")
        logger.info(f"   {predictions_file}")
        
        return results_df, predictions_df

def run_prophet_backtest():
    """Run Prophet Walk-Forward Backtest"""
    
    backtester = ProphetWalkForward()
    
    # Test JPM first
    test_symbols = ['JPM']
    
    logger.info("\n" + "="*60)
    logger.info("PROPHET WALK-FORWARD DAILY BACKTESTING")
    logger.info("Train: 6 Monate | Test: 1 Tag | Step: 1 Tag")
    logger.info("="*60)
    
    all_results = {}
    
    for symbol in test_symbols:
        results_df, predictions_df = backtester.walk_forward_test(symbol)
        if results_df is not None:
            all_results[symbol] = results_df
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROPHET BACKTEST COMPLETE")
    logger.info("="*60)
    
    for symbol, results_df in all_results.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Avg R²: {results_df['r2'].mean():.3f}")
        logger.info(f"  Avg Direction: {results_df['direction_accuracy'].mean():.1%}")
        logger.info(f"  Days with R² > 0: {(results_df['r2'] > 0).sum()} ({(results_df['r2'] > 0).sum() / len(results_df):.1%})")
    
    logger.info("\n🎉 PROPHET WALK-FORWARD COMPLETE!")
    logger.info("Results in: data/backtest_results/")

if __name__ == "__main__":
    run_prophet_backtest()