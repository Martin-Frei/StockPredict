import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardBacktest:
    """
    Walk-Forward Backtesting für XGBoost
    
    Gleitende Fenster mit vorwärts gehender Rücktesting!
    - Training Window: 6 Monate
    - Test Window: 1 Monat
    - Step: 1 Monat (sliding)
    """
    
    def __init__(self):
        self.data_processed = config.data_processed
        self.results_path = Path('data/backtest_results')
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # XGBoost Params
        self.params = {
            'n_estimators': 30,
            'max_depth': 3,
            'learning_rate': 0.03,
            'min_child_weight': 5,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'reg:squarederror'
        }
        
        logger.info("Walk-Forward Backtester initialized")
        logger.info(f"Training Window: 6 Monate")
        logger.info(f"Test Window: 1 Monat")
    
    def load_data(self, symbol):
        """Load enhanced data"""
        csv_file = self.data_processed / f"{symbol}_enhanced.csv"
        
        if not csv_file.exists():
            return None
        
        df = pd.read_csv(csv_file)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime')
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for XGBoost"""
        
        exclude_cols = [
            'DateTime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Target_7h', 'Target_Direction_7h', 'Target_1h', 'Target_4h', 'Dividends', 'Stock Splits'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        df_clean = df.dropna(subset=['Target_7h'])
        
        X = df_clean[feature_cols].fillna(0)
        y = df_clean['Target_7h']
        timestamps = df_clean['DateTime']
        
        return X, y, timestamps, feature_cols
    
    def walk_forward_test(self, symbol, train_hours=780, test_hours=7, step_hours=7):
        """
        Walk-Forward Backtesting - TAGES-GENAU!
        
        Args:
            symbol: Bank symbol
            train_hours: Training window (780h = 6 Monate)
            test_hours: Test window (7h = 1 Tag)
            step_hours: Step size (7h = 1 Tag)
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD BACKTEST: {symbol}")
        logger.info(f"Training: {train_hours}h (~{train_hours//130} Monate)")
        logger.info(f"Test: {test_hours}h (1 Tag)")
        logger.info(f"Step: {step_hours}h (täglich)")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_data(symbol)
        if df is None:
            return None
        
        X, y, timestamps, feature_cols = self.prepare_features(df)
        
        # Combine for splitting
        data = pd.DataFrame(X)
        data['Target_7h'] = y.values
        data['DateTime'] = timestamps.values
        
        results = []
        predictions_all = []
        
        start_idx = 0
        window_num = 0
        
        # Sliding Window
        while start_idx + train_hours + test_hours <= len(data):
            window_num += 1
            
            # Define windows
            train_start = start_idx
            train_end = start_idx + train_hours
            test_start = train_end
            test_end = test_start + test_hours
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(test_data) == 0:
                break
            
            # Prepare
            X_train = train_data[feature_cols]
            y_train = train_data['Target_7h']
            X_test = test_data[feature_cols]
            y_test = test_data['Target_7h']
            test_times = test_data['DateTime']
            
            # Train
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params['n_estimators'],
                verbose_eval=False
            )
            
            # Predict
            y_pred = model.predict(dtest)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Trading metrics
            correct_direction = sum((y_pred > 0) == (y_test > 0)) / len(y_test)
            avg_return = y_test.mean()
            pred_return = y_pred.mean()
            
            # Store
            results.append({
                'window': window_num,
                'train_start': train_data['DateTime'].iloc[0],
                'train_end': train_data['DateTime'].iloc[-1],
                'test_date': test_data['DateTime'].iloc[0],
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': correct_direction,
                'actual_return': avg_return,
                'predicted_return': pred_return
            })
            
            # Store predictions
            for time, actual, pred in zip(test_times, y_test, y_pred):
                predictions_all.append({
                    'DateTime': time,
                    'Window': window_num,
                    'Actual': actual,
                    'Predicted': pred,
                    'Error': actual - pred,
                    'Direction_Correct': (pred > 0) == (actual > 0)
                })
            
            # Log every 50 days
            if window_num % 50 == 0:
                logger.info(f"Window {window_num}: "
                        f"{test_data['DateTime'].iloc[0].strftime('%Y-%m-%d')} | "
                        f"R²={r2:.3f} | "
                        f"Dir={correct_direction:.0%}")
            
            # Slide by step_hours (1 Tag = 7h)
            start_idx += step_hours
        
        # Results DataFrame
        results_df = pd.DataFrame(results)
        predictions_df = pd.DataFrame(predictions_all)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTEST SUMMARY: {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Days Tested: {len(results_df)}")
        logger.info(f"Avg R²: {results_df['r2'].mean():.3f}")
        logger.info(f"Median R²: {results_df['r2'].median():.3f}")
        logger.info(f"Avg RMSE: {results_df['rmse'].mean():.6f}")
        logger.info(f"Avg Direction Accuracy: {results_df['direction_accuracy'].mean():.1%}")
        logger.info(f"Best R²: {results_df['r2'].max():.3f} on {results_df.loc[results_df['r2'].idxmax(), 'test_date']}")
        logger.info(f"Worst R²: {results_df['r2'].min():.3f} on {results_df.loc[results_df['r2'].idxmin(), 'test_date']}")
        logger.info(f"Days with R² > 0: {(results_df['r2'] > 0).sum()} ({(results_df['r2'] > 0).sum() / len(results_df):.1%})")
        
        # Save
        results_file = self.results_path / f'{symbol}_walkforward_daily.csv'
        results_df.to_csv(results_file, index=False)
        
        predictions_file = self.results_path / f'{symbol}_predictions_daily.csv'
        predictions_df.to_csv(predictions_file, index=False)
        
        logger.info(f"\n✅ Results saved:")
        logger.info(f"   {results_file}")
        logger.info(f"   {predictions_file}")
        
        return results_df, predictions_df

def run_walkforward_backtest():
    """Run walk-forward backtest - DAILY STEP"""
    
    backtester = WalkForwardBacktest()
    
    # Test JPM first, dann alle wenn's funktioniert
    test_symbols = ['JPM']  # Erst einer!
    
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD DAILY BACKTESTING")
    logger.info("Train: 6 Monate | Test: 1 Tag | Step: 1 Tag")
    logger.info("="*60)
    
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD BACKTESTING - 3 BANKS")
    logger.info("="*60)
    
    all_results = {}
    
    for symbol in test_symbols:
        results_df, predictions_df = backtester.walk_forward_test(symbol)
        if results_df is not None:
            all_results[symbol] = results_df
    
    # Combined summary
    logger.info("\n" + "="*60)
    logger.info("COMBINED SUMMARY - ALL BANKS")
    logger.info("="*60)
    
    for symbol, results_df in all_results.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Avg R²: {results_df['r2'].mean():.3f}")
        logger.info(f"  Avg Direction Accuracy: {results_df['direction_accuracy'].mean():.1%}")
        logger.info(f"  Windows Tested: {len(results_df)}")
    
    logger.info("\n" + "="*60)
    logger.info("🎉 WALK-FORWARD BACKTESTING COMPLETE!")
    logger.info("="*60)
    logger.info("Results in: data/backtest_results/")

if __name__ == "__main__":
    run_walkforward_backtest()