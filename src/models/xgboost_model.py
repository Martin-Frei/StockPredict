import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostPredictor:
    """
    XGBoost Model for 7h Stock Predictions
    - Gradient Boosting optimized for financial data
    - Uses all technical + macro features
    - Feature importance analysis
    """
    
    def __init__(self):
        self.data_processed = config.data_processed
        self.predictions_path = config.predictions_path / 'xgboost_predictions'
        self.predictions_path.mkdir(parents=True, exist_ok=True)
        
        self.bank_symbols = config.bank_symbols
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}
        
        # XGBoost hyperparameters from config
        self.params = config.xgboost_params
        
        logger.info("XGBoost Predictor initialized")
        logger.info(f"Banks: {len(self.bank_symbols)}")
        logger.info(f"Params: {self.params}")
    
    def load_enhanced_data(self, symbol):
        """Load enhanced features"""
        csv_file = self.data_processed / f"{symbol}_enhanced.csv"
        
        if not csv_file.exists():
            logger.warning(f"{symbol}: Enhanced CSV not found")
            return None
        
        try:
            df = pd.read_csv(csv_file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            logger.info(f"{symbol}: Loaded {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"{symbol}: Error loading - {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for XGBoost"""
        
        # Columns to exclude
        exclude_cols = [
            'DateTime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Target_7h', 'Target_Direction_7h', 'Target_1h', 'Target_4h'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=['Target_7h'])
        
        X = df_clean[feature_cols]
        y = df_clean['Target_7h']
        timestamps = df_clean['DateTime']
        
        # Fill remaining NaN in features
        X = X.fillna(0)
        
        logger.info(f"   Features: {len(feature_cols)} columns")
        logger.info(f"   Samples: {len(X)} rows")
        
        return X, y, timestamps, feature_cols
    
    def train_model(self, symbol):
        """Train XGBoost model for one bank"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING XGBOOST MODEL: {symbol}")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_enhanced_data(symbol)
        if df is None:
            return None
        
        # Prepare features
        X, y, timestamps, feature_cols = self.prepare_features(df)
        
        if len(X) < 100:
            logger.warning(f"{symbol}: Too few samples ({len(X)})")
            return None
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
            X, y, timestamps, test_size=0.2, shuffle=False
        )
        
        logger.info(f"Train: {len(X_train)} samples")
        logger.info(f"Test: {len(X_test)} samples")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train
        logger.info("Training XGBoost...")
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 100),
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Predictions
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        logger.info(f"\nRESULTS:")
        logger.info(f"  Train RMSE: {train_rmse:.6f}")
        logger.info(f"  Test RMSE:  {test_rmse:.6f}")
        logger.info(f"  Train R²:   {train_r2:.4f}")
        logger.info(f"  Test R²:    {test_r2:.4f}")
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        logger.info(f"\nTOP 10 FEATURES:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        # Store
        self.models[symbol] = {
            'model': model,
            'feature_cols': feature_cols,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        }
        self.feature_importance[symbol] = importance_df
        
        logger.info(f"\n✅ {symbol}: Model trained successfully")
        return model
    
    
    def predict_next_7h(self, symbol):
            """Predict next 7 hours for a bank"""
            
            if symbol not in self.models:
                logger.warning(f"{symbol}: Model not trained")
                return None
            
            logger.info(f"\nPredicting next 7h for {symbol}...")
            
            # Load data
            df = self.load_enhanced_data(symbol)
            if df is None:
                return None
            
            # Get last available data point
            last_row = df.iloc[-1:].copy()
            
            # Prepare features
            model_info = self.models[symbol]
            feature_cols = model_info['feature_cols']
            
            X_pred = last_row[feature_cols].fillna(0)
            
            # Predict 7 hours
            predictions = []
            current_features = X_pred.copy()
            
            for hour in range(7):
                dpred = xgb.DMatrix(current_features)
                pred = model_info['model'].predict(dpred)[0]
                predictions.append(pred)
                
                # Simple feature update (rolling window)
                # In reality, features would change - this is simplified
                if hour < 6:
                    current_features = X_pred.copy()
            
            # Create prediction DataFrame
            last_datetime = last_row['DateTime'].iloc[0]
            future_times = pd.date_range(
                start=last_datetime + pd.Timedelta(hours=1),
                periods=7,
                freq='1H'
            )
            
            pred_df = pd.DataFrame({
                'DateTime': future_times,
                'Predicted_Return_7h': predictions,
                'Cumulative_Return': np.cumsum(predictions),
                'Direction': [1 if p > 0 else 0 for p in predictions],
                'Confidence': [abs(p) for p in predictions]
            })
            
            total_return = sum(predictions)
            bullish_hours = sum(1 for p in predictions if p > 0)
            
            logger.info(f"✅ {symbol} Predictions:")
            logger.info(f"   Total 7h Return: {total_return:.4f} ({total_return*100:.2f}%)")
            logger.info(f"   Bullish Hours: {bullish_hours}/7")
            logger.info(f"   Direction: {'📈 BULLISH' if total_return > 0 else '📉 BEARISH'}")
            
            self.predictions[symbol] = pred_df
            return pred_df
        
    def analyze_feature_importance_all_banks(self):
        """Analyze feature importance across all banks"""
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE ANALYSIS - ALL BANKS")
        logger.info("="*60)
        
        if not self.feature_importance:
            logger.warning("No feature importance data available")
            return None
        
        # Combine importance from all banks
        all_importance = []
        for symbol, imp_df in self.feature_importance.items():
            imp_df['symbol'] = symbol
            all_importance.append(imp_df)
        
        combined = pd.concat(all_importance, ignore_index=True)
        
        # Average importance per feature
        avg_importance = combined.groupby('feature')['importance'].agg([
            'mean', 'std', 'count'
        ]).sort_values('mean', ascending=False)
        
        logger.info("\nTOP 30 FEATURES (averaged across all banks):")
        logger.info("-" * 60)
        
        for idx, (feature, row) in enumerate(avg_importance.head(30).iterrows(), 1):
            logger.info(f"{idx:2d}. {feature:30s} | "
                        f"Importance: {row['mean']:6.2f} ± {row['std']:5.2f} | "
                        f"Used by: {int(row['count'])}/12 banks")
        
        # Save to CSV
        output_file = self.predictions_path / 'feature_importance_summary.csv'
        avg_importance.to_csv(output_file)
        logger.info(f"\n✅ Feature importance saved to: {output_file}")
        
        return avg_importance

    def get_feature_correlations(self, symbol):
        """Analyze feature correlations"""
        
        df = self.load_enhanced_data(symbol)
        if df is None:
            return None
        
        # Get feature columns
        exclude_cols = [
            'DateTime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Target_7h', 'Target_Direction_7h', 'Target_1h', 'Target_4h'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr()
        
        # Find highly correlated pairs (>0.90)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.90:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        high_corr_df = pd.DataFrame(high_corr).sort_values('correlation', ascending=False)
        
        logger.info(f"\n{symbol}: HIGH CORRELATIONS (>0.90):")
        logger.info("-" * 70)
        for idx, row in high_corr_df.iterrows():
            logger.info(f"{row['feature1']:25s} ↔ {row['feature2']:25s} | r={row['correlation']:.3f}")
        
        return high_corr_df

    def train_all_banks(self):
        """Train XGBoost for all banks"""
        
        logger.info("\n" + "="*60)
        logger.info("XGBOOST TRAINING PIPELINE - ALL BANKS")
        logger.info("="*60)
        
        successful = 0
        
        for i, symbol in enumerate(self.bank_symbols, 1):
            logger.info(f"\n[{i}/{len(self.bank_symbols)}] {symbol}")
            
            model = self.train_model(symbol)
            if model is not None:
                successful += 1
        
        logger.info("\n" + "="*60)
        logger.info(f"TRAINING COMPLETE: {successful}/{len(self.bank_symbols)} banks")
        logger.info("="*60)
        
        return successful > 0

    def predict_all_banks(self):
        """Generate predictions for all trained banks"""
        
        logger.info("\n" + "="*60)
        logger.info("GENERATING 7H PREDICTIONS - ALL BANKS")
        logger.info("="*60)
        
        for symbol in self.models.keys():
            self.predict_next_7h(symbol)
        
        logger.info("\n✅ All predictions generated!")

    def save_predictions(self):
        """Save all predictions to CSV"""
        
        if not self.predictions:
            logger.warning("No predictions to save")
            return
        
        logger.info("\nSaving predictions...")
        
        for symbol, pred_df in self.predictions.items():
            output_file = self.predictions_path / f"{symbol}_7h_xgboost_prediction.csv"
            pred_df.to_csv(output_file, index=False)
            logger.info(f"✅ {symbol}: {output_file.name}")
        
        # Create trading summary
        summary_data = []
        for symbol, pred_df in self.predictions.items():
            total_return = pred_df['Predicted_Return_7h'].sum()
            bullish_hours = pred_df['Direction'].sum()
            avg_confidence = pred_df['Confidence'].mean()
            
            metrics = self.models[symbol]['metrics']
            
            summary_data.append({
                'Symbol': symbol,
                'Total_7h_Return': total_return,
                'Total_7h_Return_Pct': total_return * 100,
                'Bullish_Hours': f"{bullish_hours}/7",
                'Avg_Confidence': avg_confidence,
                'Direction': 'Bullish' if total_return > 0 else 'Bearish',
                'Train_R2': metrics['train_r2'],
                'Test_R2': metrics['test_r2'],
                'Test_RMSE': metrics['test_rmse']
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('Total_7h_Return', ascending=False)
        
        summary_file = self.predictions_path / 'xgboost_trading_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"\n✅ Trading summary: {summary_file}")
        
        # Print top performers
        logger.info("\n" + "="*60)
        logger.info("TOP 5 XGBOOST PREDICTIONS:")
        logger.info("="*60)
        for idx, row in summary_df.head(5).iterrows():
            logger.info(f"{row['Symbol']:5s} | Return: {row['Total_7h_Return_Pct']:+6.2f}% | "
                        f"Direction: {row['Direction']:8s} | "
                        f"Bullish: {row['Bullish_Hours']} | "
                        f"R²: {row['Test_R2']:.3f}")
        
        return summary_df

def run_xgboost_pipeline():
    """Run complete XGBoost pipeline"""
    predictor = XGBoostPredictor()
    
    # Step 1: Train all banks
    if not predictor.train_all_banks():
        logger.error("Training failed!")
        return
    
    # Step 2: Analyze feature importance
    predictor.analyze_feature_importance_all_banks()
    
    # Step 3: Check correlations for one bank (example)
    predictor.get_feature_correlations('JPM')
    
    # Step 4: Generate predictions
    predictor.predict_all_banks()
    
    # Step 5: Save everything
    predictor.save_predictions()
    
    logger.info("\n" + "="*60)
    logger.info("🎉 XGBOOST PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info("Results saved in: data/predictions/xgboost_predictions/")
    logger.info("- Individual predictions: *_7h_xgboost_prediction.csv")
    logger.info("- Trading summary: xgboost_trading_summary.csv")
    logger.info("- Feature importance: feature_importance_summary.csv")

if __name__ == "__main__":
    run_xgboost_pipeline()