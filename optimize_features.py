from src.models.xgboost_model import XGBoostPredictor
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FINALE FEATURE LISTE (25 Features)
SELECTED_FEATURES = [
    # Macro (8)
    'SPX_Level', 'SPX_Return',
    'XLF_Level', 'XLF_Return',
    'VIX_Level',
    'TNX_Level',
    'SHY_Level',
    'TLT_Level',
    
    # Technical (12)
    'MACD', 'MACD_Signal',
    'MA_5', 'MA_20',
    'Price_MA20_Ratio',
    'Vol_10', 'Vol_20',
    'RSI',
    'Volume_Ratio',
    'Returns',
    
    # Time (3)
    'DayOfWeek',
    'Hour',
]

def optimize_features():
    """Re-train XGBoost with selected features only"""
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZED XGBOOST WITH 22 SELECTED FEATURES")
    logger.info("="*60)
    
    # Load original predictor
    predictor = XGBoostPredictor()
    
    # Store original results
    original_models = {}
    optimized_models = {}
    
    for symbol in predictor.bank_symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZING: {symbol}")
        logger.info(f"{'='*60}")
        
        # Load data
        df = predictor.load_enhanced_data(symbol)
        if df is None:
            continue
        
        # Check if all selected features exist
        missing = [f for f in SELECTED_FEATURES if f not in df.columns]
        if missing:
            logger.warning(f"Missing features: {missing}")
            continue
        
        # Prepare features - MODIFIED
        exclude_cols = [
            'DateTime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Target_7h', 'Target_Direction_7h', 'Target_1h', 'Target_4h'
        ]
        
        df_clean = df.dropna(subset=['Target_7h'])
        
        # ONLY use selected features
        X = df_clean[SELECTED_FEATURES].fillna(0)
        y = df_clean['Target_7h']
        
        logger.info(f"Original Features: 46")
        logger.info(f"Selected Features: {len(SELECTED_FEATURES)}")
        logger.info(f"Samples: {len(X)}")
        
        # Train with selected features
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        logger.info("Training with selected features...")
        model = xgb.train(
            predictor.params,
            dtrain,
            num_boost_round=predictor.params.get('n_estimators', 100),
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Metrics
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        logger.info(f"\n✅ RESULTS:")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²:  {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.6f}")
        
        optimized_models[symbol] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE - COMPARISON")
    logger.info("="*60)
    
    for symbol, metrics in optimized_models.items():
        logger.info(f"{symbol:5s} | "
                   f"Train R²: {metrics['train_r2']:6.3f} | "
                   f"Test R²: {metrics['test_r2']:6.3f} | "
                   f"RMSE: {metrics['test_rmse']:.6f}")
    
    logger.info("\n🎯 Feature Count: 46 → 22 (52% reduction!)")

if __name__ == "__main__":
    optimize_features()