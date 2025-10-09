"""
Example script demonstrating how to use the StockPredict system.

This script shows different ways to use the stock prediction system:
1. Quick prediction with default settings
2. Custom stock and date range
3. Individual model usage
4. Ensemble with custom weights
"""

from main import StockPredictor
import pandas as pd


def example_1_basic():
    """Example 1: Basic usage with default settings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage (AAPL stock)")
    print("="*80)
    
    predictor = StockPredictor(ticker='AAPL', start_date='2020-01-01')
    predictions, ensemble_pred, metrics = predictor.run(forecast_days=30)
    
    print("\n✓ Example 1 completed!")
    return predictions, ensemble_pred, metrics


def example_2_custom_stock():
    """Example 2: Predict different stock with custom date range."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Stock (TSLA)")
    print("="*80)
    
    predictor = StockPredictor(
        ticker='TSLA',
        start_date='2021-01-01',
        end_date='2024-01-01'
    )
    predictions, ensemble_pred, metrics = predictor.run(forecast_days=60)
    
    print("\n✓ Example 2 completed!")
    return predictions, ensemble_pred, metrics


def example_3_custom_ensemble_weights():
    """Example 3: Use custom ensemble weights."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Ensemble Weights")
    print("="*80)
    
    # Define custom weights favoring LSTM
    custom_weights = {
        'ARIMA': 0.2,
        'LSTM': 0.6,
        'Prophet': 0.2
    }
    
    predictor = StockPredictor(ticker='GOOGL', start_date='2020-01-01')
    predictions, ensemble_pred, metrics = predictor.run(
        forecast_days=30,
        ensemble_weights=custom_weights
    )
    
    print("\n✓ Example 3 completed!")
    return predictions, ensemble_pred, metrics


def example_4_individual_models():
    """Example 4: Use individual models separately."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Individual Model Usage")
    print("="*80)
    
    from src.utils.data_preprocessing import DataPreprocessor
    from src.models.arima_model import ARIMAModel
    
    # Prepare data
    preprocessor = DataPreprocessor(ticker='MSFT', start_date='2022-01-01')
    preprocessor.fetch_data()
    preprocessor.clean_data()
    preprocessor.add_features()
    
    # Train only ARIMA model
    train_data, test_data = preprocessor.split_data(train_ratio=0.8)
    
    arima = ARIMAModel(order=(5, 1, 0))
    arima.train(train_data['Close'].values)
    
    # Make predictions
    predictions = arima.predict(steps=30)
    
    # Evaluate
    metrics = arima.evaluate(test_data['Close'].values)
    
    print("\n✓ Example 4 completed!")
    return predictions, metrics


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("STOCK PREDICTION EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates different ways to use StockPredict.")
    print("Running examples may take several minutes...")
    
    # Run example 1 - Basic usage
    try:
        example_1_basic()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Uncomment to run other examples:
    
    # # Run example 2 - Custom stock
    # try:
    #     example_2_custom_stock()
    # except Exception as e:
    #     print(f"Example 2 failed: {e}")
    
    # # Run example 3 - Custom weights
    # try:
    #     example_3_custom_ensemble_weights()
    # except Exception as e:
    #     print(f"Example 3 failed: {e}")
    
    # # Run example 4 - Individual models
    # try:
    #     example_4_individual_models()
    # except Exception as e:
    #     print(f"Example 4 failed: {e}")
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\nCheck the 'outputs/' directory for generated visualizations.")


if __name__ == "__main__":
    main()
