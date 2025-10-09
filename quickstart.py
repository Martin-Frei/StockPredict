"""
Quick start script - The simplest way to run the stock prediction system.

This script uses the configuration from config.py and runs the complete pipeline.
"""

import sys
import os

# Import configuration
try:
    import config
except ImportError:
    print("Error: config.py not found. Using default configuration.")
    class config:
        TICKER = "AAPL"
        START_DATE = "2020-01-01"
        END_DATE = None
        FORECAST_DAYS = 30
        ENSEMBLE_WEIGHTS = None

# Import the main predictor
from main import StockPredictor


def main():
    """Run the stock prediction system with configuration."""
    print("\n" + "="*80)
    print("STOCKPREDICT - QUICK START")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Ticker: {config.TICKER}")
    print(f"  Start Date: {config.START_DATE}")
    print(f"  End Date: {config.END_DATE if config.END_DATE else 'Today'}")
    print(f"  Forecast Days: {config.FORECAST_DAYS}")
    print("\nInitializing prediction system...")
    
    try:
        # Create predictor
        predictor = StockPredictor(
            ticker=config.TICKER,
            start_date=config.START_DATE,
            end_date=config.END_DATE
        )
        
        # Run the complete pipeline
        predictions, ensemble_pred, metrics = predictor.run(
            forecast_days=config.FORECAST_DAYS,
            ensemble_weights=config.ENSEMBLE_WEIGHTS
        )
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nPredictions generated for {config.TICKER}")
        print(f"Check the '{config.OUTPUT_DIR}/' directory for visualizations")
        
        # Display summary
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric, value in model_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return predictions, ensemble_pred, metrics
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR!")
        print("="*80)
        print(f"\nAn error occurred: {str(e)}")
        print("\nPlease check:")
        print("  1. All dependencies are installed (pip install -r requirements.txt)")
        print("  2. Internet connection is available for data fetching")
        print("  3. The ticker symbol is valid")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)
