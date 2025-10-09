# StockPredict

A comprehensive stock prediction system that combines multiple forecasting techniques to predict stock prices with high accuracy.

## Features

- **Multiple Models**: Combines ARIMA, LSTM Neural Networks, and Facebook Prophet for robust predictions
- **Ensemble Learning**: Intelligently combines predictions from multiple models for improved accuracy
- **Data Preprocessing**: Automatic data fetching, cleaning, and feature engineering
- **Visualization**: Beautiful charts and graphs to understand model performance and predictions
- **Easy to Use**: Simple API to train models and make predictions

## Architecture

The system uses three complementary approaches:

1. **ARIMA (AutoRegressive Integrated Moving Average)**: Traditional time series forecasting method that captures linear patterns and trends
2. **LSTM (Long Short-Term Memory)**: Deep learning approach that learns complex patterns and long-term dependencies
3. **Prophet**: Facebook's forecasting tool that handles seasonality and holidays effectively

These models are combined using an **Ensemble Method** that weights predictions to minimize error.

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Martin-Frei/StockPredict.git
cd StockPredict
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the prediction system with default settings (AAPL stock):

```bash
python main.py
```

### Custom Usage

```python
from main import StockPredictor

# Initialize predictor for a specific stock
predictor = StockPredictor(
    ticker='TSLA',              # Stock ticker symbol
    start_date='2020-01-01',    # Start date for historical data
    end_date=None               # End date (None = today)
)

# Run the complete prediction pipeline
predictions, ensemble_pred, metrics = predictor.run(
    forecast_days=30,           # Number of days to forecast
    ensemble_weights=None       # Custom weights for ensemble (None = auto-optimize)
)
```

### Using Individual Models

```python
from src.utils.data_preprocessing import DataPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.prophet_model import ProphetModel

# Data preprocessing
preprocessor = DataPreprocessor(ticker='AAPL', start_date='2020-01-01')
preprocessor.fetch_data()
preprocessor.clean_data()
preprocessor.add_features()

# Train ARIMA
arima = ARIMAModel(order=(5, 1, 0))
arima.train(preprocessor.data['Close'].values)
arima_predictions = arima.predict(steps=30)

# Train LSTM
lstm = LSTMModel(sequence_length=60, units=50)
X, y = preprocessor.prepare_sequences(sequence_length=60)
lstm.train(X, y, epochs=50)
lstm_predictions = lstm.predict(X)

# Train Prophet
prophet = ProphetModel()
prophet.train(preprocessor.data, date_column='Date', value_column='Close')
prophet_forecast = prophet.predict(periods=30)
```

## Project Structure

```
StockPredict/
├── src/
│   ├── models/
│   │   ├── arima_model.py      # ARIMA implementation
│   │   ├── lstm_model.py       # LSTM neural network
│   │   ├── prophet_model.py    # Prophet model
│   │   └── ensemble_model.py   # Ensemble combination
│   └── utils/
│       ├── data_preprocessing.py  # Data loading and preprocessing
│       └── visualization.py       # Plotting and visualization
├── data/                       # Data directory (auto-created)
├── outputs/                    # Output plots and results
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Output

The system generates several outputs in the `outputs/` directory:

1. **stock_history.png**: Historical stock price with moving averages
2. **lstm_training_history.png**: LSTM training and validation loss curves
3. **metrics_comparison.png**: Comparison of evaluation metrics across models
4. **all_predictions.png**: Combined visualization of all model predictions vs actual values

### Sample Metrics

The system evaluates models using:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **MSE (Mean Squared Error)**: Average squared differences

## Models Explained

### ARIMA
- Best for: Short-term predictions, stationary data
- Strengths: Fast, interpretable, good for linear trends
- Parameters: (p=5, d=1, q=0) where p=autoregressive terms, d=differencing order, q=moving average terms

### LSTM
- Best for: Complex patterns, long-term dependencies
- Strengths: Captures non-linear relationships, learns from sequences
- Architecture: 3 LSTM layers (50 units each) with dropout (0.2) for regularization

### Prophet
- Best for: Handling seasonality, missing data, outliers
- Strengths: Robust to missing data, automatic seasonality detection
- Features: Multiplicative seasonality, yearly and weekly patterns

### Ensemble
- Best for: Overall best performance
- Strengths: Combines strengths of all models, reduces individual model weaknesses
- Methods: Weighted average, simple average, median (auto-selects best)

## Advanced Configuration

### Custom Ensemble Weights

```python
# Define custom weights for each model
ensemble_weights = {
    'ARIMA': 0.3,
    'LSTM': 0.5,
    'Prophet': 0.2
}

predictor.run(ensemble_weights=ensemble_weights)
```

### LSTM Hyperparameters

```python
from src.models.lstm_model import LSTMModel

lstm = LSTMModel(
    sequence_length=60,    # Look back 60 days
    units=100,             # 100 LSTM units per layer
    dropout_rate=0.3       # 30% dropout
)
```

### ARIMA Order Selection

```python
from src.models.arima_model import ARIMAModel

# Try different orders
arima = ARIMAModel(order=(5, 1, 2))  # (p, d, q)
```

## Performance Tips

1. **More Data**: Use longer historical periods (2+ years) for better training
2. **Hyperparameter Tuning**: Experiment with different LSTM units, sequence lengths, and dropout rates
3. **Feature Engineering**: Add more technical indicators in `data_preprocessing.py`
4. **Ensemble Weights**: Fine-tune weights based on your specific stock characteristics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Stock market predictions are inherently uncertain. Do not use this software for actual trading decisions without proper financial advice. The authors are not responsible for any financial losses incurred from using this software.

## Acknowledgments

- ARIMA implementation uses statsmodels
- LSTM implementation uses TensorFlow/Keras
- Prophet by Facebook Research
- Data sourced from Yahoo Finance via yfinance

## Contact

For questions or issues, please open an issue on GitHub.