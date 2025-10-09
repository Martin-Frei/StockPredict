# Configuration file for StockPredict

# Stock configuration
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = None  # None means today

# Data preprocessing
TRAIN_RATIO = 0.8  # 80% train, 20% test

# ARIMA configuration
ARIMA_ORDER = (5, 1, 0)  # (p, d, q)

# LSTM configuration
SEQUENCE_LENGTH = 60  # Number of days to look back
LSTM_UNITS = 50  # Number of LSTM units per layer
DROPOUT_RATE = 0.2  # Dropout rate for regularization
EPOCHS = 50  # Training epochs
BATCH_SIZE = 32  # Batch size for training

# Prophet configuration
SEASONALITY_MODE = "multiplicative"  # 'additive' or 'multiplicative'
YEARLY_SEASONALITY = True
WEEKLY_SEASONALITY = True
DAILY_SEASONALITY = False

# Ensemble configuration
ENSEMBLE_WEIGHTS = None  # None for auto-optimization, or dict like {'ARIMA': 0.3, 'LSTM': 0.4, 'Prophet': 0.3}
ENSEMBLE_METHOD = "weighted_average"  # 'weighted_average', 'simple_average', or 'median'

# Forecasting
FORECAST_DAYS = 30  # Number of days to forecast into the future

# Output configuration
OUTPUT_DIR = "outputs"
SAVE_PLOTS = True
