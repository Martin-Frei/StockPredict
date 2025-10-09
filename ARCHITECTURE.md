# System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STOCKPREDICT SYSTEM                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         1. DATA LAYER                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  DataPreprocessor (src/utils/data_preprocessing.py)                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ • fetch_data()      → Download from Yahoo Finance                │  │
│  │ • clean_data()      → Handle missing values, outliers            │  │
│  │ • add_features()    → Technical indicators (MA, EMA, etc.)       │  │
│  │ • scale_data()      → Normalize for neural networks              │  │
│  │ • split_data()      → Train/Test split                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         2. MODEL LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐   │
│  │  ARIMA Model     │   │   LSTM Model     │   │  Prophet Model  │   │
│  ├──────────────────┤   ├──────────────────┤   ├─────────────────┤   │
│  │ • Stationarity   │   │ • 3 LSTM Layers  │   │ • Seasonality   │   │
│  │ • (p,d,q) Order  │   │ • Dropout 0.2    │   │ • Trend         │   │
│  │ • ADF Test       │   │ • Early Stop     │   │ • Holidays      │   │
│  │ • Forecasting    │   │ • Seq Learning   │   │ • Uncertainty   │   │
│  └──────────────────┘   └──────────────────┘   └─────────────────┘   │
│          ↓                       ↓                        ↓            │
│     Predictions              Predictions              Predictions       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         3. ENSEMBLE LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│  EnsembleModel (src/models/ensemble_model.py)                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Combination Methods:                                              │  │
│  │  • Weighted Average  → Optimal weights for each model            │  │
│  │  • Simple Average    → Equal weight to all models                │  │
│  │  • Median           → Robust to outliers                         │  │
│  │                                                                   │  │
│  │ Auto-optimization:                                                │  │
│  │  → Tests all methods                                             │  │
│  │  → Selects best based on RMSE                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      4. EVALUATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Metrics:                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ • MAE  (Mean Absolute Error)                                     │  │
│  │ • RMSE (Root Mean Squared Error)                                 │  │
│  │ • MAPE (Mean Absolute Percentage Error)                          │  │
│  │ • MSE  (Mean Squared Error)                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      5. VISUALIZATION LAYER                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Visualizer (src/utils/visualization.py)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Outputs:                                                          │  │
│  │  • stock_history.png       → Historical prices + indicators      │  │
│  │  • lstm_training_history.png → Loss curves                       │  │
│  │  • metrics_comparison.png   → Model performance bars             │  │
│  │  • all_predictions.png     → All models vs actual                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  main.py / quickstart.py / example.py                                   │
│       ↓                                                                  │
│  StockPredictor.run()                                                   │
│       ↓                                                                  │
│  1. load_and_prepare_data()    → Fetch & preprocess                    │
│       ↓                                                                  │
│  2. train_arima()              → Train ARIMA model                      │
│       ↓                                                                  │
│  3. train_lstm()               → Train LSTM model                       │
│       ↓                                                                  │
│  4. train_prophet()            → Train Prophet model                    │
│       ↓                                                                  │
│  5. make_predictions()         → Generate all predictions               │
│       ↓                                                                  │
│  6. evaluate_models()          → Calculate metrics                      │
│       ↓                                                                  │
│  7. create_ensemble()          → Combine predictions                    │
│       ↓                                                                  │
│  8. visualize_results()        → Generate plots                         │
│       ↓                                                                  │
│  Results saved to outputs/                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION OPTIONS                               │
├─────────────────────────────────────────────────────────────────────────┤
│  config.py provides:                                                     │
│  • Ticker symbol (default: AAPL)                                        │
│  • Date range                                                            │
│  • ARIMA order (p, d, q)                                                │
│  • LSTM architecture (units, dropout, epochs)                           │
│  • Prophet seasonality settings                                         │
│  • Ensemble weights                                                      │
│  • Output preferences                                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA FLOW EXAMPLE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Input: ticker='AAPL', start='2020-01-01'                          │
│       ↓                                                                  │
│  Yahoo Finance API                                                       │
│       ↓                                                                  │
│  Raw Data: Date, Open, High, Low, Close, Volume                         │
│       ↓                                                                  │
│  Cleaned Data + Features (MA7, MA21, EMA, Volatility, Momentum)        │
│       ↓                                                                  │
│  Train Split (80%) + Test Split (20%)                                   │
│       ↓                                                                  │
│  ┌─────────────┬─────────────┬─────────────┐                          │
│  │   ARIMA     │    LSTM     │   Prophet   │                           │
│  │ Training... │ Training... │ Training... │                           │
│  └─────────────┴─────────────┴─────────────┘                           │
│       ↓              ↓              ↓                                    │
│  Pred[30d]      Pred[30d]      Pred[30d]                               │
│       └──────────────┴──────────────┘                                   │
│                      ↓                                                   │
│              Ensemble Prediction                                         │
│                      ↓                                                   │
│     Evaluation Metrics + Visualization                                  │
│                      ↓                                                   │
│        outputs/all_predictions.png                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Advantages

1. **Modular Design**: Each component is independent and reusable
2. **Extensible**: Easy to add new models or features
3. **Production-Ready**: Comprehensive error handling and logging
4. **Well-Documented**: Inline docs + 3 markdown guides
5. **Tested**: Unit tests for core functionality
6. **Secure**: All dependencies updated to patched versions

## Performance Characteristics

- **ARIMA**: Fast, good for short-term linear trends
- **LSTM**: Slower training, excellent for complex patterns
- **Prophet**: Medium speed, handles seasonality well
- **Ensemble**: Best overall accuracy, minimal overhead

## Scalability

- Single stock prediction: ~2-5 minutes
- Can be parallelized for multiple stocks
- GPU acceleration available for LSTM
- Caching supported for data reuse
