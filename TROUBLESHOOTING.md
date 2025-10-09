# Troubleshooting Guide

This guide helps you resolve common issues when using StockPredict.

## Installation Issues

### Issue: `pip install` fails

**Solution:**
```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v

# Try installing packages one by one
pip install numpy pandas scikit-learn
pip install tensorflow
pip install prophet
```

### Issue: TensorFlow installation fails

**Solution:**
```bash
# For CPU-only version (smaller, faster to install)
pip install tensorflow-cpu

# For specific Python version compatibility
pip install tensorflow==2.12.1
```

### Issue: Prophet installation fails on Windows

**Solution:**
```bash
# Install Microsoft C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use conda
conda install -c conda-forge prophet
```

## Runtime Issues

### Issue: `ModuleNotFoundError: No module named 'pandas'`

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify installation
python -c "import pandas; print(pandas.__version__)"
```

### Issue: `No data found for ticker`

**Causes & Solutions:**
1. **Invalid ticker symbol**
   - Verify ticker on Yahoo Finance: https://finance.yahoo.com/
   - Use correct format (e.g., 'AAPL' not 'Apple')

2. **Network issues**
   - Check internet connection
   - Try with VPN if blocked in your region

3. **Date range issues**
   ```python
   # Use valid date range
   predictor = StockPredictor(
       ticker='AAPL',
       start_date='2020-01-01',  # Not too far in past
       end_date=None  # Use current date
   )
   ```

### Issue: `LSTM training is very slow`

**Solutions:**
1. **Reduce epochs**
   ```python
   lstm.train(X_train, y_train, epochs=20)  # Instead of 50
   ```

2. **Use GPU if available**
   ```bash
   # Install TensorFlow with GPU support
   pip install tensorflow-gpu
   ```

3. **Reduce data size**
   ```python
   # Use shorter date range
   predictor = StockPredictor(
       ticker='AAPL',
       start_date='2022-01-01'  # More recent data
   )
   ```

### Issue: Memory errors during training

**Solutions:**
1. **Reduce batch size**
   ```python
   lstm.train(X_train, y_train, batch_size=16)  # Instead of 32
   ```

2. **Use less historical data**
   ```python
   predictor = StockPredictor(
       ticker='AAPL',
       start_date='2022-01-01'  # Shorter period
   )
   ```

3. **Reduce sequence length**
   ```python
   lstm = LSTMModel(sequence_length=30)  # Instead of 60
   ```

### Issue: Predictions are inaccurate

**Possible Causes & Solutions:**

1. **Insufficient training data**
   - Use at least 2 years of historical data
   - More data generally leads to better predictions

2. **Market volatility**
   - Stock markets are inherently unpredictable
   - Use ensemble predictions for better results
   - Consider predictions as guidance, not certainty

3. **Model hyperparameters need tuning**
   ```python
   # Try different ARIMA orders
   arima = ARIMAModel(order=(7, 1, 1))
   
   # Adjust LSTM parameters
   lstm = LSTMModel(
       sequence_length=90,
       units=100,
       dropout_rate=0.3
   )
   ```

4. **Feature engineering needed**
   - Add more technical indicators in `data_preprocessing.py`
   - Include external factors (sentiment, news, etc.)

## Visualization Issues

### Issue: Plots not displaying

**Solutions:**
1. **Check if running in headless environment**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Use non-interactive backend
   ```

2. **Verify output directory exists**
   ```bash
   mkdir -p outputs
   ```

3. **Check file permissions**
   ```bash
   chmod 755 outputs/
   ```

### Issue: Plots are cut off or poorly formatted

**Solutions:**
1. **Increase figure size**
   ```python
   plt.figure(figsize=(16, 8))  # Larger figure
   ```

2. **Use tight layout**
   ```python
   plt.tight_layout()
   ```

3. **Adjust DPI**
   ```python
   plt.savefig('output.png', dpi=300, bbox_inches='tight')
   ```

## Configuration Issues

### Issue: Custom configuration not working

**Solution:**
```python
# Verify config.py is in the correct location
import config
print(config.TICKER)

# Or pass parameters directly
predictor = StockPredictor(
    ticker='TSLA',
    start_date='2020-01-01'
)
```

## Performance Optimization

### Slow data fetching

**Solutions:**
1. **Cache data locally**
   ```python
   # Save data for reuse
   preprocessor.data.to_csv('data/stock_data.csv')
   
   # Load cached data
   import pandas as pd
   data = pd.read_csv('data/stock_data.csv')
   ```

2. **Use smaller date range for testing**

### Training takes too long

**Solutions:**
1. **Use subset for initial testing**
   ```python
   # Quick test with small dataset
   predictor = StockPredictor(
       ticker='AAPL',
       start_date='2023-01-01'
   )
   ```

2. **Reduce model complexity**
   ```python
   lstm = LSTMModel(
       sequence_length=30,
       units=25  # Fewer units
   )
   ```

## Common Warnings

### Warning: `FutureWarning` from pandas

**Solution:**
- Usually safe to ignore
- Update pandas to latest version: `pip install --upgrade pandas`

### Warning: TensorFlow messages

**Solution:**
```python
# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### Warning: Prophet seasonality

**Solution:**
- Ignore if dataset is small
- Use larger historical dataset for better seasonality detection

## Debugging Tips

### Enable verbose output

```python
# In data preprocessing
preprocessor.fetch_data()  # Shows progress

# In ARIMA
arima.train(data)  # Shows model summary

# In LSTM
lstm.train(X, y, verbose=1)  # Shows training progress
```

### Check data quality

```python
# Verify data
print(f"Data shape: {preprocessor.data.shape}")
print(f"Missing values: {preprocessor.data.isnull().sum()}")
print(f"Date range: {preprocessor.data['Date'].min()} to {preprocessor.data['Date'].max()}")
```

### Verify models are training

```python
# After training, check if model exists
assert arima.fitted_model is not None
assert lstm.model is not None
assert prophet.fitted == True
```

## Getting Help

If you're still experiencing issues:

1. **Check existing issues**: https://github.com/Martin-Frei/StockPredict/issues
2. **Create new issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Error messages (full traceback)
   - Environment details (OS, Python version)
   - What you've already tried

3. **Include diagnostic information**:
   ```bash
   python --version
   pip list | grep -E "(pandas|numpy|tensorflow|prophet|scikit-learn)"
   ```

## FAQ

**Q: How accurate are the predictions?**
A: Stock prediction is inherently uncertain. Use predictions as guidance, not absolute truth. Past performance doesn't guarantee future results.

**Q: Can I use this for actual trading?**
A: This is for educational purposes only. Do not use for actual trading without proper financial advice and risk assessment.

**Q: Which model performs best?**
A: It varies by stock and market conditions. The ensemble method usually provides the best balance by combining strengths of all models.

**Q: How often should I retrain models?**
A: Retrain regularly (daily/weekly) as market conditions change. More recent data improves prediction accuracy.

**Q: Can I predict multiple stocks at once?**
A: Currently, the system predicts one stock at a time. You can run it multiple times for different stocks or extend the code to support multiple stocks.

**Q: How far into the future can I predict?**
A: Short-term predictions (1-30 days) are more reliable. Long-term predictions become increasingly uncertain.

## Still Need Help?

- Read the [README.md](README.md) for usage instructions
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub with your question
