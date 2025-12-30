from src.data.alpha_vantage import AlphaVantageLoader

# Initialize loader
loader = AlphaVantageLoader()

# Test with one stock
print('\nTesting with JPM...')
data = loader.update_stock_data('JPM', months_back=1)

if not data.empty:
    print(f'\nSuccess! Loaded {len(data)} data points')
    print(f'Date range: {data["DateTime"].min()} to {data["DateTime"].max()}')
    print(f'\nLast 3 rows:')
    print(data[['DateTime', 'Close', 'Volume']].tail(3))
else:
    print('Failed to load data')