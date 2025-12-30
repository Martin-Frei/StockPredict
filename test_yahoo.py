from src.data.yahoo_finance import YahooFinanceLoader

# Initialize loader
loader = YahooFinanceLoader()

# Test with one stock (hourly data, 1 month)
print('\nTesting with JPM (1 hour interval, 1 month)...')
data = loader.update_stock_data('JPM', hours_back=720, interval='1h')

if not data.empty:
    print(f'\nSuccess! Loaded {len(data)} data points')
    print(f'Date range: {data["DateTime"].min()} to {data["DateTime"].max()}')
    print(f'\nLast 3 rows:')
    print(data[['DateTime', 'Close', 'Volume']].tail(3))
    print(f'\nFile saved to: data/raw/JPM.csv')
else:
    print('Failed to load data')
    