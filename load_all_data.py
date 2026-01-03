from src.data.yahoo_finance import YahooFinanceLoader
from src.utils.config import config
import time

def load_all_stocks():
    """Load all bank stocks and macro features"""
    
    loader = YahooFinanceLoader()
    
    # Get symbols from config
    bank_symbols = config.bank_symbols
    macro_symbols = config.macro_symbols
    
    print(f"\n{'='*60}")
    print(f"LOADING ALL DATA - NEW YEAR DATA REFRESH")
    print(f"{'='*60}\n")
    
    # Load bank stocks (12 banks)
    print(f"📊 LOADING {len(bank_symbols)} BANK STOCKS...")
    print(f"Symbols: {', '.join(bank_symbols)}\n")
    
    bank_results = {}
    for i, symbol in enumerate(bank_symbols, 1):
        print(f"[{i}/{len(bank_symbols)}] Loading {symbol}...")
        data = loader.update_stock_data(symbol, hours_back=2160, interval='1h')  # 3 months
        if not data.empty:
            bank_results[symbol] = data
            print(f"   ✅ {symbol}: {len(data)} data points")
        else:
            print(f"   ❌ {symbol}: Failed")
        time.sleep(1)  # Be nice to Yahoo
    
    print(f"\n✅ Bank stocks: {len(bank_results)}/{len(bank_symbols)} successful\n")
    
    # Load macro features
    print(f"📈 LOADING {len(macro_symbols)} MACRO FEATURES...")
    print(f"Symbols: {', '.join(macro_symbols)}\n")
    
    macro_results = {}
    for i, symbol in enumerate(macro_symbols, 1):
        # Yahoo Finance symbol mapping
        yahoo_symbol = {
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',
            'TNX': '^TNX',
            'SPX': '^GSPC'
        }.get(symbol, symbol)
        
        print(f"[{i}/{len(macro_symbols)}] Loading {symbol} ({yahoo_symbol})...")
        data = loader.update_stock_data(yahoo_symbol, hours_back=2160, interval='1h')
        if not data.empty:
            # Save with standardized name
            loader.save_to_csv(data, symbol)
            macro_results[symbol] = data
            print(f"   ✅ {symbol}: {len(data)} data points")
        else:
            print(f"   ❌ {symbol}: Failed")
        time.sleep(1)
    
    print(f"\n✅ Macro features: {len(macro_results)}/{len(macro_symbols)} successful\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DATA LOADING COMPLETE")
    print(f"{'='*60}")
    print(f"Bank Stocks: {len(bank_results)}/{len(bank_symbols)}")
    print(f"Macro Features: {len(macro_results)}/{len(macro_symbols)}")
    print(f"Total: {len(bank_results) + len(macro_results)}/{len(bank_symbols) + len(macro_symbols)}")
    print(f"\nAll data saved to: data/raw/")
    print(f"{'='*60}\n")
    
    return bank_results, macro_results

if __name__ == "__main__":
    bank_data, macro_data = load_all_stocks()