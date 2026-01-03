import time
from pathlib import Path
from src.data.yahoo_finance import YahooFinanceLoader
from src.utils.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_data():
    """
    Load 2 YEARS of hourly data for all stocks and macro features
    """
    
    loader = YahooFinanceLoader()
    
    # Symbols to load
    bank_symbols = config.bank_symbols
    macro_symbols = config.macro_symbols
    all_symbols = bank_symbols + macro_symbols
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LOADING 2 YEARS HOURLY DATA")
    logger.info(f"{'='*60}")
    logger.info(f"Bank Stocks: {len(bank_symbols)}")
    logger.info(f"Macro Features: {len(macro_symbols)}")
    logger.info(f"Total: {len(all_symbols)} symbols")
    logger.info(f"Period: 2 YEARS (max available from Yahoo)")
    logger.info(f"{'='*60}\n")
    
    successful = 0
    failed = []
    
    for i, symbol in enumerate(all_symbols, 1):
        logger.info(f"[{i}/{len(all_symbols)}] Loading {symbol}...")
        
        try:
            # Fetch 2 years with period parameter
            import yfinance as yf
            
            # Map our symbols to Yahoo symbols
            yahoo_symbol = symbol
            if symbol == 'VIX':
                yahoo_symbol = '^VIX'
            elif symbol == 'DXY':
                yahoo_symbol = 'DX-Y.NYB'
            elif symbol == 'TNX':
                yahoo_symbol = '^TNX'
            elif symbol == 'SPX':
                yahoo_symbol = '^GSPC'
            
            ticker = yf.Ticker(yahoo_symbol)
            
            # Fetch 2 years hourly
            data = ticker.history(period='2y', interval='1h')
            
            if data.empty:
                logger.warning(f"  {symbol}: No data returned")
                failed.append(symbol)
                continue
            
            # Save to CSV
            csv_file = config.data_raw / f"{symbol}.csv"
            data.to_csv(csv_file)
            
            logger.info(f"  ✅ {symbol}: {len(data)} data points saved")
            successful += 1
            
        except Exception as e:
            logger.error(f"  ❌ {symbol}: Error - {e}")
            failed.append(symbol)
            continue
        
        # Be polite to Yahoo
        time.sleep(1)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LOADING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {successful}/{len(all_symbols)}")
    if failed:
        logger.info(f"Failed: {', '.join(failed)}")
    logger.info(f"{'='*60}\n")

if __name__ == "__main__":
    load_all_data()