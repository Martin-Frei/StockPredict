import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaVantageLoader:
    """
    Modern Alpha Vantage Data Loader
    - Uses centralized config
    - Better error handling
    - Cleaner code structure
    """

    def __init__(self):
        self.api_keys = config.alpha_vantage_keys
        self.current_key_index = 0
        self.base_url = "https://www.alphavantage.co/query"
        self.save_path = config.data_raw
        self.calls_file = Path("api_calls_today.json")
        self.calls_today = self._load_calls_today()

        logger.info(
            f"AlphaVantage Loader initialized with {len(self.api_keys)} API keys"
        )

    def _load_calls_today(self):
        """Load API call counter from JSON"""
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            if self.calls_file.exists():
                with open(self.calls_file, "r") as f:
                    data = json.load(f)

                if data.get("date") == today:
                    calls_today = data.get("calls", {})
                    logger.info(
                        f"Call counter loaded: {sum(int(c) for c in calls_today.values())} total calls today"
                    )
                else:
                    calls_today = {}
                    logger.info("New day detected - call counter reset")
            else:
                calls_today = {}
                logger.info("Call counter file created")

            for i in range(len(self.api_keys)):
                if str(i) not in calls_today:
                    calls_today[str(i)] = 0

            return calls_today
        except Exception as e:
            logger.warning(f"Error loading call counter: {e}")
            return {str(i): 0 for i in range(len(self.api_keys))}

    def _save_calls_today(self):
        """Save API call counter to JSON"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        data = {
            "date": today,
            "last_updated": current_time,
            "calls": self.calls_today,
            "total_calls_today": sum(int(count) for count in self.calls_today.values()),
        }

        try:
            with open(self.calls_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving call counter: {e}")

    def get_current_api_key(self):
        """Get current active API key"""
        if not self.api_keys:
            return None

        current_calls = int(self.calls_today.get(str(self.current_key_index), 0))
        if current_calls >= 25:
            next_key = (self.current_key_index + 1) % len(self.api_keys)
            next_calls = int(self.calls_today.get(str(next_key), 0))

            if next_calls < 25:
                logger.info(f"Switching to API Key {next_key + 1}")
                self.current_key_index = next_key
            else:
                logger.warning("All API keys have reached daily limit!")
                return None

        return self.api_keys[self.current_key_index]

    def increment_call_counter(self):
        """Increment call counter for current key"""
        key_str = str(self.current_key_index)
        current_calls = int(self.calls_today.get(key_str, 0))
        self.calls_today[key_str] = current_calls + 1
        self._save_calls_today()

        remaining = 25 - self.calls_today[key_str]
        logger.info(
            f"API Calls: {self.calls_today[key_str]}/25 (Key {self.current_key_index + 1}, {remaining} remaining)"
        )

    def fetch_stock_data(self, symbol, months_back=3):
        """Fetch stock data from Alpha Vantage"""
        api_key = self.get_current_api_key()
        if not api_key:
            logger.error("No available API key!")
            return pd.DataFrame()

        logger.info(
            f"Fetching {symbol} from Alpha Vantage (Key {self.current_key_index + 1})..."
        )

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            self.increment_call_counter()

            if response.status_code != 200:
                logger.error(f"HTTP Error: {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage Error: {data['Error Message']}")
                return pd.DataFrame()

            if "Note" in data:
                logger.warning(f"Alpha Vantage Note: {data['Note']}")
                return pd.DataFrame()

            if "Time Series (60min)" not in data:
                logger.error(f"Unexpected API response: {list(data.keys())}")
                # Print actual message
                if "Information" in data:
                    logger.error(f"Information message: {data['Information']}")
                logger.error(f"Full response: {data}")
                return pd.DataFrame()

            time_series = data["Time Series (60min)"]

            df_list = []
            for timestamp_str, values in time_series.items():
                try:
                    df_list.append(
                        {
                            "DateTime": pd.to_datetime(timestamp_str),
                            "Open": float(values["1. open"]),
                            "High": float(values["2. high"]),
                            "Low": float(values["3. low"]),
                            "Close": float(values["4. close"]),
                            "Adj Close": float(values["4. close"]),
                            "Volume": int(values["5. volume"]),
                        }
                    )
                except (ValueError, KeyError):
                    continue

            if not df_list:
                logger.error("No valid data converted!")
                return pd.DataFrame()

            df = pd.DataFrame(df_list)
            df = df.sort_values("DateTime")

            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            df = df[df["DateTime"] >= cutoff_date]
            df = df[df["DateTime"].dt.weekday < 5]

            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()

    def save_to_csv(self, data, symbol):
        """Save DataFrame to CSV"""
        if data.empty:
            logger.warning(f"No data to save for {symbol}")
            return None

        csv_file = self.save_path / f"{symbol}.csv"
        data.to_csv(csv_file, index=False)
        logger.info(f"Saved: {csv_file}")
        return csv_file

    def load_from_csv(self, symbol):
        """Load existing CSV file"""
        csv_file = self.save_path / f"{symbol}.csv"

        if not csv_file.exists():
            return None

        try:
            data = pd.read_csv(csv_file)
            data["DateTime"] = pd.to_datetime(data["DateTime"])
            logger.info(f"Loaded existing CSV: {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None

    def update_stock_data(self, symbol, months_back=3):
        """Update stock data (incremental)"""
        existing_data = self.load_from_csv(symbol)

        if existing_data is not None and not existing_data.empty:
            last_date = existing_data["DateTime"].max()
            logger.info(f"Last date in CSV: {last_date}")

            new_data = self.fetch_stock_data(symbol, months_back)

            if new_data.empty:
                logger.warning(f"No new data for {symbol}")
                return existing_data

            new_data = new_data[new_data["DateTime"] > last_date]

            if new_data.empty:
                logger.info(f"{symbol} already up to date")
                return existing_data

            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.sort_values("DateTime")
            combined_data = combined_data.drop_duplicates(
                subset=["DateTime"], keep="last"
            )

            logger.info(f"Added {len(new_data)} new data points")
        else:
            logger.info(f"First load for {symbol}")
            combined_data = self.fetch_stock_data(symbol, months_back)

        if not combined_data.empty:
            self.save_to_csv(combined_data, symbol)

        return combined_data
