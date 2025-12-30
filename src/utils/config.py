import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    Zentrale Configuration Management Klasse
    Lädt .env und config.yaml
    """
    
    def __init__(self):
        # Project root
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load .env file
        env_path = self.project_root / 'config' / '.env'
        load_dotenv(env_path)
        
        # Load YAML config
        config_path = self.project_root / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # API Keys
        self.alpha_vantage_keys = [
            os.getenv('ALPHA_VANTAGE_KEY_1'),
            os.getenv('ALPHA_VANTAGE_KEY_2')
        ]
        self.alpha_vantage_keys = [k for k in self.alpha_vantage_keys if k]
        
        # Paths
        self.data_raw = self.project_root / 'data' / 'raw'
        self.data_processed = self.project_root / 'data' / 'processed'
        self.predictions_path = self.project_root / 'data' / 'predictions'
        
        # Create directories if not exist
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.predictions_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def bank_symbols(self):
        return self.config['data']['bank_symbols']
    
    @property
    def macro_symbols(self):
        return self.config['data']['macro_symbols']
    
    @property
    def prediction_horizon(self):
        return self.config['data']['prediction_horizon']
    
    @property
    def prophet_params(self):
        return self.config['models']['prophet']
    
    @property
    def xgboost_params(self):
        return self.config['models']['xgboost']
    
    @property
    def arima_params(self):
        return self.config['models']['arima']

config = Config()