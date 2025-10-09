"""
Unit tests for the StockPredict system.

These tests verify the structure and logic of the prediction system
without requiring actual data or model training.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModuleImports(unittest.TestCase):
    """Test that all modules can be imported correctly."""
    
    def test_import_arima_model(self):
        """Test ARIMA model import."""
        try:
            from src.models.arima_model import ARIMAModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ARIMAModel: {e}")
    
    def test_import_lstm_model(self):
        """Test LSTM model import."""
        try:
            from src.models.lstm_model import LSTMModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import LSTMModel: {e}")
    
    def test_import_prophet_model(self):
        """Test Prophet model import."""
        try:
            from src.models.prophet_model import ProphetModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ProphetModel: {e}")
    
    def test_import_ensemble_model(self):
        """Test Ensemble model import."""
        try:
            from src.models.ensemble_model import EnsembleModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import EnsembleModel: {e}")
    
    def test_import_data_preprocessor(self):
        """Test DataPreprocessor import."""
        try:
            from src.utils.data_preprocessing import DataPreprocessor
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import DataPreprocessor: {e}")
    
    def test_import_visualizer(self):
        """Test Visualizer import."""
        try:
            from src.utils.visualization import Visualizer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import Visualizer: {e}")


class TestClassInitialization(unittest.TestCase):
    """Test that all classes can be initialized."""
    
    def test_arima_initialization(self):
        """Test ARIMA model initialization."""
        from src.models.arima_model import ARIMAModel
        model = ARIMAModel(order=(5, 1, 0))
        self.assertEqual(model.order, (5, 1, 0))
        self.assertIsNone(model.model)
        self.assertIsNone(model.fitted_model)
    
    def test_lstm_initialization(self):
        """Test LSTM model initialization."""
        from src.models.lstm_model import LSTMModel
        model = LSTMModel(sequence_length=60, units=50, dropout_rate=0.2)
        self.assertEqual(model.sequence_length, 60)
        self.assertEqual(model.units, 50)
        self.assertEqual(model.dropout_rate, 0.2)
        self.assertIsNone(model.model)
    
    def test_prophet_initialization(self):
        """Test Prophet model initialization."""
        from src.models.prophet_model import ProphetModel
        model = ProphetModel()
        self.assertFalse(model.fitted)
    
    def test_ensemble_initialization(self):
        """Test Ensemble model initialization."""
        from src.models.ensemble_model import EnsembleModel
        
        # Test with no weights
        model1 = EnsembleModel()
        self.assertIsNone(model1.weights)
        self.assertEqual(len(model1.predictions), 0)
        
        # Test with custom weights
        weights = {'arima': 0.3, 'lstm': 0.5, 'prophet': 0.2}
        model2 = EnsembleModel(weights=weights)
        self.assertEqual(model2.weights, weights)
    
    def test_data_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        from src.utils.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(ticker='AAPL', start_date='2020-01-01')
        self.assertEqual(preprocessor.ticker, 'AAPL')
        self.assertEqual(preprocessor.start_date, '2020-01-01')
        self.assertIsNone(preprocessor.data)
    
    def test_visualizer_initialization(self):
        """Test Visualizer initialization."""
        from src.utils.visualization import Visualizer
        viz = Visualizer(output_dir='test_outputs')
        self.assertEqual(viz.output_dir, 'test_outputs')


class TestEnsembleLogic(unittest.TestCase):
    """Test ensemble model logic without actual predictions."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.models.ensemble_model import EnsembleModel
        import numpy as np
        
        self.ensemble = EnsembleModel()
        # Add some dummy predictions
        self.ensemble.add_prediction('model1', np.array([1, 2, 3, 4, 5]))
        self.ensemble.add_prediction('model2', np.array([2, 3, 4, 5, 6]))
        self.ensemble.add_prediction('model3', np.array([3, 4, 5, 6, 7]))
    
    def test_simple_average(self):
        """Test simple average combination."""
        import numpy as np
        result = self.ensemble.combine_predictions(method='simple_average')
        expected = np.array([2, 3, 4, 5, 6])  # (1+2+3)/3, (2+3+4)/3, etc.
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_median(self):
        """Test median combination."""
        import numpy as np
        result = self.ensemble.combine_predictions(method='median')
        expected = np.array([2, 3, 4, 5, 6])  # median of [1,2,3], [2,3,4], etc.
        np.testing.assert_array_almost_equal(result, expected)


class TestFileStructure(unittest.TestCase):
    """Test that all required files exist."""
    
    def test_main_file_exists(self):
        """Test main.py exists."""
        self.assertTrue(os.path.exists('main.py'))
    
    def test_example_file_exists(self):
        """Test example.py exists."""
        self.assertTrue(os.path.exists('example.py'))
    
    def test_config_file_exists(self):
        """Test config.py exists."""
        self.assertTrue(os.path.exists('config.py'))
    
    def test_requirements_file_exists(self):
        """Test requirements.txt exists."""
        self.assertTrue(os.path.exists('requirements.txt'))
    
    def test_readme_file_exists(self):
        """Test README.md exists."""
        self.assertTrue(os.path.exists('README.md'))
    
    def test_models_directory_exists(self):
        """Test src/models directory exists."""
        self.assertTrue(os.path.exists('src/models'))
    
    def test_utils_directory_exists(self):
        """Test src/utils directory exists."""
        self.assertTrue(os.path.exists('src/utils'))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
