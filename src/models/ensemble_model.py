import numpy as np
import pandas as pd


class EnsembleModel:
    """Ensemble model that combines predictions from multiple models."""
    
    def __init__(self, weights=None):
        """
        Initialize ensemble model.
        
        Args:
            weights: Dictionary of model weights (e.g., {'arima': 0.3, 'lstm': 0.4, 'prophet': 0.3})
                     If None, equal weights are used
        """
        self.weights = weights
        self.predictions = {}
        
    def add_prediction(self, model_name, prediction):
        """
        Add a model's prediction to the ensemble.
        
        Args:
            model_name: Name of the model (e.g., 'arima', 'lstm', 'prophet')
            prediction: Model's prediction array
        """
        self.predictions[model_name] = np.array(prediction).flatten()
        print(f"Added {model_name} predictions: {len(prediction)} values")
    
    def combine_predictions(self, method='weighted_average'):
        """
        Combine predictions from all models.
        
        Args:
            method: Combination method - 'weighted_average', 'simple_average', or 'median'
            
        Returns:
            Combined predictions
        """
        if not self.predictions:
            raise ValueError("No predictions added. Call add_prediction() first.")
        
        # Ensure all predictions have the same length
        min_length = min(len(pred) for pred in self.predictions.values())
        aligned_predictions = {
            name: pred[:min_length] 
            for name, pred in self.predictions.items()
        }
        
        if method == 'weighted_average':
            return self._weighted_average(aligned_predictions)
        elif method == 'simple_average':
            return self._simple_average(aligned_predictions)
        elif method == 'median':
            return self._median(aligned_predictions)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _weighted_average(self, predictions):
        """Compute weighted average of predictions."""
        if self.weights is None:
            # Use equal weights if not specified
            n_models = len(predictions)
            self.weights = {name: 1.0/n_models for name in predictions.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        normalized_weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"Using weights: {normalized_weights}")
        
        # Compute weighted average
        ensemble_pred = np.zeros(len(next(iter(predictions.values()))))
        for name, pred in predictions.items():
            weight = normalized_weights.get(name, 0)
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def _simple_average(self, predictions):
        """Compute simple average of predictions."""
        print("Using simple average")
        pred_array = np.array(list(predictions.values()))
        return np.mean(pred_array, axis=0)
    
    def _median(self, predictions):
        """Compute median of predictions."""
        print("Using median")
        pred_array = np.array(list(predictions.values()))
        return np.median(pred_array, axis=0)
    
    def evaluate(self, actual_values, method='weighted_average'):
        """
        Evaluate ensemble performance.
        
        Args:
            actual_values: Actual values for comparison
            method: Combination method to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        ensemble_pred = self.combine_predictions(method=method)
        
        # Align lengths
        min_length = min(len(ensemble_pred), len(actual_values))
        ensemble_pred = ensemble_pred[:min_length]
        actual_values = np.array(actual_values).flatten()[:min_length]
        
        # Calculate metrics
        mae = np.mean(np.abs(ensemble_pred - actual_values))
        mse = np.mean((ensemble_pred - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - ensemble_pred) / actual_values)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print(f"\nEnsemble Model Evaluation ({method}):")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, ensemble_pred
    
    def optimize_weights(self, actual_values, methods=['weighted_average', 'simple_average', 'median']):
        """
        Find the best combination method based on evaluation metrics.
        
        Args:
            actual_values: Actual values for comparison
            methods: List of methods to try
            
        Returns:
            Best method and its metrics
        """
        best_method = None
        best_rmse = float('inf')
        results = {}
        
        for method in methods:
            metrics, _ = self.evaluate(actual_values, method=method)
            results[method] = metrics
            
            if metrics['RMSE'] < best_rmse:
                best_rmse = metrics['RMSE']
                best_method = method
        
        print(f"\nBest method: {best_method} (RMSE: {best_rmse:.4f})")
        return best_method, results
