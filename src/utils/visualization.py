import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


class Visualizer:
    """Visualization tools for stock prediction results."""
    
    def __init__(self, output_dir='outputs'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save output plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 7)
    
    def plot_stock_data(self, data, title='Stock Price History', save_as=None):
        """
        Plot stock price history.
        
        Args:
            data: DataFrame with Date and Close columns
            title: Plot title
            save_as: Filename to save the plot (optional)
        """
        plt.figure(figsize=(14, 7))
        plt.plot(data['Date'], data['Close'], label='Close Price', linewidth=2)
        
        if 'MA7' in data.columns:
            plt.plot(data['Date'], data['MA7'], label='7-Day MA', alpha=0.7)
        if 'MA21' in data.columns:
            plt.plot(data['Date'], data['MA21'], label='21-Day MA', alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_as:
            filepath = os.path.join(self.output_dir, save_as)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_predictions(self, dates, actual, predictions_dict, title='Model Predictions', save_as=None):
        """
        Plot predictions from multiple models.
        
        Args:
            dates: Date values for x-axis
            actual: Actual values
            predictions_dict: Dictionary of {model_name: predictions}
            title: Plot title
            save_as: Filename to save the plot (optional)
        """
        plt.figure(figsize=(14, 7))
        
        # Plot actual values
        plt.plot(dates, actual, label='Actual', color='black', linewidth=2, marker='o', markersize=4)
        
        # Plot predictions from each model
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            color = colors[i % len(colors)]
            # Align predictions with dates
            pred_length = min(len(dates), len(predictions))
            plt.plot(dates[:pred_length], predictions[:pred_length], 
                    label=f'{model_name}', color=color, linewidth=2, 
                    alpha=0.7, linestyle='--', marker='s', markersize=3)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_as:
            filepath = os.path.join(self.output_dir, save_as)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict, save_as=None):
        """
        Plot comparison of metrics across models.
        
        Args:
            metrics_dict: Dictionary of {model_name: metrics_dict}
            save_as: Filename to save the plot (optional)
        """
        # Extract metrics
        models = list(metrics_dict.keys())
        metrics = ['MAE', 'RMSE', 'MAPE']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            axes[i].bar(models, values, color=sns.color_palette('husl', len(models)))
            axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric, fontsize=10)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_as:
            filepath = os.path.join(self.output_dir, save_as)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_training_history(self, history, save_as=None):
        """
        Plot LSTM training history.
        
        Args:
            history: Training history from LSTM model
            save_as: Filename to save the plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss', fontsize=10)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
            axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_title('Model MAE', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=10)
            axes[1].set_ylabel('MAE', fontsize=10)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_as:
            filepath = os.path.join(self.output_dir, save_as)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def plot_forecast(self, historical_dates, historical_values, 
                     forecast_dates, forecast_values, 
                     title='Stock Price Forecast', save_as=None):
        """
        Plot historical data with future forecast.
        
        Args:
            historical_dates: Historical dates
            historical_values: Historical values
            forecast_dates: Forecast dates
            forecast_values: Forecast values
            title: Plot title
            save_as: Filename to save the plot (optional)
        """
        plt.figure(figsize=(14, 7))
        
        # Plot historical data
        plt.plot(historical_dates, historical_values, 
                label='Historical', color='black', linewidth=2)
        
        # Plot forecast
        plt.plot(forecast_dates, forecast_values, 
                label='Forecast', color='red', linewidth=2, linestyle='--', marker='o')
        
        # Add vertical line to separate historical and forecast
        if len(historical_dates) > 0:
            plt.axvline(x=historical_dates[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_as:
            filepath = os.path.join(self.output_dir, save_as)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
        plt.close()
