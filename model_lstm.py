import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
    print("[OK] Scikit-Learn Neural Network verfügbar")
except ImportError:
    print("[ERROR] Scikit-Learn nicht verfügbar")
    SKLEARN_AVAILABLE = False

class BankNeuralNetworkPredictor:
    """
    Neural Network Model für 7h Bank-Predictions - ASCII VERSION
    Scikit-Learn MLPRegressor für komplexe Pattern-Erkennung (ohne TensorFlow)
    """
    
    def __init__(self, features_path="features_enhanced"):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-Learn muss verfügbar sein")
        
        self.features_path = features_path
        self.bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
        self.models = {}
        self.predictions = {}
        self.performance = {}
        self.scalers = {}
        
        # Neural Network Parameter
        self.sequence_length = 12  # 12 Stunden Lookback (weniger als LSTM für Performance)
        self.test_size = 0.2
        self.random_state = 42
        
        # MLPRegressor Parameter (Neural Network)
        self.nn_params = {
            'hidden_layer_sizes': (64, 32, 16),  # 3 Hidden Layers
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,  # Regularization
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 20,
            'random_state': self.random_state
        }
        
        # Features für Neural Network (alle außer DateTime und Targets)
        self.excluded_features = [
            'DateTime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'Target_1h', 'Target_4h', 'Target_7h', 'Target_24h', 
            'Target_Direction_1h', 'Target_Direction_4h', 'Target_Direction_7h'
        ]
        
        print(f"[NN] Neural Network Predictor initialisiert")
        print(f"   [PATH] Features Pfad: {features_path}")
        print(f"   [BANKS] Banken: {len(self.bank_symbols)}")
        print(f"   [LAYERS] Hidden Layers: {self.nn_params['hidden_layer_sizes']}")
        print(f"   [SEQUENCE] Sequence Length: {self.sequence_length} Stunden")
    
    def load_bank_data(self, symbol):
        """Lädt Enhanced Features für eine Bank"""
        
        csv_file = os.path.join(self.features_path, f"{symbol}_enhanced.csv")
        
        if not os.path.exists(csv_file):
            print(f"   [ERROR] {symbol}: Datei nicht gefunden: {csv_file}")
            return None
        
        try:
            df = pd.read_csv(csv_file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Sortieren nach Zeit
            df = df.sort_values('DateTime')
            
            # Prüfe Target-Spalte
            if 'Target_7h' not in df.columns:
                print(f"   [WARN] {symbol}: Target_7h Spalte fehlt")
                return None
            
            # NaN-Handling
            df = df.dropna(subset=['DateTime', 'Close', 'Target_7h'])
            
            print(f"   [OK] {symbol}: {len(df)} Zeilen geladen")
            return df
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: Fehler beim Laden: {str(e)}")
            return None
    
    def prepare_features(self, df, symbol):
        """Bereitet Features für Neural Network vor"""
        
        # Verfügbare Feature-Spalten ermitteln
        available_columns = df.columns.tolist()
        feature_columns = [col for col in available_columns if col not in self.excluded_features]
        
        print(f"   [FEATURES] {symbol}: {len(feature_columns)} Features für Neural Network")
        
        # Features extrahieren
        features_df = df[feature_columns].copy()
        
        # Robustes NaN-Handling
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Infinite-Werte behandeln
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Target extrahieren
        target = df['Target_7h'].values
        
        print(f"      Feature-Shape: {features_df.shape}")
        print(f"      Target-Shape: {target.shape}")
        
        return features_df.values, target, feature_columns
    
    def create_sequence_features(self, features, target, symbol):
        """Erstellt Sequence-basierte Features für Neural Network"""
        
        sequence_features = []
        sequence_targets = []
        
        # Erstelle Features mit Lookback-Window
        for i in range(self.sequence_length, len(features)):
            # Aktuelle Features
            current_features = features[i]
            
            # Historische Features (Lookback)
            lookback_features = []
            for j in range(1, self.sequence_length + 1):
                if i - j >= 0:
                    lookback_features.extend(features[i - j])
                else:
                    # Padding mit aktuellen Features falls nicht genug Historie
                    lookback_features.extend(current_features)
            
            # Kombiniere aktuelle + historische Features
            combined_features = np.concatenate([current_features, lookback_features])
            
            # Target für diese Position
            target_value = target[i]
            
            if not np.isnan(target_value):
                sequence_features.append(combined_features)
                sequence_targets.append(target_value)
        
        sequence_features = np.array(sequence_features)
        sequence_targets = np.array(sequence_targets)
        
        print(f"   [SEQUENCE] {symbol}: {len(sequence_features)} Sequenz-Features erstellt")
        print(f"      Feature-Dimension: {sequence_features.shape[1]}")
        
        return sequence_features, sequence_targets
    
    def train_neural_network_model(self, symbol):
        """Trainiert Neural Network Model für eine Bank"""
        
        print(f"\n[TRAIN] Trainiere Neural Network Model für {symbol}...")
        
        # Daten laden
        df = self.load_bank_data(symbol)
        if df is None:
            return None
        
        # Features vorbereiten
        features, target, feature_columns = self.prepare_features(df, symbol)
        if len(features) < self.sequence_length + 50:
            print(f"   [ERROR] {symbol}: Zu wenig Daten für Training ({len(features)})")
            return None
        
        # Sequence Features erstellen
        X, y = self.create_sequence_features(features, target, symbol)
        if len(X) < 50:
            print(f"   [ERROR] {symbol}: Zu wenig Sequenzen für Training ({len(X)})")
            return None
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )
        
        print(f"   [SPLIT] {symbol}: Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Features skalieren
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Scaler speichern
        self.scalers[symbol] = scaler
        
        # Neural Network Model erstellen und trainieren
        try:
            print(f"   [START] Starte Neural Network Training...")
            
            model = MLPRegressor(**self.nn_params)
            model.fit(X_train_scaled, y_train)
            
            # Predictions für Evaluation
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Metriken berechnen
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            print(f"   [SUCCESS] {symbol}: Neural Network Training erfolgreich")
            print(f"      Iterationen: {model.n_iter_}")
            print(f"      Train MSE: {train_mse:.6f}")
            print(f"      Test MSE: {test_mse:.6f}")
            print(f"      Test MAE: {test_mae:.6f}")
            print(f"      Test R²: {test_r2:.4f}")
            
            # Model speichern
            self.models[symbol] = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'feature_count': len(feature_columns),
                'sequence_length': self.sequence_length,
                'metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'iterations': model.n_iter_
                },
                'raw_data': df
            }
            
            return model
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: Neural Network Training-Fehler: {str(e)}")
            return None
    
    def get_next_trading_hour(self, last_time):
        """Bestimmt nächste Handelsstunde"""
        
        if last_time.weekday() >= 5:
            days_until_monday = 7 - last_time.weekday()
            next_monday = last_time + timedelta(days=days_until_monday)
            return next_monday.replace(hour=13, minute=30, second=0, microsecond=0)
        
        if last_time.hour >= 20:
            next_day = last_time + timedelta(days=1)
            if next_day.weekday() >= 5:
                days_until_monday = 7 - next_day.weekday()
                next_monday = next_day + timedelta(days=days_until_monday)
                return next_monday.replace(hour=13, minute=30, second=0, microsecond=0)
            else:
                return next_day.replace(hour=13, minute=30, second=0, microsecond=0)
        
        if last_time.hour < 13 or (last_time.hour == 13 and last_time.minute < 30):
            return last_time.replace(hour=13, minute=30, second=0, microsecond=0)
        
        next_hour = last_time.replace(minute=30, second=0, microsecond=0) + timedelta(hours=1)
        
        if next_hour.hour <= 19:
            return next_hour
        else:
            return self.get_next_trading_hour(next_hour)
    
    def make_prediction(self, symbol, prediction_start_time=None):
        """Macht 7h Neural Network Prediction für eine Bank"""
        
        if symbol not in self.models:
            print(f"[ERROR] {symbol}: Neural Network Model nicht trainiert")
            return None
        
        print(f"\n[PREDICT] Neural Network Prediction für {symbol}...")
        
        try:
            model_data = self.models[symbol]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            raw_data = model_data['raw_data']
            sequence_length = model_data['sequence_length']
            
            # Bestimme Prediction-Start-Zeit
            if prediction_start_time is None:
                last_time = raw_data['DateTime'].max()
                next_trading_hour = self.get_next_trading_hour(last_time)
                prediction_start_time = next_trading_hour
            
            print(f"   [TIME] {symbol}: Neural Network Prediction ab {prediction_start_time}")
            
            # Letzte sequence_length + 1 Zeilen für Feature-Erstellung
            recent_data = raw_data.tail(sequence_length + 1)
            
            # Features für Prediction vorbereiten
            feature_data = recent_data[feature_columns].fillna(method='ffill').fillna(0)
            feature_data = feature_data.replace([np.inf, -np.inf], 0)
            features_array = feature_data.values
            
            # Aktuelle Features (letzte Zeile)
            current_features = features_array[-1]
            
            # Historische Features (Lookback)
            lookback_features = []
            for j in range(1, sequence_length + 1):
                if len(features_array) - 1 - j >= 0:
                    lookback_features.extend(features_array[-1 - j])
                else:
                    lookback_features.extend(current_features)
            
            # Kombiniere für Prediction
            prediction_features = np.concatenate([current_features, lookback_features])
            prediction_input = prediction_features.reshape(1, -1)
            
            # Skalieren
            prediction_input_scaled = scaler.transform(prediction_input)
            
            # Neural Network Prediction
            nn_prediction = model.predict(prediction_input_scaled)[0]
            
            # Für 7h-Predictions: Gleiche Prediction für alle 7 Stunden
            seven_hour_predictions = [nn_prediction] * 7
            
            # Future Timestamps erstellen
            future_times = []
            current_time = prediction_start_time
            for i in range(7):
                future_times.append(current_time)
                current_time += timedelta(hours=1)
            
            # Confidence basierend auf Test-Performance
            test_mae = model_data['metrics']['test_mae']
            confidence_interval = test_mae * 1.96  # 95% CI approximation
            
            lower_bounds = [pred - confidence_interval for pred in seven_hour_predictions]
            upper_bounds = [pred + confidence_interval for pred in seven_hour_predictions]
            
            # Ergebnisse formatieren
            prediction_result = pd.DataFrame({
                'DateTime': future_times,
                'Predicted_Return_7h': seven_hour_predictions,
                'Lower_Bound': lower_bounds,
                'Upper_Bound': upper_bounds,
                'Confidence': [confidence_interval * 2] * 7,
                'Direction': [1 if x > 0 else 0 for x in seven_hour_predictions]
            })
            
            # Zusätzliche Metriken
            prediction_result['Return_Magnitude'] = abs(prediction_result['Predicted_Return_7h'])
            prediction_result['Confidence_Score'] = 1 / (prediction_result['Confidence'] + 0.001)
            prediction_result['Model_Type'] = 'Neural_Network'
            prediction_result['NN_Iterations'] = model_data['metrics']['iterations']
            prediction_result['NN_Test_R2'] = model_data['metrics']['test_r2']
            
            self.predictions[symbol] = prediction_result
            
            print(f"   [SUCCESS] {symbol}: Neural Network 7h Prediction erfolgreich")
            print(f"      7h Return Prediction: {nn_prediction:.4f}")
            print(f"      Direction: {'Bullish' if nn_prediction > 0 else 'Bearish'}")
            print(f"      Confidence: ±{confidence_interval:.4f}")
            print(f"      Test R²: {model_data['metrics']['test_r2']:.4f}")
            
            return prediction_result
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: Neural Network Prediction-Fehler: {str(e)}")
            return None
    
    def train_all_banks(self):
        """Trainiert Neural Network Models für alle Banken"""
        
        print(f"[START] NEURAL NETWORK TRAINING FÜR ALLE BANKEN")
        print("=" * 60)
        
        successful_trainings = 0
        
        for symbol in self.bank_symbols:
            model = self.train_neural_network_model(symbol)
            if model is not None:
                successful_trainings += 1
        
        print(f"\n[RESULT] Neural Network Training abgeschlossen:")
        print(f"   [SUCCESS] Erfolgreich: {successful_trainings}/{len(self.bank_symbols)} Banken")
        print(f"   [MODELS] Trainierte Models: {list(self.models.keys())}")
        
        # Model-Übersicht
        if self.models:
            print(f"\n[OVERVIEW] Neural Network Model-Übersicht:")
            for symbol, data in self.models.items():
                metrics = data['metrics']
                print(f"   {symbol}: Iter={metrics['iterations']}, R²={metrics['test_r2']:.3f}, MAE={metrics['test_mae']:.4f}")
        
        return successful_trainings > 0
    
    def predict_all_banks(self, prediction_start_time=None):
        """Macht Neural Network Predictions für alle trainierten Banken"""
        
        print(f"\n[START] 7H-NEURAL-NETWORK-PREDICTIONS FÜR ALLE BANKEN")
        print("=" * 65)
        
        if not self.models:
            print("[ERROR] Keine trainierten Neural Network Models verfügbar!")
            return {}
        
        successful_predictions = 0
        
        for symbol in self.models.keys():
            prediction = self.make_prediction(symbol, prediction_start_time)
            if prediction is not None:
                successful_predictions += 1
        
        print(f"\n[RESULT] Neural Network Predictions abgeschlossen:")
        print(f"   [SUCCESS] Erfolgreich: {successful_predictions}/{len(self.models)} Banken")
        
        return self.predictions
    
    def get_trading_summary(self):
        """Erstellt Neural Network Trading-Summary aller Predictions"""
        
        if not self.predictions:
            print("[ERROR] Keine Neural Network Predictions verfügbar!")
            return None
        
        print(f"\n[SUMMARY] NEURAL NETWORK TRADING SUMMARY")
        print("=" * 55)
        
        summary_data = []
        
        for symbol, pred in self.predictions.items():
            total_return = pred['Predicted_Return_7h'].sum()
            avg_confidence = pred['Confidence_Score'].mean()
            bullish_hours = pred['Direction'].sum()
            max_single_return = pred['Predicted_Return_7h'].max()
            min_single_return = pred['Predicted_Return_7h'].min()
            nn_iterations = pred['NN_Iterations'].iloc[0]
            nn_r2 = pred['NN_Test_R2'].iloc[0]
            
            summary_data.append({
                'Symbol': symbol,
                'Total_7h_Return': total_return,
                'Avg_Confidence': avg_confidence,
                'Bullish_Hours': f"{bullish_hours}/7",
                'Max_Hourly_Return': max_single_return,
                'Min_Hourly_Return': min_single_return,
                'Net_Direction': 'Bullish' if total_return > 0 else 'Bearish',
                'Strength': abs(total_return),
                'NN_Iterations': nn_iterations,
                'NN_Test_R2': nn_r2,
                'Model_Type': 'Neural_Network'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total_7h_Return', ascending=False)
        
        print("[TOP] NEURAL NETWORK TOP PERFORMERS (7h Total Return):")
        print(summary_df[['Symbol', 'Total_7h_Return', 'Net_Direction', 'Bullish_Hours', 'NN_Test_R2']].head())
        
        print(f"\n[PORTFOLIO] NEURAL NETWORK PORTFOLIO OVERVIEW:")
        total_portfolio_return = summary_df['Total_7h_Return'].sum()
        bullish_banks = len(summary_df[summary_df['Total_7h_Return'] > 0])
        avg_r2 = summary_df['NN_Test_R2'].mean()
        avg_iterations = summary_df['NN_Iterations'].mean()
        
        print(f"   Neural Network Portfolio 7h Return: {total_portfolio_return:.4f}")
        print(f"   Neural Network Bullish Banks: {bullish_banks}/{len(summary_df)}")
        print(f"   Neural Network Average Confidence: {summary_df['Avg_Confidence'].mean():.2f}")
        print(f"   Neural Network Average R²: {avg_r2:.3f}")
        print(f"   Neural Network Average Iterations: {avg_iterations:.0f}")
        
        return summary_df
    
    def save_predictions(self, output_path="neural_network_predictions"):
        """Speichert alle Neural Network Predictions als CSV"""
        
        if not self.predictions:
            print("[ERROR] Keine Neural Network Predictions zum Speichern!")
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n[SAVE] Speichere Neural Network Predictions...")
        
        # Einzelne Bank-Predictions
        for symbol, pred in self.predictions.items():
            csv_file = os.path.join(output_path, f"{symbol}_7h_neural_network_prediction.csv")
            pred.to_csv(csv_file, index=False)
            print(f"   [OK] {csv_file}")
        
        # Kombinierte Summary
        summary = self.get_trading_summary()
        if summary is not None:
            summary_file = os.path.join(output_path, "neural_network_trading_summary.csv")
            summary.to_csv(summary_file, index=False)
            print(f"   [OK] {summary_file}")
        
        # Model-Details speichern
        model_details = []
        for symbol, data in self.models.items():
            model_details.append({
                'Symbol': symbol,
                'Hidden_Layers': str(self.nn_params['hidden_layer_sizes']),
                'Feature_Count': data['feature_count'],
                'Sequence_Length': data['sequence_length'],
                'Iterations': data['metrics']['iterations'],
                'Test_MSE': data['metrics']['test_mse'],
                'Test_MAE': data['metrics']['test_mae'],
                'Test_R2': data['metrics']['test_r2']
            })
        
        model_df = pd.DataFrame(model_details)
        model_file = os.path.join(output_path, "neural_network_model_details.csv")
        model_df.to_csv(model_file, index=False)
        print(f"   [OK] {model_file}")
        
        print(f"   [DONE] Alle Neural Network Predictions gespeichert in: {output_path}")

def run_neural_network_pipeline():
    """Hauptfunktion für Neural Network Pipeline"""
    
    print("[START] NEURAL NETWORK BANK-PREDICTION PIPELINE")
    print("=" * 60)
    
    # Neural Network Predictor initialisieren
    predictor = BankNeuralNetworkPredictor()
    
    # Prüfe ob Features verfügbar
    if not os.path.exists(predictor.features_path):
        print(f"[ERROR] Features-Ordner nicht gefunden: {predictor.features_path}")
        print("   Führe zuerst Enhanced Feature-Engineering aus!")
        return
    
    # Training
    print(f"[START] Starte Neural Network Training...")
    training_success = predictor.train_all_banks()
    
    if not training_success:
        print("[ERROR] Neural Network Training fehlgeschlagen!")
        return
    
    # Predictions für Montag (nächste Handelsstunden)
    print(f"[START] Starte Neural Network 7h-Predictions...")
    predictions = predictor.predict_all_banks()
    
    if not predictions:
        print("[ERROR] Neural Network Predictions fehlgeschlagen!")
        return
    
    # Trading Summary
    summary = predictor.get_trading_summary()
    
    # Speichern
    predictor.save_predictions()
    
    print(f"\n[SUCCESS] NEURAL NETWORK PIPELINE ERFOLGREICH!")
    print(f"   [RESULTS] {len(predictions)} Neural Network Bank-Predictions erstellt")
    print(f"   [SAVE] Ergebnisse gespeichert in: neural_network_predictions/")
    print(f"   [READY] Bereit für Ensemble mit Prophet + ARIMA!")
    print(f"   [INFO] Funktioniert OHNE TensorFlow (nur Scikit-Learn)")

if __name__ == "__main__":
    if not SKLEARN_AVAILABLE:
        print("[ERROR] Scikit-Learn nicht verfügbar!")
    else:
        run_neural_network_pipeline()