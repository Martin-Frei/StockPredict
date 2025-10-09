import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

class BankEnsemblePredictor:
    """
    Ensemble Model für 7h Bank-Predictions
    Kombiniert NUR Prophet + ARIMA (Neural Network ausgeschlossen)
    """
    
    def __init__(self, exclude_neural_network=True):
        self.bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
        
        # Prediction-Ordner
        self.prophet_path = "prophet_predictions"
        self.arima_path = "arima_predictions"
        self.nn_path = "neural_network_predictions"
        
        # NEUE OPTION: Neural Network ausschließen
        self.exclude_neural_network = exclude_neural_network
        
        # Ensemble-Daten
        self.prophet_data = {}
        self.arima_data = {}
        self.nn_data = {}
        self.ensemble_predictions = {}
        
        # Ensemble-Parameter
        self.ensemble_methods = ['average', 'weighted', 'voting']
        
        if self.exclude_neural_network:
            print(f"ENSEMBLE PREDICTOR (OHNE Neural Network)")
        else:
            print(f"ENSEMBLE PREDICTOR (MIT Neural Network)")
            
        print(f"   Banken: {len(self.bank_symbols)}")
        print(f"   Methoden: {', '.join(self.ensemble_methods)}")
        if self.exclude_neural_network:
            print(f"   HINWEIS: Neural Network wird ausgeschlossen wegen schlechter Performance")
    
    def load_prophet_predictions(self):
        """Lädt Prophet Predictions"""
        
        print(f"\nLade Prophet Predictions...")
        
        if not os.path.exists(self.prophet_path):
            print(f"   Prophet Ordner nicht gefunden: {self.prophet_path}")
            return False
        
        # Trading Summary laden
        summary_file = os.path.join(self.prophet_path, "trading_summary.csv")
        if os.path.exists(summary_file):
            prophet_summary = pd.read_csv(summary_file)
            print(f"   Prophet Summary: {len(prophet_summary)} Banken")
        else:
            print(f"   Prophet Summary nicht gefunden")
            prophet_summary = None
        
        # Einzelne Predictions laden
        loaded_count = 0
        for symbol in self.bank_symbols:
            pred_file = os.path.join(self.prophet_path, f"{symbol}_7h_prediction.csv")
            if os.path.exists(pred_file):
                try:
                    pred_data = pd.read_csv(pred_file)
                    pred_data['DateTime'] = pd.to_datetime(pred_data['DateTime'])
                    self.prophet_data[symbol] = pred_data
                    loaded_count += 1
                except Exception as e:
                    print(f"   Fehler bei {symbol}: {str(e)}")
            else:
                print(f"   {symbol} Prophet Prediction nicht gefunden")
        
        print(f"   Prophet: {loaded_count}/{len(self.bank_symbols)} Banken geladen")
        return loaded_count > 0
    
    def load_arima_predictions(self):
        """Lädt ARIMA Predictions"""
        
        print(f"\nLade ARIMA Predictions...")
        
        if not os.path.exists(self.arima_path):
            print(f"   ARIMA Ordner nicht gefunden: {self.arima_path}")
            return False
        
        # Trading Summary laden
        summary_file = os.path.join(self.arima_path, "arima_trading_summary.csv")
        if os.path.exists(summary_file):
            arima_summary = pd.read_csv(summary_file)
            print(f"   ARIMA Summary: {len(arima_summary)} Banken")
        else:
            print(f"   ARIMA Summary nicht gefunden")
        
        # Einzelne Predictions laden
        loaded_count = 0
        for symbol in self.bank_symbols:
            pred_file = os.path.join(self.arima_path, f"{symbol}_7h_arima_prediction.csv")
            if os.path.exists(pred_file):
                try:
                    pred_data = pd.read_csv(pred_file)
                    pred_data['DateTime'] = pd.to_datetime(pred_data['DateTime'])
                    self.arima_data[symbol] = pred_data
                    loaded_count += 1
                except Exception as e:
                    print(f"   Fehler bei {symbol}: {str(e)}")
            else:
                print(f"   {symbol} ARIMA Prediction nicht gefunden")
        
        print(f"   ARIMA: {loaded_count}/{len(self.bank_symbols)} Banken geladen")
        return loaded_count > 0
    
    def load_neural_network_predictions(self):
        """Lädt Neural Network Predictions (falls nicht ausgeschlossen)"""
        
        if self.exclude_neural_network:
            print(f"\nNeural Network Predictions werden AUSGESCHLOSSEN")
            print(f"   Grund: Schlechte R² Werte und unrealistische Predictions")
            return False
        
        print(f"\nLade Neural Network Predictions...")
        
        if not os.path.exists(self.nn_path):
            print(f"   Neural Network Ordner nicht gefunden: {self.nn_path}")
            return False
        
        # Trading Summary laden
        summary_file = os.path.join(self.nn_path, "neural_network_trading_summary.csv")
        if os.path.exists(summary_file):
            nn_summary = pd.read_csv(summary_file)
            print(f"   Neural Network Summary: {len(nn_summary)} Banken")
            
            # Zeige R² Werte zur Information
            if 'NN_Test_R2' in nn_summary.columns:
                avg_r2 = nn_summary['NN_Test_R2'].mean()
                print(f"   Durchschnittliches R²: {avg_r2:.2f}")
                if avg_r2 < 0:
                    print(f"   WARNUNG: Negative R² Werte - Neural Network Qualität fragwürdig")
        else:
            print(f"   Neural Network Summary nicht gefunden")
        
        # Einzelne Predictions laden
        loaded_count = 0
        for symbol in self.bank_symbols:
            pred_file = os.path.join(self.nn_path, f"{symbol}_7h_neural_network_prediction.csv")
            if os.path.exists(pred_file):
                try:
                    pred_data = pd.read_csv(pred_file)
                    pred_data['DateTime'] = pd.to_datetime(pred_data['DateTime'])
                    self.nn_data[symbol] = pred_data
                    loaded_count += 1
                except Exception as e:
                    print(f"   Fehler bei {symbol}: {str(e)}")
            else:
                print(f"   {symbol} Neural Network Prediction nicht gefunden")
        
        print(f"   Neural Network: {loaded_count}/{len(self.bank_symbols)} Banken geladen")
        return loaded_count > 0
    
    def create_ensemble_prediction(self, symbol):
        """Erstellt Ensemble-Prediction für eine Bank (OHNE Neural Network)"""
        
        available_models = []
        predictions_data = {}
        
        # Prophet Daten
        if symbol in self.prophet_data:
            prophet_pred = self.prophet_data[symbol]['Predicted_Return_7h'].values
            predictions_data['Prophet'] = prophet_pred
            available_models.append('Prophet')
        
        # ARIMA Daten
        if symbol in self.arima_data:
            arima_pred = self.arima_data[symbol]['Predicted_Return_7h'].values
            predictions_data['ARIMA'] = arima_pred
            available_models.append('ARIMA')
        
        # Neural Network AUSGESCHLOSSEN (oder optional)
        if not self.exclude_neural_network and symbol in self.nn_data:
            nn_pred = self.nn_data[symbol]['Predicted_Return_7h'].values
            predictions_data['Neural_Network'] = nn_pred
            available_models.append('Neural_Network')
            print(f"   WARNUNG: {symbol}: Neural Network trotz schlechter Performance verwendet")
        
        if len(available_models) < 2:
            if len(available_models) == 1:
                print(f"   INFO: {symbol}: Nur {available_models[0]} verfügbar - verwende Single Model")
                # Verwende das einzige verfügbare Modell
                single_model = available_models[0]
                predictions_data = {single_model: list(predictions_data.values())[0]}
            else:
                print(f"   FEHLER: {symbol}: Keine Modelle verfügbar")
                return None
        
        print(f"   ENSEMBLE: {symbol}: {available_models}")
        
        # Timestamps von erstem verfügbaren Model nehmen
        if symbol in self.prophet_data:
            timestamps = self.prophet_data[symbol]['DateTime'].values
        elif symbol in self.arima_data:
            timestamps = self.arima_data[symbol]['DateTime'].values
        else:
            timestamps = self.nn_data[symbol]['DateTime'].values
        
        # Ensemble-Methoden anwenden
        ensemble_results = {}
        
        if len(available_models) >= 2:
            # 1. SIMPLE AVERAGE
            all_predictions = np.array(list(predictions_data.values()))
            avg_prediction = np.mean(all_predictions, axis=0)
            ensemble_results['Average'] = avg_prediction
            
            # 2. WEIGHTED AVERAGE (Prophet und ARIMA gleich gewichtet)
            weights = self.calculate_model_weights(available_models)
            weighted_pred = np.average(all_predictions, axis=0, weights=weights)
            ensemble_results['Weighted'] = weighted_pred
            
            # 3. VOTING SYSTEM
            voting_pred = self.voting_ensemble(predictions_data)
            ensemble_results['Voting'] = voting_pred
            
            # Finale Ensemble-Prediction (Weighted als Standard)
            final_prediction = ensemble_results['Weighted']
        else:
            # Single Model Fall
            single_pred = list(predictions_data.values())[0]
            ensemble_results['Average'] = single_pred
            ensemble_results['Weighted'] = single_pred
            ensemble_results['Voting'] = single_pred
            final_prediction = single_pred
        
        # Confidence berechnen (Durchschnitt aller Modelle)
        confidence_values = []
        for model in available_models:
            if symbol in self.prophet_data and model == 'Prophet':
                confidence_values.append(self.prophet_data[symbol]['Confidence'].values)
            elif symbol in self.arima_data and model == 'ARIMA':
                confidence_values.append(self.arima_data[symbol]['Confidence'].values)
            elif symbol in self.nn_data and model == 'Neural_Network':
                confidence_values.append(self.nn_data[symbol]['Confidence'].values)
        
        if confidence_values:
            avg_confidence = np.mean(confidence_values, axis=0)
        else:
            avg_confidence = np.ones(7) * 0.01  # Fallback
        
        # Ergebnis formatieren
        result = pd.DataFrame({
            'DateTime': timestamps,
            'Predicted_Return_7h': final_prediction,
            'Ensemble_Average': ensemble_results['Average'],
            'Ensemble_Weighted': ensemble_results['Weighted'], 
            'Ensemble_Voting': ensemble_results['Voting'],
            'Lower_Bound': final_prediction - (avg_confidence / 2),
            'Upper_Bound': final_prediction + (avg_confidence / 2),
            'Confidence': avg_confidence,
            'Direction': (final_prediction > 0).astype(int),
            'Models_Used': [', '.join(available_models)] * 7,
            'NN_Excluded': [self.exclude_neural_network] * 7
        })
        
        # Model-spezifische Predictions hinzufügen
        for model_name, pred in predictions_data.items():
            result[f'{model_name}_Prediction'] = pred
        
        result['Return_Magnitude'] = abs(result['Predicted_Return_7h'])
        result['Confidence_Score'] = 1 / (result['Confidence'] + 0.001)
        result['Model_Type'] = 'Ensemble_Clean' if self.exclude_neural_network else 'Ensemble_Full'
        
        return result
    
    def calculate_model_weights(self, available_models):
        """Berechnet Gewichte für Modelle (OHNE Neural Network bias)"""
        
        # NEUE Gewichte: Prophet und ARIMA gleich, Neural Network stark reduziert
        weight_map = {
            'Prophet': 0.5,      # 50% wenn nur Prophet + ARIMA
            'ARIMA': 0.5,        # 50% wenn nur Prophet + ARIMA
            'Neural_Network': 0.0  # Nur 0% falls doch verwendet
        }
        
        weights = []
        for model in available_models:
            weights.append(weight_map.get(model, 0.33))
        
        # Normalisieren (wichtig wenn Neural Network ausgeschlossen)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print(f"      Gewichte: {dict(zip(available_models, weights))}")
        return weights
    
    def voting_ensemble(self, predictions_data):
        """Voting-basiertes Ensemble (Majority Vote für Direction)"""
        
        # Direction für jedes Model
        directions = []
        magnitudes = []
        
        for model_name, pred in predictions_data.items():
            directions.append((pred > 0).astype(int))
            magnitudes.append(np.abs(pred))
        
        # Majority Vote für Direction
        direction_votes = np.array(directions)
        majority_direction = (np.mean(direction_votes, axis=0) >= 0.5).astype(int)
        
        # Durchschnittliche Magnitude
        avg_magnitude = np.mean(magnitudes, axis=0)
        
        # Kombiniere Direction und Magnitude
        voting_result = np.where(majority_direction == 1, avg_magnitude, -avg_magnitude)
        
        return voting_result
    
    def create_ensemble_for_all_banks(self):
        """Erstellt Ensemble-Predictions für alle Banken"""
        
        model_type = "ohne_NN" if self.exclude_neural_network else "mit_NN"
        print(f"\nENSEMBLE-PREDICTIONS FÜR ALLE BANKEN ({model_type})")
        print("=" * 70)
        
        successful_ensembles = 0
        
        for symbol in self.bank_symbols:
            print(f"\nErstelle Ensemble für {symbol}...")
            
            ensemble_result = self.create_ensemble_prediction(symbol)
            if ensemble_result is not None:
                self.ensemble_predictions[symbol] = ensemble_result
                successful_ensembles += 1
                
                # Kurze Zusammenfassung
                total_return = ensemble_result['Predicted_Return_7h'].sum()
                bullish_hours = ensemble_result['Direction'].sum()
                models_used = ensemble_result['Models_Used'].iloc[0]
                
                print(f"   OK: {symbol}: Total Return {total_return:.4f}, Bullish {bullish_hours}/7")
                print(f"      Modelle: {models_used}")
            else:
                print(f"   FEHLER: {symbol}: Ensemble fehlgeschlagen")
        
        print(f"\nEnsemble abgeschlossen ({model_type}):")
        print(f"   Erfolgreich: {successful_ensembles}/{len(self.bank_symbols)} Banken")
        
        return successful_ensembles > 0
    
    def get_ensemble_trading_summary(self):
        """Erstellt Trading-Summary für Ensemble"""
        
        if not self.ensemble_predictions:
            print(f"Keine Ensemble Predictions verfügbar")
            return None
        
        model_type = "ohne_NN" if self.exclude_neural_network else "mit_NN"
        print(f"\nENSEMBLE TRADING SUMMARY ({model_type})")
        print("=" * 60)
        
        summary_data = []
        
        for symbol, pred in self.ensemble_predictions.items():
            total_return = pred['Predicted_Return_7h'].sum()
            avg_confidence = pred['Confidence_Score'].mean()
            bullish_hours = pred['Direction'].sum()
            max_single_return = pred['Predicted_Return_7h'].max()
            min_single_return = pred['Predicted_Return_7h'].min()
            models_used = pred['Models_Used'].iloc[0]
            
            summary_data.append({
                'Symbol': symbol,
                'Total_7h_Return': total_return,
                'Avg_Confidence': avg_confidence,
                'Bullish_Hours': f"{bullish_hours}/7",
                'Max_Hourly_Return': max_single_return,
                'Min_Hourly_Return': min_single_return,
                'Net_Direction': 'Bullish' if total_return > 0 else 'Bearish',
                'Strength': abs(total_return),
                'Models_Used': models_used,
                'NN_Excluded': self.exclude_neural_network,
                'Model_Type': f'Ensemble_Clean' if self.exclude_neural_network else 'Ensemble_Full'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total_7h_Return', ascending=False)
        
        print("TOP PERFORMERS (7h Total Return):")
        print(summary_df[['Symbol', 'Total_7h_Return', 'Net_Direction', 'Bullish_Hours', 'Models_Used']].head())
        
        print(f"\nENSEMBLE PORTFOLIO OVERVIEW:")
        total_portfolio_return = summary_df['Total_7h_Return'].sum()
        bullish_banks = len(summary_df[summary_df['Total_7h_Return'] > 0])
        
        print(f"   Ensemble Portfolio 7h Return: {total_portfolio_return:.4f}")
        print(f"   Ensemble Bullish Banks: {bullish_banks}/{len(summary_df)}")
        print(f"   Ensemble Average Confidence: {summary_df['Avg_Confidence'].mean():.2f}")
        print(f"   Neural Network ausgeschlossen: {self.exclude_neural_network}")
        
        return summary_df
    
    def save_ensemble_predictions(self, output_path="ensemble_predictions_clean"):
        """Speichert Ensemble Predictions"""
        
        if not self.ensemble_predictions:
            print(f"Keine Ensemble Predictions zum Speichern")
            return
        
        # Ausgabe-Ordner erstellen
        if self.exclude_neural_network:
            output_dir = "ensemble_predictions_without_NN"
        else:
            output_dir = "ensemble_predictions_with_NN"
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSpeichere Ensemble Predictions...")
        
        # Einzelne Bank-Predictions
        for symbol, pred in self.ensemble_predictions.items():
            csv_file = os.path.join(output_dir, f"{symbol}_7h_ensemble_prediction.csv")
            pred.to_csv(csv_file, index=False)
            print(f"   OK: {csv_file}")
        
        # Trading Summary
        summary = self.get_ensemble_trading_summary()
        if summary is not None:
            summary_file = os.path.join(output_dir, f"ensemble_trading_summary_without_NN.csv")
            summary.to_csv(summary_file, index=False)
            print(f"   OK: {summary_file}")
        
        print(f"   Alle Ensemble Predictions gespeichert in: {output_dir}")

def run_ensemble_pipeline():
    """Hauptfunktion für Ensemble Pipeline (OHNE Neural Network)"""
    
    print("ENSEMBLE BANK-PREDICTION PIPELINE (CLEAN)")
    print("=" * 55)
    
    # Ensemble Predictor initialisieren (Neural Network ausgeschlossen)
    ensemble = BankEnsemblePredictor(exclude_neural_network=True)
    
    # Prophet Predictions laden
    prophet_loaded = ensemble.load_prophet_predictions()
    
    # ARIMA Predictions laden
    arima_loaded = ensemble.load_arima_predictions()
    
    # Neural Network Predictions NICHT laden (ausgeschlossen)
    nn_loaded = False
    
    # Mindestens 1 Modell nötig (falls nur Prophet oder ARIMA)
    available_models = sum([prophet_loaded, arima_loaded])
    if available_models < 1:
        print(f"\nFEHLER: Mindestens 1 Modell benötigt für Ensemble!")
        print(f"   Verfügbar: {available_models}/2 Modelle")
        return
    
    if available_models == 1:
        print(f"\nWARNUNG: Nur 1 Modell verfügbar - Single Model statt Ensemble")
    else:
        print(f"\nOK: {available_models}/2 Modelle verfügbar für sauberes Ensemble")
    
    # Ensemble OHNE Neural Network erstellen
    print(f"\nErstelle sauberes Ensemble (Prophet + ARIMA)...")
    success = ensemble.create_ensemble_for_all_banks()
    
    if success:
        ensemble.save_ensemble_predictions()
        
        print(f"\nENSEMBLE PIPELINE ERFOLGREICH!")
        print(f"   Ensemble Predictions erstellt (OHNE Neural Network)")
        print(f"   Ergebnisse: ensemble_predictions_without_NN/")
        print(f"   SAUBERE TRADING-SIGNALE ohne extreme NN-Predictions!")
    else:
        print(f"\nENSEMBLE PIPELINE FEHLGESCHLAGEN!")

if __name__ == "__main__":
    run_ensemble_pipeline()