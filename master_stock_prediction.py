import os
import sys
import json
import subprocess
import webbrowser
import pandas as pd
from datetime import datetime
import time

class StockPredictionMaster:
    """
    Master Pipeline für Stock Prediction System
    Führt alle Schritte automatisch aus und startet Dashboard
    """
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.steps_completed = []
        self.step_results = {}
        
        # Verfügbare Scripte
        self.scripts = {
            'data_update': {
                'alpha_vantage': 'alpha_vantage_loader.py',
                'yahoo_finance': 'yahoo_finance_loader.py'
            },
            'feature_engineering': 'enhanced_feature_engenerring.py',
            'models': {
                'prophet': 'model_prophet.py',
                'arima': 'model_arima.py', 
                'neural_network': 'model_lstm.py'
            },
            'ensemble': 'ensemble_predictor.py'
        }
        
        self.output_dirs = [
            'csv_alpha',
            'features_enhanced', 
            'prophet_predictions',
            'arima_predictions',
            'neural_network_predictions',
            'ensemble_predictions_without_NN',
            'ensemble_predictions_with_NN'
        ]
        
        # Bank-Symbole für Validation
        self.bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
        
        print("MASTER STOCK PREDICTION PIPELINE")
        print("=" * 55)
        print("   Automatische Ausführung aller ML-Modelle")
        print("   Generierung interaktives Dashboard")
        print("   Echte Trading-Signale für Montag")
    
    def check_prerequisites(self):
        """Prüft ob alle notwendigen Dateien vorhanden sind"""
        
        print("\nPrüfe Voraussetzungen...")
        
        missing_files = []
        
        # Prüfe Hauptscripte
        main_scripts = [
            'enhanced_feature_engenerring.py',
            'ensemble_predictor.py'
        ]
        
        for script in main_scripts:
            if not os.path.exists(os.path.join(self.script_dir, script)):
                missing_files.append(script)
        
        # Prüfe Model-Scripte (optional - könnten anders heißen)
        model_scripts = ['model_prophet.py', 'model_arima.py', 'model_lstm.py']
        available_models = []
        
        for script in model_scripts:
            if os.path.exists(os.path.join(self.script_dir, script)):
                available_models.append(script)
        
        # Prüfe API Keys
        if not os.path.exists(os.path.join(self.script_dir, 'api_key.py')):
            missing_files.append('api_key.py (für Alpha Vantage)')
        
        # Ergebnisse
        if missing_files:
            print(f"   Fehlende Dateien: {', '.join(missing_files)}")
            return False
        
        print(f"   Alle Hauptdateien gefunden")
        print(f"   Verfügbare Modelle: {len(available_models)}")
        
        return True
    
    def check_enhanced_features_status(self):
        """Prüft ob Enhanced Features bereits vorhanden und aktuell sind"""
        
        print("\nPrüfe Enhanced Features Status...")
        
        features_path = os.path.join(self.script_dir, 'features_enhanced')
        
        # Prüfe ob Ordner existiert
        if not os.path.exists(features_path):
            print(f"   features_enhanced/ Ordner nicht gefunden")
            return False, "Ordner fehlt"
        
        # Prüfe ob Bank-Features vorhanden
        missing_banks = []
        existing_banks = []
        
        for bank in self.bank_symbols:
            bank_file = os.path.join(features_path, f"{bank}_enhanced.csv")
            if os.path.exists(bank_file):
                try:
                    # Prüfe ob Datei lesbar und hat erwartete Struktur
                    df = pd.read_csv(bank_file)
                    if len(df) > 100 and 'Target_7h' in df.columns:  # Mindestanforderungen
                        existing_banks.append(bank)
                    else:
                        missing_banks.append(f"{bank} (unvollständig)")
                except:
                    missing_banks.append(f"{bank} (korrupt)")
            else:
                missing_banks.append(bank)
        
        # Prüfe Summary-Datei
        summary_file = os.path.join(features_path, 'summary.csv')
        summary_exists = os.path.exists(summary_file)
        
        # Bewertung
        success_rate = len(existing_banks) / len(self.bank_symbols)
        
        print(f"   Enhanced Features Status:")
        print(f"      Verfügbare Banken: {len(existing_banks)}/{len(self.bank_symbols)} ({success_rate:.1%})")
        print(f"      Funktionierende Features: {', '.join(existing_banks[:5])}{'...' if len(existing_banks) > 5 else ''}")
        
        if missing_banks:
            print(f"      Fehlende/Problematische: {', '.join(missing_banks[:3])}{'...' if len(missing_banks) > 3 else ''}")
        
        print(f"      Summary vorhanden: {'Ja' if summary_exists else 'Nein'}")
        
        # Entscheidungslogik
        if success_rate >= 0.8:  # 80% der Banken verfügbar
            print(f"   Enhanced Features sind ausreichend vorhanden ({success_rate:.1%})")
            print(f"   Feature-Engineering wird ÜBERSPRUNGEN")
            return True, f"{len(existing_banks)} Banken bereit"
        else:
            print(f"   Enhanced Features unvollständig ({success_rate:.1%})")
            print(f"   Feature-Engineering wird AUSGEFÜHRT")
            return False, f"Nur {len(existing_banks)} Banken verfügbar"
    
    def run_step(self, step_name, script_path, description):
        """Führt einen Pipeline-Schritt aus"""
        
        print(f"\n{step_name}: {description}")
        print(f"   Script: {script_path}")
        
        if not os.path.exists(script_path):
            print(f"   Script nicht gefunden - überspringe")
            return False
        
        try:
            # Script ausführen
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, cwd=self.script_dir)
            
            if result.returncode == 0:
                print(f"   {step_name} erfolgreich")
                self.steps_completed.append(step_name)
                self.step_results[step_name] = {
                    'success': True,
                    'output': result.stdout[-500:] if result.stdout else ""  # Letzte 500 Zeichen
                }
                return True
            else:
                print(f"   {step_name} fehlgeschlagen")
                print(f"   Return Code: {result.returncode}")
                print(f"   Fehler: {result.stderr[:200]}...")
                self.step_results[step_name] = {
                    'success': False,
                    'error': result.stderr,
                    'return_code': result.returncode
                }
                return False
                
        except Exception as e:
            print(f"   {step_name} Ausnahme: {str(e)}")
            self.step_results[step_name] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def update_data(self):
        """Aktualisiert Rohdaten von Alpha Vantage und Yahoo Finance"""
        
        print(f"\nSCHRITT 1: DATEN-UPDATE")
        print("=" * 40)
        
        # Alpha Vantage (Bank-Daten)
        alpha_script = self.scripts['data_update']['alpha_vantage']
        if os.path.exists(alpha_script):
            self.run_step('Alpha_Vantage', alpha_script, 'Bank-Daten aktualisieren')
        
        # Yahoo Finance (Makro-Daten)  
        yahoo_script = self.scripts['data_update']['yahoo_finance']
        if os.path.exists(yahoo_script):
            self.run_step('Yahoo_Finance', yahoo_script, 'Makro-Daten aktualisieren')
        
        # Prüfe ob csv_alpha Daten vorhanden
        csv_alpha_path = os.path.join(self.script_dir, 'csv_alpha')
        if os.path.exists(csv_alpha_path):
            csv_files = [f for f in os.listdir(csv_alpha_path) if f.endswith('.csv')]
            print(f"   csv_alpha: {len(csv_files)} CSV-Dateien verfügbar")
            return len(csv_files) >= 10  # Mindestens 10 Dateien für Pipeline
        
        return False
    
    def run_feature_engineering(self):
        """Führt Feature-Engineering aus (SMART mit Existenz-Check)"""
        
        print(f"\nSCHRITT 2: FEATURE-ENGINEERING")
        print("=" * 40)
        
        # NEUE INTELLIGENTE PRÜFUNG
        features_ready, status_msg = self.check_enhanced_features_status()
        
        if features_ready:
            print(f"   Enhanced Features bereits vorhanden: {status_msg}")
            print(f"   Schritt wird ÜBERSPRUNGEN - spare Zeit!")
            
            # Markiere als erfolgreich (ohne auszuführen)
            self.steps_completed.append('Feature_Engineering_Skipped')
            self.step_results['Feature_Engineering'] = {
                'success': True,
                'skipped': True,
                'reason': f"Features bereits vorhanden: {status_msg}",
                'output': f"Enhanced Features für {status_msg} bereits verfügbar"
            }
            return True
        
        else:
            print(f"   Enhanced Features unvollständig: {status_msg}")
            print(f"   Führe Feature-Engineering aus...")
            
            script_path = os.path.join(self.script_dir, self.scripts['feature_engineering'])
            return self.run_step('Feature_Engineering', script_path, 'Enhanced Features erstellen')
    
    def run_ml_models(self):
        """Führt alle ML-Modelle aus"""
        
        print(f"\nSCHRITT 3: ML-MODELLE TRAINIEREN")
        print("=" * 40)
        
        model_results = {}
        
        # Prophet Model
        prophet_script = os.path.join(self.script_dir, 'model_prophet.py')
        if os.path.exists(prophet_script):
            model_results['Prophet'] = self.run_step('Prophet_Model', prophet_script, 'Prophet Zeitreihen-Modell')
        else:
            # Fallback: Suche nach anderen Prophet-Scripten
            for filename in os.listdir(self.script_dir):
                if 'prophet' in filename.lower() and filename.endswith('.py'):
                    prophet_script = os.path.join(self.script_dir, filename)
                    model_results['Prophet'] = self.run_step('Prophet_Model', prophet_script, 'Prophet Zeitreihen-Modell')
                    break
        
        # ARIMA Model
        arima_script = os.path.join(self.script_dir, 'model_arima.py')
        if not os.path.exists(arima_script):
            # Fallback: arima_model_fixed.py
            arima_script = os.path.join(self.script_dir, 'arima_model_fixed.py')
        
        if os.path.exists(arima_script):
            model_results['ARIMA'] = self.run_step('ARIMA_Model', arima_script, 'ARIMA Zeitreihen-Modell')
        
        # Neural Network Model
        nn_script = os.path.join(self.script_dir, 'model_lstm.py')
        if not os.path.exists(nn_script):
            # Fallback: neural_network_predictor.py
            nn_script = os.path.join(self.script_dir, 'neural_network_predictor.py')
        
        if os.path.exists(nn_script):
            model_results['Neural_Network'] = self.run_step('Neural_Network_Model', nn_script, 'Neural Network Modell')
        
        successful_models = sum(model_results.values())
        print(f"\n   ML-Modelle abgeschlossen: {successful_models}/{len(model_results)} erfolgreich")
        
        return successful_models >= 2  # Mindestens 2 Modelle für Ensemble
    
    def run_ensemble(self):
        """Führt Ensemble-Model aus"""
        
        print(f"\nSCHRITT 4: ENSEMBLE ERSTELLEN")
        print("=" * 40)
        
        script_path = os.path.join(self.script_dir, self.scripts['ensemble'])
        return self.run_step('Ensemble_Model', script_path, 'Ensemble-Predictions erstellen')
    
    def collect_detailed_predictions(self):
        """Sammelt ALLE 7h-Predictions für das Dashboard"""
        
        print(f"\n[DATA] Sammle detaillierte Predictions für Dashboard...")
        
        detailed_data = {
            'models': ['Prophet', 'ARIMA', 'Neural Network', 'Ensemble'],
            'banks': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF'],
            'predictions': {}
        }
        
        # Prophet Predictions laden
        prophet_path = os.path.join(self.script_dir, 'prophet_predictions')
        if os.path.exists(prophet_path):
            detailed_data['predictions']['Prophet'] = self.load_model_predictions(prophet_path, '_7h_prediction.csv')
            print(f"   [OK] Prophet: {len(detailed_data['predictions']['Prophet'])} Banken geladen")
        else:
            detailed_data['predictions']['Prophet'] = {}
            print(f"   [WARN] Prophet Predictions nicht gefunden")
        
        # ARIMA Predictions laden
        arima_path = os.path.join(self.script_dir, 'arima_predictions')
        if os.path.exists(arima_path):
            detailed_data['predictions']['ARIMA'] = self.load_model_predictions(arima_path, '_7h_arima_prediction.csv')
            print(f"   [OK] ARIMA: {len(detailed_data['predictions']['ARIMA'])} Banken geladen")
        else:
            detailed_data['predictions']['ARIMA'] = {}
            print(f"   [WARN] ARIMA Predictions nicht gefunden")
        
        # Neural Network Predictions laden
        nn_path = os.path.join(self.script_dir, 'neural_network_predictions')
        if os.path.exists(nn_path):
            detailed_data['predictions']['Neural Network'] = self.load_model_predictions(nn_path, '_7h_neural_network_prediction.csv')
            print(f"   [OK] Neural Network: {len(detailed_data['predictions']['Neural Network'])} Banken geladen")
        else:
            detailed_data['predictions']['Neural Network'] = {}
            print(f"   [WARN] Neural Network Predictions nicht gefunden")
        
        # Ensemble Predictions laden
        ensemble_path = os.path.join(self.script_dir, 'ensemble_predictions_without_NN')
        if os.path.exists(ensemble_path):
            detailed_data['predictions']['Ensemble'] = self.load_model_predictions(ensemble_path, '_7h_ensemble_prediction.csv')
            print(f"   [OK] Ensemble: {len(detailed_data['predictions']['Ensemble'])} Banken geladen")
        else:
            detailed_data['predictions']['Ensemble'] = {}
            print(f"   [WARN] Ensemble Predictions nicht gefunden")
        
        # Detailed Data als JSON speichern
        detailed_file = os.path.join(self.script_dir, 'detailed_predictions.json')
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        print(f"   [SAVE] Detaillierte Predictions gespeichert: detailed_predictions.json")
        
        # Statistiken
        total_predictions = sum(len(model_data) for model_data in detailed_data['predictions'].values())
        print(f"   [STATS] Gesamt: {total_predictions} Bank-Model-Kombinationen")
        
        return detailed_data
    
    def load_model_predictions(self, model_path, file_suffix):
        """Lädt Predictions für ein spezifisches Modell"""
        
        model_predictions = {}
        bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
        
        for bank in bank_symbols:
            prediction_file = os.path.join(model_path, f"{bank}{file_suffix}")
            
            if os.path.exists(prediction_file):
                try:
                    df = pd.read_csv(prediction_file)
                    
                    # Extrahiere 7h-Predictions
                    if 'Predicted_Return_7h' in df.columns and len(df) >= 7:
                        hourly_predictions = df['Predicted_Return_7h'].head(7).tolist()
                        total_return = sum(hourly_predictions)
                        bullish_hours = sum(1 for x in hourly_predictions if x > 0)
                        
                        # Confidence (falls verfügbar)
                        if 'Confidence' in df.columns:
                            avg_confidence = df['Confidence'].head(7).mean()
                        elif 'Confidence_Score' in df.columns:
                            avg_confidence = df['Confidence_Score'].head(7).mean()
                        else:
                            avg_confidence = 75.0  # Fallback
                        
                        model_predictions[bank] = {
                            'hourlyPredictions': hourly_predictions,
                            'totalReturn': total_return,
                            'bullishHours': bullish_hours,
                            'direction': 'Bullish' if total_return > 0 else 'Bearish',
                            'confidence': avg_confidence
                        }
                        
                except Exception as e:
                    print(f"      [ERROR] {bank}: {str(e)[:50]}...")
                    continue
            else:
                print(f"      [MISS] {bank}: Datei nicht gefunden")
        
        return model_predictions

    def collect_results(self):
        """Sammelt alle Prediction-Ergebnisse für Dashboard - ERWEITERT"""
        
        print(f"\n[COLLECT] SCHRITT 5: ERGEBNISSE SAMMELN")
        print("=" * 40)
        
        # Sammle Summary-Daten (wie bisher)
        results = {
            'prophet': {'available': False},
            'arima': {'available': False},
            'neural_network': {'available': False},
            'ensemble': {'available': False}
        }
        
        # Prophet Ergebnisse
        prophet_summary = os.path.join(self.script_dir, 'prophet_predictions', 'trading_summary.csv')
        if os.path.exists(prophet_summary):
            try:
                df = pd.read_csv(prophet_summary)
                results['prophet'] = {
                    'available': True,
                    'portfolio_return': df['Total_7h_Return'].sum(),
                    'bullish_banks': len(df[df['Total_7h_Return'] > 0]),
                    'top_performers': df.head().to_dict('records')
                }
                print(f"   [OK] Prophet: {results['prophet']['portfolio_return']:.4f}% Portfolio Return")
            except Exception as e:
                print(f"   [WARN] Prophet Daten-Fehler: {str(e)}")
        
        # ARIMA Ergebnisse
        arima_summary = os.path.join(self.script_dir, 'arima_predictions', 'arima_trading_summary.csv')
        if os.path.exists(arima_summary):
            try:
                df = pd.read_csv(arima_summary)
                results['arima'] = {
                    'available': True,
                    'portfolio_return': df['Total_7h_Return'].sum(),
                    'bullish_banks': len(df[df['Total_7h_Return'] > 0]),
                    'top_performers': df.head().to_dict('records')
                }
                print(f"   [OK] ARIMA: {results['arima']['portfolio_return']:.4f}% Portfolio Return")
            except Exception as e:
                print(f"   [WARN] ARIMA Daten-Fehler: {str(e)}")
        
        # Neural Network Ergebnisse
        nn_summary = os.path.join(self.script_dir, 'neural_network_predictions', 'neural_network_trading_summary.csv')
        if os.path.exists(nn_summary):
            try:
                df = pd.read_csv(nn_summary)
                results['neural_network'] = {
                    'available': True,
                    'portfolio_return': df['Total_7h_Return'].sum(),
                    'bullish_banks': len(df[df['Total_7h_Return'] > 0]),
                    'top_performers': df.head().to_dict('records')
                }
                print(f"   [OK] Neural Network: {results['neural_network']['portfolio_return']:.4f}% Portfolio Return")
            except Exception as e:
                print(f"   [WARN] Neural Network Daten-Fehler: {str(e)}")
        
        # Ensemble Ergebnisse
        ensemble_summary = os.path.join(self.script_dir, 'ensemble_predictions_without_NN', 'ensemble_trading_summary_without_NN.csv')
        if os.path.exists(ensemble_summary):
            try:
                df = pd.read_csv(ensemble_summary)
                results['ensemble'] = {
                    'available': True,
                    'portfolio_return': df['Total_7h_Return'].sum(),
                    'bullish_banks': len(df[df['Total_7h_Return'] > 0]),
                    'top_performers': df.head().to_dict('records')
                }
                print(f"   [OK] Ensemble: {results['ensemble']['portfolio_return']:.4f}% Portfolio Return")
            except Exception as e:
                print(f"   [WARN] Ensemble Daten-Fehler: {str(e)}")
        
        # Ergebnisse für Dashboard speichern (Summary)
        results_file = os.path.join(self.script_dir, 'dashboard_data.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   [SAVE] Dashboard-Summary gespeichert: dashboard_data.json")
        
        # NEU: Sammle detaillierte Predictions
        detailed_data = self.collect_detailed_predictions()
        
        available_models = sum(1 for model in results.values() if model['available'])
        return available_models >= 2
    
    
    def open_dashboard(self, dashboard_file):
        """Öffnet Dashboard im Browser"""
        
        print(f"\nSCHRITT 7: DASHBOARD STARTEN")
        print("=" * 40)
        
        try:
            webbrowser.open(f'file://{os.path.abspath(dashboard_file)}')
            print(f"   Dashboard geöffnet im Browser")
            print(f"   URL: file://{os.path.abspath(dashboard_file)}")
            return True
        except Exception as e:
            print(f"   Browser-Start fehlgeschlagen: {str(e)}")
            print(f"   Öffne manuell: {dashboard_file}")
            return False
    
    def create_dashboard_with_data(self):
        """Erstellt HTML-Dashboard mit echten Daten - FIXED"""
        
        print(f"\n[DASHBOARD] SCHRITT 6: DASHBOARD ERSTELLEN")
        print("=" * 40)
        
        # Lade echte Daten
        results_file = os.path.join(self.script_dir, 'dashboard_data.json')
        detailed_file = os.path.join(self.script_dir, 'detailed_predictions.json')
        
        # Erstelle eingebettete JSON-Daten für das Dashboard
        dashboard_data = {}
        detailed_data = {}
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                dashboard_data = json.load(f)
        
        if os.path.exists(detailed_file):
            with open(detailed_file, 'r') as f:
                detailed_data = json.load(f)
        
        # FIXED DASHBOARD HTML mit eingebetteten Daten
        dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Stock Prediction Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }}

        .dashboard-container {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #fff, #ffd700);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .controls {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .control-label {{
            font-weight: 500;
            opacity: 0.9;
        }}

        select, .btn {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 8px 15px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        select {{
            min-width: 150px;
        }}

        select:hover, .btn:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-1px);
        }}

        .prediction-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}

        .model-column {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .model-header {{
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        }}

        .bank-tile {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            border: 2px solid transparent;
        }}

        .bank-tile:hover {{
            transform: scale(1.02);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
        }}

        .bank-symbol {{
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }}

        .bank-return {{
            font-size: 1rem;
            font-weight: bold;
            margin-bottom: 8px;
        }}

        .sparkline-container {{
            height: 30px;
            margin-bottom: 5px;
        }}

        .sparkline {{
            width: 100%;
            height: 100%;
        }}

        .direction-indicator {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}

        .very-bearish {{ background: linear-gradient(145deg, #7f1d1d, #991b1b); }}
        .bearish {{ background: linear-gradient(145deg, #dc2626, #ef4444); }}
        .neutral {{ background: linear-gradient(145deg, #6b7280, #9ca3af); }}
        .bullish {{ background: linear-gradient(145deg, #059669, #10b981); }}
        .very-bullish {{ background: linear-gradient(145deg, #064e3b, #047857); }}

        .very-bearish .bank-return {{ color: #fca5a5; }}
        .bearish .bank-return {{ color: #fca5a5; }}
        .neutral .bank-return {{ color: #d1d5db; }}
        .bullish .bank-return {{ color: #86efac; }}
        .very-bullish .bank-return {{ color: #86efac; }}

        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 15px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
            display: none;
        }}

        .tooltip-header {{
            font-weight: bold;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 5px;
        }}

        .tooltip-chart {{
            height: 150px;
            margin-bottom: 10px;
        }}

        .tooltip-stats {{
            font-size: 0.9rem;
            line-height: 1.4;
        }}

        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 10px;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}

        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .stat-value {{
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-label {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}

        @media (max-width: 1200px) {{
            .prediction-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 768px) {{
            .prediction-grid {{
                grid-template-columns: 1fr;
            }}
            
            .controls {{
                flex-direction: column;
                align-items: center;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header">
            <h1>Bank Stock Prediction Dashboard</h1>
            <p>Prophet • ARIMA • Neural Network • Ensemble Analysis</p>
            <p style="font-size: 0.9rem; margin-top: 10px;">7-Hour Trading Predictions with Interactive Details</p>
        </div>

        <!-- Controls -->
        <div class="controls">
            <div class="control-group">
                <span class="control-label">Sort by:</span>
                <select id="sortSelect" onchange="sortBanks()">
                    <option value="alphabetical">Alphabetical</option>
                    <option value="performance">Best Performance</option>
                    <option value="consistency">Model Consistency</option>
                </select>
            </div>
            <div class="control-group">
                <span class="control-label">View:</span>
                <button class="btn" onclick="toggleCompactView()">Toggle Compact</button>
            </div>
        </div>

        <!-- Legend -->
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color very-bullish"></div>
                <span>Very Bullish (>+0.2%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color bullish"></div>
                <span>Bullish (0% to +0.2%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color neutral"></div>
                <span>Neutral (0%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color bearish"></div>
                <span>Bearish (0% to -0.2%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color very-bearish"></div>
                <span>Very Bearish (<-0.2%)</span>
            </div>
        </div>

        <!-- Summary Stats -->
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value" id="bestModel">ARIMA</div>
                <div class="stat-label">Best Performing Model</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgReturn">+0.12%</div>
                <div class="stat-label">Average 7h Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="bullishCount">32/48</div>
                <div class="stat-label">Bullish Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="topBank">USB</div>
                <div class="stat-label">Most Consistent Performer</div>
            </div>
        </div>

        <!-- Main Prediction Grid -->
        <div class="prediction-grid" id="predictionGrid">
            <!-- Prophet Column -->
            <div class="model-column">
                <div class="model-header">Prophet</div>
                <div id="prophetBanks"></div>
            </div>

            <!-- ARIMA Column -->
            <div class="model-column">
                <div class="model-header">ARIMA</div>
                <div id="arimaBanks"></div>
            </div>

            <!-- Neural Network Column -->
            <div class="model-column">
                <div class="model-header">Neural Network</div>
                <div id="neuralBanks"></div>
            </div>

            <!-- Ensemble Column -->
            <div class="model-column">
                <div class="model-header">Ensemble</div>
                <div id="ensembleBanks"></div>
            </div>
        </div>
    </div>

    <!-- Tooltip -->
    <div class="tooltip" id="tooltip">
        <div class="tooltip-header" id="tooltipHeader"></div>
        <div class="tooltip-chart">
            <canvas id="tooltipChart"></canvas>
        </div>
        <div class="tooltip-stats" id="tooltipStats"></div>
    </div>

    <script>
        // EINGEBETTETE DATEN - Keine CORS-Probleme mehr
        const embeddedDashboardData = {json.dumps(dashboard_data, indent=8)};
        const embeddedDetailedData = {json.dumps(detailed_data, indent=8)};
        
        let predictionData = {{}};
        let realData = embeddedDashboardData;
        
        const bankSymbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF'];
        const models = ['Prophet', 'ARIMA', 'Neural Network', 'Ensemble'];
        
        let currentSort = 'alphabetical';
        let compactView = false;
        let tooltipChart = null;

        function initializePredictionData() {{
            // Verwende eingebettete Daten oder generiere Mock-Daten
            if (embeddedDetailedData && embeddedDetailedData.predictions) {{
                predictionData = embeddedDetailedData.predictions;
                console.log('Echte Daten geladen:', Object.keys(predictionData));
            }} else {{
                console.log('Keine echten Daten, generiere Mock-Daten');
                generateMockData();
            }}
        }}

        function generateMockData() {{
            models.forEach(model => {{
                predictionData[model] = {{}};
                bankSymbols.forEach(bank => {{
                    const hourlyPredictions = [];
                    let baseReturn = (Math.random() - 0.5) * 0.8;
                    
                    for (let hour = 0; hour < 7; hour++) {{
                        baseReturn += (Math.random() - 0.5) * 0.1;
                        hourlyPredictions.push(baseReturn / 100);
                    }}
                    
                    const totalReturn = hourlyPredictions.reduce((sum, val) => sum + val, 0);
                    const bullishHours = hourlyPredictions.filter(val => val > 0).length;
                    
                    predictionData[model][bank] = {{
                        hourlyPredictions,
                        totalReturn,
                        bullishHours,
                        direction: totalReturn > 0 ? 'Bullish' : 'Bearish',
                        confidence: Math.random() * 40 + 60
                    }};
                }});
            }});
        }}

        function getPerformanceClass(totalReturn) {{
            if (totalReturn > 0.002) return 'very-bullish';
            if (totalReturn > 0) return 'bullish';
            if (totalReturn === 0) return 'neutral';
            if (totalReturn > -0.002) return 'bearish';
            return 'very-bearish';
        }}

        function createSparkline(container, data) {{
            const canvas = document.createElement('canvas');
            canvas.className = 'sparkline';
            container.appendChild(canvas);
            
            const ctx = canvas.getContext('2d');
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
            
            if (data.length === 0) return;
            
            const max = Math.max(...data);
            const min = Math.min(...data);
            const range = max - min || 0.001;
            
            const width = canvas.width;
            const height = canvas.height;
            const stepX = width / (data.length - 1);
            
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            data.forEach((value, index) => {{
                const x = index * stepX;
                const y = height - ((value - min) / range) * height;
                
                if (index === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            
            ctx.stroke();
        }}

        function createBankTile(model, bank, data) {{
            const tile = document.createElement('div');
            tile.className = `bank-tile ${{getPerformanceClass(data.totalReturn)}}`;
            tile.setAttribute('data-model', model);
            tile.setAttribute('data-bank', bank);
            
            const returnPercent = (data.totalReturn * 100).toFixed(3);
            
            tile.innerHTML = `
                <div class="bank-symbol">${{bank}}</div>
                <div class="bank-return">${{data.totalReturn >= 0 ? '+' : ''}}${{returnPercent}}%</div>
                <div class="sparkline-container"></div>
                <div class="direction-indicator">${{data.bullishHours}}/7 bullish hours</div>
            `;
            
            // Add sparkline
            const sparklineContainer = tile.querySelector('.sparkline-container');
            if (data.hourlyPredictions && data.hourlyPredictions.length > 0) {{
                setTimeout(() => createSparkline(sparklineContainer, data.hourlyPredictions), 100);
            }}
            
            // Add hover events
            tile.addEventListener('mouseenter', (e) => showTooltip(e, model, bank, data));
            tile.addEventListener('mouseleave', hideTooltip);
            tile.addEventListener('mousemove', (e) => moveTooltip(e));
            
            return tile;
        }}

        function showTooltip(event, model, bank, data) {{
            const tooltip = document.getElementById('tooltip');
            const header = document.getElementById('tooltipHeader');
            const stats = document.getElementById('tooltipStats');
            
            header.textContent = `${{bank}} - ${{model}}`;
            
            const returnPercent = (data.totalReturn * 100).toFixed(3);
            const hourlyPercents = data.hourlyPredictions ? 
                data.hourlyPredictions.map(val => (val * 100).toFixed(3)) : 
                ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'];
            
            stats.innerHTML = `
                <strong>7-Hour Forecast:</strong><br>
                Total Return: ${{data.totalReturn >= 0 ? '+' : ''}}${{returnPercent}}%<br>
                Bullish Hours: ${{data.bullishHours}}/7<br>
                Direction: ${{data.direction}}<br>
                Confidence: ${{data.confidence.toFixed(1)}}%<br>
                <br>
                <strong>Hourly Breakdown:</strong><br>
                ${{hourlyPercents.map((val, i) => 
                    `Hour ${{i+1}}: ${{parseFloat(val) >= 0 ? '+' : ''}}${{val}}%`
                ).join('<br>')}}
            `;
            
            if (data.hourlyPredictions && data.hourlyPredictions.length > 0) {{
                createTooltipChart(data.hourlyPredictions.map(val => val * 100));
            }}
            
            tooltip.style.display = 'block';
            moveTooltip(event);
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
            if (tooltipChart) {{
                tooltipChart.destroy();
                tooltipChart = null;
            }}
        }}

        function moveTooltip(event) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}

        function createTooltipChart(data) {{
            if (tooltipChart) {{
                tooltipChart.destroy();
            }}
            
            const ctx = document.getElementById('tooltipChart').getContext('2d');
            
            tooltipChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: ['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6', 'Hour 7'],
                    datasets: [{{
                        label: 'Predicted Return (%)',
                        data: data,
                        borderColor: '#60a5fa',
                        backgroundColor: 'rgba(96, 165, 250, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                            ticks: {{ color: 'white', font: {{ size: 10 }} }}
                        }},
                        x: {{
                            grid: {{ color: 'rgba(255, 255, 255, 0.1)' }},
                            ticks: {{ color: 'white', font: {{ size: 10 }} }}
                        }}
                    }}
                }}
            }});
        }}

        function renderPredictions() {{
            const containers = {{
                'Prophet': document.getElementById('prophetBanks'),
                'ARIMA': document.getElementById('arimaBanks'),
                'Neural Network': document.getElementById('neuralBanks'),
                'Ensemble': document.getElementById('ensembleBanks')
            }};
            
            Object.values(containers).forEach(container => {{
                if (container) container.innerHTML = '';
            }});
            
            const sortedBanks = getSortedBanks();
            
            models.forEach(model => {{
                if (predictionData[model] && containers[model]) {{
                    sortedBanks.forEach(bank => {{
                        if (predictionData[model][bank]) {{
                            const data = predictionData[model][bank];
                            const tile = createBankTile(model, bank, data);
                            containers[model].appendChild(tile);
                        }} else {{
                            const placeholder = document.createElement('div');
                            placeholder.className = 'bank-tile neutral';
                            placeholder.innerHTML = `
                                <div class="bank-symbol">${{bank}}</div>
                                <div class="bank-return">No Data</div>
                                <div class="direction-indicator">N/A</div>
                            `;
                            containers[model].appendChild(placeholder);
                        }}
                    }});
                }}
            }});
        }}

        function getSortedBanks() {{
            let banks = [...bankSymbols];
            
            switch (currentSort) {{
                case 'performance':
                    banks.sort((a, b) => {{
                        const avgA = models.reduce((sum, model) => {{
                            const data = predictionData[model] && predictionData[model][a];
                            return sum + (data ? data.totalReturn : 0);
                        }}, 0) / models.length;
                        const avgB = models.reduce((sum, model) => {{
                            const data = predictionData[model] && predictionData[model][b];
                            return sum + (data ? data.totalReturn : 0);
                        }}, 0) / models.length;
                        return avgB - avgA;
                    }});
                    break;
                    
                case 'consistency':
                    banks.sort((a, b) => {{
                        const bullishA = models.filter(model => {{
                            const data = predictionData[model] && predictionData[model][a];
                            return data && data.totalReturn > 0;
                        }}).length;
                        const bullishB = models.filter(model => {{
                            const data = predictionData[model] && predictionData[model][b];
                            return data && data.totalReturn > 0;
                        }}).length;
                        return bullishB - bullishA;
                    }});
                    break;
                    
                default:
                    banks.sort();
            }}
            
            return banks;
        }}

        function sortBanks() {{
            currentSort = document.getElementById('sortSelect').value;
            renderPredictions();
        }}

        function toggleCompactView() {{
            compactView = !compactView;
            document.body.classList.toggle('compact-view', compactView);
        }}

        function updateSummaryStats() {{
            if (!realData) return;
            
            const modelReturns = {{}};
            ['prophet', 'arima', 'neural_network', 'ensemble'].forEach(model => {{
                if (realData[model] && realData[model].available) {{
                    modelReturns[model] = realData[model].portfolio_return;
                }}
            }});
            
            if (Object.keys(modelReturns).length > 0) {{
                const bestModel = Object.keys(modelReturns).reduce((a, b) => 
                    modelReturns[a] > modelReturns[b] ? a : b);
                
                document.getElementById('bestModel').textContent = bestModel.toUpperCase();
                
                const avgReturn = Object.values(modelReturns).reduce((a, b) => a + b, 0) / Object.keys(modelReturns).length;
                document.getElementById('avgReturn').textContent = 
                    (avgReturn >= 0 ? '+' : '') + avgReturn.toFixed(3) + '%';
                
                const totalBullish = Object.values(realData).reduce((sum, model) => {{
                    return sum + (model.available ? model.bullish_banks : 0);
                }}, 0);
                document.getElementById('bullishCount').textContent = `${{totalBullish}}/48`;
            }}
        }}

        function initializeDashboard() {{
            initializePredictionData();
            renderPredictions();
            updateSummaryStats();
        }}

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializeDashboard);
    </script>
</body>
</html>"""
        
        # Dashboard speichern
        dashboard_file = os.path.join(self.script_dir, 'live_dashboard.html')
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"   [OK] Enhanced Dashboard erstellt: live_dashboard.html")
        return dashboard_file
    
    def run_full_pipeline(self, skip_data_update=False, auto_mode=False):
        """Führt komplette Pipeline aus"""
        
        print(f"\nSTARTE VOLLSTÄNDIGE PREDICTION PIPELINE")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Schritt 0: Voraussetzungen prüfen
        if not self.check_prerequisites():
            print(f"\nPipeline abgebrochen - Voraussetzungen nicht erfüllt")
            return False
        
        # Schritt 1: Daten-Update (optional überspringen)
        if not skip_data_update:
            if not self.update_data():
                print(f"\nDaten-Update problematisch - verwende bestehende Daten")
        
        # Schritt 2: Feature-Engineering (SMART mit Check)
        if not self.run_feature_engineering():
            print(f"\nPipeline abgebrochen - Feature-Engineering fehlgeschlagen")
            return False
        
        # Schritt 3: ML-Modelle
        if not self.run_ml_models():
            print(f"\nPipeline abgebrochen - ML-Modelle fehlgeschlagen")
            return False
        
        # Schritt 4: Ensemble
        if not self.run_ensemble():
            print(f"\nEnsemble fehlgeschlagen - verwende Einzelmodelle")
        
        # Schritt 5: Ergebnisse sammeln
        if not self.collect_results():
            print(f"\nPipeline abgebrochen - Ergebnisse unvollständig")
            return False
        
        # Schritt 6: Dashboard erstellen
        dashboard_file = self.create_dashboard_with_data()
        
        # Schritt 7: Dashboard öffnen
        self.open_dashboard(dashboard_file)
        
        # Abschluss
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nPIPELINE ERFOLGREICH ABGESCHLOSSEN!")
        print("=" * 50)
        print(f"   Dauer: {duration:.1f} Sekunden")
        print(f"   Schritte erfolgreich: {len(self.steps_completed)}")
        print(f"   Dashboard: live_dashboard.html")
        print(f"   Trading-Signale bereit für Montag!")
        
        # NEUE: Zeige Pipeline-Zusammenfassung
        self.show_pipeline_summary()
        
        return True
    
    def show_pipeline_summary(self):
        """Zeigt detaillierte Pipeline-Zusammenfassung"""
        
        print(f"\nPIPELINE-ZUSAMMENFASSUNG:")
        print("=" * 40)
        
        for step_name, result in self.step_results.items():
            if result['success']:
                status = "Erfolgreich"
                if result.get('skipped'):
                    status = "Übersprungen (bereits vorhanden)"
            else:
                status = f"Fehlgeschlagen (Code: {result.get('return_code', 'N/A')})"
            
            print(f"   {step_name}: {status}")
            
            if result.get('reason'):
                print(f"      Grund: {result['reason']}")


def main():
    """Hauptfunktion für Master Pipeline"""
    
    # Master Pipeline initialisieren
    master = StockPredictionMaster()
    
    # Benutzer-Optionen
    print(f"\nPIPELINE-OPTIONEN:")
    print(f"1. Vollständige Pipeline (Daten-Update + Modelle + Dashboard)")
    print(f"2. Schnelle Pipeline (nur Modelle + Dashboard)")
    print(f"3. Nur Dashboard (mit bestehenden Daten)")
    print(f"4. Abbrechen")
    
    try:
        choice = input(f"\nWähle Option (1-4): ").strip()
    except KeyboardInterrupt:
        print(f"\nPipeline abgebrochen")
        return
    
    if choice == "1":
        # Vollständige Pipeline
        print(f"\nStarte vollständige Pipeline...")
        master.run_full_pipeline(skip_data_update=False)
    
    elif choice == "2":
        # Schnelle Pipeline (ohne Daten-Update)
        print(f"\nStarte schnelle Pipeline...")
        master.run_full_pipeline(skip_data_update=True)
    
    elif choice == "3":
        # Nur Dashboard
        print(f"\nErstelle nur Dashboard...")
        if master.collect_results():
            dashboard_file = master.create_dashboard_with_data()
            master.open_dashboard(dashboard_file)
        else:
            print(f"Keine Prediction-Daten gefunden!")
    
    else:
        print(f"Pipeline abgebrochen")


if __name__ == "__main__":
    main()