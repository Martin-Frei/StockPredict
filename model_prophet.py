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
    import sys
    sys.path.insert(0, r'C:\temp\prophet_install')
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("[OK] Prophet verfügbar")
except ImportError:
    print("[ERROR] Prophet nicht installiert")
    PROPHET_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BankProphetPredictorFixed:
    """
    Prophet Model für 7h Bank-Predictions - ASCII VERSION
    Robustes NaN-Handling für Makro-Features
    """
    
    def __init__(self, features_path="features_enhanced"):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet muss installiert sein")
        
        self.features_path = features_path
        self.bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
        self.models = {}
        self.predictions = {}
        self.performance = {}
        
        # Makro-Features für externe Regressoren (prioritized)
        self.macro_features = [
            'VIX_Level', 'VIX_Return', 'DXY_Level', 'DXY_Return', 
            'TNX_Level', 'TNX_Return', 'SPX_Return', 'Sector_Return',
            'Yield_Curve_Proxy', 'Market_Beta_5D', 'Risk_On_Regime'
        ]
        
        print("[PROPHET] Prophet Predictor FIXED initialisiert")
        print("   [PATH] Features Pfad: " + features_path)
        print("   [BANKS] Banken: " + str(len(self.bank_symbols)))
        print("   [MACRO] Makro-Features: " + str(len(self.macro_features)))
    
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
            
            # Entferne NaN in kritischen Spalten
            df = df.dropna(subset=['DateTime', 'Close', 'Target_7h'])
            
            print(f"   [OK] {symbol}: {len(df)} Zeilen geladen")
            return df
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: Fehler beim Laden: {str(e)}")
            return None
    
    def clean_and_prepare_features(self, df, symbol):
        """Bereinigt und bereitet Makro-Features vor (ROBUST)"""
        
        print(f"   [CLEAN] {symbol}: Bereinige Makro-Features...")
        
        available_features = []
        feature_stats = {}
        
        for feature in self.macro_features:
            if feature in df.columns:
                # Original-Werte vor Bereinigung
                original_count = len(df[feature])
                nan_count = df[feature].isna().sum()
                
                # ROBUSTE BEREINIGUNG
                # 1. Forward-Fill (bis zu 5 aufeinanderfolgende NaN)
                df[feature] = df[feature].fillna(method='ffill', limit=5)
                
                # 2. Backward-Fill (für Anfangs-NaN)
                df[feature] = df[feature].fillna(method='bfill', limit=3)
                
                # 3. Median-Fill für verbleibende NaN
                if df[feature].isna().any():
                    median_value = df[feature].median()
                    if pd.notna(median_value):
                        df[feature] = df[feature].fillna(median_value)
                    else:
                        # Fallback: 0
                        df[feature] = df[feature].fillna(0)
                
                # 4. Infinite-Werte behandeln
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                if df[feature].isna().any():
                    df[feature] = df[feature].fillna(df[feature].median())
                
                # Final check
                remaining_nan = df[feature].isna().sum()
                
                if remaining_nan == 0:
                    available_features.append(feature)
                    feature_stats[feature] = {
                        'original_nan': nan_count,
                        'filled': nan_count - remaining_nan,
                        'success': True
                    }
                    print(f"      [OK] {feature}: {nan_count} NaN -> 0 NaN")
                else:
                    print(f"      [ERROR] {feature}: {remaining_nan} NaN verbleiben")
                    feature_stats[feature] = {
                        'original_nan': nan_count,
                        'remaining': remaining_nan,
                        'success': False
                    }
            else:
                print(f"      [WARN] {feature}: Spalte nicht gefunden")
        
        print(f"   [STATS] {symbol}: {len(available_features)}/{len(self.macro_features)} Features bereit")
        
        return df, available_features, feature_stats
    
    def prepare_prophet_data(self, df, symbol):
        """Bereitet Daten für Prophet vor (ROBUST)"""
        
        # Makro-Features bereinigen
        df_clean, available_regressors, stats = self.clean_and_prepare_features(df, symbol)
        
        # Prophet Format: ds (datetime), y (target)
        prophet_df = pd.DataFrame({
            'ds': df_clean['DateTime'],
            'y': df_clean['Target_7h']  # 7h Returns als Target
        })
        
        # Externe Regressoren hinzufügen
        for feature in available_regressors:
            prophet_df[feature] = df_clean[feature]
        
        # Final NaN check
        initial_rows = len(prophet_df)
        prophet_df = prophet_df.dropna()
        final_rows = len(prophet_df)
        
        if initial_rows != final_rows:
            print(f"   [CLEAN] {symbol}: {initial_rows - final_rows} Zeilen mit NaN entfernt")
        
        print(f"   [PREP] {symbol}: Prophet-Daten vorbereitet")
        print(f"      Zeilen: {len(prophet_df)}")
        print(f"      Externe Regressoren: {len(available_regressors)}")
        
        return prophet_df, available_regressors
    
    def create_prophet_model(self, symbol, available_regressors):
        """Erstellt Prophet Model mit Bank-spezifischen Parametern"""
        
        # Bank-spezifische Prophet-Parameter
        model_params = {
            'growth': 'linear',
            'yearly_seasonality': False,  
            'weekly_seasonality': True,   
            'daily_seasonality': True,    
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,  
            'seasonality_prior_scale': 10.0,
            'interval_width': 0.95
        }
        
        model = Prophet(**model_params)
        
        # Externe Regressoren hinzufügen (mit Fehlerbehandlung)
        successfully_added = []
        
        for regressor in available_regressors:
            try:
                if 'Level' in regressor:
                    model.add_regressor(regressor, prior_scale=0.5, mode='additive')
                elif 'Return' in regressor:
                    model.add_regressor(regressor, prior_scale=1.0, mode='additive')
                elif 'Corr' in regressor or 'Beta' in regressor:
                    model.add_regressor(regressor, prior_scale=0.3, mode='additive')
                else:
                    model.add_regressor(regressor, prior_scale=0.1, mode='additive')
                
                successfully_added.append(regressor)
                print(f"      [OK] Regressor hinzugefügt: {regressor}")
                
            except Exception as e:
                print(f"      [WARN] Regressor-Fehler {regressor}: {str(e)}")
        
        print(f"      [STATS] {len(successfully_added)}/{len(available_regressors)} Regressoren erfolgreich")
        
        return model, successfully_added
    
    def train_prophet_model(self, symbol):
        """Trainiert Prophet Model für eine Bank"""
        
        print(f"\n[TRAIN] Trainiere Prophet Model für {symbol}...")
        
        # Daten laden
        df = self.load_bank_data(symbol)
        if df is None:
            return None
        
        # Prophet-Daten vorbereiten
        prophet_df, available_regressors = self.prepare_prophet_data(df, symbol)
        if len(prophet_df) < 50:
            print(f"   [ERROR] {symbol}: Zu wenig Daten für Training ({len(prophet_df)})")
            return None
        
        # Model erstellen
        model, successful_regressors = self.create_prophet_model(symbol, available_regressors)
        
        # Training
        try:
            print(f"   [START] Starte Training...")
            model.fit(prophet_df)
            
            # Model speichern
            self.models[symbol] = {
                'model': model,
                'regressors': successful_regressors,
                'train_data': prophet_df,
                'raw_data': df
            }
            
            print(f"   [SUCCESS] {symbol}: Training erfolgreich")
            return model
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: Training-Fehler: {str(e)}")
            return None
    
    def create_robust_future_values(self, last_data, regressors):
        """Erstellt robuste Future-Werte für Makro-Features (WEEKEND-GAP LOGIC)"""
        
        future_values = {}
        
        for regressor in regressors:
            if regressor in last_data.columns:
                # Alle verfügbaren Werte für dieses Feature
                feature_series = last_data[regressor].dropna()
                
                if len(feature_series) > 0:
                    # WEEKEND-GAP STRATEGIE: Letzter verfügbarer Freitags-Wert
                    last_value = feature_series.iloc[-1]
                    
                    # Zusätzliche Robustheit
                    if pd.isna(last_value) or np.isinf(last_value):
                        # Fallback: Median der letzten 10 Werte
                        if len(feature_series) >= 10:
                            last_value = feature_series.tail(10).median()
                        else:
                            last_value = feature_series.median()
                    
                    # Final Fallback
                    if pd.isna(last_value) or np.isinf(last_value):
                        if 'Level' in regressor:
                            last_value = 100  # Typical market level
                        else:
                            last_value = 0    # Returns/Ratios
                    
                    future_values[regressor] = float(last_value)
                    print(f"      [FUTURE] {regressor}: {last_value:.4f} (Weekend-Gap Wert)")
                    
                else:
                    # Kein Wert verfügbar - Fallback
                    if 'Level' in regressor:
                        future_values[regressor] = 100.0
                    else:
                        future_values[regressor] = 0.0
                    print(f"      [WARN] {regressor}: Fallback-Wert verwendet")
            else:
                # Feature nicht vorhanden
                if 'Level' in regressor:
                    future_values[regressor] = 100.0
                else:
                    future_values[regressor] = 0.0
                print(f"      [ERROR] {regressor}: Feature fehlt - Fallback")
        
        return future_values
    
    def create_future_dataframe(self, symbol, prediction_start_time=None):
        """Erstellt Future DataFrame für 7h Prediction (ROBUST)"""
        
        if symbol not in self.models:
            print(f"[ERROR] {symbol}: Model nicht verfügbar")
            return None
        
        model_data = self.models[symbol]
        model = model_data['model']
        regressors = model_data['regressors']
        last_data = model_data['raw_data']
        
        # Bestimme Prediction-Start-Zeit
        if prediction_start_time is None:
            last_time = last_data['DateTime'].max()
            next_trading_hour = self.get_next_trading_hour(last_time)
            prediction_start_time = next_trading_hour
        
        print(f"   [TIME] {symbol}: Prediction ab {prediction_start_time}")
        
        # 7 Stunden Future DataFrame
        future_times = []
        current_time = prediction_start_time
        
        for i in range(7):
            future_times.append(current_time)
            current_time += timedelta(hours=1)
        
        future_df = pd.DataFrame({'ds': future_times})
        
        # ROBUSTE Future-Werte für alle Regressoren
        future_values = self.create_robust_future_values(last_data, regressors)
        
        # Future DataFrame befüllen
        for regressor, value in future_values.items():
            future_df[regressor] = value
        
        # Final validation
        nan_cols = future_df.columns[future_df.isna().any()].tolist()
        if nan_cols:
            print(f"   [WARN] {symbol}: NaN in Future-DataFrame: {nan_cols}")
            # Emergency fix
            future_df = future_df.fillna(0)
        
        print(f"   [FUTURE] {symbol}: Robustes Future DataFrame erstellt")
        print(f"      Features: {len(future_values)} Makro-Werte gesetzt")
        
        return future_df
    
    def get_next_trading_hour(self, last_time):
        """Bestimmt nächste Handelsstunde (13:30-19:30 UTC)"""
        
        # Wenn Wochenende, dann Montag 13:30 UTC
        if last_time.weekday() >= 5:  # Samstag=5, Sonntag=6
            days_until_monday = 7 - last_time.weekday()
            next_monday = last_time + timedelta(days=days_until_monday)
            return next_monday.replace(hour=13, minute=30, second=0, microsecond=0)
        
        # Wenn Handelstag aber nach 19:30, dann nächster Tag 13:30
        if last_time.hour >= 20:
            next_day = last_time + timedelta(days=1)
            if next_day.weekday() >= 5:
                days_until_monday = 7 - next_day.weekday()
                next_monday = next_day + timedelta(days=days_until_monday)
                return next_monday.replace(hour=13, minute=30, second=0, microsecond=0)
            else:
                return next_day.replace(hour=13, minute=30, second=0, microsecond=0)
        
        # Wenn vor Handelsstart (13:30), dann heute 13:30
        if last_time.hour < 13 or (last_time.hour == 13 and last_time.minute < 30):
            return last_time.replace(hour=13, minute=30, second=0, microsecond=0)
        
        # Sonst nächste Stunde
        next_hour = last_time.replace(minute=30, second=0, microsecond=0) + timedelta(hours=1)
        
        if next_hour.hour <= 19:
            return next_hour
        else:
            return self.get_next_trading_hour(next_hour)
    
    def make_prediction(self, symbol, prediction_start_time=None):
        """Macht 7h Prediction für eine Bank (ROBUST)"""
        
        if symbol not in self.models:
            print(f"[ERROR] {symbol}: Model nicht trainiert")
            return None
        
        print(f"\n[PREDICT] Prediction für {symbol}...")
        
        try:
            # Future DataFrame erstellen
            future_df = self.create_future_dataframe(symbol, prediction_start_time)
            if future_df is None:
                return None
            
            # Prediction
            model = self.models[symbol]['model']
            forecast = model.predict(future_df)
            
            # Ergebnisse formatieren
            prediction_result = pd.DataFrame({
                'DateTime': future_df['ds'],
                'Predicted_Return_7h': forecast['yhat'],
                'Lower_Bound': forecast['yhat_lower'],
                'Upper_Bound': forecast['yhat_upper'],
                'Confidence': forecast['yhat_upper'] - forecast['yhat_lower'],
                'Direction': (forecast['yhat'] > 0).astype(int)
            })
            
            # Zusätzliche Metriken
            prediction_result['Return_Magnitude'] = abs(prediction_result['Predicted_Return_7h'])
            prediction_result['Confidence_Score'] = 1 / (prediction_result['Confidence'] + 0.001)
            
            self.predictions[symbol] = prediction_result
            
            print(f"   [SUCCESS] {symbol}: 7h Prediction erfolgreich")
            print(f"      Durchschnittlicher Return: {prediction_result['Predicted_Return_7h'].mean():.4f}")
            print(f"      Bullish Stunden: {prediction_result['Direction'].sum()}/7")
            
            return prediction_result
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: Prediction-Fehler: {str(e)}")
            return None
    
    def train_all_banks(self):
        """Trainiert Prophet Models für alle Banken"""
        
        print(f"[START] PROPHET TRAINING FÜR ALLE BANKEN (FIXED)")
        print("=" * 55)
        
        successful_trainings = 0
        
        for symbol in self.bank_symbols:
            model = self.train_prophet_model(symbol)
            if model is not None:
                successful_trainings += 1
        
        print(f"\n[RESULT] Training abgeschlossen:")
        print(f"   [SUCCESS] Erfolgreich: {successful_trainings}/{len(self.bank_symbols)} Banken")
        print(f"   [MODELS] Trainierte Models: {list(self.models.keys())}")
        
        return successful_trainings > 0
    
    def predict_all_banks(self, prediction_start_time=None):
        """Macht Predictions für alle trainierten Banken (ROBUST)"""
        
        print(f"\n[START] 7H-PREDICTIONS FÜR ALLE BANKEN (ROBUST)")
        print("=" * 55)
        
        if not self.models:
            print("[ERROR] Keine trainierten Models verfügbar!")
            return {}
        
        successful_predictions = 0
        
        for symbol in self.models.keys():
            prediction = self.make_prediction(symbol, prediction_start_time)
            if prediction is not None:
                successful_predictions += 1
        
        print(f"\n[RESULT] Predictions abgeschlossen:")
        print(f"   [SUCCESS] Erfolgreich: {successful_predictions}/{len(self.models)} Banken")
        
        return self.predictions
    
    def get_trading_summary(self):
        """Erstellt Trading-Summary aller Predictions"""
        
        if not self.predictions:
            print("[ERROR] Keine Predictions verfügbar!")
            return None
        
        print(f"\n[SUMMARY] TRADING SUMMARY")
        print("=" * 40)
        
        summary_data = []
        
        for symbol, pred in self.predictions.items():
            total_return = pred['Predicted_Return_7h'].sum()
            avg_confidence = pred['Confidence_Score'].mean()
            bullish_hours = pred['Direction'].sum()
            max_single_return = pred['Predicted_Return_7h'].max()
            min_single_return = pred['Predicted_Return_7h'].min()
            
            summary_data.append({
                'Symbol': symbol,
                'Total_7h_Return': total_return,
                'Avg_Confidence': avg_confidence,
                'Bullish_Hours': f"{bullish_hours}/7",
                'Max_Hourly_Return': max_single_return,
                'Min_Hourly_Return': min_single_return,
                'Net_Direction': 'Bullish' if total_return > 0 else 'Bearish',
                'Strength': abs(total_return)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total_7h_Return', ascending=False)
        
        print("[TOP] TOP PERFORMERS (7h Total Return):")
        print(summary_df[['Symbol', 'Total_7h_Return', 'Net_Direction', 'Bullish_Hours']].head())
        
        print(f"\n[PORTFOLIO] PORTFOLIO OVERVIEW:")
        total_portfolio_return = summary_df['Total_7h_Return'].sum()
        bullish_banks = len(summary_df[summary_df['Total_7h_Return'] > 0])
        print(f"   Portfolio 7h Return: {total_portfolio_return:.4f}")
        print(f"   Bullish Banks: {bullish_banks}/{len(summary_df)}")
        print(f"   Average Confidence: {summary_df['Avg_Confidence'].mean():.2f}")
        
        return summary_df
    
    def save_predictions(self, output_path="prophet_predictions"):
        """Speichert alle Predictions als CSV"""
        
        if not self.predictions:
            print("[ERROR] Keine Predictions zum Speichern!")
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n[SAVE] Speichere Predictions...")
        
        # Einzelne Bank-Predictions
        for symbol, pred in self.predictions.items():
            csv_file = os.path.join(output_path, f"{symbol}_7h_prediction.csv")
            pred.to_csv(csv_file, index=False)
            print(f"   [OK] {csv_file}")
        
        # Kombinierte Summary
        summary = self.get_trading_summary()
        if summary is not None:
            summary_file = os.path.join(output_path, "trading_summary.csv")
            summary.to_csv(summary_file, index=False)
            print(f"   [OK] {summary_file}")
        
        print(f"   [DONE] Alle Predictions gespeichert in: {output_path}")

def run_prophet_pipeline_fixed():
    """Hauptfunktion für ROBUSTE Prophet Pipeline"""
    
    print("[START] PROPHET BANK-PREDICTION PIPELINE (ROBUST)")
    print("=" * 60)
    
    # Prophet Predictor initialisieren
    predictor = BankProphetPredictorFixed()
    
    # Prüfe ob Features verfügbar
    if not os.path.exists(predictor.features_path):
        print(f"[ERROR] Features-Ordner nicht gefunden: {predictor.features_path}")
        print("   Führe zuerst Enhanced Feature-Engineering aus!")
        return
    
    # Training
    print(f"[START] Starte ROBUSTES Prophet Training...")
    training_success = predictor.train_all_banks()
    
    if not training_success:
        print("[ERROR] Training fehlgeschlagen!")
        return
    
    # Predictions für Montag (nächste Handelsstunden)
    print(f"[START] Starte ROBUSTE 7h-Predictions...")
    predictions = predictor.predict_all_banks()
    
    if not predictions:
        print("[ERROR] Predictions fehlgeschlagen!")
        return
    
    # Trading Summary
    summary = predictor.get_trading_summary()
    
    # Speichern
    predictor.save_predictions()
    
    print(f"\n[SUCCESS] PROPHET PIPELINE ERFOLGREICH!")
    print(f"   [RESULTS] {len(predictions)} Bank-Predictions erstellt")
    print(f"   [SAVE] Ergebnisse gespeichert in: prophet_predictions/")
    print(f"   [READY] BEREIT FÜR TRADING-ENTSCHEIDUNGEN!")
    print(f"   [ROBUST] ROBUST gegen NaN-Probleme")

if __name__ == "__main__":
    if not PROPHET_AVAILABLE:
        print("[ERROR] Prophet nicht installiert!")
        print("   Installiere mit der temp-Installation")
    else:
        run_prophet_pipeline_fixed()