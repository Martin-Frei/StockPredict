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
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    ARIMA_AVAILABLE = True
    print("[OK] ARIMA Libraries verfügbar (nur statsmodels)")
except ImportError:
    print("[ERROR] statsmodels nicht installiert: pip install statsmodels")
    ARIMA_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BankARIMAPredictorFixed:
    """
    ARIMA Model für 7h Bank-Predictions - ASCII VERSION
    Nutzt bewährte Standard-Parameter für Financial Returns
    """
    
    def __init__(self, features_path="features_enhanced"):
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels muss installiert sein: pip install statsmodels")
        
        self.features_path = features_path
        self.bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
        self.models = {}
        self.predictions = {}
        self.performance = {}
        
        # Standard ARIMA-Parameter für Financial Returns (bewährt)
        self.standard_orders = [
            (1, 1, 1),  # Klassischer ARIMA für Returns
            (1, 1, 0),  # Einfacher AR mit Differenzierung
            (2, 1, 1),  # Komplexere Patterns
            (0, 1, 1),  # Moving Average Fokus
            (2, 1, 0),  # Auto-Regression Fokus
        ]
        
        print(f"[ARIMA] ARIMA Predictor FIXED initialisiert (ohne pmdarima)")
        print(f"   [PATH] Features Pfad: {features_path}")
        print(f"   [BANKS] Banken: {len(self.bank_symbols)}")
        print(f"   [ORDERS] Standard ARIMA Orders: {self.standard_orders}")
    
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
    
    def prepare_time_series(self, df, symbol):
        """Bereitet Zeitreihen für ARIMA vor"""
        
        # Hauptzeitreihe: Returns (nicht Target_7h für Training)
        time_series = df['Returns'].dropna()
        
        # Zeitindex setzen
        time_index = df.loc[time_series.index, 'DateTime']
        ts_data = pd.Series(time_series.values, index=time_index)
        
        print(f"   [PREP] {symbol}: Zeitreihe vorbereitet")
        print(f"      Länge: {len(ts_data)}")
        print(f"      Zeitraum: {ts_data.index.min()} bis {ts_data.index.max()}")
        
        return ts_data
    
    def check_stationarity(self, ts_data, symbol):
        """Prüft Stationarität der Zeitreihe"""
        
        print(f"   [TEST] {symbol}: Stationaritäts-Test...")
        
        try:
            # Augmented Dickey-Fuller Test
            adf_result = adfuller(ts_data.dropna())
            adf_pvalue = adf_result[1]
            
            # Interpretation
            is_stationary = adf_pvalue < 0.05
            
            print(f"      ADF p-value: {adf_pvalue:.6f}")
            if is_stationary:
                print(f"      [OK] Zeitreihe ist stationär")
                return True, 0
            else:
                print(f"      [WARN] Zeitreihe ist nicht stationär - Differenzierung empfohlen")
                return True, 1  # Returns meist mit d=1 gut
                    
        except Exception as e:
            print(f"      [ERROR] Stationaritäts-Test Fehler: {str(e)}")
            return True, 1  # Fallback
    
    def select_best_arima_order(self, ts_data, symbol):
        """Wählt beste ARIMA-Parameter durch AIC-Vergleich"""
        
        print(f"   [SELECT] {symbol}: Teste Standard ARIMA-Parameter...")
        
        best_order = None
        best_aic = np.inf
        results = []
        
        for order in self.standard_orders:
            try:
                # ARIMA Model mit aktuellen Parametern
                model = ARIMA(ts_data, order=order)
                fitted = model.fit()
                
                aic = fitted.aic
                results.append({
                    'order': order,
                    'aic': aic,
                    'success': True
                })
                
                print(f"      [TEST] ARIMA{order}: AIC={aic:.2f}")
                
                # Bestes Model bisher?
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                
            except Exception as e:
                results.append({
                    'order': order,
                    'aic': np.inf,
                    'success': False,
                    'error': str(e)
                })
                print(f"      [ERROR] ARIMA{order}: {str(e)[:50]}...")
        
        if best_order is not None:
            print(f"      [BEST] Beste Parameter: ARIMA{best_order} (AIC={best_aic:.2f})")
            return best_order, best_aic
        else:
            print(f"      [FALLBACK] Fallback auf ARIMA(1,1,0)")
            return (1, 1, 0), None
    
    def train_arima_model(self, symbol):
        """Trainiert ARIMA Model für eine Bank"""
        
        print(f"\n[TRAIN] Trainiere ARIMA Model für {symbol}...")
        
        # Daten laden
        df = self.load_bank_data(symbol)
        if df is None:
            return None
        
        # Zeitreihe vorbereiten
        ts_data = self.prepare_time_series(df, symbol)
        if len(ts_data) < 50:
            print(f"   [ERROR] {symbol}: Zu wenig Daten für ARIMA Training ({len(ts_data)})")
            return None
        
        # Stationarität prüfen
        is_stationary, suggested_d = self.check_stationarity(ts_data, symbol)
        
        # Beste ARIMA-Parameter bestimmen
        order, aic = self.select_best_arima_order(ts_data, symbol)
        
        # ARIMA Model trainieren
        try:
            print(f"   [START] Starte ARIMA Training mit {order}...")
            
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            # Model-Diagnostik
            try:
                ljung_box = acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)
                residual_autocorr = ljung_box['lb_pvalue'].iloc[-1]
            except:
                residual_autocorr = 0.5  # Fallback
            
            print(f"   [SUCCESS] {symbol}: ARIMA Training erfolgreich")
            print(f"      [AIC] AIC: {fitted_model.aic:.2f}")
            print(f"      [TEST] Residual Test p-value: {residual_autocorr:.4f}")
            
            # Model speichern
            self.models[symbol] = {
                'model': fitted_model,
                'order': order,
                'aic': fitted_model.aic,
                'time_series': ts_data,
                'raw_data': df,
                'diagnostics': {
                    'residual_autocorr': residual_autocorr,
                    'stationarity_d': suggested_d,
                    'selection_method': 'AIC_comparison'
                }
            }
            
            return fitted_model
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: ARIMA Training-Fehler: {str(e)}")
            
            # Einfachster Fallback: ARIMA(0,1,0) = Random Walk
            try:
                print(f"   [RETRY] {symbol}: Versuche Random Walk ARIMA(0,1,0)...")
                model = ARIMA(ts_data, order=(0, 1, 0))
                fitted_model = model.fit()
                
                self.models[symbol] = {
                    'model': fitted_model,
                    'order': (0, 1, 0),
                    'aic': fitted_model.aic,
                    'time_series': ts_data,
                    'raw_data': df,
                    'diagnostics': {'fallback': 'random_walk'}
                }
                
                print(f"   [SUCCESS] {symbol}: Random Walk Fallback erfolgreich")
                return fitted_model
                
            except Exception as e2:
                print(f"   [ERROR] {symbol}: Auch Random Walk fehlgeschlagen: {str(e2)}")
                return None
    
    def get_next_trading_hour(self, last_time):
        """Bestimmt nächste Handelsstunde (identisch mit Prophet)"""
        
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
        """Macht 7h ARIMA Prediction für eine Bank"""
        
        if symbol not in self.models:
            print(f"[ERROR] {symbol}: ARIMA Model nicht trainiert")
            return None
        
        print(f"\n[PREDICT] ARIMA Prediction für {symbol}...")
        
        try:
            model_data = self.models[symbol]
            fitted_model = model_data['model']
            time_series = model_data['time_series']
            raw_data = model_data['raw_data']
            
            # Bestimme Prediction-Start-Zeit
            if prediction_start_time is None:
                last_time = time_series.index.max()
                next_trading_hour = self.get_next_trading_hour(last_time)
                prediction_start_time = next_trading_hour
            
            print(f"   [TIME] {symbol}: ARIMA Prediction ab {prediction_start_time}")
            
            # 7-Stunden Forecast
            forecast_result = fitted_model.forecast(steps=7)
            forecast_values = forecast_result
            
            # Confidence Intervals berechnen
            try:
                forecast_obj = fitted_model.get_forecast(steps=7)
                conf_int = forecast_obj.conf_int()
                lower_bounds = conf_int.iloc[:, 0].values
                upper_bounds = conf_int.iloc[:, 1].values
            except:
                # Fallback: Historische Volatilität für Confidence
                hist_vol = time_series.std()
                lower_bounds = forecast_values - (1.96 * hist_vol)
                upper_bounds = forecast_values + (1.96 * hist_vol)
            
            # Future Timestamps erstellen
            future_times = []
            current_time = prediction_start_time
            for i in range(7):
                future_times.append(current_time)
                current_time += timedelta(hours=1)
            
            # WICHTIG: ARIMA vorhersagt 1h-Returns, wir brauchen 7h-Returns
            # Lösung: Rolling 7-hour cumulative returns
            seven_hour_returns = []
            
            for i in range(7):
                # Für jede Stunde: Summe der nächsten 7 Stunden
                if i + 7 <= len(forecast_values):
                    # Volle 7 Stunden verfügbar
                    seven_h_return = forecast_values[i:i+7].sum()
                else:
                    # Weniger als 7 Stunden: verfügbare + durchschnittlicher Rest
                    available_hours = len(forecast_values) - i
                    available_sum = forecast_values[i:].sum()
                    missing_hours = 7 - available_hours
                    avg_return = forecast_values.mean()
                    seven_h_return = available_sum + (missing_hours * avg_return)
                
                seven_hour_returns.append(seven_h_return)
            
            # Confidence für 7h-Returns (approximiert)
            confidence_bounds = []
            for i in range(7):
                lower_7h = sum(lower_bounds[max(0, i):min(len(lower_bounds), i+7)])
                upper_7h = sum(upper_bounds[max(0, i):min(len(upper_bounds), i+7)])
                confidence_bounds.append((lower_7h, upper_7h))
            
            # Ergebnisse formatieren
            prediction_result = pd.DataFrame({
                'DateTime': future_times,
                'Predicted_Return_7h': seven_hour_returns,
                'Lower_Bound': [cb[0] for cb in confidence_bounds],
                'Upper_Bound': [cb[1] for cb in confidence_bounds],
                'Confidence': [cb[1] - cb[0] for cb in confidence_bounds],
                'Direction': [1 if x > 0 else 0 for x in seven_hour_returns]
            })
            
            # Zusätzliche Metriken
            prediction_result['Return_Magnitude'] = abs(prediction_result['Predicted_Return_7h'])
            prediction_result['Confidence_Score'] = 1 / (prediction_result['Confidence'] + 0.001)
            prediction_result['Model_Type'] = 'ARIMA'
            prediction_result['ARIMA_Order'] = str(model_data['order'])
            
            self.predictions[symbol] = prediction_result
            
            print(f"   [SUCCESS] {symbol}: ARIMA 7h Prediction erfolgreich")
            print(f"      Durchschnittlicher 7h Return: {prediction_result['Predicted_Return_7h'].mean():.4f}")
            print(f"      Bullish Stunden: {prediction_result['Direction'].sum()}/7")
            print(f"      ARIMA Order: {model_data['order']}")
            
            return prediction_result
            
        except Exception as e:
            print(f"   [ERROR] {symbol}: ARIMA Prediction-Fehler: {str(e)}")
            return None
    
    def train_all_banks(self):
        """Trainiert ARIMA Models für alle Banken"""
        
        print(f"[START] ARIMA TRAINING FÜR ALLE BANKEN (FIXED)")
        print("=" * 55)
        
        successful_trainings = 0
        
        for symbol in self.bank_symbols:
            model = self.train_arima_model(symbol)
            if model is not None:
                successful_trainings += 1
        
        print(f"\n[RESULT] ARIMA Training abgeschlossen:")
        print(f"   [SUCCESS] Erfolgreich: {successful_trainings}/{len(self.bank_symbols)} Banken")
        print(f"   [MODELS] Trainierte Models: {list(self.models.keys())}")
        
        # Model-Übersicht
        if self.models:
            print(f"\n[OVERVIEW] ARIMA Model-Übersicht:")
            for symbol, data in self.models.items():
                order = data['order']
                aic = data['aic']
                print(f"   {symbol}: ARIMA{order}, AIC={aic:.1f}")
        
        return successful_trainings > 0
    
    def predict_all_banks(self, prediction_start_time=None):
        """Macht ARIMA Predictions für alle trainierten Banken"""
        
        print(f"\n[START] 7H-ARIMA-PREDICTIONS FÜR ALLE BANKEN")
        print("=" * 55)
        
        if not self.models:
            print("[ERROR] Keine trainierten ARIMA Models verfügbar!")
            return {}
        
        successful_predictions = 0
        
        for symbol in self.models.keys():
            prediction = self.make_prediction(symbol, prediction_start_time)
            if prediction is not None:
                successful_predictions += 1
        
        print(f"\n[RESULT] ARIMA Predictions abgeschlossen:")
        print(f"   [SUCCESS] Erfolgreich: {successful_predictions}/{len(self.models)} Banken")
        
        return self.predictions
    
    def get_trading_summary(self):
        """Erstellt ARIMA Trading-Summary aller Predictions"""
        
        if not self.predictions:
            print("[ERROR] Keine ARIMA Predictions verfügbar!")
            return None
        
        print(f"\n[SUMMARY] ARIMA TRADING SUMMARY")
        print("=" * 45)
        
        summary_data = []
        
        for symbol, pred in self.predictions.items():
            total_return = pred['Predicted_Return_7h'].sum()
            avg_confidence = pred['Confidence_Score'].mean()
            bullish_hours = pred['Direction'].sum()
            max_single_return = pred['Predicted_Return_7h'].max()
            min_single_return = pred['Predicted_Return_7h'].min()
            arima_order = pred['ARIMA_Order'].iloc[0]
            
            summary_data.append({
                'Symbol': symbol,
                'Total_7h_Return': total_return,
                'Avg_Confidence': avg_confidence,
                'Bullish_Hours': f"{bullish_hours}/7",
                'Max_Hourly_Return': max_single_return,
                'Min_Hourly_Return': min_single_return,
                'Net_Direction': 'Bullish' if total_return > 0 else 'Bearish',
                'Strength': abs(total_return),
                'ARIMA_Order': arima_order,
                'Model_Type': 'ARIMA'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total_7h_Return', ascending=False)
        
        print("[TOP] ARIMA TOP PERFORMERS (7h Total Return):")
        print(summary_df[['Symbol', 'Total_7h_Return', 'Net_Direction', 'Bullish_Hours', 'ARIMA_Order']].head())
        
        print(f"\n[PORTFOLIO] ARIMA PORTFOLIO OVERVIEW:")
        total_portfolio_return = summary_df['Total_7h_Return'].sum()
        bullish_banks = len(summary_df[summary_df['Total_7h_Return'] > 0])
        print(f"   ARIMA Portfolio 7h Return: {total_portfolio_return:.4f}")
        print(f"   ARIMA Bullish Banks: {bullish_banks}/{len(summary_df)}")
        print(f"   ARIMA Average Confidence: {summary_df['Avg_Confidence'].mean():.2f}")
        
        return summary_df
    
    def save_predictions(self, output_path="arima_predictions"):
        """Speichert alle ARIMA Predictions als CSV"""
        
        if not self.predictions:
            print("[ERROR] Keine ARIMA Predictions zum Speichern!")
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n[SAVE] Speichere ARIMA Predictions...")
        
        # Einzelne Bank-Predictions
        for symbol, pred in self.predictions.items():
            csv_file = os.path.join(output_path, f"{symbol}_7h_arima_prediction.csv")
            pred.to_csv(csv_file, index=False)
            print(f"   [OK] {csv_file}")
        
        # Kombinierte Summary
        summary = self.get_trading_summary()
        if summary is not None:
            summary_file = os.path.join(output_path, "arima_trading_summary.csv")
            summary.to_csv(summary_file, index=False)
            print(f"   [OK] {summary_file}")
        
        # Model-Details speichern
        model_details = []
        for symbol, data in self.models.items():
            model_details.append({
                'Symbol': symbol,
                'ARIMA_Order': str(data['order']),
                'AIC': data['aic'],
                'Training_Samples': len(data['time_series']),
                'Selection_Method': data['diagnostics'].get('selection_method', 'unknown'),
                'Diagnostics': str(data.get('diagnostics', {}))
            })
        
        model_df = pd.DataFrame(model_details)
        model_file = os.path.join(output_path, "arima_model_details.csv")
        model_df.to_csv(model_file, index=False)
        print(f"   [OK] {model_file}")
        
        print(f"   [DONE] Alle ARIMA Predictions gespeichert in: {output_path}")

def run_arima_pipeline_fixed():
    """Hauptfunktion für ARIMA Pipeline (OHNE pmdarima)"""
    
    print("[START] ARIMA BANK-PREDICTION PIPELINE (FIXED)")
    print("=" * 55)
    
    # ARIMA Predictor initialisieren
    predictor = BankARIMAPredictorFixed()
    
    # Prüfe ob Features verfügbar
    if not os.path.exists(predictor.features_path):
        print(f"[ERROR] Features-Ordner nicht gefunden: {predictor.features_path}")
        print("   Führe zuerst Enhanced Feature-Engineering aus!")
        return
    
    # Training
    print(f"[START] Starte ARIMA Training (Standard-Parameter)...")
    training_success = predictor.train_all_banks()
    
    if not training_success:
        print("[ERROR] ARIMA Training fehlgeschlagen!")
        return
    
    # Predictions für Montag (nächste Handelsstunden)
    print(f"[START] Starte ARIMA 7h-Predictions...")
    predictions = predictor.predict_all_banks()
    
    if not predictions:
        print("[ERROR] ARIMA Predictions fehlgeschlagen!")
        return
    
    # Trading Summary
    summary = predictor.get_trading_summary()
    
    # Speichern
    predictor.save_predictions()
    
    print(f"\n[SUCCESS] ARIMA PIPELINE ERFOLGREICH!")
    print(f"   [RESULTS] {len(predictions)} ARIMA Bank-Predictions erstellt")
    print(f"   [SAVE] Ergebnisse gespeichert in: arima_predictions/")
    print(f"   [READY] Bereit für Ensemble mit Prophet!")
    print(f"   [INFO] Funktioniert OHNE pmdarima (nur statsmodels)")

if __name__ == "__main__":
    if not ARIMA_AVAILABLE:
        print("[ERROR] statsmodels nicht installiert!")
        print("   Installiere mit: pip install statsmodels")
    else:
        run_arima_pipeline_fixed()