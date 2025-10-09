#!/usr/bin/env python3
"""
Dashboard Integration - Verbindet echtes Backfill mit HTML Dashboard
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import http.server
import socketserver
import webbrowser
from urllib.parse import urlparse, parse_qs
import threading
import time

class DashboardDataProvider:
    def __init__(self, base_path="StockPredict"):
        self.base_path = Path(base_path)
        self.predictions_path = self.base_path / "Predictions"
        self.enhanced_features_path = self.base_path / "Enhanced_Features"
        
        self.banks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF']
        self.models = ['RandomForest', 'XGBoost', 'LSTM', 'LinearRegression', 'Ensemble']
        
    def load_all_predictions(self):
        """
        Lädt ALLE echten Predictions aus dem Predictions-Ordner
        """
        print("📊 Lade echte Predictions aus Disk...")
        
        all_predictions = []
        
        # Durchsuche alle Datums-Ordner
        for date_folder in sorted(self.predictions_path.glob("*")):
            if not date_folder.is_dir():
                continue
                
            try:
                date_str = date_folder.name
                prediction_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            
            print(f"  📅 Lade {date_str}...")
            
            # Lade alle CSV-Dateien in diesem Ordner
            for pred_file in date_folder.glob("*.csv"):
                try:
                    df = pd.read_csv(pred_file)
                    
                    # Füge zu all_predictions hinzu
                    for _, row in df.iterrows():
                        prediction_data = {
                            'date': date_str,
                            'bank': row.get('Bank', 'Unknown'),
                            'model': row.get('Model', 'Unknown'),
                            'predicted': float(row.get('Predicted_7h_Return', 0)),
                            'actual': None,  # Wird später gefüllt
                            'direction_predicted': 'bullish' if row.get('Predicted_7h_Return', 0) > 0 else 'bearish',
                            'backfilled': row.get('Backfilled', False)
                        }
                        all_predictions.append(prediction_data)
                        
                except Exception as e:
                    print(f"    ❌ Fehler bei {pred_file}: {e}")
                    continue
        
        print(f"📊 {len(all_predictions)} echte Predictions geladen")
        return all_predictions
    
    def load_actual_returns(self, predictions):
        """
        Lädt tatsächliche Returns aus Enhanced Features für Validierung
        """
        print("📈 Lade tatsächliche Returns für Validierung...")
        
        # Gruppiere Predictions nach Bank
        predictions_by_bank = {}
        for pred in predictions:
            bank = pred['bank']
            if bank not in predictions_by_bank:
                predictions_by_bank[bank] = []
            predictions_by_bank[bank].append(pred)
        
        # Lade Enhanced Features für jede Bank
        for bank in predictions_by_bank:
            enhanced_file = self.enhanced_features_path / f"{bank}_enhanced_features.csv"
            
            if not enhanced_file.exists():
                print(f"  ⚠️ {bank}: Enhanced Features nicht gefunden")
                continue
            
            try:
                df = pd.read_csv(enhanced_file)
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                
                print(f"  📊 {bank}: {len(df)} Enhanced Features geladen")
                
                # Matche Predictions mit tatsächlichen Returns
                for pred in predictions_by_bank[bank]:
                    # Berechne Validierungs-Zeitpunkt (Prediction um 9:30 + 7h = 16:30)
                    pred_datetime = datetime.strptime(f"{pred['date']} 16:30:00", "%Y-%m-%d %H:%M:%S")
                    
                    # Suche nach passendem Target_7h
                    matching_rows = df[
                        (df['DateTime'].dt.date == pred_datetime.date()) &
                        (df['DateTime'].dt.hour == 16) &
                        (df['DateTime'].dt.minute == 30)
                    ]
                    
                    if not matching_rows.empty and 'Target_7h' in df.columns:
                        actual_return = matching_rows['Target_7h'].iloc[0]
                        pred['actual'] = float(actual_return)
                        pred['direction_actual'] = 'bullish' if actual_return > 0 else 'bearish'
                    else:
                        # Fallback: Suche nächstgelegenen Zeitpunkt
                        closest_row = df.iloc[(df['DateTime'] - pred_datetime).abs().argsort()[:1]]
                        if not closest_row.empty and 'Target_7h' in df.columns:
                            pred['actual'] = float(closest_row['Target_7h'].iloc[0])
                            pred['direction_actual'] = 'bullish' if pred['actual'] > 0 else 'bearish'
                
            except Exception as e:
                print(f"  ❌ {bank}: Fehler beim Laden: {e}")
                continue
        
        # Zähle erfolgreiche Matches
        matched_count = sum(1 for pred in predictions if pred['actual'] is not None)
        print(f"📈 {matched_count}/{len(predictions)} Predictions mit tatsächlichen Returns gematcht")
        
        return predictions
    
    def calculate_real_performance_metrics(self, predictions):
        """
        Berechnet echte Performance-Metriken aus den geladenen Daten
        """
        print("🔢 Berechne echte Performance-Metriken...")
        
        # Filtere nur Predictions mit tatsächlichen Returns
        valid_predictions = [p for p in predictions if p['actual'] is not None]
        
        if not valid_predictions:
            print("❌ Keine gültigen Predictions für Metrik-Berechnung")
            return {}
        
        # Gruppiere nach Modell
        model_stats = {}
        for model in self.models:
            model_preds = [p for p in valid_predictions if p['model'] == model]
            
            if not model_preds:
                continue
            
            # Berechne Metriken
            total_predictions = len(model_preds)
            correct_directions = sum(1 for p in model_preds 
                                   if p['direction_predicted'] == p['direction_actual'])
            hit_rate = (correct_directions / total_predictions) * 100
            
            predicted_returns = [p['predicted'] for p in model_preds]
            actual_returns = [p['actual'] for p in model_preds]
            
            total_predicted = sum(predicted_returns)
            total_actual = sum(actual_returns)
            
            # MAPE (Mean Absolute Percentage Error)
            absolute_errors = [abs(p['predicted'] - p['actual']) for p in model_preds]
            mape = sum(absolute_errors) / len(absolute_errors)
            
            # Volatility
            import statistics
            volatility = statistics.stdev(actual_returns) if len(actual_returns) > 1 else 0
            
            # Sharpe Ratio (vereinfacht)
            avg_return = sum(actual_returns) / len(actual_returns)
            sharpe_ratio = (avg_return / volatility * (252**0.5)) if volatility > 0 else 0
            
            # Accuracy Score
            accuracy_score = (hit_rate * 0.6) + ((100 - mape) * 0.4)
            
            model_stats[model] = {
                'hitRate': hit_rate,
                'mape': mape,
                'avgReturn': avg_return,
                'totalReturn': total_actual,
                'sharpeRatio': sharpe_ratio,
                'accuracyScore': accuracy_score,
                'totalPredictions': total_predictions,
                'volatility': volatility
            }
        
        print(f"🔢 Performance-Metriken für {len(model_stats)} Modelle berechnet")
        return model_stats
    
    def export_for_dashboard(self):
        """
        Exportiert alle Daten im Format für das Dashboard
        """
        print("🚀 Exportiere Daten für Dashboard...")
        
        # 1. Lade echte Predictions
        predictions = self.load_all_predictions()
        
        if not predictions:
            print("❌ Keine Predictions gefunden")
            return None
        
        # 2. Lade tatsächliche Returns
        predictions = self.load_actual_returns(predictions)
        
        # 3. Berechne Performance-Metriken
        performance_metrics = self.calculate_real_performance_metrics(predictions)
        
        # 4. Erstelle Dashboard-Export
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'performance': performance_metrics,
            'summary': {
                'totalPredictions': len(predictions),
                'validPredictions': len([p for p in predictions if p['actual'] is not None]),
                'dateRange': {
                    'start': min(p['date'] for p in predictions) if predictions else None,
                    'end': max(p['date'] for p in predictions) if predictions else None
                },
                'avgHitRate': sum(m['hitRate'] for m in performance_metrics.values()) / len(performance_metrics) if performance_metrics else 0,
                'totalReturn': sum(m['totalReturn'] for m in performance_metrics.values()) if performance_metrics else 0
            }
        }
        
        print("🚀 Dashboard-Export erstellt")
        return dashboard_data

class IntegratedTVSServer:
    def __init__(self, port=8080):
        self.port = port
        self.data_provider = DashboardDataProvider()
        self.real_data = None
        
    def load_real_data(self):
        """
        Lädt echte Daten im Hintergrund
        """
        print("⏳ Lade echte Daten...")
        self.real_data = self.data_provider.export_for_dashboard()
        print("✅ Echte Daten geladen")
    
    def create_integrated_html(self):
        """
        Erstellt HTML mit echten Daten statt Simulation
        """
        
        if self.real_data is None:
            print("⚠️ Keine echten Daten verfügbar - verwende Simulation")
            real_data_js = "null"
        else:
            real_data_js = json.dumps(self.real_data, indent=2)
        
        html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Validation System (TVS) - Real Data</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        /* Ihr ursprüngliches CSS hier... */
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #fff;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .data-status {{
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
        }}
        .status-bar {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .status-card {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            flex: 1;
            min-width: 200px;
        }}
        .status-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .status-label {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Trading Validation System (TVS)</h1>
            <p>Mit echten Backfill-Daten aus deinem StockPredict System</p>
            <div class="data-status">
                <strong>✅ ECHTE DATEN GELADEN:</strong>
                <span id="dataInfo">Lade Daten...</span>
            </div>
        </div>
        
        <div class="status-bar">
            <div class="status-card">
                <div class="status-value" id="totalPredictions">0</div>
                <div class="status-label">Echte Predictions</div>
            </div>
            <div class="status-card">
                <div class="status-value" id="validPredictions">0</div>
                <div class="status-label">Mit Actual Returns</div>
            </div>
            <div class="status-card">
                <div class="status-value" id="hitRate">0%</div>
                <div class="status-label">Hit Rate (Real)</div>
            </div>
            <div class="status-card">
                <div class="status-value" id="totalReturn">0%</div>
                <div class="status-label">Total Return (Real)</div>
            </div>
            <div class="status-card">
                <div class="status-value" id="bestModel">--</div>
                <div class="status-label">Best Model</div>
            </div>
        </div>
        
        <div id="detailedAnalysis">
            <!-- Detaillierte Analyse wird hier eingefügt -->
        </div>
    </div>

    <script>
        // Echte Daten vom Server
        const REAL_DATA = {real_data_js};
        
        class RealTVSSystem {{
            constructor() {{
                this.realData = REAL_DATA;
                this.initializeWithRealData();
            }}
            
            initializeWithRealData() {{
                console.log('🚀 Initialisiere TVS mit echten Daten...');
                
                if (!this.realData) {{
                    console.error('❌ Keine echten Daten verfügbar');
                    document.getElementById('dataInfo').textContent = 'Keine echten Daten verfügbar';
                    return;
                }}
                
                console.log('📊 Echte Daten geladen:', this.realData.summary);
                
                // Update Data Info
                const summary = this.realData.summary;
                document.getElementById('dataInfo').innerHTML = `
                    ${{summary.totalPredictions}} Predictions von ${{summary.dateRange.start}} bis ${{summary.dateRange.end}}
                `;
                
                // Update Status Cards
                document.getElementById('totalPredictions').textContent = summary.totalPredictions.toLocaleString();
                document.getElementById('validPredictions').textContent = summary.validPredictions.toLocaleString();
                document.getElementById('hitRate').textContent = summary.avgHitRate.toFixed(1) + '%';
                document.getElementById('totalReturn').textContent = summary.totalReturn.toFixed(1) + '%';
                
                // Best Model
                const bestModel = this.findBestModel();
                document.getElementById('bestModel').textContent = bestModel || '--';
                
                // Create detailed analysis
                this.createDetailedAnalysis();
            }}
            
            findBestModel() {{
                if (!this.realData.performance) return null;
                
                let bestModel = null;
                let bestScore = -1;
                
                for (const [model, metrics] of Object.entries(this.realData.performance)) {{
                    if (metrics.accuracyScore > bestScore) {{
                        bestScore = metrics.accuracyScore;
                        bestModel = model;
                    }}
                }}
                
                return bestModel;
            }}
            
            createDetailedAnalysis() {{
                const container = document.getElementById('detailedAnalysis');
                
                let html = '<h2>🔍 Detaillierte Performance-Analyse (Echte Daten)</h2>';
                
                if (this.realData.performance) {{
                    html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">';
                    
                    // Sortiere Modelle nach Accuracy Score
                    const sortedModels = Object.entries(this.realData.performance)
                        .sort(([,a], [,b]) => b.accuracyScore - a.accuracyScore);
                    
                    sortedModels.forEach(([model, metrics], index) => {{
                        const rankColor = index === 0 ? '#FFD700' : index === 1 ? '#C0C0C0' : '#CD7F32';
                        
                        html += `
                        <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h3>${{model}}</h3>
                                <span style="background: ${{rankColor}}; color: black; padding: 5px 10px; border-radius: 20px; font-weight: bold;">
                                    #${{index + 1}}
                                </span>
                            </div>
                            <div style="display: grid; gap: 8px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Hit Rate:</span>
                                    <strong>${{metrics.hitRate.toFixed(1)}}%</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Accuracy Score:</span>
                                    <strong>${{metrics.accuracyScore.toFixed(1)}}</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Total Return:</span>
                                    <strong>${{metrics.totalReturn.toFixed(2)}}%</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Sharpe Ratio:</span>
                                    <strong>${{metrics.sharpeRatio.toFixed(3)}}</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>MAPE:</span>
                                    <strong>${{metrics.mape.toFixed(2)}}%</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Predictions:</span>
                                    <strong>${{metrics.totalPredictions}}</strong>
                                </div>
                            </div>
                        </div>`;
                    }});
                    
                    html += '</div>';
                }} else {{
                    html += '<p>⚠️ Keine Performance-Metriken verfügbar</p>';
                }}
                
                container.innerHTML = html;
            }}
        }}
        
        // Initialize Real TVS
        document.addEventListener('DOMContentLoaded', function() {{
            const realTVS = new RealTVSSystem();
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def run_server(self):
        """
        Startet integrierten Web-Server
        """
        print(f"🚀 Starte integrierten TVS Server auf Port {self.port}...")
        
        # Lade echte Daten im Hintergrund
        self.load_real_data()
        
        class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.server_instance = kwargs.pop('server_instance', None)
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    # Serve integrated HTML
                    html_content = self.server_instance.create_integrated_html()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.send_header('Content-length', len(html_content.encode('utf-8')))
                    self.end_headers()
                    self.wfile.write(html_content.encode('utf-8'))
                    
                elif self.path == '/api/refresh':
                    # API Endpoint für Daten-Refresh
                    self.server_instance.load_real_data()
                    
                    response = {{"status": "success", "message": "Daten aktualisiert"}}
                    response_json = json.dumps(response)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Content-length', len(response_json))
                    self.end_headers()
                    self.wfile.write(response_json.encode('utf-8'))
                    
                else:
                    # Standard file serving
                    super().do_GET()
        
        # Erstelle Handler mit Server-Instanz
        def handler_class(*args, **kwargs):
            return CustomHTTPRequestHandler(*args, server_instance=self, **kwargs)
        
        with socketserver.TCPServer(("", self.port), handler_class) as httpd:
            url = f"http://localhost:{self.port}"
            print(f"✅ Server läuft auf {url}")
            print("🌐 Öffne Browser...")
            
            # Öffne Browser nach kurzer Verzögerung
            def open_browser():
                time.sleep(1)
                webbrowser.open(url)
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\\n🛑 Server gestoppt")

# Main Execution
if __name__ == "__main__":
    print("🔗 TVS Dashboard Integration mit echten Daten")
    print("="*60)
    
    # Erstelle integrierten Server
    server = IntegratedTVSServer(port=8080)
    
    # Starte Server
    server.run_server()