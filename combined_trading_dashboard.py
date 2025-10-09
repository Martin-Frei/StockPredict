this.updateSummaryCards();
                this.updateMarketStatus();
                this.populateRecommendations();
                this.createAllCharts();
                this.createModelPerformanceCards();
            }}
            
            updateSummaryCards() {{
                const summary = this.data.summary;
                
                document.getElementById('totalPredictions').textContent = summary.total_historical_predictions.toLocaleString();
                document.getElementById('currentPredictions').textContent = summary.current_predictions.toLocaleString();
                document.getElementById('avgHitRate').textContent = summary.average_hit_rate.toFixed(1) + '%';
                document.getElementById('bestModel').textContent = summary.best_model || 'N/A';
                
                document.getElementById('lastUpdate').textContent = new Date(this.data.timestamp).toLocaleString('de-DE');
            }}
            
            updateMarketStatus() {{
                const marketStatus = this.data.trading_section.market_status;
                const statusElement = document.getElementById('marketStatus');
                
                statusElement.textContent = marketStatus === 'OPEN' ? '🟢 Markt GEÖFFNET' : '🔴 Markt GESCHLOSSEN';
                statusElement.className = `market-status ${{marketStatus === 'OPEN' ? 'market-open' : 'market-closed'}}`;
            }}
            
            populateRecommendations() {{
                const container = document.getElementById('recommendationsContainer');
                const recommendations = this.data.trading_section.recommendations;
                
                if (!recommendations || recommendations.length === 0) {{
                    container.innerHTML = '<div class="alert alert-warning">Keine Trading-Empfehlungen verfügbar</div>';
                    return;
                }}
                
                let html = '';
                
                recommendations.slice(0, 8).forEach(rec => {{
                    const actionClass = rec.direction.toLowerCase();
                    const confidenceColor = rec.confidence > 70 ? '#00ff88' : rec.confidence > 40 ? '#FFD700' : '#ff6b6b';
                    
                    html += `
                    <div class="recommendation-item rec-${{actionClass}}">
                        <div class="rec-header">
                            <div class="bank-symbol">${{rec.bank}}</div>
                            <div class="rec-action action-${{actionClass}}">${{rec.direction}}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>Gewichtete Prediction:</span>
                            <strong style="color: ${{rec.weighted_prediction >= 0 ? '#00ff88' : '#ff6b6b'}}">
                                ${{(rec.weighted_prediction * 100).toFixed(3)}}%
                            </strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>Confidence:</span>
                            <strong style="color: ${{confidenceColor}}">${{rec.confidence.toFixed(1)}}%</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Bestes Modell:</span>
                            <strong>${{rec.best_model}}</strong>
                        </div>
                    </div>`;
                }});
                
                container.innerHTML = html;
            }}
            
            createAllCharts() {{
                this.createCurrentPredictionsChart();
                this.createHourlyPredictionsChart();
                this.createHitRateOverTimeChart();
                this.createPredictedVsActualChart();
                this.createCumulativePerformanceChart();
                this.createAccuracyDistributionChart();
                this.createModelRankingsChart();
                this.createRiskReturnChart();
            }}
            
            createCurrentPredictionsChart() {{
                const ctx = document.getElementById('currentPredictionsChart').getContext('2d');
                const currentPreds = this.data.trading_section.current_predictions;
                
                // Gruppiere nach Modell
                const modelData = {{}};
                currentPreds.forEach(pred => {{
                    if (!modelData[pred.model]) {{
                        modelData[pred.model] = [];
                    }}
                    modelData[pred.model].push(pred.predicted_7h);
                }});
                
                const datasets = Object.entries(modelData).map(([model, values], index) => ({{
                    label: model,
                    data: values,
                    backgroundColor: ['#FFD700', '#FFA500', '#FF6347', '#32CD32'][index % 4],
                    borderColor: ['#FFD700', '#FFA500', '#FF6347', '#32CD32'][index % 4],
                    borderWidth: 2
                }}));
                
                charts.currentPredictions = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: Object.keys(modelData).map(() => 'Predictions'),
                        datasets: datasets
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }},
                            title: {{ display: true, text: 'Model Predictions Vergleich', color: '#fff' }}
                        }},
                        scales: {{
                            x: {{ ticks: {{ color: '#fff' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                            y: {{ 
                                ticks: {{ color: '#fff', callback: function(value) {{ return (value * 100).toFixed(2) + '%'; }} }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            createHourlyPredictionsChart() {{
                const ctx = document.getElementById('hourlyPredictionsChart').getContext('2d');
                const currentPreds = this.data.trading_section.current_predictions;
                
                // Nimm erste Bank mit hourly predictions
                const samplePred = currentPreds.find(p => p.hourly_predictions && p.hourly_predictions.length > 0);
                
                if (!samplePred) {{
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#fff';
                    ctx.fillText('Keine stündlichen Predictions verfügbar', 50, 100);
                    return;
                }}
                
                const hours = samplePred.hourly_predictions.map((_, i) => `+${{i+1}}h`);
                
                // Erstelle Datasets für alle Modelle dieser Bank
                const bankPreds = currentPreds.filter(p => p.bank === samplePred.bank);
                const datasets = bankPreds.map((pred, index) => ({{
                    label: `${{pred.model}} (${{pred.bank}})`,
                    data: pred.hourly_predictions,
                    borderColor: ['#FFD700', '#FFA500', '#FF6347', '#32CD32'][index % 4],
                    backgroundColor: ['#FFD700', '#FFA500', '#FF6347', '#32CD32'][index % 4] + '20',
                    tension: 0.4,
                    fill: false
                }}));
                
                charts.hourlyPredictions = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: hours,
                        datasets: datasets
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }}
                        }},
                        scales: {{
                            x: {{ ticks: {{ color: '#fff' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                            y: {{ 
                                ticks: {{ color: '#fff', callback: function(value) {{ return (value * 100).toFixed(3) + '%'; }} }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            createHitRateOverTimeChart() {{
                const ctx = document.getElementById('hitRateOverTimeChart').getContext('2d');
                const matchedData = this.data.validation_section.matched_predictions;
                
                if (!matchedData || matchedData.length === 0) {{
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#fff';
                    ctx.fillText('Keine historischen Daten verfügbar', 50, 100);
                    return;
                }}
                
                // Gruppiere nach Datum und berechne Hit Rate
                const dateHitRates = {{}};
                matchedData.forEach(item => {{
                    if (!dateHitRates[item.date]) {{
                        dateHitRates[item.date] = {{ correct: 0, total: 0 }};
                    }}
                    dateHitRates[item.date].total++;
                    if (item.prediction_correct) {{
                        dateHitRates[item.date].correct++;
                    }}
                }});
                
                const dates = Object.keys(dateHitRates).sort();
                const hitRates = dates.map(date => {{
                    const data = dateHitRates[date];
                    return (data.correct / data.total) * 100;
                }});
                
                charts.hitRateOverTime = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: dates.map(date => new Date(date).toLocaleDateString('de-DE')),
                        datasets: [{{
                            label: 'Hit Rate (%)',
                            data: hitRates,
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            tension: 0.4,
                            fill: true
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }}
                        }},
                        scales: {{
                            x: {{ ticks: {{ color: '#fff' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                            y: {{ 
                                min: 0, max: 100,
                                ticks: {{ color: '#fff', callback: function(value) {{ return value + '%'; }} }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            createPredictedVsActualChart() {{
                const ctx = document.getElementById('predictedVsActualChart').getContext('2d');
                const matchedData = this.data.validation_section.matched_predictions;
                
                if (!matchedData || matchedData.length === 0) {{
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#fff';
                    ctx.fillText('Keine Validation-Daten verfügbar', 50, 100);
                    return;
                }}
                
                // Erstelle Scatter Plot: Predicted vs Actual
                const scatterData = matchedData.map(item => ({{
                    x: item.predicted_7h * 100,  // Convert to percentage
                    y: item.actual_7h * 100,     // Convert to percentage
                    model: item.model,
                    bank: item.bank
                }}));
                
                // Gruppiere nach Modell
                const modelColors = {{ 'Prophet': '#FFD700', 'ARIMA': '#FFA500', 'Neural_Network': '#FF6347', 'Ensemble': '#32CD32' }};
                const datasets = Object.keys(modelColors).map(model => {{
                    const modelData = scatterData.filter(item => item.model === model);
                    return {{
                        label: model,
                        data: modelData,
                        backgroundColor: modelColors[model],
                        borderColor: modelColors[model],
                        pointRadius: 4
                    }};
                }});
                
                charts.predictedVsActual = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{ datasets }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const point = context.parsed;
                                        return `${{context.dataset.label}}: Pred ${{point.x.toFixed(3)}}%, Actual ${{point.y.toFixed(3)}}%`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{ 
                                title: {{ display: true, text: 'Predicted Return (%)', color: '#fff' }},
                                ticks: {{ color: '#fff' }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            y: {{ 
                                title: {{ display: true, text: 'Actual Return (%)', color: '#fff' }},
                                ticks: {{ color: '#fff' }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            createCumulativePerformanceChart() {{
                const ctx = document.getElementById('cumulativePerformanceChart').getContext('2d');
                const matchedData = this.data.validation_section.matched_predictions;
                
                if (!matchedData || matchedData.length === 0) {{
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#fff';
                    ctx.fillText('Keine Performance-Daten verfügbar', 50, 100);
                    return;
                }}
                
                // Gruppiere nach Modell und berechne kumulative Returns
                const models = [...new Set(matchedData.map(item => item.model))];
                const modelColors = {{ 'Prophet': '#FFD700', 'ARIMA': '#FFA500', 'Neural_Network': '#FF6347', 'Ensemble': '#32CD32' }};
                
                const datasets = models.map(model => {{
                    const modelData = matchedData.filter(item => item.model === model).sort((a, b) => new Date(a.date) - new Date(b.date));
                    
                    let cumulative = 0;
                    const cumulativeReturns = modelData.map(item => {{
                        cumulative += item.actual_7h;
                        return cumulative * 100; // Convert to percentage
                    }});
                    
                    return {{
                        label: model,
                        data: cumulativeReturns,
                        borderColor: modelColors[model] || '#87CEEB',
                        backgroundColor: (modelColors[model] || '#87CEEB') + '20',
                        tension: 0.4,
                        fill: false
                    }};
                }});
                
                // Use dates from first model for x-axis
                const dates = matchedData.filter(item => item.model === models[0])
                    .sort((a, b) => new Date(a.date) - new Date(b.date))
                    .map(item => new Date(item.date).toLocaleDateString('de-DE'));
                
                charts.cumulativePerformance = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: dates,
                        datasets: datasets
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }}
                        }},
                        scales: {{
                            x: {{ ticks: {{ color: '#fff' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                            y: {{ 
                                ticks: {{ color: '#fff', callback: function(value) {{ return value.toFixed(2) + '%'; }} }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            createAccuracyDistributionChart() {{
                const ctx = document.getElementById('accuracyDistributionChart').getContext('2d');
                const performance = this.data.validation_section.performance_metrics;
                
                if (!performance || Object.keys(performance).length === 0) {{
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#fff';
                    ctx.fillText('Keine Performance-Metriken verfügbar', 50, 100);
                    return;
                }}
                
                const models = Object.keys(performance);
                const hitRates = models.map(model => performance[model].hit_rate);
                const mapes = models.map(model => performance[model].mape);
                
                charts.accuracyDistribution = new Chart(ctx, {{
                    type: 'radar',
                    data: {{
                        labels: models,
                        datasets: [
                            {{
                                label: 'Hit Rate (%)',
                                data: hitRates,
                                borderColor: '#00ff88',
                                backgroundColor: 'rgba(0, 255, 136, 0.2)',
                                pointBackgroundColor: '#00ff88'
                            }},
                            {{
                                label: 'Accuracy (100-MAPE)',
                                data: mapes.map(mape => 100 - mape),
                                borderColor: '#FFD700',
                                backgroundColor: 'rgba(255, 215, 0, 0.2)',
                                pointBackgroundColor: '#FFD700'
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }}
                        }},
                        scales: {{
                            r: {{
                                min: 0,
                                max: 100,
                                ticks: {{ color: '#fff' }},
                                grid: {{ color: 'rgba(255,255,255,0.2)' }},
                                pointLabels: {{ color: '#fff' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            createModelRankingsChart() {{
                const ctx = document.getElementById('modelRankingsChart').getContext('2d');
                const performance = this.data.validation_section.performance_metrics;
                
                if (!performance || Object.keys(performance).length === 0) {{
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#fff';
                    ctx.fillText('Keine Performance-Daten verfügbar', 50, 100);
                    return;
                }}
                
                // Sortiere nach Accuracy Score
                const sortedModels = Object.entries(performance)
                    .sort(([,a], [,b]) => b.accuracy_score - a.accuracy_score);
                
                const models = sortedModels.map(([model]) => model);
                const accuracyScores = sortedModels.map(([,data]) => data.accuracy_score);
                
                charts.modelRankings = new Chart(ctx, {{
                    type: 'horizontalBar',
                    data: {{
                        labels: models,
                        datasets: [{{
                            label: 'Accuracy Score',
                            data: accuracyScores,
                            backgroundColor: ['#FFD700', '#C0C0C0', '#CD7F32', '#87CEEB'],
                            borderColor: ['#FFD700', '#C0C0C0', '#CD7F32', '#87CEEB'],
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }}
                        }},
                        scales: {{
                            x: {{ ticks: {{ color: '#fff' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                            y: {{ ticks: {{ color: '#fff' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }}
                        }}
                    }}
                }});
            }}
            
            createRiskReturnChart() {{
                const ctx = document.getElementById('riskReturnChart').getContext('2d');
                const performance = this.data.validation_section.performance_metrics;
                
                if (!performance || Object.keys(performance).length === 0) {{
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#fff';
                    ctx.fillText('Keine Risk-Return-Daten verfügbar', 50, 100);
                    return;
                }}
                
                const scatterData = Object.entries(performance).map(([model, data]) => ({{
                    x: data.volatility_actual * 100,  // Risk (Volatility)
                    y: data.avg_actual * 100,         // Return
                    label: model
                }}));
                
                charts.riskReturn = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{
                        datasets: [{{
                            label: 'Models',
                            data: scatterData,
                            backgroundColor: '#FFD700',
                            borderColor: '#FFA500',
                            pointRadius: 8
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{ labels: {{ color: '#fff' }} }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const point = scatterData[context.dataIndex];
                                        return `${{point.label}}: Risk ${{point.x.toFixed(3)}}%, Return ${{point.y.toFixed(3)}}%`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{ 
                                title: {{ display: true, text: 'Risk (Volatility %)', color: '#fff' }},
                                ticks: {{ color: '#fff' }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            y: {{ 
                                title: {{ display: true, text: 'Return (%)', color: '#fff' }},
                                ticks: {{ color: '#fff' }}, 
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }}
            
            createModelPerformanceCards() {{
                const container = document.getElementById('modelPerformanceGrid');
                const performance = this.data.validation_section.performance_metrics;
                
                if (!performance || Object.keys(performance).length === 0) {{
                    container.innerHTML = '<div class="alert alert-warning">Keine Model-Performance-Daten verfügbar</div>';
                    return;
                }}
                
                // Sortiere nach Accuracy Score
                const sortedModels = Object.entries(performance)
                    .sort(([,a], [,b]) => b.accuracy_score - a.accuracy_score);
                
                let html = '';
                
                sortedModels.forEach(([model, data], index) => {{
                    const rankClass = index === 0 ? 'rank-1' : index === 1 ? 'rank-2' : index === 2 ? 'rank-3' : 'rank-other';
                    
                    // Neural Network Warning
                    const isNeuralNetwork = model.includes('Neural');
                    const warningHtml = isNeuralNetwork && Math.abs(data.avg_predicted) > 0.1 ? 
                        '<div class="alert alert-danger">⚠️ Möglicherweise Overfitting - sehr hohe Predictions!</div>' : '';
                    
                    html += `
                    <div class="model-performance-card">
                        <div class="model-header">
                            <div class="model-name">${{model}}</div>
                            <div class="model-rank ${{rankClass}}">#${{index + 1}}</div>
                        </div>
                        ${{warningHtml}}
                        <div class="metric-row">
                            <span>Hit Rate:</span>
                            <strong>${{data.hit_rate.toFixed(1)}}%</strong>
                        </div>
                        <div class="metric-row">
                            <span>Accuracy Score:</span>
                            <strong>${{data.accuracy_score.toFixed(1)}}</strong>
                        </div>
                        <div class="metric-row">
                            <span>MAPE:</span>
                            <strong>${{data.mape.toFixed(2)}}%</strong>
                        </div>
                        <div class="metric-row">
                            <span>Avg Predicted:</span>
                            <strong style="color: ${{data.avg_predicted >= 0 ? '#00ff88' : '#ff6b6b'}}">${{{(data.avg_predicted * 100).toFixed(3)}}%</strong>
                        </div>
                        <div class="metric-row">
                            <span>Avg Actual:</span>
                            <strong style="color: ${{data.avg_actual >= 0 ? '#00ff88' : '#ff6b6b'}}">${{{(data.avg_actual * 100).toFixed(3)}}%</strong>
                        </div>
                        <div class="metric-row">
                            <span>Prediction Bias:</span>
                            <strong style="color: ${{Math.abs(data.prediction_bias) < 0.001 ? '#00ff88' : '#ff6b6b'}}">${{{(data.prediction_bias * 100).toFixed(3)}}%</strong>
                        </div>
                        <div class="metric-row">
                            <span>Sharpe Ratio:</span>
                            <strong>${{data.sharpe_ratio.toFixed(3)}}</strong>
                        </div>
                        <div class="metric-row">
                            <span>Total Predictions:</span>
                            <strong>${{data.total_predictions}}</strong>
                        </div>
                    </div>`;
                }});
                
                container.innerHTML = html;
            }}
        }}
        
        // Tab Management
        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Remove active class from all buttons
            document.querySelectorAll('.tab-button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }}
        
        // Initialize Dashboard
        document.addEventListener('DOMContentLoaded', function() {{
            new CombinedDashboard();
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def run_server(self):
        """
        Startet den Combined Dashboard Server
        """
        print("🚀 Starte Combined Trading & Validation Dashboard Server...")
        
        try:
            # Erstelle Dashboard Data
            dashboard_data = self.create_dashboard_data()
        except Exception as e:
            print(f"❌ Fehler beim Erstellen der Dashboard-Daten: {e}")
            import traceback
            traceback.print_exc()
            return
        
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.dashboard_system = kwargs.pop('dashboard_system', None)
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    try:
                        html_content = self.dashboard_system.create_html_dashboard(dashboard_data)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html; charset=utf-8')
                        self.send_header('Content-length', len(html_content.encode('utf-8')))
                        self.end_headers()
                        self.wfile.write(html_content.encode('utf-8'))
                    except Exception as e:
                        error_html = f"<html><body><h1>Dashboard Error</h1><p>{str(e)}</p></body></html>"
                        self.send_response(500)
                        self.send_header('Content-type', 'text/html')
                        self.send_header('Content-length', len(error_html))
                        self.end_headers()
                        self.wfile.write(error_html.encode('utf-8'))
                        
                elif self.path == '/api/data':
                    # Data API
                    response_json = json.dumps(dashboard_data, default=str)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Content-length', len(response_json))
                    self.end_headers()
                    self.wfile.write(response_json.encode('utf-8'))
                    
                else:
                    super().do_GET()
        
        def handler_with_system(*args, **kwargs):
            return CustomHandler(*args, dashboard_system=self, **kwargs)
        
        port = 8080
        try:
            with socketserver.TCPServer(("", port), handler_with_system) as httpd:
                url = f"http://localhost:{port}"
                print(f"✅ Combined Dashboard Server läuft auf {url}")
                print("🌐 Öffne Browser...")
                
                def open_browser():
                    time.sleep(2)
                    webbrowser.open(url)
                
                threading.Thread(target=open_browser, daemon=True).start()
                
                print("\\n🎯 Dashboard Status:")
                print(f"  📊 Historische Predictions: {dashboard_data['summary']['total_historical_predictions']}")
                print(f"  📈 Aktuelle Predictions: {dashboard_data['summary']['current_predictions']}")
                print(f"  🎯 Durchschnittliche Hit Rate: {dashboard_data['summary']['average_hit_rate']:.1f}%")
                print(f"  🏆 Bestes Modell: {dashboard_data['summary']['best_model']}")
                print("\\n⌨️  Drücke Ctrl+C zum Beenden")
                
                httpd.serve_forever()
                
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"❌ Port {port} bereits in Verwendung - versuche Port 8081...")
                port = 8081
                with socketserver.TCPServer(("", port), handler_with_system) as httpd:
                    url = f"http://localhost:{port}"
                    print(f"✅ Combined Dashboard Server läuft auf {url}")
                    webbrowser.open(url)
                    httpd.serve_forever()
            else:
                raise e
                
        except KeyboardInterrupt:
            print("\\n🛑 Combined Dashboard Server gestoppt")
            print("👋 Auf Wiedersehen!")

# Main Execution
if __name__ == "__main__":
    print("🚀 Combined Trading & Validation Dashboard")
    print("="*60)
    print("📊 Trading: Aktuelle Predictions für heute")
    print("📈 Validation: Historische Performance seit 17.07.2025")
    print("🔍 Comparison: Model-Vergleich und Rankings")
    print("="*60)
    
    try:
        # Erstelle Combined Dashboard System
        dashboard_system = CombinedTradingDashboard()
        
        # Starte Server
        dashboard_system.run_server()
        
    except Exception as e:
        print(f"❌ Kritischer Fehler: {e}")
        print("🔧 Debug-Informationen:")
        
        import traceback
        traceback.print_exc()
        
        print("\\n💡 Lösungsvorschläge:")
        print("1. Prüfe ob alle Datenordner existieren:")
        print("   - features_enhanced/")
        print("   - prophet_predictions/")
        print("   - arima_predictions/")
        print("   - neural_network_predictions/")
        print("   - ensemble_predictions_with_NN/")
        print("2. Prüfe ob dashboard_data.json und detailed_predictions.json existieren")
        print("3. Prüfe Dateiberechtigungen")
        print("4. Führe zuerst deine Master Pipeline aus um aktuelle Daten zu generieren")
#!/usr/bin/env python3
"""
Combined Trading & Validation Dashboard
- Trading: Aktuelle Predictions für heute
- Validation: Historische Performance seit 17.07.2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import http.server
import socketserver
import webbrowser
import threading
import time

class CombinedTradingDashboard:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        
        # Martin's echte Pfade
        self.dashboard_data_file = self.base_path / "dashboard_data.json"
        self.detailed_predictions_file = self.base_path / "detailed_predictions.json"
        self.features_enhanced_path = self.base_path / "features_enhanced"
        
        # Prediction Ordner
        self.prediction_paths = {
            'Prophet': self.base_path / "prophet_predictions",
            'ARIMA': self.base_path / "arima_predictions", 
            'Neural_Network': self.base_path / "neural_network_predictions",
            'Ensemble': self.base_path / "ensemble_predictions_with_NN"
        }
        
        print("🚀 Combined Trading & Validation Dashboard")
        print(f"📂 Base Path: {self.base_path.absolute()}")
        
    def load_current_predictions(self):
        """
        Lädt aktuelle Predictions aus detailed_predictions.json
        """
        print("📊 Lade aktuelle Trading-Predictions...")
        
        try:
            with open(self.detailed_predictions_file, 'r') as f:
                data = json.load(f)
            
            current_predictions = []
            models = data.get('models', [])
            banks = data.get('banks', [])
            predictions_data = data.get('predictions', {})
            
            for model in models:
                if model in predictions_data:
                    for bank in banks:
                        if bank in predictions_data[model]:
                            pred_data = predictions_data[model][bank]
                            
                            current_pred = {
                                'date': datetime.now().strftime("%Y-%m-%d"),
                                'time': datetime.now().strftime("%H:%M"),
                                'bank': bank,
                                'model': model,
                                'predicted_7h': pred_data.get('totalReturn', 0.0),
                                'direction': pred_data.get('direction', 'Unknown'),
                                'confidence': pred_data.get('confidence', 0.0),
                                'bullish_hours': pred_data.get('bullishHours', 0),
                                'hourly_predictions': pred_data.get('hourlyPredictions', [])
                            }
                            current_predictions.append(current_pred)
            
            print(f"  ✅ {len(current_predictions)} aktuelle Predictions geladen")
            return current_predictions
            
        except Exception as e:
            print(f"  ❌ Fehler beim Laden aktueller Predictions: {e}")
            return []
    
    def load_historical_actual_returns(self):
        """
        Lädt historische tatsächliche Returns aus features_enhanced
        """
        print("📈 Lade historische tatsächliche Returns...")
        
        historical_actuals = {}
        
        # Lade für jede Bank die Enhanced Features
        for enhanced_file in self.features_enhanced_path.glob("*_enhanced.csv"):
            try:
                bank_name = enhanced_file.stem.replace('_enhanced', '').upper()
                df = pd.read_csv(enhanced_file)
                
                if 'DateTime' in df.columns and 'Target_7h' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    
                    # Konvertiere zu Dictionary für schnelle Lookups
                    historical_actuals[bank_name] = {}
                    for _, row in df.iterrows():
                        date_str = row['DateTime'].strftime("%Y-%m-%d")
                        hour_str = row['DateTime'].strftime("%H:%M")
                        
                        if date_str not in historical_actuals[bank_name]:
                            historical_actuals[bank_name][date_str] = {}
                        
                        historical_actuals[bank_name][date_str][hour_str] = {
                            'target_7h': row['Target_7h'],
                            'returns': row.get('Returns', 0),
                            'datetime': row['DateTime']
                        }
                    
                    print(f"  ✅ {bank_name}: {len(df)} historische Punkte")
                
            except Exception as e:
                print(f"  ❌ Fehler bei {enhanced_file.name}: {e}")
                continue
        
        print(f"📈 Historische Daten für {len(historical_actuals)} Banken geladen")
        return historical_actuals
    
    def load_historical_predictions(self):
        """
        Lädt historische Predictions aus allen Model-Ordnern
        """
        print("🔍 Lade historische Predictions aus Model-Ordnern...")
        
        historical_predictions = []
        
        for model_name, model_path in self.prediction_paths.items():
            if not model_path.exists():
                print(f"  ⚠️ {model_name}: Ordner nicht gefunden")
                continue
            
            model_predictions = 0
            
            # Lade alle Bank-CSV-Dateien für dieses Modell
            for pred_file in model_path.glob("*_7h_*.csv"):
                try:
                    # Extrahiere Bank-Name aus Dateiname
                    bank_name = pred_file.stem.split('_')[0].upper()
                    
                    df = pd.read_csv(pred_file)
                    
                    if 'DateTime' in df.columns and 'Predicted_Return_7h' in df.columns:
                        df['DateTime'] = pd.to_datetime(df['DateTime'])
                        
                        for _, row in df.iterrows():
                            hist_pred = {
                                'date': row['DateTime'].strftime("%Y-%m-%d"),
                                'time': row['DateTime'].strftime("%H:%M"),
                                'bank': bank_name,
                                'model': model_name,
                                'predicted_7h': row['Predicted_Return_7h'],
                                'direction': 'Bullish' if row.get('Direction', 1) > 0 else 'Bearish',
                                'confidence': row.get('Confidence', 0),
                                'confidence_score': row.get('Confidence_Score', 0),
                                'datetime': row['DateTime']
                            }
                            historical_predictions.append(hist_pred)
                            model_predictions += 1
                
                except Exception as e:
                    print(f"    ❌ {pred_file.name}: {e}")
                    continue
            
            print(f"  ✅ {model_name}: {model_predictions} Predictions")
        
        print(f"🔍 Total historische Predictions: {len(historical_predictions)}")
        return historical_predictions
    
    def match_predictions_with_actuals(self, historical_predictions, historical_actuals):
        """
        Matcht historische Predictions mit tatsächlichen Returns
        """
        print("🔗 Matche Predictions mit tatsächlichen Returns...")
        
        matched_data = []
        matches = 0
        
        for pred in historical_predictions:
            bank = pred['bank']
            date = pred['date']
            time = pred['time']
            
            if bank in historical_actuals:
                if date in historical_actuals[bank]:
                    # Suche exakte Zeit oder nächstgelegene Zeit
                    if time in historical_actuals[bank][date]:
                        actual_data = historical_actuals[bank][date][time]
                        
                        matched_entry = {
                            **pred,
                            'actual_7h': actual_data['target_7h'],
                            'actual_returns': actual_data['returns'],
                            'prediction_correct': (
                                (pred['direction'] == 'Bullish' and actual_data['target_7h'] > 0) or
                                (pred['direction'] == 'Bearish' and actual_data['target_7h'] < 0)
                            ),
                            'prediction_error': abs(pred['predicted_7h'] - actual_data['target_7h'])
                        }
                        matched_data.append(matched_entry)
                        matches += 1
        
        print(f"🔗 {matches} Predictions erfolgreich gematcht")
        return matched_data
    
    def calculate_performance_metrics(self, matched_data):
        """
        Berechnet Performance-Metriken pro Modell
        """
        print("📊 Berechne Performance-Metriken...")
        
        models = set(item['model'] for item in matched_data)
        performance_metrics = {}
        
        for model in models:
            model_data = [item for item in matched_data if item['model'] == model]
            
            if not model_data:
                continue
            
            # Hit Rate (Richtung korrekt)
            correct_predictions = sum(1 for item in model_data if item['prediction_correct'])
            hit_rate = (correct_predictions / len(model_data)) * 100
            
            # Return Accuracy
            predicted_returns = [item['predicted_7h'] for item in model_data]
            actual_returns = [item['actual_7h'] for item in model_data]
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean([abs((p - a) / (abs(a) + 1e-8)) for p, a in zip(predicted_returns, actual_returns)]) * 100
            
            # Total Returns
            total_predicted = sum(predicted_returns)
            total_actual = sum(actual_returns)
            
            # Average Returns
            avg_predicted = np.mean(predicted_returns)
            avg_actual = np.mean(actual_returns)
            
            # Volatility
            volatility_predicted = np.std(predicted_returns) if len(predicted_returns) > 1 else 0
            volatility_actual = np.std(actual_returns) if len(actual_returns) > 1 else 0
            
            # Sharpe Ratio (simplified)
            sharpe_ratio = (avg_actual / volatility_actual * np.sqrt(252)) if volatility_actual > 0 else 0
            
            # Accuracy Score (composite)
            accuracy_score = (hit_rate * 0.7) + ((100 - min(mape, 100)) * 0.3)
            
            performance_metrics[model] = {
                'total_predictions': len(model_data),
                'hit_rate': hit_rate,
                'mape': mape,
                'total_predicted': total_predicted,
                'total_actual': total_actual,
                'avg_predicted': avg_predicted,
                'avg_actual': avg_actual,
                'volatility_predicted': volatility_predicted,
                'volatility_actual': volatility_actual,
                'sharpe_ratio': sharpe_ratio,
                'accuracy_score': accuracy_score,
                'prediction_bias': avg_predicted - avg_actual
            }
            
            print(f"  📊 {model}: {hit_rate:.1f}% Hit Rate, {accuracy_score:.1f} Accuracy Score")
        
        return performance_metrics
    
    def generate_trading_recommendations(self, current_predictions, performance_metrics):
        """
        Generiert Trading-Empfehlungen basierend auf aktuellen Predictions und historischer Performance
        """
        print("💡 Generiere Trading-Empfehlungen...")
        
        # Sortiere Modelle nach Accuracy Score
        model_rankings = sorted(performance_metrics.items(), 
                               key=lambda x: x[1]['accuracy_score'], 
                               reverse=True)
        
        best_model = model_rankings[0][0] if model_rankings else None
        
        recommendations = []
        
        # Bank-spezifische Empfehlungen
        banks = set(pred['bank'] for pred in current_predictions)
        
        for bank in banks:
            bank_predictions = [p for p in current_predictions if p['bank'] == bank]
            
            # Gewichtete Prediction basierend auf Model-Performance
            weighted_prediction = 0
            total_weight = 0
            
            for pred in bank_predictions:
                model = pred['model']
                if model in performance_metrics:
                    weight = performance_metrics[model]['accuracy_score'] / 100
                    weighted_prediction += pred['predicted_7h'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_prediction /= total_weight
                
                recommendation = {
                    'bank': bank,
                    'weighted_prediction': weighted_prediction,
                    'direction': 'BUY' if weighted_prediction > 0.001 else 'HOLD' if weighted_prediction > -0.001 else 'SELL',
                    'confidence': min(abs(weighted_prediction) * 1000, 100),
                    'best_model': best_model,
                    'individual_predictions': {p['model']: p['predicted_7h'] for p in bank_predictions}
                }
                recommendations.append(recommendation)
        
        # Sortiere nach Confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"💡 {len(recommendations)} Trading-Empfehlungen generiert")
        return recommendations
    
    def create_dashboard_data(self):
        """
        Erstellt alle Dashboard-Daten
        """
        print("🚀 Erstelle Combined Dashboard Data...")
        
        # 1. Lade aktuelle Trading-Predictions
        current_predictions = self.load_current_predictions()
        
        # 2. Lade historische tatsächliche Returns
        historical_actuals = self.load_historical_actual_returns()
        
        # 3. Lade historische Predictions
        historical_predictions = self.load_historical_predictions()
        
        # 4. Matche Predictions mit Actuals
        matched_data = self.match_predictions_with_actuals(historical_predictions, historical_actuals)
        
        # 5. Berechne Performance-Metriken
        performance_metrics = self.calculate_performance_metrics(matched_data)
        
        # 6. Generiere Trading-Empfehlungen
        trading_recommendations = self.generate_trading_recommendations(current_predictions, performance_metrics)
        
        # 7. Berechne Dashboard-Summary
        total_predictions = len(matched_data)
        avg_hit_rate = np.mean([m['hit_rate'] for m in performance_metrics.values()]) if performance_metrics else 0
        best_model = max(performance_metrics.items(), key=lambda x: x[1]['accuracy_score'])[0] if performance_metrics else None
        worst_model = min(performance_metrics.items(), key=lambda x: x[1]['accuracy_score'])[0] if performance_metrics else None
        
        # 8. Erstelle finales Dashboard-Data-Objekt
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'trading_section': {
                'current_predictions': current_predictions,
                'recommendations': trading_recommendations,
                'market_status': 'OPEN' if 9 <= datetime.now().hour <= 16 else 'CLOSED'
            },
            'validation_section': {
                'matched_predictions': matched_data,
                'performance_metrics': performance_metrics,
                'date_range': {
                    'start': min(item['date'] for item in matched_data) if matched_data else None,
                    'end': max(item['date'] for item in matched_data) if matched_data else None
                }
            },
            'summary': {
                'total_historical_predictions': total_predictions,
                'current_predictions': len(current_predictions),
                'average_hit_rate': avg_hit_rate,
                'best_model': best_model,
                'worst_model': worst_model,
                'models_count': len(performance_metrics),
                'banks_count': len(set(pred['bank'] for pred in current_predictions))
            }
        }
        
        print("🎯 Combined Dashboard Data erstellt:")
        print(f"  📊 Historische Predictions: {total_predictions}")
        print(f"  📈 Aktuelle Predictions: {len(current_predictions)}")
        print(f"  🎯 Durchschnittliche Hit Rate: {avg_hit_rate:.1f}%")
        print(f"  🏆 Bestes Modell: {best_model}")
        print(f"  📉 Schlechtestes Modell: {worst_model}")
        
        return dashboard_data
    
    def create_html_dashboard(self, dashboard_data):
        """
        Erstellt das HTML Dashboard
        """
        dashboard_data_js = json.dumps(dashboard_data, indent=2, default=str)
        
        html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Trading & Validation Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh; color: #fff; padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        
        .header {{
            text-align: center; background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .header h1 {{
            font-size: 3rem; margin-bottom: 10px;
            background: linear-gradient(45deg, #FFD700, #FFA500);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .header .subtitle {{ font-size: 1.2rem; opacity: 0.9; margin-bottom: 20px; }}
        
        .market-status {{
            display: inline-block; padding: 10px 20px; border-radius: 25px;
            font-weight: bold; margin-top: 10px;
        }}
        .market-open {{ background: rgba(0, 255, 136, 0.3); border: 1px solid #00ff88; }}
        .market-closed {{ background: rgba(255, 107, 107, 0.3); border: 1px solid #ff6b6b; }}
        
        .dashboard-tabs {{
            display: flex; justify-content: center; gap: 20px; margin-bottom: 30px;
        }}
        .tab-button {{
            background: rgba(255,255,255,0.1); border: none; color: white;
            padding: 15px 30px; border-radius: 25px; cursor: pointer;
            font-size: 1.1rem; font-weight: bold; transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .tab-button.active {{
            background: linear-gradient(45deg, #FFD700, #FFA500);
            color: #000; transform: translateY(-2px);
        }}
        .tab-button:hover {{
            background: rgba(255,255,255,0.2); transform: translateY(-1px);
        }}
        
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .summary-cards {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 25px; text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }}
        .summary-card:hover {{ transform: translateY(-5px); }}
        .card-value {{ font-size: 2.5rem; font-weight: bold; margin-bottom: 10px; }}
        .card-label {{ font-size: 1rem; opacity: 0.8; }}
        
        .trading-grid {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;
        }}
        .validation-grid {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;
        }}
        
        .chart-container {{
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 25px; border: 1px solid rgba(255,255,255,0.2);
        }}
        .chart-title {{ font-size: 1.4rem; margin-bottom: 20px; color: #FFD700; text-align: center; }}
        
        .recommendations-container {{
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 25px; border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .recommendation-item {{
            background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px;
            margin-bottom: 15px; border-left: 4px solid;
        }}
        .rec-buy {{ border-left-color: #00ff88; }}
        .rec-hold {{ border-left-color: #FFD700; }}
        .rec-sell {{ border-left-color: #ff6b6b; }}
        
        .rec-header {{
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;
        }}
        .bank-symbol {{ font-size: 1.3rem; font-weight: bold; }}
        .rec-action {{
            padding: 5px 15px; border-radius: 15px; font-weight: bold;
            font-size: 0.9rem;
        }}
        .action-buy {{ background: #00ff88; color: black; }}
        .action-hold {{ background: #FFD700; color: black; }}
        .action-sell {{ background: #ff6b6b; color: white; }}
        
        .performance-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }}
        .model-performance-card {{
            background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .model-header {{
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;
        }}
        .model-name {{ font-size: 1.3rem; font-weight: bold; }}
        .model-rank {{
            padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9rem;
        }}
        .rank-1 {{ background: #FFD700; color: black; }}
        .rank-2 {{ background: #C0C0C0; color: black; }}
        .rank-3 {{ background: #CD7F32; color: black; }}
        .rank-other {{ background: #87CEEB; color: black; }}
        
        .metric-row {{
            display: flex; justify-content: space-between; padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .alert {{
            padding: 15px; border-radius: 10px; margin: 15px 0; font-weight: bold;
        }}
        .alert-warning {{ background: rgba(255, 193, 7, 0.2); border: 1px solid #ffc107; }}
        .alert-danger {{ background: rgba(220, 53, 69, 0.2); border: 1px solid #dc3545; }}
        
        @media (max-width: 768px) {{
            .trading-grid, .validation-grid {{ grid-template-columns: 1fr; }}
            .dashboard-tabs {{ flex-direction: column; align-items: center; }}
            .summary-cards {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Combined Trading & Validation Dashboard</h1>
            <div class="subtitle">Live Trading Signals & Historical Performance Validation</div>
            <div id="marketStatus" class="market-status">Lade Status...</div>
            <div style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                Letzte Aktualisierung: <span id="lastUpdate">--</span>
            </div>
        </div>
        
        <div class="dashboard-tabs">
            <button class="tab-button active" onclick="showTab('trading')">
                📈 Trading Dashboard
            </button>
            <button class="tab-button" onclick="showTab('validation')">
                📊 Performance Validation
            </button>
            <button class="tab-button" onclick="showTab('comparison')">
                🔍 Model Comparison
            </button>
        </div>
        
        <div class="summary-cards">
            <div class="summary-card">
                <div class="card-value" id="totalPredictions">0</div>
                <div class="card-label">Historische Predictions</div>
            </div>
            <div class="summary-card">
                <div class="card-value" id="currentPredictions">0</div>
                <div class="card-label">Aktuelle Predictions</div>
            </div>
            <div class="summary-card">
                <div class="card-value" id="avgHitRate">0%</div>
                <div class="card-label">Durchschnittliche Hit Rate</div>
            </div>
            <div class="summary-card">
                <div class="card-value" id="bestModel">--</div>
                <div class="card-label">Bestes Modell</div>
            </div>
        </div>
        
        <!-- Trading Tab -->
        <div id="trading" class="tab-content active">
            <div class="trading-grid">
                <div class="chart-container">
                    <div class="chart-title">🎯 Heutige Trading Empfehlungen</div>
                    <div id="recommendationsContainer">
                        <!-- Recommendations will be populated here -->
                    </div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">📊 Aktuelle Model Predictions</div>
                    <canvas id="currentPredictionsChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">⏱️ Stündliche Predictions (Nächste 7h)</div>
                <canvas id="hourlyPredictionsChart"></canvas>
            </div>
        </div>
        
        <!-- Validation Tab -->
        <div id="validation" class="tab-content">
            <div class="validation-grid">
                <div class="chart-container">
                    <div class="chart-title">🎯 Hit Rate Over Time</div>
                    <canvas id="hitRateOverTimeChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">📈 Predicted vs Actual Returns</div>
                    <canvas id="predictedVsActualChart"></canvas>
                </div>
            </div>
            <div class="validation-grid">
                <div class="chart-container">
                    <div class="chart-title">📊 Cumulative Performance</div>
                    <canvas id="cumulativePerformanceChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">⚖️ Prediction Accuracy Distribution</div>
                    <canvas id="accuracyDistributionChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Comparison Tab -->
        <div id="comparison" class="tab-content">
            <div class="performance-grid" id="modelPerformanceGrid">
                <!-- Model performance cards will be populated here -->
            </div>
            <div class="validation-grid">
                <div class="chart-container">
                    <div class="chart-title">🏆 Model Rankings</div>
                    <canvas id="modelRankingsChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">⚡ Risk vs Return Analysis</div>
                    <canvas id="riskReturnChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard Data
        const DASHBOARD_DATA = {dashboard_data_js};
        
        // Global Chart References
        let charts = {{}};
        
        class CombinedDashboard {{
            constructor() {{
                this.data = DASHBOARD_DATA;
                this.currentTab = 'trading';
                this.initialize();
            }}
            
            initialize() {{
                console.log('🚀 Combined Dashboard initialized');
                console.log('📊 Data:', this.data.summary);
                
                this.update