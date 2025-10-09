#!/usr/bin/env python3
"""
Debug Script - Analysiert das Format deiner bestehenden Daten
"""

import json
from pathlib import Path

def analyze_data_structure():
    base_path = Path(".")
    
    print("🔍 Analysiere Datenstruktur...")
    print("="*50)
    
    # 1. Dashboard Data analysieren
    dashboard_file = base_path / "dashboard_data.json"
    if dashboard_file.exists():
        print("📊 Dashboard Data Structure:")
        with open(dashboard_file, 'r') as f:
            dashboard_data = json.load(f)
        
        print(f"  Type: {type(dashboard_data)}")
        print(f"  Keys: {list(dashboard_data.keys()) if isinstance(dashboard_data, dict) else 'Not a dict'}")
        
        if isinstance(dashboard_data, dict):
            for key, value in dashboard_data.items():
                print(f"    {key}: {type(value)} - {str(value)[:100]}...")
        
        print("\n  Full Content (first 500 chars):")
        print(str(dashboard_data)[:500] + "...")
    else:
        print("❌ dashboard_data.json nicht gefunden")
    
    print("\n" + "="*50)
    
    # 2. Detailed Predictions analysieren
    predictions_file = base_path / "detailed_predictions.json"
    if predictions_file.exists():
        print("📈 Detailed Predictions Structure:")
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        print(f"  Type: {type(predictions_data)}")
        print(f"  Keys: {list(predictions_data.keys()) if isinstance(predictions_data, dict) else 'Not a dict'}")
        
        if isinstance(predictions_data, dict):
            for key, value in predictions_data.items():
                print(f"    {key}: {type(value)}")
                
                if isinstance(value, list) and len(value) > 0:
                    print(f"      List length: {len(value)}")
                    print(f"      First item type: {type(value[0])}")
                    print(f"      First item: {value[0]}")
                    
                    if len(value) > 1:
                        print(f"      Second item: {value[1]}")
        
        print("\n  Full Content (first 1000 chars):")
        print(str(predictions_data)[:1000] + "...")
    else:
        print("❌ detailed_predictions.json nicht gefunden")
    
    print("\n" + "="*50)
    
    # 3. Features Enhanced analysieren (deine echte Struktur)
    enhanced_path = base_path / "features_enhanced"
    if enhanced_path.exists():
        print("📂 Features Enhanced Structure:")
        feature_files = list(enhanced_path.glob("*_enhanced.csv"))
        print(f"  Found {len(feature_files)} files:")
        
        for file in feature_files[:3]:  # Nur erste 3
            print(f"    {file.name}")
            
            try:
                import pandas as pd
                df = pd.read_csv(file)
                print(f"      Rows: {len(df)}, Columns: {len(df.columns)}")
                print(f"      Columns: {list(df.columns)[:10]}...")  # Erste 10 Spalten
                
                if 'DateTime' in df.columns:
                    print(f"      Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
                
                # Suche nach Target-Spalten
                target_cols = [col for col in df.columns if 'target' in col.lower() or 'return' in col.lower()]
                if target_cols:
                    print(f"      Target columns: {target_cols}")
                    for col in target_cols[:2]:  # Erste 2 Target-Spalten
                        print(f"        {col} sample: {df[col].head(3).tolist()}")
                
            except Exception as e:
                print(f"      Error reading: {e}")
    else:
        print("❌ features_enhanced ordner nicht gefunden")
    
    print("\n" + "="*50)
    
    # 4. Prediction Ordner analysieren
    prediction_folders = ['prophet_predictions', 'arima_predictions', 'neural_network_predictions', 'ensemble_predictions_with_NN']
    
    for folder in prediction_folders:
        folder_path = base_path / folder
        if folder_path.exists():
            print(f"📈 {folder} Structure:")
            csv_files = list(folder_path.glob("*.csv"))
            print(f"  Found {len(csv_files)} CSV files")
            
            # Analysiere erste Datei
            if csv_files:
                first_file = csv_files[0]
                print(f"    Sample file: {first_file.name}")
                
                try:
                    import pandas as pd
                    df = pd.read_csv(first_file)
                    print(f"      Rows: {len(df)}, Columns: {list(df.columns)}")
                    print(f"      Sample data:")
                    print(f"        {df.head(2).to_dict('records')}")
                except Exception as e:
                    print(f"      Error reading: {e}")
        else:
            print(f"❌ {folder} nicht gefunden")
    
    print("\n" + "="*60)
    print("🎯 Zusammenfassung:")
    print("Diese Analyse zeigt Martin's echte Datenstruktur")
    print("Basierend darauf wird das kombinierte Dashboard erstellt")

if __name__ == "__main__":
    analyze_data_structure()