import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def enhanced_feature_engineering():
    """Enhanced Feature-Engineering Pipeline mit allen Makro-Features"""
    
    print("ENHANCED FEATURE-ENGINEERING PIPELINE")
    print("=" * 50)
    
    csv_path = "csv_alpha"
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    
    if len(csv_files) == 0:
        print("FEHLER: Keine CSV-Dateien gefunden")
        return
    
    print(f"Verarbeite {len(csv_files)} CSV-Dateien...")
    
    features_data = {}
    bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
    
    # Erweiterte Makro-Symbole (alle verfügbaren)
    extended_macro_symbols = ['VIX', 'DXY', 'TNX', 'TLT', 'SHY', 'VFH', 'IYF', 'SPX', 'XLF']
    
    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).replace('.csv', '')
        
        try:
            print(f"Verarbeite {symbol}...")
            
            # 1. LADEN
            df = pd.read_csv(csv_file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df = df.sort_values('DateTime')
            
            print(f"  Geladen: {len(df)} Zeilen")
            
            # 2. BASIS-FEATURES (VORSICHTIG)
            df['Returns'] = df['Close'].pct_change()
            
            # 3. SANFTE BEREINIGUNG
            # Entferne nur erste Zeile (pct_change NaN)
            if df['Returns'].isna().iloc[0]:
                df = df.iloc[1:]
                print(f"  Nach NaN-Bereinigung: {len(df)} Zeilen")
            
            # Entferne Infinite (aber nicht alle NaN)
            df['Returns'] = df['Returns'].replace([np.inf, -np.inf], np.nan)
            
            # 4. TECHNISCHE INDIKATOREN (ROBUST)
            # Moving Averages
            df['MA_5'] = df['Close'].rolling(5).mean()
            df['MA_10'] = df['Close'].rolling(10).mean()
            df['MA_20'] = df['Close'].rolling(20).mean()
            
            # Price Ratios (mit NaN-Schutz)
            df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
            df['Price_MA10_Ratio'] = df['Close'] / df['MA_10']
            df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
            
            # Volatility
            df['Vol_5'] = df['Returns'].rolling(5).std()
            df['Vol_10'] = df['Returns'].rolling(10).std()
            df['Vol_20'] = df['Returns'].rolling(20).std()
            
            # Volume Features
            df['Volume_MA'] = df['Volume'].rolling(10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # High-Low Spread
            df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
            df['HL_Spread_MA'] = df['HL_Spread'].rolling(5).mean()
            
            # RSI (vereinfacht)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)  # Vermeide Division durch 0
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Zeit-Features
            df['Hour'] = df['DateTime'].dt.hour
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
            df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
            
            print(f"  Nach technischen Indikatoren: {len(df)} Zeilen")
            
            # 5. TARGET-VARIABLEN (nur für Banken)
            if symbol in bank_symbols:
                df['Target_1h'] = df['Returns'].shift(-1)
                df['Target_4h'] = df['Returns'].shift(-4)
                df['Target_24h'] = df['Returns'].shift(-24)
                df['Target_Direction_1h'] = (df['Target_1h'] > 0).astype(int)
                df['Target_Direction_4h'] = (df['Target_4h'] > 0).astype(int)
                
                # NEU: 7h Target für Ihre Anforderung
                df['Target_7h'] = df['Returns'].shift(-7)
                df['Target_Direction_7h'] = (df['Target_7h'] > 0).astype(int)
                
                print(f"  Target-Variablen erstellt (inkl. 7h)")
            
            # 6. FINALE BEREINIGUNG (NUR WICHTIGE SPALTEN)
            # Definiere kritische Spalten die nicht NaN sein dürfen
            critical_cols = ['Close', 'Returns']
            
            # Entferne Zeilen wo kritische Spalten NaN sind
            df_clean = df.dropna(subset=critical_cols)
            
            print(f"  Nach kritischer Bereinigung: {len(df_clean)} Zeilen")
            
            # Entferne Zeilen mit zu vielen NaN (80% Threshold statt 50%)
            threshold = len(df_clean.columns) * 0.8
            df_clean = df_clean.dropna(thresh=threshold)
            
            print(f"  Nach 80%-Threshold: {len(df_clean)} Zeilen")
            
            # Prüfe Mindestanforderung
            if len(df_clean) >= 50:
                features_data[symbol] = df_clean
                feature_count = len([c for c in df_clean.columns 
                                   if c not in ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']])
                print(f"  ERFOLG: {symbol} mit {len(df_clean)} Zeilen und {feature_count} Features gespeichert")
            else:
                print(f"  WARNUNG: {symbol} hat nur {len(df_clean)} Zeilen (< 50)")
                
        except Exception as e:
            print(f"  FEHLER bei {symbol}: {str(e)}")
    
    # 7. CROSS-ASSET FEATURES (XLF - bestehende Logik)
    if 'XLF' in features_data:
        print(f"\nErstelle Cross-Asset Features mit XLF...")
        
        xlf_data = features_data['XLF'].set_index('DateTime')
        
        for symbol in bank_symbols:
            if symbol in features_data:
                print(f"  Cross-Asset für {symbol}...")
                
                bank_data = features_data[symbol].set_index('DateTime')
                
                # XLF Returns für Alignment
                xlf_returns = xlf_data['Returns'].reindex(bank_data.index)
                
                # Relative Performance Features
                bank_data['Sector_Return'] = xlf_returns
                bank_data['Bank_vs_Sector'] = bank_data['Returns'] - xlf_returns
                bank_data['Relative_Strength_5D'] = (bank_data['Returns'].rolling(5).sum() - 
                                                   xlf_returns.rolling(5).sum())
                
                # Speichere zurück
                features_data[symbol] = bank_data.reset_index()
        
        print(f"  Cross-Asset Features abgeschlossen")
    
    # 8. NEUE: ERWEITERTE MAKRO-FEATURES
    add_extended_macro_features(features_data)
    
    # 9. SPEICHERN
    if features_data:
        os.makedirs("features_enhanced", exist_ok=True)
        
        print(f"\nSpeichere Enhanced Features...")
        
        for symbol, data in features_data.items():
            csv_file = f"features_enhanced/{symbol}_enhanced.csv"
            data.to_csv(csv_file, index=False)
            print(f"  {csv_file} gespeichert")
        
        # Summary erstellen
        summary_data = []
        for symbol, data in features_data.items():
            feature_count = len([c for c in data.columns 
                               if c not in ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']])
            summary_data.append({
                'Symbol': symbol,
                'Type': 'Bank' if symbol in bank_symbols else 'Macro',
                'Rows': len(data),
                'Features': feature_count,
                'Date_Start': data['DateTime'].min(),
                'Date_End': data['DateTime'].max()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("features_enhanced/summary.csv", index=False)
        
        print(f"\nENHANCED FEATURE-ENGINEERING ERFOLGREICH!")
        print(f"  Symbole verarbeitet: {len(features_data)}")
        print(f"  Gespeichert in: features_enhanced/")
        print(f"  Summary: features_enhanced/summary.csv")
        
        # Detailliertes Summary
        print(f"\nDETAILS:")
        banks_processed = [s for s in features_data.keys() if s in bank_symbols]
        macro_processed = [s for s in features_data.keys() if s not in bank_symbols]
        
        print(f"  Banken ({len(banks_processed)}): {', '.join(banks_processed)}")
        print(f"  Makro ({len(macro_processed)}): {', '.join(macro_processed)}")
        
        for symbol, data in features_data.items():
            feature_count = len([c for c in data.columns 
                               if c not in ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']])
            print(f"  {symbol}: {len(data)} Zeilen, {feature_count} Features")
        
        if len(banks_processed) >= 8:
            print(f"\nBEREIT FÜR ML-TRAINING!")
            print(f"  ✅ Ausreichend Bank-Daten für Prophet/LSTM/ARIMA")
            print(f"  ✅ 7h-Targets verfügbar")
            print(f"  ✅ Erweiterte Makro-Features verfügbar")
        
    else:
        print("FEHLER: Keine Daten konnten verarbeitet werden!")

def add_extended_macro_features(features_data):
    """Erweiterte Cross-Asset Features mit allen Makro-Indikatoren"""
    
    print(f"\n🔗 Erstelle erweiterte Makro-Features...")
    
    # Verfügbare Makro-Features prüfen
    available_macro = ['VIX', 'DXY', 'TNX', 'TLT', 'SHY', 'VFH', 'IYF', 'SPX']
    macro_data = {}
    
    for macro_symbol in available_macro:
        if macro_symbol in features_data:
            macro_data[macro_symbol] = features_data[macro_symbol].set_index('DateTime')
            print(f"  ✅ {macro_symbol} verfügbar für Cross-Asset Features")
    
    if not macro_data:
        print(f"  ⚠️ Keine zusätzlichen Makro-Features verfügbar")
        return
    
    print(f"  📊 Verwende {len(macro_data)} Makro-Indikatoren für erweiterte Features")
    
    # Erweiterte Features für alle Banken
    bank_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BLK', 'AXP', 'COF']
    
    for bank in bank_symbols:
        if bank not in features_data:
            continue
            
        print(f"  🏦 Erweitere {bank} mit Makro-Features...")
        bank_data = features_data[bank].set_index('DateTime')
        
        # VIX Features (Volatility Sensitivity)
        if 'VIX' in macro_data:
            vix_returns = macro_data['VIX']['Returns'].reindex(bank_data.index)
            vix_close = macro_data['VIX']['Close'].reindex(bank_data.index)
            
            bank_data['VIX_Level'] = vix_close
            bank_data['VIX_Return'] = vix_returns
            bank_data['Bank_VIX_Corr_5D'] = bank_data['Returns'].rolling(5).corr(vix_returns)
            bank_data['VIX_Volatility_Regime'] = (vix_close > vix_close.rolling(20).mean()).astype(int)
            
            print(f"    ✅ VIX Features hinzugefügt")
        
        # DXY Features (Dollar Sensitivity)
        if 'DXY' in macro_data:
            dxy_returns = macro_data['DXY']['Returns'].reindex(bank_data.index)
            dxy_close = macro_data['DXY']['Close'].reindex(bank_data.index)
            
            bank_data['DXY_Level'] = dxy_close
            bank_data['DXY_Return'] = dxy_returns
            bank_data['Bank_DXY_Corr_5D'] = bank_data['Returns'].rolling(5).corr(dxy_returns)
            bank_data['DXY_Strength_Regime'] = (dxy_close > dxy_close.rolling(20).mean()).astype(int)
            
            print(f"    ✅ DXY Features hinzugefügt")
        
        # TNX Features (Interest Rate Sensitivity)
        if 'TNX' in macro_data:
            tnx_returns = macro_data['TNX']['Returns'].reindex(bank_data.index)
            tnx_close = macro_data['TNX']['Close'].reindex(bank_data.index)
            
            bank_data['TNX_Level'] = tnx_close
            bank_data['TNX_Return'] = tnx_returns
            bank_data['Bank_TNX_Corr_5D'] = bank_data['Returns'].rolling(5).corr(tnx_returns)
            bank_data['Interest_Rate_Regime'] = (tnx_close > tnx_close.rolling(20).mean()).astype(int)
            
            print(f"    ✅ TNX (10Y Yield) Features hinzugefügt")
        
        # Treasury Spread Features (TLT vs SHY - Yield Curve)
        if 'TLT' in macro_data and 'SHY' in macro_data:
            tlt_returns = macro_data['TLT']['Returns'].reindex(bank_data.index)
            shy_returns = macro_data['SHY']['Returns'].reindex(bank_data.index)
            tlt_close = macro_data['TLT']['Close'].reindex(bank_data.index)
            shy_close = macro_data['SHY']['Close'].reindex(bank_data.index)
            
            # Yield Curve Proxy (TLT/SHY Ratio)
            yield_curve_proxy = tlt_close / shy_close
            yield_curve_returns = tlt_returns - shy_returns
            
            bank_data['Yield_Curve_Proxy'] = yield_curve_proxy
            bank_data['Yield_Curve_Return'] = yield_curve_returns
            bank_data['Bank_YieldCurve_Corr_5D'] = bank_data['Returns'].rolling(5).corr(yield_curve_returns)
            bank_data['Yield_Curve_Steepening'] = (yield_curve_proxy > yield_curve_proxy.rolling(10).mean()).astype(int)
            
            print(f"    ✅ Yield Curve (TLT/SHY) Features hinzugefügt")
        
        # Financial Sector Strength (VFH, IYF - zusätzlich zu XLF)
        if 'VFH' in macro_data:
            vfh_returns = macro_data['VFH']['Returns'].reindex(bank_data.index)
            bank_data['VFH_Return'] = vfh_returns
            bank_data['Bank_vs_VFH'] = bank_data['Returns'] - vfh_returns
            bank_data['Bank_VFH_Outperformance'] = (bank_data['Bank_vs_VFH'] > 0).astype(int)
            
            print(f"    ✅ VFH Sector Features hinzugefügt")
        
        if 'IYF' in macro_data:
            iyf_returns = macro_data['IYF']['Returns'].reindex(bank_data.index)
            bank_data['IYF_Return'] = iyf_returns
            bank_data['Bank_vs_IYF'] = bank_data['Returns'] - iyf_returns
            
            print(f"    ✅ IYF Sector Features hinzugefügt")
        
        # Market Beta (SPX)
        if 'SPX' in macro_data:
            spx_returns = macro_data['SPX']['Returns'].reindex(bank_data.index)
            spx_close = macro_data['SPX']['Close'].reindex(bank_data.index)
            
            bank_data['SPX_Return'] = spx_returns
            bank_data['Market_Beta_5D'] = bank_data['Returns'].rolling(5).corr(spx_returns)
            bank_data['Market_Beta_20D'] = bank_data['Returns'].rolling(20).corr(spx_returns)
            bank_data['SPX_Bull_Market'] = (spx_close > spx_close.rolling(50).mean()).astype(int)
            
            print(f"    ✅ SPX Market Beta Features hinzugefügt")
        
        # Multi-Factor Risk Model (wenn mehrere Makro-Features verfügbar)
        if len(macro_data) >= 3:
            # Risk-On/Risk-Off Regime
            risk_on_signals = []
            
            if 'VIX' in macro_data:
                vix_signal = (macro_data['VIX']['Close'].reindex(bank_data.index) < 
                             macro_data['VIX']['Close'].reindex(bank_data.index).rolling(20).mean())
                risk_on_signals.append(vix_signal)
            
            if 'SPX' in macro_data:
                spx_signal = (macro_data['SPX']['Returns'].reindex(bank_data.index) > 0)
                risk_on_signals.append(spx_signal)
            
            if risk_on_signals:
                bank_data['Risk_On_Regime'] = pd.concat(risk_on_signals, axis=1).mean(axis=1)
                print(f"    ✅ Risk-On/Risk-Off Regime hinzugefügt")
        
        # Zurück speichern
        features_data[bank] = bank_data.reset_index()
    
    print(f"  🎯 Erweiterte Makro-Features für alle Banken abgeschlossen!")
    print(f"  📈 Zusätzliche Features pro Bank: ~15-20 neue Makro-Features")

if __name__ == "__main__":
    enhanced_feature_engineering()