import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    print("❌ yfinance nicht installiert!")
    print("   Installiere mit: pip install yfinance")
    YFINANCE_AVAILABLE = False

class YahooFinanceLoader:
    """
    Yahoo Finance Loader - Kompatibel mit Alpha Vantage System
    Lädt Makro-Features und speichert im gleichen Format wie Alpha Vantage
    """
    
    def __init__(self, save_path=None):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance muss installiert sein: pip install yfinance")
        
        # Pfad-Handling - gleich wie Alpha Vantage System
        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_path = os.path.join(script_dir, "csv_alpha")
        else:
            self.save_path = save_path
        
        # Makro-Symbole (Yahoo Finance Notation → CSV Name)
        self.macro_symbols = {
            '^VIX': 'VIX',           # CBOE Volatility Index
            'DX-Y.NYB': 'DXY',       # US Dollar Index
            '^TNX': 'TNX',           # 10-Year Treasury Note Yield
            'TLT': 'TLT',            # 20+ Year Treasury Bond ETF
            'SHY': 'SHY',            # 1-3 Year Treasury Bond ETF
            'VFH': 'VFH',            # Vanguard Financials ETF
            'IYF': 'IYF',            # iShares US Financials ETF
            '^GSPC': 'SPX'           # S&P 500 Index
        }
        
        print(f"🌐 Yahoo Finance Loader initialisiert")
        print(f"   📁 Speicherpfad: {self.save_path}")
        print(f"   📊 Makro-Symbole: {len(self.macro_symbols)}")
        
        # Erstelle Speicherordner falls nicht vorhanden
        os.makedirs(self.save_path, exist_ok=True)
    
    def fetch_yahoo_data(self, yahoo_symbol, csv_name, hours_back=2160):
        """
        Lädt Daten von Yahoo Finance im Alpha Vantage kompatiblen Format
        
        Args:
            yahoo_symbol: Yahoo Finance Symbol (z.B. '^VIX')
            csv_name: Name für CSV-Datei (z.B. 'VIX')
            hours_back: Anzahl Stunden zurück (default: 2160 = ~3 Monate)
        """
        
        print(f"📡 Lade {csv_name} ({yahoo_symbol}) von Yahoo Finance...")
        
        try:
            # Berechne Zeitraum
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)
            
            print(f"   📅 Zeitraum: {start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}")
            
            # Yahoo Finance Ticker erstellen
            ticker = yf.Ticker(yahoo_symbol)
            
            # Lade 1-Stunden Daten
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1h',
                auto_adjust=False,  # Keine automatische Anpassung
                prepost=False       # Keine Pre/Post-Market Daten
            )
            
            if data.empty:
                print(f"   ❌ Keine Daten für {yahoo_symbol} erhalten")
                return pd.DataFrame()
            
            # Format konvertieren (Alpha Vantage kompatibel)
            df = data.reset_index()
            
            # Spalten umbenennen für Kompatibilität
            column_mapping = {
                'Datetime': 'DateTime',
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Adj Close': 'Adj Close',
                'Volume': 'Volume'
            }
            
            # Falls 'Adj Close' nicht existiert, verwende 'Close'
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            df = df.rename(columns=column_mapping)
            
            # Datetime formatieren
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Nur Handelstage (Mo-Fr) für Konsistenz mit Alpha Vantage
            df = df[df['DateTime'].dt.weekday < 5]
            
            # Sortieren nach DateTime
            df = df.sort_values('DateTime')
            
            # NaN-Werte behandeln
            df = df.dropna()
            
            print(f"   ✅ {len(df)} Datenpunkte für {csv_name} erhalten")
            print(f"   📊 Zeitspanne: {df['DateTime'].min()} bis {df['DateTime'].max()}")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Fehler beim Laden von {yahoo_symbol}: {str(e)}")
            return pd.DataFrame()
    
    def save_to_csv(self, data, csv_name):
        """
        Speichert DataFrame als CSV (identisch mit Alpha Vantage System)
        """
        if data.empty:
            print(f"   ❌ Keine Daten zum Speichern für {csv_name}!")
            return None
        
        csv_file = os.path.join(self.save_path, f"{csv_name}.csv")
        data.to_csv(csv_file, index=False)
        print(f"   ✅ CSV gespeichert: {csv_file}")
        
        return csv_file
    
    def load_from_csv(self, csv_name):
        """
        Lädt bestehende CSV-Datei (identisch mit Alpha Vantage System)
        """
        csv_file = os.path.join(self.save_path, f"{csv_name}.csv")
        
        if not os.path.exists(csv_file):
            return None
        
        try:
            data = pd.read_csv(csv_file)
            data['DateTime'] = pd.to_datetime(data['DateTime'])
            print(f"   ✅ Bestehende CSV geladen: {len(data)} Zeilen")
            return data
        except Exception as e:
            print(f"   ❌ Fehler beim Laden der CSV: {str(e)}")
            return None
    
    def update_yahoo_data(self, yahoo_symbol, csv_name, hours_back=2160):
        """
        Aktualisiert Daten (inkrementell wie Alpha Vantage System)
        """
        existing_data = self.load_from_csv(csv_name)
        
        if existing_data is not None and not existing_data.empty:
            last_date = existing_data['DateTime'].max()
            print(f"   📅 Letztes Datum in CSV: {last_date}")
            
            # Lade neue Daten
            new_data = self.fetch_yahoo_data(yahoo_symbol, csv_name, hours_back)
            
            if new_data.empty:
                print(f"   ⚠️ Keine neuen Daten für {csv_name}")
                return existing_data
            
            # Filtere nur neue Daten
            new_data = new_data[new_data['DateTime'] > last_date]
            
            if new_data.empty:
                print(f"   ✅ {csv_name} bereits aktuell")
                return existing_data
            
            # Kombiniere Daten
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.sort_values('DateTime')
            combined_data = combined_data.drop_duplicates(subset=['DateTime'], keep='last')
            
            print(f"   ✅ {len(new_data)} neue Datenpunkte hinzugefügt")
            
        else:
            print(f"   📥 Erste Ladung für {csv_name}")
            combined_data = self.fetch_yahoo_data(yahoo_symbol, csv_name, hours_back)
        
        if not combined_data.empty:
            self.save_to_csv(combined_data, csv_name)
        
        return combined_data
    
    def load_all_macro_features(self, hours_back=2160):
        """
        Lädt alle Makro-Features von Yahoo Finance
        """
        print(f"🚀 Lade {len(self.macro_symbols)} Makro-Features von Yahoo Finance...")
        print(f"   📅 Zeitraum: {hours_back} Stunden zurück (~{hours_back//24} Tage)")
        
        results = {}
        total = len(self.macro_symbols)
        
        for i, (yahoo_symbol, csv_name) in enumerate(self.macro_symbols.items(), 1):
            print(f"\n[{i}/{total}] {csv_name} ({yahoo_symbol})...")
            
            try:
                data = self.update_yahoo_data(yahoo_symbol, csv_name, hours_back)
                if not data.empty:
                    results[csv_name] = data
                    print(f"   ✅ Erfolgreich")
                else:
                    print(f"   ❌ Keine Daten")
                    
                # Kurze Pause um Yahoo Finance nicht zu überlasten
                if i < total:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"   ❌ Fehler: {str(e)}")
        
        print(f"\n🎯 Yahoo Finance Download abgeschlossen!")
        print(f"   ✅ {len(results)} von {total} Makro-Features erfolgreich")
        
        return results
    
    def get_available_symbols(self):
        """
        Zeigt verfügbare Makro-Symbole
        """
        print(f"📊 Verfügbare Makro-Features:")
        for yahoo_symbol, csv_name in self.macro_symbols.items():
            description = self.get_symbol_description(csv_name)
            print(f"   {csv_name:4} | {yahoo_symbol:10} | {description}")
    
    def get_symbol_description(self, csv_name):
        """
        Gibt Beschreibung für Symbol zurück
        """
        descriptions = {
            'VIX': 'CBOE Volatility Index - Market Fear Gauge',
            'DXY': 'US Dollar Index - Dollar Strength',
            'TNX': '10-Year Treasury Yield - Interest Rate Level',
            'TLT': '20+ Year Treasury ETF - Long-term Bonds',
            'SHY': '1-3 Year Treasury ETF - Short-term Bonds',
            'VFH': 'Vanguard Financials ETF - Financial Sector',
            'IYF': 'iShares US Financials ETF - Financial Sector',
            'SPX': 'S&P 500 Index - Market Benchmark'
        }
        return descriptions.get(csv_name, 'Unknown')
    
    def validate_csv_compatibility(self):
        """
        Prüft ob CSV-Dateien mit Alpha Vantage Format kompatibel sind
        """
        print(f"\n🔍 Validiere CSV-Kompatibilität mit Alpha Vantage Format...")
        
        csv_files = [f for f in os.listdir(self.save_path) 
                    if f.endswith('.csv') and f.replace('.csv', '') in self.macro_symbols.values()]
        
        if not csv_files:
            print(f"   ⚠️ Keine Yahoo Finance CSV-Dateien gefunden")
            return False
        
        required_columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        all_compatible = True
        
        for csv_file in csv_files:
            symbol = csv_file.replace('.csv', '')
            csv_path = os.path.join(self.save_path, csv_file)
            
            try:
                df = pd.read_csv(csv_path)
                
                # Prüfe Spalten
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"   ❌ {symbol}: Fehlende Spalten: {missing_columns}")
                    all_compatible = False
                else:
                    print(f"   ✅ {symbol}: {len(df)} Zeilen, kompatibel")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Fehler beim Lesen: {str(e)}")
                all_compatible = False
        
        if all_compatible:
            print(f"   🎯 Alle Yahoo Finance CSV-Dateien sind Alpha Vantage kompatibel!")
        
        return all_compatible

def test_yahoo_finance_loader():
    """
    Test-Funktion für Yahoo Finance Loader
    """
    print("🧪 YAHOO FINANCE LOADER TEST")
    print("=" * 50)
    
    if not YFINANCE_AVAILABLE:
        print("❌ yfinance nicht verfügbar - installiere zuerst: pip install yfinance")
        return
    
    loader = YahooFinanceLoader()
    
    # Zeige verfügbare Symbole
    loader.get_available_symbols()
    
    # Test mit einem Symbol
    print(f"\n🧪 Teste VIX (Volatility Index)...")
    vix_data = loader.update_yahoo_data('^VIX', 'VIX', hours_back=720)  # 30 Tage
    
    if not vix_data.empty:
        print(f"\n📊 VIX Test-Ergebnisse:")
        print(f"   Zeilen: {len(vix_data)}")
        print(f"   Zeitraum: {vix_data['DateTime'].min()} bis {vix_data['DateTime'].max()}")
        print(f"   Spalten: {list(vix_data.columns)}")
        print(f"   Letzte 3 Werte:")
        print(vix_data[['DateTime', 'Close', 'Volume']].tail(3))
        
        print(f"\n✅ VIX Test erfolgreich!")
        return True
    else:
        print(f"\n❌ VIX Test fehlgeschlagen!")
        return False

def load_macro_features_only():
    """
    Lädt nur Makro-Features (für schnellen Test)
    """
    print("📈 MAKRO-FEATURES LOADER")
    print("=" * 40)
    
    loader = YahooFinanceLoader()
    
    # Frage Benutzer
    print(f"\n📋 Lade alle {len(loader.macro_symbols)} Makro-Features:")
    loader.get_available_symbols()
    
    response = input(f"\n🎯 Alle Makro-Features laden? (j/n): ").lower()
    
    if response != 'j':
        print("❌ Abgebrochen")
        return {}
    
    # Lade alle Makro-Features
    results = loader.load_all_macro_features(hours_back=2160)  # 3 Monate
    
    # Validiere Kompatibilität
    if results:
        loader.validate_csv_compatibility()
        
        print(f"\n🎯 BEREIT FÜR FEATURE-ENGINEERING!")
        print(f"   📁 Makro-CSV-Dateien in: {loader.save_path}")
        print(f"   🔧 Kompatibel mit Alpha Vantage System")
        print(f"   ✅ Kann jetzt Feature-Engineering mit allen Daten ausführen")
    
    return results

if __name__ == "__main__":
    print("🌐 YAHOO FINANCE LOADER FÜR MAKRO-FEATURES")
    print("=" * 55)
    
    print("📋 Optionen:")
    print("1. 🧪 Test (nur VIX)")
    print("2. 📈 Alle Makro-Features laden")
    print("3. 🔍 Nur verfügbare Symbole anzeigen")
    
    choice = input("\nWähle Option (1-3): ").strip()
    
    if choice == "1":
        test_yahoo_finance_loader()
    elif choice == "2":
        load_macro_features_only()
    elif choice == "3":
        if YFINANCE_AVAILABLE:
            loader = YahooFinanceLoader()
            loader.get_available_symbols()
        else:
            print("❌ yfinance nicht verfügbar")
    else:
        print("❌ Ungültige Auswahl")