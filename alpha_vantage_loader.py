import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import os

class AlphaVantageLoader:
    """
    Alpha Vantage Loader V7 - Mit persistenter Call-Zählung
    """
    
    def __init__(self, api_keys_file="api_keys.py", save_path=None):
        self.api_keys_file = api_keys_file
        self.current_key_index = 0
        self.base_url = 'https://www.alphavantage.co/query'
        
        # Pfad-Handling - immer relativ zur Script-Datei
        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_path = os.path.join(script_dir, "csv_alpha")
        else:
            self.save_path = save_path
        
        # Call-Counter JSON-Datei
        self.calls_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_calls_today.json")
        
        # Initialisierung in korrekter Reihenfolge
        self.api_keys = self.load_api_keys()
        self.calls_today = self.load_calls_today()
    
    def load_api_keys(self):
        """
        Lädt API Keys aus separater Datei
        """
        try:
            # Import der api_keys.py Datei
            import importlib.util
            spec = importlib.util.spec_from_file_location("api_keys", self.api_keys_file)
            api_keys_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_keys_module)
            
            # Keys aus dem Modul holen
            keys = api_keys_module.ALPHA_VANTAGE_KEYS
            print(f"✅ {len(keys)} API Keys geladen")
            
            return keys
            
        except FileNotFoundError:
            print(f"❌ API Keys Datei nicht gefunden: {self.api_keys_file}")
            print("   Erstelle Beispiel-Datei...")
            self.create_example_keys_file()
            return []
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der API Keys: {str(e)}")
            return []
    
    def create_example_keys_file(self):
        """
        Erstellt Beispiel api_keys.py Datei
        """
        example_content = '''# Alpha Vantage API Keys
# Registrierung: https://www.alphavantage.co/support/#api-key

# Liste von API Keys (Primary, Secondary, etc.)
ALPHA_VANTAGE_KEYS = [
    "YOUR_PRIMARY_API_KEY_HERE",
    "YOUR_SECONDARY_API_KEY_HERE",  # Optional: Backup Key
    # "YOUR_THIRD_API_KEY_HERE",    # Optional: Weitere Keys
]

# Tägliches Limit pro Key (Alpha Vantage Free: 25 calls/day)
DAILY_LIMIT_PER_KEY = 25
'''
        
        with open(self.api_keys_file, 'w') as f:
            f.write(example_content)
        
        print(f"✅ Beispiel-Datei erstellt: {self.api_keys_file}")
        print("   Bitte trage deine echten API Keys ein!")
    
    def load_calls_today(self):
        """
        Lädt Call-Counter aus JSON-Datei oder erstellt neue
        Resettet automatisch um Mitternacht
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        try:
            if os.path.exists(self.calls_file):
                with open(self.calls_file, 'r') as f:
                    data = json.load(f)
                
                # Prüfe ob heute oder veraltet
                if data.get('date') == today:
                    calls_today = data.get('calls', {})
                    total_calls = sum(int(count) for count in calls_today.values())
                    print(f"📅 Call-Counter geladen (heute): {total_calls} total calls")
                else:
                    # Neuer Tag - Reset auf 0
                    calls_today = {}
                    print(f"🌅 Neuer Tag! Call-Counter zurückgesetzt (war: {data.get('date', 'unknown')})")
            else:
                # Erste Verwendung
                calls_today = {}
                print(f"📅 Neue Call-Counter-Datei erstellt")
            
            # Sicherstellen dass alle Keys einen Counter haben
            for i in range(len(self.api_keys)):
                if str(i) not in calls_today:
                    calls_today[str(i)] = 0
            
            return calls_today
            
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Call-Counter: {str(e)}")
            print(f"   Erstelle neuen Counter...")
            # Fallback: Neuer Counter
            calls_today = {}
            for i in range(len(self.api_keys)):
                calls_today[str(i)] = 0
            return calls_today
    
    def save_calls_today(self):
        """
        Speichert aktuellen Call-Counter in JSON-Datei
        """
        today = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        
        data = {
            'date': today,
            'last_updated': current_time,
            'calls': self.calls_today,
            'total_calls_today': sum(int(count) for count in self.calls_today.values())
        }
        
        try:
            with open(self.calls_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Fehler beim Speichern der Call-Counter: {str(e)}")
    
    def reset_calls_if_new_day(self):
        """
        Prüft ob neuer Tag und resettet Counter falls nötig
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        try:
            if os.path.exists(self.calls_file):
                with open(self.calls_file, 'r') as f:
                    data = json.load(f)
                
                if data.get('date') != today:
                    # Neuer Tag erkannt!
                    print(f"🌅 Neuer Tag erkannt! Reset von {data.get('date')} auf {today}")
                    for key in self.calls_today:
                        self.calls_today[key] = 0
                    self.save_calls_today()
                    return True
        except:
            pass
        
        return False
    
    def get_current_api_key(self):
        """
        Gibt den aktuell aktiven API Key zurück
        """
        if not self.api_keys:
            return None
        
        # Prüfe ob neuer Tag (automatischer Reset)
        self.reset_calls_if_new_day()
        
        # Check if current key has reached limit
        current_calls = int(self.calls_today.get(str(self.current_key_index), 0))
        if current_calls >= 25:
            # Try next key
            next_key = (self.current_key_index + 1) % len(self.api_keys)
            next_calls = int(self.calls_today.get(str(next_key), 0))
            
            if next_calls < 25:
                print(f"🔄 Wechsel zu API Key {next_key + 1} (Key {self.current_key_index + 1} Limit erreicht)")
                self.current_key_index = next_key
            else:
                print("⚠️ Alle API Keys haben Tageslimit erreicht!")
                return None
        
        return self.api_keys[self.current_key_index]
    
    def increment_call_counter(self):
        """
        Erhöht Call-Counter für aktuellen Key und speichert in JSON
        """
        key_str = str(self.current_key_index)
        current_calls = int(self.calls_today.get(key_str, 0))
        self.calls_today[key_str] = current_calls + 1
        
        # Sofort in JSON speichern
        self.save_calls_today()
        
        remaining = 25 - self.calls_today[key_str]
        total_calls = sum(int(count) for count in self.calls_today.values())
        
        print(f"   API Calls heute: {self.calls_today[key_str]}/25 (Key {self.current_key_index + 1}, {remaining} übrig) | Total: {total_calls}")
    
    def fetch_stock_data(self, symbol, months_back=3):
        """
        Lädt Aktiendaten von Alpha Vantage
        """
        api_key = self.get_current_api_key()
        if not api_key:
            print("❌ Kein verfügbarer API Key!")
            return pd.DataFrame()
        
        print(f"📡 Lade {symbol} von Alpha Vantage (Key {self.current_key_index + 1})...")
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '60min',
            'outputsize': 'full',
            'apikey': api_key
        }
        
        try:
            # API Call
            response = requests.get(self.base_url, params=params, timeout=30)
            self.increment_call_counter()
            
            if response.status_code != 200:
                print(f"❌ HTTP Error: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Error Handling
            if 'Error Message' in data:
                print(f"❌ Alpha Vantage Error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                print(f"⚠️ Alpha Vantage Note: {data['Note']}")
                if 'call frequency' in data['Note'].lower():
                    next_key = (self.current_key_index + 1) % len(self.api_keys)
                    if next_key != self.current_key_index:
                        print(f"🔄 Versuche mit Key {next_key + 1}...")
                        self.current_key_index = next_key
                        return self.fetch_stock_data(symbol, months_back)
                return pd.DataFrame()
            
            if 'Time Series (60min)' not in data:
                print(f"❌ Unerwartete API Response: {list(data.keys())}")
                return pd.DataFrame()
            
            # Daten konvertieren
            time_series = data['Time Series (60min)']
            
            df_list = []
            for timestamp_str, values in time_series.items():
                try:
                    df_list.append({
                        'DateTime': pd.to_datetime(timestamp_str),
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Adj Close': float(values['4. close']),
                        'Volume': int(values['5. volume'])
                    })
                except (ValueError, KeyError) as e:
                    continue
            
            if not df_list:
                print("❌ Keine gültigen Daten konvertiert!")
                return pd.DataFrame()
            
            df = pd.DataFrame(df_list)
            df = df.sort_values('DateTime')
            
            # Auf gewünschten Zeitraum filtern
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            df = df[df['DateTime'] >= cutoff_date]
            
            # Nur Handelstage (Mo-Fr)
            df = df[df['DateTime'].dt.weekday < 5]
            
            print(f"✅ {len(df)} Datenpunkte für {symbol} erhalten")
            return df
            
        except Exception as e:
            print(f"❌ Fehler beim Laden von {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def save_to_csv(self, data, symbol, filename=None):
        """
        Speichert DataFrame als CSV
        """
        if data.empty:
            print("❌ Keine Daten zum Speichern!")
            return None
        
        os.makedirs(self.save_path, exist_ok=True)
        
        if filename is None:
            filename = f"{symbol}.csv"
        
        csv_file = os.path.join(self.save_path, filename)
        data.to_csv(csv_file, index=False)
        print(f"✅ CSV gespeichert: {csv_file}")
        
        return csv_file
    
    def load_from_csv(self, symbol):
        """
        Lädt bestehende CSV-Datei
        """
        csv_file = os.path.join(self.save_path, f"{symbol}.csv")
        
        if not os.path.exists(csv_file):
            return None
        
        try:
            data = pd.read_csv(csv_file)
            data['DateTime'] = pd.to_datetime(data['DateTime'])
            print(f"✅ Bestehende CSV geladen: {len(data)} Zeilen")
            return data
        except Exception as e:
            print(f"❌ Fehler beim Laden der CSV: {str(e)}")
            return None
    
    def update_stock_data(self, symbol, months_back=3):
        """
        Lädt neue Daten und kombiniert mit bestehender CSV
        """
        existing_data = self.load_from_csv(symbol)
        
        if existing_data is not None and not existing_data.empty:
            last_date = existing_data['DateTime'].max()
            print(f"📅 Letztes Datum in CSV: {last_date}")
            
            new_data = self.fetch_stock_data(symbol, months_back)
            
            if new_data.empty:
                print(f"⚠️ Keine neuen Daten für {symbol}")
                return existing_data
            
            new_data = new_data[new_data['DateTime'] > last_date]
            
            if new_data.empty:
                print(f"✅ {symbol} bereits aktuell")
                return existing_data
            
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.sort_values('DateTime')
            combined_data = combined_data.drop_duplicates(subset=['DateTime'], keep='last')
            
            print(f"✅ {len(new_data)} neue Datenpunkte hinzugefügt")
            
        else:
            print(f"📥 Erste Ladung für {symbol}")
            combined_data = self.fetch_stock_data(symbol, months_back)
        
        if not combined_data.empty:
            self.save_to_csv(combined_data, symbol)
        
        return combined_data
    
    def get_status(self):
        """
        Zeigt Status aller Keys mit persistenten Daten
        """
        self.reset_calls_if_new_day()
        
        today = datetime.now().strftime('%Y-%m-%d')
        total_calls = sum(int(count) for count in self.calls_today.values())
        
        print(f"📊 API Key Status ({today}):")
        for i, key in enumerate(self.api_keys):
            calls = int(self.calls_today.get(str(i), 0))
            remaining = 25 - calls
            status = "✅ Aktiv" if i == self.current_key_index else "⏸️ Standby"
            print(f"   Key {i + 1}: {calls}/25 calls, {remaining} übrig ({status})")
        
        print(f"   📈 Gesamt heute: {total_calls}/50 calls")
        
        if os.path.exists(self.calls_file):
            try:
                with open(self.calls_file, 'r') as f:
                    data = json.load(f)
                last_updated = data.get('last_updated', 'unknown')
                print(f"   💾 Zuletzt gespeichert: {last_updated}")
            except:
                pass
    
    def get_save_path(self):
        """
        Gibt den aktuellen Speicherpfad zurück
        """
        return self.save_path

def load_multiple_stocks(symbols, months_back=3):
    """
    Lädt mehrere Aktien mit Alpha Vantage
    """
    loader = AlphaVantageLoader()
    
    if not loader.api_keys:
        print("❌ Keine API Keys verfügbar!")
        return {}
    
    results = {}
    total = len(symbols)
    
    print(f"🚀 Lade {total} Aktien...")
    loader.get_status()
    print()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] {symbol}...", end=" ")
        
        try:
            data = loader.update_stock_data(symbol, months_back)
            if not data.empty:
                results[symbol] = data
                print("✅")
            else:
                print("❌ Keine Daten")
        except Exception as e:
            print(f"❌ Fehler: {str(e)}")
    
    print(f"\n🎯 Fertig! {len(results)} von {total} Aktien erfolgreich geladen")
    loader.get_status()
    
    return results

def test_alpha_vantage_v7():
    """
    Test der V7 Alpha Vantage Implementation mit persistenter Call-Zählung
    """
    print("🚀 Alpha Vantage V7 Test")
    print("=" * 50)
    
    loader = AlphaVantageLoader()
    
    if not loader.api_keys:
        print("❌ Keine API Keys verfügbar!")
        return None
    
    loader.get_status()
    print(f"💾 Speicherpfad: {loader.get_save_path()}")
    
    print(f"\n📈 Teste AAL Update...")
    aal_data = loader.update_stock_data('AAL', months_back=3)
    
    if aal_data.empty:
        print("❌ AAL Test fehlgeschlagen!")
        return None
    
    print(f"\n📊 AAL Datenanalyse:")
    print(f"   Zeilen: {len(aal_data)}")
    print(f"   Zeitraum: {aal_data['DateTime'].min()} bis {aal_data['DateTime'].max()}")
    print(f"   Letzte 3 Zeilen:")
    print(aal_data[['DateTime', 'Close', 'Volume']].tail(3))
    
    print(f"\n📊 API Status nach Test:")
    loader.get_status()
    
    print(f"\n✅ V7 Test erfolgreich!")
    return aal_data

if __name__ == "__main__":
    result = test_alpha_vantage_v7()
    
    if result is not None:
        print("\n🎯 V7 BEREIT FÜR PRODUKTION!")
        print("   - Persistente Call-Zählung ✅")
        print("   - Täglicher Auto-Reset ✅")
        print("   - Multi-Key-System ✅") 
        print("   - Incremental Updates ✅")
    else:
        print("\n❌ V7 TEST FEHLGESCHLAGEN!")