# Alpha Vantage API Keys
# Registrierung: https://www.alphavantage.co/support/#api-key
# WICHTIG: Diese Datei NICHT in Git committen! (.gitignore hinzufügen)

# Liste von API Keys (Primary, Secondary, etc.)
ALPHA_VANTAGE_KEYS = [
    "N6WMF8CM65X3SP6F",                 # Ihr Haupt-API-Key
    "VHCK6V1AYM9HCS4L",                 # Optional: Backup Key
    # "YOUR_THIRD_API_KEY_HERE",        # Optional: Weitere Keys für mehr Calls/Tag
]

# Zusätzliche Konfiguration
ACTIVE_KEY_INDEX = 0                    # Startet mit erstem Key (0)
DAILY_LIMIT_PER_KEY = 25               # Alpha Vantage Free: 25 calls/day
AUTO_ROTATE_KEYS = True                # Automatisch zwischen Keys wechseln

# Beispiel für echte Keys (ersetzen Sie die Platzhalter):
# ALPHA_VANTAGE_KEYS = [
#     "ABCD1234EFGH5678",               # Ihr echter Primary Key
#     "IJKL9012MNOP3456",               # Ihr echter Secondary Key  
# ]

# Pro-Tip: Sie können sich mehrere kostenlose Keys registrieren:
# - Verwenden Sie verschiedene Email-Adressen
# - Jeder Key: 25 calls/day = 50 calls total mit 2 Keys
# - 23 Aktien täglich updaten = 23 calls → passt gut!

 