import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendSignalSystem:
    """
    5-Stufen Ampel-System für Trend & Position-Sizing
    
    Signals:
        DARK_GREEN:  Score +3 to +5  (100% Long)
        LIGHT_GREEN: Score +1.5 to +3 (50% Long)
        YELLOW:      Score -1 to +1.5 (25% Long or Flat)
        ORANGE:      Score -2.5 to -1 (50% Short)
        RED:         Score -5 to -2.5 (100% Short)
    """
    
    def __init__(self):
        # Signal weights (wie wichtig ist jeder Indikator?)
        self.weights = {
            'price_vs_ma': 1.5,      # Trend ist König!
            'macd': 1.2,             # Momentum wichtig
            'rsi': 1.0,              # Standard
            'volume': 0.8,           # Bestätigung
            'vix': 1.0               # Risk Sentiment
        }
        
        logger.info("Trend Signal System initialized")
    
    def calculate_signal(self, data):
        """
        Berechnet Ampel-Signal für eine Zeile (oder DataFrame)
        
        Args:
            data: Series oder DataFrame mit Features
        
        Returns:
            dict mit signal, score, confidence, action, position_size
        """
        
        if isinstance(data, pd.DataFrame):
            # Wenn DataFrame → letzte Zeile nehmen
            data = data.iloc[-1]
        
        signals = []
        
        # Signal 1: Price vs MA_20 (Trend)
        if data['Close'] > data['MA_20'] * 1.02:
            signals.append((+1, self.weights['price_vs_ma']))
        elif data['Close'] < data['MA_20'] * 0.98:
            signals.append((-1, self.weights['price_vs_ma']))
        else:
            signals.append((0, self.weights['price_vs_ma']))
        
        # Signal 2: MACD
        if data['MACD'] > data['MACD_Signal'] and data['MACD'] > 0:
            signals.append((+1, self.weights['macd']))
        elif data['MACD'] < data['MACD_Signal'] and data['MACD'] < 0:
            signals.append((-1, self.weights['macd']))
        else:
            signals.append((0, self.weights['macd']))
        
        # Signal 3: RSI
        rsi = data['RSI']
        if 45 < rsi < 65:
            signals.append((+1, self.weights['rsi']))
        elif 35 < rsi < 55:
            signals.append((-1, self.weights['rsi']))
        elif rsi < 25:  # Oversold → Bounce
            signals.append((+1, self.weights['rsi'] * 0.8))
        elif rsi > 75:  # Overbought → Drop
            signals.append((-1, self.weights['rsi'] * 0.8))
        else:
            signals.append((0, self.weights['rsi']))
        
        # Signal 4: Volume Confirmation
        if data['Volume_Ratio'] > 1.3:
            volume_signal = +1 if data['Returns'] > 0 else -1
            signals.append((volume_signal, self.weights['volume']))
        else:
            signals.append((0, self.weights['volume'] * 0.5))
        
        # Signal 5: VIX (Risk)
        vix = data.get('VIX_Level', 15)  # Default 15 falls nicht vorhanden
        if vix < 14:
            signals.append((+1, self.weights['vix']))
        elif vix < 18:
            signals.append((+0.5, self.weights['vix']))
        elif vix > 25:
            signals.append((-1, self.weights['vix']))
        else:
            signals.append((0, self.weights['vix']))
        
        # Gewichteter Score berechnen
        total_score = sum(s * w for s, w in signals)
        max_score = sum(w for _, w in signals)
        
        # Normalisieren auf -5 bis +5
        normalized_score = (total_score / max_score) * 5
        
        # Confidence (0-100%)
        confidence = abs(normalized_score) / 5 * 100
        
        # Ampel-Signal bestimmen
        if normalized_score >= 3:
            signal = 'DARK_GREEN'
            action = 'STRONG BUY'
            position_size = 1.0  # 100%
        elif normalized_score >= 1.5:
            signal = 'LIGHT_GREEN'
            action = 'BUY'
            position_size = 0.5  # 50%
        elif normalized_score >= -1:
            signal = 'YELLOW'
            action = 'HOLD'
            position_size = 0.25  # 25% (oder 0 wenn du flat willst)
        elif normalized_score >= -2.5:
            signal = 'ORANGE'
            action = 'SELL'
            position_size = -0.5  # -50% (Short)
        else:
            signal = 'RED'
            action = 'STRONG SELL'
            position_size = -1.0  # -100% (Full Short)
        
        return {
            'signal': signal,
            'score': round(normalized_score, 2),
            'confidence': round(confidence, 1),
            'action': action,
            'position_size': position_size
        }
    
    def calculate_signals_batch(self, df):
        """
        Berechnet Signale für kompletten DataFrame
        
        Args:
            df: DataFrame mit Features
        
        Returns:
            DataFrame mit zusätzlichen Signal-Spalten
        """
        
        logger.info(f"Calculating signals for {len(df)} rows...")
        
        results = []
        for idx, row in df.iterrows():
            signal_data = self.calculate_signal(row)
            results.append(signal_data)
        
        # Als DataFrame zurück
        signals_df = pd.DataFrame(results)
        
        # Merge mit original DataFrame
        df_with_signals = pd.concat([df.reset_index(drop=True), signals_df], axis=1)
        
        logger.info(f"Signal distribution:")
        logger.info(df_with_signals['signal'].value_counts())
        
        return df_with_signals
    
    def get_position_recommendation(self, data, current_position=0):
        """
        Gibt Position-Empfehlung basierend auf Signal
        
        Args:
            data: Aktuelle Marktdaten
            current_position: Aktuelle Position (-1 bis +1)
        
        Returns:
            dict mit trade_action, target_position, risk_level
        """
        
        signal_data = self.calculate_signal(data)
        target_position = signal_data['position_size']
        
        # Trade Action bestimmen
        if target_position > current_position:
            trade_action = 'BUY' if current_position >= 0 else 'COVER'
            amount = target_position - current_position
        elif target_position < current_position:
            trade_action = 'SELL' if current_position > 0 else 'SHORT'
            amount = abs(target_position - current_position)
        else:
            trade_action = 'HOLD'
            amount = 0
        
        # Risk Level
        if signal_data['confidence'] > 70:
            risk_level = 'HIGH_CONVICTION'
        elif signal_data['confidence'] > 50:
            risk_level = 'MEDIUM_CONVICTION'
        else:
            risk_level = 'LOW_CONVICTION'
        
        return {
            'signal': signal_data['signal'],
            'score': signal_data['score'],
            'confidence': signal_data['confidence'],
            'current_position': current_position,
            'target_position': target_position,
            'trade_action': trade_action,
            'amount': round(amount, 2),
            'risk_level': risk_level
        }

# Test function
def test_signal_system():
    """Test Ampel-System mit Beispiel-Daten"""
    
    # Beispiel-Daten (Bullish Szenario)
    test_data_bullish = pd.Series({
        'Close': 105,
        'MA_20': 100,
        'MACD': 0.5,
        'MACD_Signal': 0.3,
        'RSI': 58,
        'Volume_Ratio': 1.5,
        'Returns': 0.01,
        'VIX_Level': 13
    })
    
    # Test
    system = TrendSignalSystem()
    
    logger.info("\n" + "="*60)
    logger.info("TEST 1: BULLISH SCENARIO")
    logger.info("="*60)
    result = system.calculate_signal(test_data_bullish)
    for key, value in result.items():
        logger.info(f"{key:15s}: {value}")
    
    # Beispiel-Daten (Bearish Szenario)
    test_data_bearish = pd.Series({
        'Close': 95,
        'MA_20': 100,
        'MACD': -0.4,
        'MACD_Signal': -0.2,
        'RSI': 38,
        'Volume_Ratio': 1.6,
        'Returns': -0.015,
        'VIX_Level': 28
    })
    
    logger.info("\n" + "="*60)
    logger.info("TEST 2: BEARISH SCENARIO")
    logger.info("="*60)
    result = system.calculate_signal(test_data_bearish)
    for key, value in result.items():
        logger.info(f"{key:15s}: {value}")
    
    # Position Recommendation
    logger.info("\n" + "="*60)
    logger.info("TEST 3: POSITION RECOMMENDATION")
    logger.info("="*60)
    recommendation = system.get_position_recommendation(
        test_data_bullish, 
        current_position=0.25
    )
    for key, value in recommendation.items():
        logger.info(f"{key:20s}: {value}")

if __name__ == "__main__":
    test_signal_system()