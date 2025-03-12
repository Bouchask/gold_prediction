from .data_preparation import DataPreparation
from .model import GoldPricePredictor
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import traceback

class TradingPredictor:
    def __init__(self, confidence_threshold=0.55):
        self.confidence_threshold = confidence_threshold
        self.data_prep = DataPreparation(
            symbol="PAXGUSDT",
            interval="5m",
            lookback_period=288
        )
        self.model = GoldPricePredictor()
        self.sequence_length = 30
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def get_latest_prediction(self):
        try:
            logging.info("Récupération des données en direct...")
            df = self.data_prep.fetch_data()
            
            if len(df) < self.sequence_length:
                logging.error(f"Pas assez de données: {len(df)} points au lieu de {self.sequence_length}")
                return None
                
            logging.info(f"Données récupérées: {len(df)} points")
            
            logging.info("Calcul des indicateurs techniques...")
            df = self.data_prep.calculate_indicators(df)
            
            logging.info("Préparation des données pour la prédiction...")
            X_live, latest_timestamp = self.data_prep.prepare_live_data(self.sequence_length)
            logging.info(f"Données préparées, shape: {X_live.shape}")
            
            logging.info("Génération de la prédiction...")
            prediction = self.model.predict(X_live)[0][0]
            logging.info(f"Prédiction brute: {prediction:.4f}")
            
            current_data = df.tail(1)
            
            signal = None
            if prediction > self.confidence_threshold:
                logging.info(f"Signal potentiel ACHAT détecté avec confiance: {prediction*100:.2f}%")
                signal = self._calculate_signal(current_data, prediction)
            elif prediction < (1 - self.confidence_threshold):
                logging.info(f"Signal potentiel VENTE détecté avec confiance: {(1-prediction)*100:.2f}%")
                signal = self._calculate_signal(current_data, 1 - prediction)
            else:
                logging.info("Pas de signal clair détecté")
            
            result = {
                'timestamp': latest_timestamp,
                'prediction': float(prediction),
                'signal': signal,
                'last_close': float(current_data['close'].iloc[0])
            }
            logging.info(f"Résultat final: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Erreur dans la prédiction: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def _calculate_signal(self, data, confidence):
        """Calcule les niveaux de Take Profit et Stop Loss et détermine l'action"""
        try:
            logging.info("Calcul des niveaux et détermination de l'action...")
            entry_price = float(data['close'].iloc[0])
            
            # Utilisation de l'ATR pour le calcul des niveaux
            atr = float(data['atr'].iloc[0])
            if pd.isna(atr) or atr == 0:
                atr = entry_price * 0.001  # Utiliser 0.1% si ATR n'est pas disponible
            
            # Calcul des niveaux potentiels pour BUY et SELL
            buy_tp = entry_price + (2.5 * atr)  # 2.5x ATR pour TP en BUY
            buy_sl = entry_price - (1.0 * atr)  # 1.0x ATR pour SL en BUY
            
            sell_tp = entry_price - (2.5 * atr)  # 2.5x ATR pour TP en SELL
            sell_sl = entry_price + (1.0 * atr)  # 1.0x ATR pour SL en SELL
            
            # Calcul des ratios risque/récompense
            buy_rr = (buy_tp - entry_price) / (entry_price - buy_sl)
            sell_rr = (entry_price - sell_tp) / (sell_sl - entry_price)
            
            # Détermination de l'action finale basée sur le meilleur ratio risque/récompense
            if buy_rr > sell_rr and buy_rr >= 2.0:
                action = 'BUY'
                take_profit = buy_tp
                stop_loss = buy_sl
            elif sell_rr >= 2.0:
                action = 'SELL'
                take_profit = sell_tp
                stop_loss = sell_sl
            else:
                logging.info("Pas de setup valide trouvé (ratio risque/récompense < 2.0)")
                return None
            
            signal = {
                'action': action,
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'risk_reward': buy_rr if action == 'BUY' else sell_rr
            }
            
            logging.info(f"Signal généré: {signal}")
            return signal
            
        except Exception as e:
            logging.error(f"Erreur dans le calcul du signal: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def get_historical_predictions(self, lookback_hours=24):
        try:
            logging.info(f"Récupération de l'historique sur {lookback_hours} heures...")
            df = self.data_prep.fetch_data()
            
            if len(df) < self.sequence_length:
                logging.error(f"Pas assez de données historiques: {len(df)} points")
                return None
                
            df = self.data_prep.calculate_indicators(df)
            df = df.tail(int(lookback_hours * 12))  # 12 périodes de 5 minutes par heure
            
            logging.info(f"Données historiques récupérées: {len(df)} points")
            predictions = []
            
            for i in range(self.sequence_length, len(df)):
                sequence_data = df.iloc[i-self.sequence_length:i]
                if len(sequence_data) == self.sequence_length:
                    X = self.data_prep.prepare_live_data(self.sequence_length)[0]
                    pred = self.model.predict(X)[0][0]
                    
                    signal = None
                    if pred > self.confidence_threshold or pred < (1 - self.confidence_threshold):
                        current_data = df.iloc[[i-1]]
                        if pred > self.confidence_threshold:
                            signal = self._calculate_signal(current_data, pred)
                        else:
                            signal = self._calculate_signal(current_data, 1 - pred)
                    
                    predictions.append({
                        'timestamp': df.index[i-1],
                        'prediction': float(pred),
                        'close': float(df['close'].iloc[i-1]),
                        'signal': signal
                    })
            
            logging.info(f"Nombre de prédictions historiques: {len(predictions)}")
            return predictions
            
        except Exception as e:
            logging.error(f"Erreur dans les prédictions historiques: {str(e)}")
            logging.error(traceback.format_exc())
            return None
