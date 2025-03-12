import pandas as pd
import numpy as np
from binance.client import Client
import ta
from sklearn.preprocessing import MinMaxScaler
import logging

class DataPreparation:
    def __init__(self, symbol="PAXGUSDT", interval="1h", lookback_period=2000):
        self.symbol = symbol
        self.interval = interval
        self.lookback_period = lookback_period
        self.client = Client()
        self.scaler = MinMaxScaler()
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def fetch_data(self):
        """Récupère les données historiques de Binance"""
        try:
            logging.info(f"Récupération des données pour {self.symbol}, période: {self.lookback_period}")
            klines = self.client.get_historical_klines(
                self.symbol,
                self.interval,
                f"{self.lookback_period} hours ago UTC"
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Conversion des types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            return df
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des données: {str(e)}")
            raise
    
    def add_technical_indicators(self, df):
        """Ajoute les indicateurs techniques"""
        try:
            # Tendance
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Momentum
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # Volatilité
            df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            
            return df
            
        except Exception as e:
            logging.error(f"Erreur lors de l'ajout des indicateurs techniques: {str(e)}")
            raise
    
    def calculate_indicators(self, df):
        """Calcule les indicateurs techniques"""
        try:
            # Copie du DataFrame pour éviter les modifications en place
            df = df.copy()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            
            # Moyennes mobiles
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            
            # Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
            
            # ATR pour la volatilité
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Suppression des lignes avec des valeurs manquantes
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
            raise
    
    def prepare_target(self, df, threshold=0.005):
        """Prépare la variable cible (1 pour hausse significative, 0 sinon)"""
        try:
            # Calcul du rendement futur sur 12 périodes
            df['future_return'] = df['close'].shift(-12).div(df['close']) - 1
            
            # Création de la variable cible
            df['target'] = (df['future_return'] > threshold).astype(int)
            
            return df
            
        except Exception as e:
            logging.error(f"Erreur lors de la préparation de la cible: {str(e)}")
            raise
    
    def prepare_sequences(self, df, sequence_length=30):
        """Prépare les séquences pour l'entraînement"""
        try:
            feature_columns = [
                'close', 'volume', 'sma_20', 'sma_50', 'sma_200',
                'ema_12', 'ema_26', 'rsi', 'macd', 'williams_r',
                'bb_high', 'bb_low', 'atr', 'obv', 'mfi'
            ]
            
            # Normalisation des features
            df_normalized = pd.DataFrame(
                self.scaler.fit_transform(df[feature_columns]),
                columns=feature_columns,
                index=df.index
            )
            
            X, y = [], []
            for i in range(len(df_normalized) - sequence_length - 12):
                X.append(df_normalized.iloc[i:(i + sequence_length)].values)
                y.append(df.iloc[i + sequence_length]['target'])
            
            return np.array(X), np.array(y), feature_columns
            
        except Exception as e:
            logging.error(f"Erreur lors de la préparation des séquences: {str(e)}")
            raise
    
    def prepare_live_data(self, sequence_length):
        """Prépare les données pour la prédiction en direct"""
        try:
            # Récupération des données
            df = self.fetch_data()
            
            # Calcul des indicateurs
            df = self.calculate_indicators(df)
            
            # Sélection des features
            features = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_diff',
                       'bb_high', 'bb_low', 'bb_mid', 'sma_20', 'sma_50', 'ema_20',
                       'stoch_rsi_k', 'stoch_rsi_d', 'atr']
            
            # Normalisation des données
            df_norm = df[features].copy()
            for col in features:
                mean = df_norm[col].mean()
                std = df_norm[col].std()
                if std == 0:  # Éviter la division par zéro
                    std = 1
                df_norm[col] = (df_norm[col] - mean) / std
            
            # Préparation de la séquence
            sequence = df_norm.values[-sequence_length:]
            X = np.expand_dims(sequence, axis=0)
            
            return X, df.index[-1]
            
        except Exception as e:
            logging.error(f"Erreur lors de la préparation des données live: {str(e)}")
            raise
    
    def prepare_data(self, sequence_length=30, train_size=0.8, val_size=0.1):
        """Prépare toutes les données pour l'entraînement"""
        try:
            # Récupération et préparation des données
            df = self.fetch_data()
            df = self.add_technical_indicators(df)
            df = self.prepare_target(df)
            
            # Suppression des lignes avec des valeurs manquantes
            df = df.dropna()
            
            # Préparation des séquences
            X, y, feature_columns = self.prepare_sequences(df, sequence_length)
            
            # Split temporel des données
            train_end = int(len(X) * train_size)
            val_end = int(len(X) * (train_size + val_size))
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]
            
            logging.info(f"Forme des données - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_columns
            
        except Exception as e:
            logging.error(f"Erreur lors de la préparation des données: {str(e)}")
            raise
