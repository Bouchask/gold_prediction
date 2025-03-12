import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import logging

class GoldPricePredictor:
    def __init__(self, sequence_length=30, n_features=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.model_path = os.path.join('models', 'gold_price_model.h5')
        
    def build_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        if not self.model:
            self.build_model()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Entraînement
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        return history
    
    def load_saved_model(self):
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            logging.info(f"Modèle chargé depuis {self.model_path}")
            return True
        return False
    
    def predict(self, X):
        if not self.model:
            if not self.load_saved_model():
                raise ValueError("Aucun modèle n'est chargé ou disponible")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        if not self.model:
            if not self.load_saved_model():
                raise ValueError("Aucun modèle n'est chargé ou disponible")
        return self.model.evaluate(X_test, y_test)

def train_test_split_temporal(X, y, train_size=0.7, val_size=0.15):
    """Split temporel des données"""
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
