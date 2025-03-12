import logging
import numpy as np
from datetime import datetime
from data_preparation import DataPreparation
from model import GoldPricePredictor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_training_history(history, save_dir='plots'):
    """Génère et sauvegarde les graphiques de l'historique d'entraînement"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot de l'accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy du Modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
    
    # Plot de la loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss du Modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir='plots'):
    """Génère et sauvegarde la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    try:
        # Préparation des données avec 2000 heures d'historique
        logging.info("Chargement des données...")
        data_prep = DataPreparation(
            symbol="PAXGUSDT",
            interval="1h",
            lookback_period=2000  # 2000 heures d'historique
        )
        
        # Récupération et préparation des données
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_columns = data_prep.prepare_data(
            sequence_length=30,
            train_size=0.7,  # 70% pour l'entraînement
            val_size=0.15    # 15% pour la validation, 15% pour le test
        )
        
        logging.info(f"Forme des données d'entraînement: {X_train.shape}")
        logging.info(f"Forme des données de validation: {X_val.shape}")
        logging.info(f"Forme des données de test: {X_test.shape}")
        logging.info(f"Nombre de features: {len(feature_columns)}")
        
        # Création et entraînement du modèle
        logging.info("Création du modèle...")
        model = GoldPricePredictor(
            sequence_length=30,
            n_features=len(feature_columns)
        )
        
        # Entraînement avec plus d'epochs
        logging.info("Début de l'entraînement...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=200,  # Augmentation du nombre d'epochs
            batch_size=32
        )
        
        # Évaluation sur l'ensemble de test
        logging.info("Évaluation du modèle...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        logging.info(f"Test Loss: {test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)
        y_pred_classes = (y_pred > 0.6).astype(int)  # Seuil de confiance à 60%
        
        # Génération des graphiques
        logging.info("Génération des graphiques d'analyse...")
        plot_training_history(history)
        plot_confusion_matrix(y_test, y_pred_classes)
        
        # Affichage du rapport de classification
        report = classification_report(y_test, y_pred_classes)
        logging.info("\nRapport de Classification:\n" + report)
        
        # Sauvegarde des métriques dans un fichier
        with open('models/training_metrics.txt', 'w') as f:
            f.write(f"Date d'entraînement: {datetime.now()}\n")
            f.write(f"Nombre total d'échantillons: {len(X_train) + len(X_val) + len(X_test)}\n")
            f.write(f"Échantillons d'entraînement: {len(X_train)}\n")
            f.write(f"Échantillons de validation: {len(X_val)}\n")
            f.write(f"Échantillons de test: {len(X_test)}\n")
            f.write(f"Accuracy sur le test: {test_accuracy:.4f}\n")
            f.write(f"Loss sur le test: {test_loss:.4f}\n")
            f.write("\nRapport de Classification:\n")
            f.write(report)
        
        logging.info("Entraînement terminé avec succès!")
        
    except Exception as e:
        logging.error(f"Erreur pendant l'entraînement: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
