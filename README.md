# Prédiction du Prix de l'Or en Temps Réel

## Objectif du Projet
Ce projet vise à prédire les mouvements du prix de l'or en utilisant l'intelligence artificielle et le deep learning. Le système analyse en temps réel les données du marché pour générer des signaux de trading fiables avec une confiance supérieure à 80%.

### Partie 1 : Analyse en Temps Réel et Prédictions
- Affichage des prix de l'or en temps réel via l'API Binance
- Graphiques interactifs avec indicateurs techniques (MA, EMA, RSI, etc.)
- Mise à jour automatique des données toutes les 15 secondes
- Interface utilisateur moderne et intuitive avec Streamlit

### Partie 2 : Modèle de Deep Learning
- Architecture LSTM bidirectionnelle pour la prédiction des mouvements de prix
- Plus de 40 indicateurs techniques pour l'analyse
- Signaux de trading avec niveau de confiance >80%
- Stop Loss et Take Profit calculés automatiquement basés sur l'ATR

## Caractéristiques Principales
1. **Trading Signals**
   - Signaux BUY/SELL avec confiance >80%
   - Stop Loss et Take Profit optimisés
   - Gestion du risque intégrée

2. **Analyse Technique**
   - Moyennes mobiles (SMA, EMA)
   - Indicateurs de momentum (RSI, MACD)
   - Indicateurs de volatilité (ATR, Bollinger Bands)

3. **Interface Temps Réel**
   - Prix actuels et historiques
   - Graphiques interactifs
   - Tableau de bord des performances

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation
1. Lancer l'application :
```bash
streamlit run app.py
```

2. L'interface affichera :
   - Prix en temps réel
   - Graphiques avec indicateurs
   - Signaux de trading (>80% confiance)
   - Niveaux de Stop Loss et Take Profit
