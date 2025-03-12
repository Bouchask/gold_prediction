import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
from model_deeplearning.predict import TradingPredictor

# Configuration de la page
st.set_page_config(
    page_title="Gold Price AI Predictions",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé style TradingView
st.markdown("""
    <style>
    .tradingview-widget-container {
        background-color: #131722;
    }
    .main {
        background-color: #131722;
    }
    .stApp {
        background-color: #131722;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 10px;
        border-radius: 5px;
    }
    .signal-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
        background-color: #1e222d;
    }
    .buy-signal {
        border-left: 4px solid #00ff00;
    }
    .sell-signal {
        border-left: 4px solid #ff0000;
    }
    h1, h2, h3 {
        color: #d1d4dc !important;
    }
    p {
        color: #d1d4dc;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation du prédicteur
@st.cache_resource
def get_predictor():
    return TradingPredictor(confidence_threshold=0.8)

predictor = get_predictor()

# Sidebar
st.sidebar.title("⚙️ Paramètres")
lookback_hours = st.sidebar.slider("Période d'historique (heures)", 24, 168, 48)
auto_refresh = st.sidebar.checkbox("Actualisation automatique", value=True)
refresh_interval = st.sidebar.slider("Intervalle d'actualisation (sec)", 5, 60, 15)

# En-tête principal
st.title("🤖 Prédictions IA du Prix de l'Or")

try:
    # Récupération de la dernière prédiction
    with st.spinner("Analyse du marché en cours..."):
        prediction = predictor.get_latest_prediction()
        historical_preds = predictor.get_historical_predictions(lookback_hours)

    if prediction and prediction['signal']:
        signal = prediction['signal']
        
        # Affichage du signal principal
        st.subheader("🎯 Signal de Trading Actuel")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            signal_color = "green" if signal['action'] == 'BUY' else "red"
            st.metric(
                "Signal",
                signal['action'],
                f"{signal['confidence']*100:.1f}% confiance",
                delta_color=signal_color.replace('green', 'normal')
            )
        
        with col2:
            profit_pct = abs(signal['take_profit'] - signal['entry_price']) / signal['entry_price'] * 100
            st.metric(
                "Take Profit",
                f"${signal['take_profit']:.2f}",
                f"+{profit_pct:.1f}%"
            )
        
        with col3:
            loss_pct = abs(signal['stop_loss'] - signal['entry_price']) / signal['entry_price'] * 100
            st.metric(
                "Stop Loss",
                f"${signal['stop_loss']:.2f}",
                f"-{loss_pct:.1f}%",
                delta_color="inverse"
            )
        
        # Affichage des détails du signal
        signal_type = "buy-signal" if signal['action'] == 'BUY' else "sell-signal"
        st.markdown(f"""
        <div class="signal-box {signal_type}">
            <h3>{signal['action']} Signal - {signal['confidence']*100:.1f}% Confiance</h3>
            <p><b>Prix d'entrée:</b> ${signal['entry_price']:.2f}</p>
            <p><b>Take Profit:</b> ${signal['take_profit']:.2f} ({profit_pct:.1f}%)</p>
            <p><b>Stop Loss:</b> ${signal['stop_loss']:.2f} ({loss_pct:.1f}%)</p>
            <p><b>Ratio Risque/Récompense:</b> {profit_pct/loss_pct:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Affichage des prédictions historiques
    if historical_preds:
        st.subheader("📈 Historique des Prédictions")
        
        # Conversion en DataFrame
        df_hist = pd.DataFrame(historical_preds)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        df_hist.set_index('timestamp', inplace=True)
        
        # Création du graphique
        fig = go.Figure()
        
        # Ligne du prix
        fig.add_trace(go.Scatter(
            x=df_hist.index,
            y=df_hist['close'],
            name='Prix',
            line=dict(color='#d1d4dc', width=1)
        ))
        
        # Ligne de prédiction
        fig.add_trace(go.Scatter(
            x=df_hist.index,
            y=df_hist['prediction'],
            name='Probabilité Achat',
            line=dict(color='#00ff00', width=1, dash='dot')
        ))
        
        # Mise à jour du layout
        fig.update_layout(
            title="Historique des Prédictions vs Prix",
            yaxis_title="Prix / Probabilité",
            xaxis_title="Date",
            height=400,
            template="plotly_dark",
            plot_bgcolor='#131722',
            paper_bgcolor='#131722',
            font=dict(color='#d1d4dc')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques des prédictions
        high_conf_preds = df_hist[df_hist['prediction'] >= 0.8]
        low_conf_preds = df_hist[df_hist['prediction'] <= 0.2]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Signaux d'achat (>80%)",
                len(high_conf_preds),
                f"{len(high_conf_preds)/len(df_hist)*100:.1f}% du total"
            )
        
        with col2:
            st.metric(
                "Signaux de vente (<20%)",
                len(low_conf_preds),
                f"{len(low_conf_preds)/len(df_hist)*100:.1f}% du total"
            )
        
        with col3:
            st.metric(
                "Période analysée",
                f"{lookback_hours}h",
                f"{len(df_hist)} points de données"
            )
    
    else:
        st.warning("Pas de prédictions historiques disponibles.")

except Exception as e:
    st.error(f"Une erreur s'est produite : {str(e)}")

# Footer avec dernière mise à jour
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #808a9d;'>Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.experimental_rerun()
