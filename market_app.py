import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging
from model_deeplearning.predict import TradingPredictor
import plotly.express as px

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(
    page_title="Gold Trading Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .high-confidence {
        background-color: rgba(38, 166, 154, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stProgress .st-bo {
        background-color: #26a69a;
    }
    .stProgress .st-bp {
        background-color: rgba(38, 166, 154, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

def format_price(price):
    return f"${price:,.2f}"

def create_candlestick_chart(df, signals=None):
    fig = go.Figure()
    
    # Chandelier principal
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="PAXG/USDT",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Ajout des signaux si disponibles
    if signals is not None:
        buy_signals = signals[signals['action'] == 'BUY']
        sell_signals = signals[signals['action'] == 'SELL']
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#26a69a',
                    line=dict(color='#26a69a', width=1)
                ),
                name='Signaux d\'achat'
            ))
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#ef5350',
                    line=dict(color='#ef5350', width=1)
                ),
                name='Signaux de vente'
            ))
    
    # Configuration du graphique
    fig.update_layout(
        title="Prix de l'Or PAXG/USDT",
        yaxis_title="Prix (USDT)",
        xaxis_title="Date",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.2)'
        ),
        height=600
    )
    
    return fig

def main():
    st.title("Gold Trading Assistant ")
    
    # Initialisation du prédicteur
    if 'predictor' not in st.session_state:
        logger.info("Initialisation du prédicteur...")
        st.session_state.predictor = TradingPredictor()
    
    # Récupération des données et prédictions
    try:
        current_data = st.session_state.predictor.get_current_data()
        predictions = st.session_state.predictor.get_predictions()
        
        if current_data is not None and predictions is not None:
            # Layout principal
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Graphique du Prix et Signaux")
                fig = create_candlestick_chart(current_data, predictions)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Signaux de Trading (>60% confiance)")
                
                # Filtrer uniquement les signaux de haute confiance
                high_confidence_signals = predictions[predictions['confidence'] > 0.60]
                
                if not high_confidence_signals.empty:
                    for idx, signal in high_confidence_signals.iterrows():
                        signal_color = "#26a69a" if signal['action'] == 'BUY' else "#ef5350"
                        signal_icon = "" if signal['action'] == 'BUY' else ""
                        
                        st.markdown(f"""
                        <div class="high-confidence" style="border-left: 4px solid {signal_color}">
                            <h3>{signal_icon} Signal de {signal['action']}</h3>
                            <p>Confiance: {signal['confidence']*100:.1f}%</p>
                            <p>Prix: {format_price(signal['price'])}</p>
                            <p>Take Profit: {format_price(signal['take_profit'])}</p>
                            <p>Stop Loss: {format_price(signal['stop_loss'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Ratio risque/récompense
                        risk = abs(signal['stop_loss'] - signal['price'])
                        reward = abs(signal['take_profit'] - signal['price'])
                        rr_ratio = reward / risk if risk > 0 else 0
                        
                        st.progress(min(rr_ratio/3, 1.0))
                        st.caption(f"Ratio Risque/Récompense: {rr_ratio:.2f}")
                else:
                    st.info("Pas de signaux de haute confiance pour le moment")
                
                # Dernière mise à jour
                st.caption(f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("En attente des données du marché...")
            
    except Exception as e:
        logger.error(f"Erreur dans l'application: {str(e)}")
        st.error("Une erreur s'est produite lors de la récupération des données")
    
    # Rafraîchissement automatique
    time.sleep(5)
    st.experimental_rerun()

if __name__ == "__main__":
    main()
