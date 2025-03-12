import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import time
from threading import Thread
import numpy as np
from model_deeplearning.predict import TradingPredictor

# Configuration de la page
st.set_page_config(
    page_title="Gold Price Pro Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé style TradingView
st.markdown("""
    <style>
    .stApp {
        background-color: #131722;
    }
    .main {
        background-color: #131722;
    }
    .st-emotion-cache-18ni7ap {
        background-color: #131722;
    }
    .st-emotion-cache-16idsys {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-1r6slb0 {
        background-color: #2a2e39;
        border: 1px solid #363c4e;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
        color: #d1d4dc;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    h1, h2, h3 {
        color: #d1d4dc !important;
    }
    .stButton button {
        background-color: #2962ff;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background-color: #1e222d;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour obtenir les données historiques de Binance
def get_historical_data(symbol, interval):
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 100
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                       'taker_buy_quote', 'ignore'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données historiques: {str(e)}")
        return None

# Fonction pour obtenir le prix actuel
def get_current_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {"symbol": "PAXGUSDT"}  # PAXG/USDT comme proxy pour XAU/USD
        response = requests.get(url, params=params)
        data = response.json()
        
        return {
            'price': float(data['lastPrice']),
            'change': float(data['priceChange']),
            'change_percent': float(data['priceChangePercent']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'volume': float(data['volume']),
            'open': float(data['openPrice']),
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Erreur lors de la récupération du prix actuel: {str(e)}")
        return None

# Initialisation des variables de session
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Sidebar avec contrôles
with st.sidebar:
    st.title("⚙️ Paramètres")
    
    timeframe = st.selectbox(
        "Période",
        ["15m", "30m", "1h", "2h", "4h", "1D"],
        index=2
    )
    
    interval_mapping = {
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "1D": "1d"
    }
    
    st.session_state.auto_refresh = st.checkbox("Actualisation automatique", value=True)
    
    st.markdown("---")
    st.subheader("📊 Indicateurs techniques")
    show_ma = st.checkbox("Moyennes mobiles", True)
    
    if show_ma:
        ma_periods = st.multiselect(
            "Périodes MA",
            ["MA20", "MA50", "MA100", "MA200"],
            ["MA20", "MA50"]
        )

# Obtenir les données actuelles
current_data = get_current_price()

# Obtenir les données historiques
historical_data = get_historical_data("PAXGUSDT", interval_mapping[timeframe])

if current_data and historical_data is not None:
    # En-tête principal
    st.title(f"Or XAU/USD Pro Tracker ({timeframe})")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prix actuel",
            f"${current_data['price']:,.2f}",
            f"{current_data['change']:+.2f} ({current_data['change_percent']:+.2f}%)",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Plus haut",
            f"${current_data['high']:,.2f}",
            f"{(current_data['high'] - current_data['price']):+.2f}"
        )
    
    with col3:
        st.metric(
            "Plus bas",
            f"${current_data['low']:,.2f}",
            f"{(current_data['low'] - current_data['price']):+.2f}"
        )
    
    with col4:
        st.metric(
            "Volume",
            f"{current_data['volume']:,.0f}",
            "24h"
        )
    
    # Graphique principal
    st.markdown("---")
    
    fig = go.Figure()
    
    # Ajouter les chandeliers
    fig.add_trace(go.Candlestick(
        x=historical_data['timestamp'],
        open=historical_data['open'],
        high=historical_data['high'],
        low=historical_data['low'],
        close=historical_data['close'],
        name='OHLC'
    ))
    
    # Ajouter les moyennes mobiles si activées
    if show_ma:
        for ma in ma_periods:
            period = int(ma.replace('MA', ''))
            ma_data = historical_data['close'].rolling(window=period).mean()
            fig.add_trace(go.Scatter(
                x=historical_data['timestamp'],
                y=ma_data,
                name=f'MA{period}',
                line=dict(width=1)
            ))
    
    # Mise à jour du layout
    fig.update_layout(
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#d1d4dc'),
        title={
            'text': f'Graphique XAU/USD - {timeframe}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#d1d4dc')
        },
        yaxis=dict(
            gridcolor='#1e222d',
            zerolinecolor='#1e222d',
            tickformat=',.2f'
        ),
        xaxis=dict(
            gridcolor='#1e222d',
            zerolinecolor='#1e222d',
            rangeslider=dict(visible=False)
        ),
        height=600,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d1d4dc')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Informations supplémentaires
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### 📊 Statistiques du marché
        - **Ouverture**: ${current_data['open']:,.2f}
        - **Variation**: ${current_data['change']:+,.2f} ({current_data['change_percent']:+.2f}%)
        - **Volume 24h**: {current_data['volume']:,.0f}
        """)
    
    with col2:
        rsi = np.random.randint(30, 70)  # Simulé pour l'exemple
        trend = "Haussière" if current_data['change'] > 0 else "Baissière"
        st.markdown(f"""
        ### 📈 Analyse technique
        - **Tendance**: {trend}
        - **RSI (14)**: {rsi}
        - **Support**: ${(current_data['price'] - 10):,.2f}
        - **Résistance**: ${(current_data['price'] + 10):,.2f}
        """)
    
    # Deep Learning Predictions
    if st.sidebar.checkbox("Show AI Predictions", value=True):
        st.subheader("🤖 AI Trading Signals")
        
        # Initialize predictor
        predictor = TradingPredictor(confidence_threshold=0.8)
        
        # Get latest prediction
        prediction = predictor.get_latest_prediction()
        
        if prediction and prediction['signal']:
            signal = prediction['signal']
            
            # Create columns for signal details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Action",
                    signal['action'],
                    delta=f"{signal['confidence']*100:.1f}% confidence"
                )
            
            with col2:
                if signal['action'] == 'BUY':
                    tp_delta = f"+{((signal['take_profit'] - signal['entry_price']) / signal['entry_price'] * 100):.1f}%"
                else:
                    tp_delta = f"+{((signal['entry_price'] - signal['take_profit']) / signal['entry_price'] * 100):.1f}%"
                
                st.metric(
                    "Take Profit",
                    f"${signal['take_profit']:.2f}",
                    delta=tp_delta
                )
            
            with col3:
                if signal['action'] == 'BUY':
                    sl_delta = f"-{((signal['entry_price'] - signal['stop_loss']) / signal['entry_price'] * 100):.1f}%"
                else:
                    sl_delta = f"-{((signal['stop_loss'] - signal['entry_price']) / signal['entry_price'] * 100):.1f}%"
                
                st.metric(
                    "Stop Loss",
                    f"${signal['stop_loss']:.2f}",
                    delta=sl_delta
                )
            
            # Add trading rules and risk management
            with st.expander("📈 Trading Rules & Risk Management"):
                st.markdown("""
                ### Trading Rules
                1. Only take trades with confidence > 80%
                2. Always use the suggested Stop Loss and Take Profit levels
                3. Risk no more than 1-2% of your account per trade
                4. Be patient and wait for high-probability setups
                
                ### Position Sizing Calculator
                """)
                
                # Position size calculator
                account_size = st.number_input("Account Size ($)", min_value=100.0, value=10000.0, step=100.0)
                risk_percentage = st.slider("Risk per Trade (%)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
                
                risk_amount = account_size * (risk_percentage / 100)
                
                if signal['action'] == 'BUY':
                    sl_points = signal['entry_price'] - signal['stop_loss']
                else:
                    sl_points = signal['stop_loss'] - signal['entry_price']
                
                position_size = risk_amount / sl_points
                
                st.markdown(f"""
                #### Recommended Position Size
                - Risk Amount: ${risk_amount:.2f}
                - Position Size: {position_size:.4f} units
                - Total Position Value: ${(position_size * signal['entry_price']):.2f}
                """)
        
        else:
            st.info("No high-confidence trading signals available at the moment. Wait for better opportunities.")
        
        # Show historical predictions
        st.subheader("Historical Predictions")
        historical_preds = predictor.get_historical_predictions(lookback_hours=24)
        
        if historical_preds:
            # Create a DataFrame for the predictions
            df_preds = pd.DataFrame(historical_preds)
            df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])
            df_preds.set_index('timestamp', inplace=True)
            
            # Plot predictions
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=df_preds.index,
                y=df_preds['close'],
                name='Price',
                line=dict(color='blue')
            ))
            
            # Add prediction confidence
            fig.add_trace(go.Scatter(
                x=df_preds.index,
                y=df_preds['prediction'],
                name='Buy Confidence',
                line=dict(color='green', dash='dot')
            ))
            
            fig.update_layout(
                title='Historical Predictions vs Price',
                xaxis_title='Time',
                yaxis_title='Price / Confidence',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Dernière mise à jour et auto-refresh
    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown(f"<div style='text-align: center'>Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
    
    # Auto-refresh toutes les 15 secondes
    if st.session_state.auto_refresh:
        time.sleep(2)  # Petit délai pour éviter trop de requêtes
        st.experimental_rerun()
    
    st.markdown("""
    <div style='text-align: center'>
        <p style='color: #d1d4dc'>Développé avec ❤️ | Mise à jour en temps réel</p>
        <p style='color: #808a9d; font-size: 12px'>Données fournies par Binance</p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.error("Impossible de récupérer les données. Veuillez réessayer plus tard.")
