import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import optuna
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

# Configuración de la página
st.set_page_config(
    page_title="Optimización de Predicciones con Interpolación Fractal",
    page_icon="📈",
    layout="wide"
)

# Título y descripción
st.title("📈 Optimización de Predicciones con Interpolación Fractal")
st.markdown("""
Esta aplicación permite optimizar predicciones de series temporales utilizando técnicas de interpolación fractal. 
Se aplican tres estrategias principales: **Closest Hurst Strategy (CHS)**, **Closest Values Strategy (CVS)** y **Formula Strategy (FS)**.
Los datos interpolados se utilizan para entrenar un modelo LSTM y mejorar la precisión de las predicciones.
""")

# Función para generar datos sintéticos
def generate_synthetic_data(n_points=100, trend=0.1, seasonality=0.5, noise=0.2):
    """Genera datos sintéticos con tendencia, estacionalidad y ruido."""
    time = np.arange(n_points)
    trend_component = trend * time
    seasonality_component = seasonality * np.sin(2 * np.pi * time / 12)
    noise_component = noise * np.random.randn(n_points)
    data = trend_component + seasonality_component + noise_component
    return pd.DataFrame({
        'Fecha': pd.date_range(start='2021-01-01', periods=n_points, freq='D'),
        'Valor': data
    })

# Función para aplicar interpolación fractal
def fractal_interpolation(data, strategy='CHS', n_interpolation=10):
    """Aplica la interpolación fractal según la estrategia seleccionada."""
    interpolated_data = data.copy()
    
    if strategy == 'CHS':
        # Closest Hurst Strategy: Duplica los datos como ejemplo
        interpolated_data = pd.concat([data, data], ignore_index=True)
    
    elif strategy == 'CVS':
        # Closest Values Strategy: Interpola linealmente entre puntos
        interpolated_data = data.set_index('Fecha').resample('D').interpolate().reset_index()
    
    elif strategy == 'FS':
        # Formula Strategy: Interpola usando una fórmula específica
        interpolated_data['Valor'] = data['Valor'].rolling(window=2).mean().shift(1)
    
    return interpolated_data

# Función para entrenar el modelo LSTM
def train_lstm_model(data, n_steps=10):
    """Entrena un modelo LSTM con los datos interpolados."""
    # Preparar los datos para LSTM
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Construir el modelo LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

# Función para visualizar resultados
def plot_results(original_data, predictions, title):
    """Visualiza los resultados de las predicciones."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_data['Fecha'], y=original_data['Valor'], mode='lines', name='Datos Originales'))
    fig.add_trace(go.Scatter(x=original_data['Fecha'], y=predictions, mode='lines', name='Predicciones'))
    fig.update_layout(title=title, xaxis_title='Fecha', yaxis_title='Valor')
    st.plotly_chart(fig)

# Sidebar con instrucciones
with st.sidebar:
    st.header("Instrucciones")
    st.markdown("""
    1. Genere datos sintéticos o cargue un archivo CSV con datos de series temporales.
    2. Seleccione una estrategia de interpolación fractal (CHS, CVS, FS).
    3. Ajuste los parámetros de interpolación y entrenamiento del modelo LSTM.
    4. Visualice los resultados y compare las predicciones.
    """)
    
    # Generar datos sintéticos
    if st.button("Generar Datos Sintéticos"):
        synthetic_data = generate_synthetic_data()
        st.session_state['data'] = synthetic_data
        st.success("Datos sintéticos generados correctamente.")

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    # Entrada de datos
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    
    uploaded_file = st.file_uploader("Cargue su archivo CSV con datos de series temporales", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
    
    if st.session_state['data'] is not None:
        st.write("Vista previa de los datos:", st.session_state['data'].head())
        
        # Selección de estrategia de interpolación
        strategy = st.selectbox("Seleccione una estrategia de interpolación", ["CHS", "CVS", "FS"])
        
        # Aplicar interpolación fractal
        if st.button("Aplicar Interpolación Fractal"):
            interpolated_data = fractal_interpolation(st.session_state['data'], strategy)
            st.write("Datos interpolados:", interpolated_data.head())
            
            # Entrenar modelo LSTM
            if st.button("Entrenar Modelo LSTM"):
                model = train_lstm_model(interpolated_data[['Valor']].values)
                predictions = model.predict(interpolated_data[['Valor']].values)
                
                # Visualizar resultados
                plot_results(st.session_state['data'], predictions, "Predicciones con Interpolación Fractal")
                
                # Exportar resultados
                results_df = pd.DataFrame({
                    'Fecha': st.session_state['data']['Fecha'],
                    'Original': st.session_state['data']['Valor'],
                    'Predicciones': predictions.flatten()
                })
                
                st.download_button(
                    label="Descargar resultados (CSV)",
                    data=results_df.to_csv(index=False).encode(),
                    file_name="resultados_predicciones.csv",
                    mime="text/csv"
                )

with col2:
    # Información adicional y ayuda
    st.header("Características")
    st.markdown("""
    - Interpolación fractal para mejorar la resolución temporal de los datos.
    - Entrenamiento de un modelo LSTM para predicciones precisas.
    - Visualización interactiva de los resultados.
    - Exportación de datos interpolados y predicciones.
    """)
    
    st.header("Limitaciones")
    st.markdown("""
    - Los datos deben estar en formato de series temporales.
    - La interpolación fractal puede no ser adecuada para todos los tipos de datos.
    - El entrenamiento del modelo LSTM puede requerir tiempo y recursos computacionales.
    """)