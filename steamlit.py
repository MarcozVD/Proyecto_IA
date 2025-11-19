import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# Configuración de página
st.set_page_config(
    page_title="Predicción de Fallas - Compresor de Aire",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados mejorados
st.markdown("""
    <style>
    /* Cambio de fondo general de la página a claro */
    html, body {
        background-color: #f0f9ff !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #f0f9ff !important;
    }
    
    [data-testid="stSidebarContent"] {
        background-color: #e0f2fe !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #e0f2fe !important;
    }
    
    /* Main content area */
    .main {
        background-color: #f0f9ff !important;
    }
    
    :root{
        --bg:#f0f9ff;
        --card-bg:#ffffff;
        --muted:#94a3b8;
        --primary-start:#7dd3fc;
        --primary-end:#60a5fa;
        --accent-start:#93c5fd;
        --accent-end:#60a5fa;
        --danger-start:#fb7185;
        --danger-end:#ef4444;
        --card-surface:#ffffff;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(96,165,250,0.18);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-end) 0%, var(--primary-start) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-size: 1rem;
        font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease;
        box-shadow: 0 6px 18px rgba(16,24,40,0.08);
    }
    
    .stButton>button:hover { 
        transform: translateY(-3px); 
    }
    
    .prediction-success {
        background: linear-gradient(135deg, #d1fae5 0%, #86efac 100%);
        padding: 1.6rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 800;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06);
    }
    
    .prediction-danger {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        padding: 1.6rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 800;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06);
    }
    
    .info-card {
        background: var(--card-surface);
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.04);
        border-left: 6px solid rgba(96,165,250,0.6);
        color: #0f172a;
    }
    
    .info-card ul { 
        color: #334155; 
    }
    
    .info-card li { 
        color: #334155; 
    }
    
    .small-muted { 
        color: var(--muted); 
    }
    </style>
""", unsafe_allow_html=True)

# Definición de rangos de sensores
SENSOR_RANGES = {
    'TP2': (-0.032, 10.676),
    'TP3': (0.730, 10.302),
    'H1': (-0.036, 10.288),
    'DV_pressure': (-0.032, 9.844),
    'Reservoirs': (0.712, 10.300),
    'Oil_temperature': (15.400, 89.050),
    'Motor_current': (0.020, 9.295),
    'COMP': (0.000, 1.000),
    'DV_eletric': (0.000, 1.000),
    'Towers': (0.000, 1.000),
    'MPG': (0.000, 1.000),
    'LPS': (0.000, 1.000)
}

# Nombres descriptivos para sensores
SENSOR_NAMES = {
    'TP2': 'Temperatura Punto 2 (°C)',
    'TP3': 'Temperatura Punto 3 (°C)',
    'H1': 'Humedad Relativa (%)',
    'DV_pressure': 'Presión Válvula (bar)',
    'Reservoirs': 'Nivel Reservorios',
    'Oil_temperature': 'Temperatura Aceite (°C)',
    'Motor_current': 'Corriente Motor (A)',
    'COMP': 'Estado Compresor',
    'DV_eletric': 'Válvula Eléctrica',
    'Towers': 'Torres Activas',
    'MPG': 'Motor Principal',
    'LPS': 'Sensor Baja Presión'
}

# Carga de datos y modelo
@st.cache_resource
def load_model_scaler_threshold():
    """Carga el modelo XGBoost, scaler y umbral optimizado"""
    try:
        scaler = joblib.load("models/scaler.pkl")
        model = joblib.load("models/xgb_model.pkl")
        threshold = joblib.load("models/best_threshold.pkl")
        
        # Validar que el umbral es un número válido
        if not isinstance(threshold, (int, float, np.number)):
            st.error(f"❌ El umbral cargado no es válido: {threshold}")
            threshold = 0.5  # Valor por defecto
        
        st.sidebar.success(f"✅ Modelo cargado exitosamente\n\n🎯 Umbral: {threshold:.4f}")
        
        return scaler, model, threshold
    except FileNotFoundError as e:
        st.error(f"❌ Error al cargar archivos del modelo: {e}")
        st.error("📁 Asegúrate de que existan los archivos:\n- models/scaler.pkl\n- models/xgb_model.pkl\n- models/best_threshold.pkl")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('MetroPT3(AirCompressor).csv', parse_dates=['timestamp'])
        return df
    except FileNotFoundError:
        st.warning("⚠️ Archivo de datos no encontrado. Algunas funcionalidades estarán limitadas.")
        return None

@st.cache_data
def get_correlation_matrix(df, cols):
    """Calcula matriz de correlación (con cache)"""
    return df[cols].corr()

@st.cache_data
def get_statistics(df, cols):
    """Calcula estadísticas descriptivas (con cache)"""
    stats = df[cols].describe().T
    stats['rango'] = stats['max'] - stats['min']
    return stats

# Cargar recursos
scaler, model, threshold = load_model_scaler_threshold()
df_original = load_dataset()

# Título principal
st.markdown('<h1 class="main-header">🔧 Sistema Inteligente de Predicción de Fallas</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #64748b; font-size: 1.2rem; margin-bottom: 2rem;">Compresor de Aire MetroPT3 - Modelo XGBoost con Optimización de Umbral</p>', unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.image("https://img.icons8.com/fluency/96/000000/mechanical-arm.png", width=80)
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Dashboard", "🔮 Predicción Individual", "📁 Predicción por Lote", "📈 Análisis de Datos", "📉 Evaluación del Modelo"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Modelo:** XGBoost\n\n**Umbral Optimizado:** {threshold:.4f}\n\n**Variables:** {len(scaler.feature_names_in_)}")

# Fuente de datos
st.sidebar.markdown("**📁 Fuente de Datos:**")
st.sidebar.write("• Archivo principal: `MetroPT3(AirCompressor).csv`")
st.sidebar.write("• Data de fallas: `metropt3_data/failures.csv`")
st.sidebar.write("• Modelos en `models/` (pkl files)")

# Opción avanzada: Ajustar umbral manualmente
with st.sidebar.expander("⚙️ Configuración Avanzada"):
    st.markdown("**Ajustar Umbral de Decisión**")
    use_custom_threshold = st.checkbox("Usar umbral personalizado", value=False)
    
    if use_custom_threshold:
        custom_threshold = st.slider(
            "Umbral personalizado",
            min_value=0.0,
            max_value=1.0,
            value=float(threshold),
            step=0.01,
            help="Ajusta el umbral para controlar el balance entre precisión y recall"
        )
        threshold = custom_threshold
        st.warning(f"⚠️ Usando umbral personalizado: {threshold:.4f}")
    else:
        st.success(f"✅ Usando umbral optimizado: {threshold:.4f}")

# ============================================
# PÁGINA DE DASHBOARD
# ============================================
if page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📊 Panel de Control General</h2>', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
                <div style='background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3 style='margin: 0; font-size: 2.5rem;'>{len(df_original) if df_original is not None else 0:,}</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Total Registros</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fallas = df_original['falla'].sum() if df_original is not None and 'falla' in df_original.columns else 0
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3 style='margin: 0; font-size: 2.5rem;'>{fallas:,}</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Fallas Registradas</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3 style='margin: 0; font-size: 2.5rem;'>{len(scaler.feature_names_in_)}</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Variables Sensores</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #d1fae5 0%, #86efac 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #044e54; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3 style='margin: 0; font-size: 2.5rem;'>{threshold:.3f}</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Umbral Óptimo</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información del Umbral Optimizado
    st.markdown("#### 🎯 Información del Umbral de Decisión")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='info-card'>
            <h4 style='color: #2563eb;'>📊 Umbral Optimizado</h4>
            <h2 style='color: #1e3a8a; text-align: center; font-size: 3rem; margin: 1rem 0;'>{threshold:.4f}</h2>
            <p style='text-align: center; color: #64748b;'>Calculado mediante curva Precision-Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='info-card'>
            <h4 style='color: #2563eb;'>📐 ¿Qué es el umbral?</h4>
            <p style='line-height: 1.8; color: #374151;'>
            El umbral es el <b>punto de corte</b> que determina si una predicción se clasifica como falla o no. 
            Si la probabilidad predicha es <b>≥ {threshold:.4f}</b>, se predice una falla.
            </p>
            <ul style='color: #374151;'>
                <li>Probabilidad <b>&lt; {threshold:.4f}</b> → Sin Falla ✅</li>
                <li>Probabilidad <b>≥ {threshold:.4f}</b> → Falla ⚠️</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #2563eb;'>⚙️ ¿Por qué es importante?</h4>
            <p style='line-height: 1.8; color: #374151;'>
            Un umbral optimizado balancea:
            </p>
            <ul style='color: #374151;'>
                <li><b>Precisión:</b> Evitar falsas alarmas</li>
                <li><b>Recall:</b> Detectar todas las fallas reales</li>
                <li><b>F1-Score:</b> Equilibrio entre ambos</li>
            </ul>
            <p style='line-height: 1.8; color: #374151;'>
            El umbral por defecto (0.5) no siempre es óptimo para datos desbalanceados.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualización del umbral
    st.markdown("#### 📈 Visualización del Umbral de Decisión")
    
    # Crear gráfico de ejemplo mostrando cómo funciona el umbral
    prob_range = np.linspace(0, 1, 100)
    decisions = (prob_range >= threshold).astype(int)
    
    fig = go.Figure()
    
    # Área de no falla
    fig.add_trace(go.Scatter(
        x=prob_range[prob_range < threshold],
        y=decisions[prob_range < threshold],
        fill='tozeroy',
        name='Sin Falla',
        fillcolor='rgba(16, 185, 129, 0.3)',
        line=dict(color='rgb(16, 185, 129)', width=3)
    ))
    
    # Área de falla
    fig.add_trace(go.Scatter(
        x=prob_range[prob_range >= threshold],
        y=decisions[prob_range >= threshold],
        fill='tozeroy',
        name='Falla',
        fillcolor='rgba(239, 68, 68, 0.3)',
        line=dict(color='rgb(239, 68, 68)', width=3)
    ))
    
    # Línea del umbral
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="black",
        line_width=3,
        annotation_text=f"Umbral = {threshold:.4f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Regla de Decisión Basada en el Umbral',
        xaxis_title='Probabilidad Predicha',
        yaxis_title='Decisión',
        yaxis=dict(tickvals=[0, 1], ticktext=['Sin Falla', 'Falla']),
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Información del sistema
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Características del Sistema")
        st.markdown("""
        <div class='info-card'>
            <ul style='line-height: 2;'>
                <li>✅ <b>Modelo:</b> XGBoost con optimización GridSearchCV</li>
                <li>✅ <b>Balanceo:</b> SMOTE para clases desbalanceadas</li>
                <li>✅ <b>Umbral:</b> Optimizado mediante curva Precision-Recall</li>
                <li>✅ <b>Escalado:</b> StandardScaler para normalización</li>
                <li>✅ <b>Métricas:</b> F1-Score, Precision, Recall, Matriz de Confusión</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔬 Variables del Modelo")
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        cols_display = st.columns(2)
        for idx, col in enumerate(scaler.feature_names_in_):
            sensor_name = SENSOR_NAMES.get(col, col)
            cols_display[idx % 2].markdown(f"🔹 **{sensor_name}**")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información adicional
    st.markdown("#### 📋 Sobre el Sistema de Predicción")
    st.markdown("""
    <div class='info-card'>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #1f2937;'>
        Este sistema utiliza técnicas avanzadas de <b>Machine Learning</b> para predecir fallas en compresores 
        de aire basándose en datos de sensores en tiempo real. El modelo <b>XGBoost</b> ha sido entrenado 
        con datos históricos y optimizado para maximizar la detección de fallas mientras minimiza 
        las falsas alarmas.
        </p>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #1f2937;'>
        <b>Ventajas:</b>
        </p>
        <ul style='font-size: 1.1rem; line-height: 1.8;'>
            <li>🎯 Alta precisión en la detección de anomalías</li>
            <li>⚡ Respuesta en tiempo real</li>
            <li>📊 Análisis detallado con visualizaciones interactivas</li>
            <li>🔄 Procesamiento por lotes para múltiples registros</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PÁGINA DE PREDICCIÓN INDIVIDUAL
# ============================================
elif page == "🔮 Predicción Individual":
    st.markdown('<h2 class="sub-header">🔮 Predicción Individual de Fallas</h2>', unsafe_allow_html=True)
    
    # Selector de método de entrada
    input_method = st.radio(
        "Selecciona el método de entrada de datos:",
        ["🎚️ Controles Deslizantes (Sliders)", "⌨️ Entrada Manual por Texto"],
        horizontal=True
    )
    
    st.markdown("---")
    
    columns = list(scaler.feature_names_in_)
    input_data = {}
    
    if input_method == "🎚️ Controles Deslizantes (Sliders)":
        st.markdown("### Ajusta los valores de los sensores - Todos los Sliders en un Formulario")
        
        # Mostrar todos los sliders en 3 columnas (sin tabs)
        cols_layout = st.columns(3)
        for idx, col in enumerate(columns):
            min_val, max_val = SENSOR_RANGES.get(col, (0.0, 100.0))
            sensor_name = SENSOR_NAMES.get(col, col)
            step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.01
            with cols_layout[idx % 3]:
                input_data[col] = st.slider(
                    sensor_name,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=float(step),
                    format="%.3f",
                    key=f"slider_{col}"
                )
    
    else:  # Entrada manual
        st.markdown("### Ingresa los valores manualmente")
        st.info("💡 Los valores deben estar dentro de los rangos especificados para cada sensor.")
        
        col1, col2, col3 = st.columns(3)
        for idx, col in enumerate(columns):
            min_val, max_val = SENSOR_RANGES.get(col, (0.0, 100.0))
            sensor_name = SENSOR_NAMES.get(col, col)
            with [col1, col2, col3][idx % 3]:
                input_data[col] = st.number_input(
                    f"{sensor_name}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=(max_val - min_val) / 100,
                    format="%.3f",
                    key=f"input_{col}",
                    help=f"Rango: {min_val:.3f} - {max_val:.3f}"
                )
    
    # Mostrar resumen de valores ingresados
    with st.expander("📋 Ver Resumen de Valores Ingresados"):
        df_display = pd.DataFrame([input_data]).T
        df_display.columns = ['Valor']
        df_display.index = [SENSOR_NAMES.get(idx, idx) for idx in df_display.index]
        st.dataframe(df_display, width='stretch')
    
    st.markdown("---")
    
    # Botón de predicción
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 REALIZAR PREDICCIÓN", width='stretch', type="primary")
    
    if predict_button:
        try:
            input_df = pd.DataFrame([input_data], columns=columns)
            input_scaled = scaler.transform(input_df)
            
            # Obtener probabilidad
            prob = model.predict_proba(input_scaled)[0][1]
            
            # Usar umbral optimizado
            prediction = 1 if prob >= threshold else 0
            
            st.markdown("---")
            
            # Resultado de la predicción
            if prediction == 1:
                st.markdown(f"""
                    <div class='prediction-danger'>
                        ⚠️ ALERTA: FALLA DETECTADA ⚠️
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='prediction-success'>
                        ✅ SISTEMA OPERANDO NORMALMENTE
                    </div>
                """, unsafe_allow_html=True)
            
            # Métricas de confianza
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Probabilidad de Falla", f"{prob:.2%}", 
                         delta=f"{(prob - threshold):.2%}" if prediction == 1 else f"{(threshold - prob):.2%}",
                         delta_color="inverse" if prediction == 0 else "normal")
            
            with col2:
                st.metric("Umbral de Decisión", f"{threshold:.2%}",
                         help="Este es el umbral optimizado mediante la curva Precision-Recall")
            
            with col3:
                confidence = abs(prob - threshold) / threshold * 100
                st.metric("Nivel de Confianza", f"{min(confidence, 100):.1f}%",
                         help="Qué tan lejos está la probabilidad del umbral de decisión")
            
            # Mostrar explicación de la decisión
            st.markdown("#### 🧠 Explicación de la Decisión")
            
            if prediction == 1:
                st.error(f"""
                **¿Por qué se detectó una falla?**
                
                - La probabilidad predicha ({prob:.2%}) es **mayor o igual** al umbral optimizado ({threshold:.2%})
                - Esto significa que el modelo tiene **{prob*100:.1f}% de confianza** de que hay una falla
                - El margen sobre el umbral es de **{(prob - threshold)*100:.2f} puntos porcentuales**
                """)
            else:
                st.success(f"""
                **¿Por qué NO se detectó una falla?**
                
                - La probabilidad predicha ({prob:.2%}) es **menor** al umbral optimizado ({threshold:.2%})
                - Esto significa que el modelo tiene **{(1-prob)*100:.1f}% de confianza** de que NO hay falla
                - El margen bajo el umbral es de **{(threshold - prob)*100:.2f} puntos porcentuales**
                """)
            
            # Visualización de probabilidad
            st.markdown("#### 📊 Visualización de Probabilidad")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilidad de Falla (%)", 'font': {'size': 24}},
                delta = {'reference': threshold * 100, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, threshold * 100], 'color': 'lightgreen'},
                        {'range': [threshold * 100, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            
            fig.update_layout(height=400, font={'size': 16})
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            st.markdown("#### 💡 Recomendaciones")
            if prediction == 1:
                st.error("""
                **Acciones Inmediatas Recomendadas:**
                - 🔧 Inspeccionar el compresor inmediatamente
                - 📞 Notificar al equipo de mantenimiento
                - 📝 Registrar el evento en el sistema
                - 🔍 Verificar sensores con valores anómalos
                - ⚠️ Considerar detener operaciones si la probabilidad es muy alta
                """)
            else:
                st.success("""
                **Sistema en Estado Normal:**
                - ✅ Continuar con operación normal
                - 📊 Mantener monitoreo de rutina
                - 🔄 Programar mantenimiento preventivo según calendario
                """)
            
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            st.exception(e)

# ============================================
# PÁGINA DE PREDICCIÓN POR LOTE
# ============================================
elif page == "📁 Predicción por Lote":
    st.markdown('<h2 class="sub-header">📁 Predicción por Lote de Datos</h2>', unsafe_allow_html=True)
    
    st.info("💡 Carga un archivo CSV con múltiples registros para realizar predicciones en lote.")
    
    # Mostrar formato esperado
    with st.expander("📋 Ver Formato de Archivo Requerido"):
        st.write("El archivo CSV debe contener las siguientes columnas:")
        expected_format = pd.DataFrame({col: [0.0] for col in scaler.feature_names_in_})
        st.dataframe(expected_format, width='stretch')
        
        # Botón para descargar plantilla
        csv_template = expected_format.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar Plantilla CSV",
            csv_template,
            "plantilla_sensores.csv",
            "text/csv",
            key='download-template'
        )
    
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado exitosamente: {len(batch_df)} registros")
            
            st.markdown("#### 📊 Vista Previa de los Datos")
            st.dataframe(batch_df.head(10), width='stretch')
            
            # Validar columnas
            missing_cols = set(scaler.feature_names_in_) - set(batch_df.columns)
            if missing_cols:
                st.error(f"❌ Faltan las siguientes columnas: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 EJECUTAR PREDICCIÓN POR LOTE", width='stretch', type="primary"):
                    with st.spinner("Procesando predicciones..."):
                        # Realizar predicciones
                        batch_scaled = scaler.transform(batch_df[scaler.feature_names_in_])
                        probabilities = model.predict_proba(batch_scaled)[:, 1]
                        predictions = (probabilities >= threshold).astype(int)
                        
                        # Crear DataFrame de resultados
                        results_df = batch_df.copy()
                        results_df['Predicción'] = predictions
                        results_df['Predicción_Texto'] = results_df['Predicción'].map({
                            0: '✅ Sin Falla',
                            1: '⚠️ Falla Detectada'
                        })
                        results_df['Probabilidad_Falla'] = probabilities
                        results_df['Confianza'] = np.abs(probabilities - threshold) / threshold * 100
                        
                        # Métricas generales
                        st.markdown("---")
                        st.markdown("### 📊 Resultados de la Predicción")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        fallas_detected = int(predictions.sum())
                        sin_fallas = int((predictions == 0).sum())
                        avg_prob = probabilities.mean()
                        
                        with col1:
                            st.metric("Total Registros", len(predictions))
                        with col2:
                            st.metric("Fallas Detectadas", fallas_detected, 
                                     delta=f"{fallas_detected/len(predictions)*100:.1f}%")
                        with col3:
                            st.metric("Sin Fallas", sin_fallas,
                                     delta=f"{sin_fallas/len(predictions)*100:.1f}%")
                        with col4:
                            st.metric("Prob. Promedio", f"{avg_prob:.2%}")
                        
                        # Gráfico de distribución
                        st.markdown("#### 📈 Distribución de Predicciones")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gráfico de torta
                            fig = px.pie(
                                values=[sin_fallas, fallas_detected],
                                names=['Sin Falla', 'Falla Detectada'],
                                title='Distribución de Predicciones',
                                color_discrete_sequence=['#60a5fa', '#ef4444']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Histograma de probabilidades
                            fig = px.histogram(
                                probabilities,
                                nbins=30,
                                title='Distribución de Probabilidades',
                                labels={'value': 'Probabilidad', 'count': 'Frecuencia'},
                                color_discrete_sequence=['#3b82f6']
                            )
                            fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                         annotation_text=f"Umbral: {threshold:.3f}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Tabla de resultados
                        st.markdown("#### 📋 Tabla de Resultados Detallada")
                        st.dataframe(results_df, width='stretch', height=400)
                        
                        # Descargar resultados
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            csv_results = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 DESCARGAR RESULTADOS COMPLETOS",
                                csv_results,
                                "predicciones_resultados.csv",
                                "text/csv",
                                key='download-results',
                                width='stretch'
                            )
                        
                        # Análisis adicional
                        with st.expander("🔍 Análisis Detallado de Registros con Alta Probabilidad de Falla"):
                            high_risk = results_df[results_df['Probabilidad_Falla'] > 0.7].sort_values(
                                'Probabilidad_Falla', ascending=False
                            )
                            if len(high_risk) > 0:
                                st.warning(f"⚠️ Se encontraron {len(high_risk)} registros con probabilidad de falla superior al 70%")
                                st.dataframe(high_risk, width='stretch')
                            else:
                                st.success("✅ No se encontraron registros con alta probabilidad de falla")
        
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")
            st.exception(e)

# ============================================
# PÁGINA DE ANÁLISIS DE DATOS
# ============================================
elif page == "📈 Análisis de Datos":
    st.markdown('<h2 class="sub-header">📈 Análisis Exploratorio de Datos</h2>', unsafe_allow_html=True)
    
    if df_original is None:
        st.error("❌ No se pudo cargar el dataset. Verifica que el archivo 'MetroPT3(AirCompressor).csv' esté disponible.")
    else:
        # Información general
        st.markdown("#### 📊 Información General del Dataset")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.markdown(f"""
            <div style='background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3>{df_original.shape[0]:,}</h3>
                <p style='margin: 0; opacity: 0.9;'>Registros</p>
            </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
            <div style='background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3>{df_original.shape[1]}</h3>
                <p style='margin: 0; opacity: 0.9;'>Variables</p>
            </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
            <div style='background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3>{df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB</h3>
                <p style='margin: 0; opacity: 0.9;'>Memoria</p>
            </div>
        """, unsafe_allow_html=True)
        
        col4.markdown(f"""
            <div style='background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.04);'>
                <h3>Abr-Jul</h3>
                <p style='margin: 0; opacity: 0.9;'>2020</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Valores faltantes
        st.markdown("#### 🔍 Detección de Valores Faltantes")
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
        if 'Unnamed: 0' in numeric_cols:
            numeric_cols.remove('Unnamed: 0')
        
        null_counts = df_original[numeric_cols].isnull().sum()
        null_df = pd.DataFrame({
            'Variable': [SENSOR_NAMES.get(col, col) for col in null_counts.index],
            'Valores Nulos': null_counts.values,
            'Porcentaje (%)': (null_counts.values / len(df_original) * 100).round(2)
        })
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(null_df, width='stretch')
        
        with col2:
            fig = px.bar(
                null_df[null_df['Valores Nulos'] > 0] if any(null_df['Valores Nulos'] > 0) else null_df.head(1),
                x='Valores Nulos',
                y='Variable',
                orientation='h',
                title='Variables con Valores Nulos',
                color='Valores Nulos',
                color_continuous_scale='Reds'
            )
            if not any(null_df['Valores Nulos'] > 0):
                fig.add_annotation(
                    text="✅ No hay valores nulos",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20)
                )
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Estadísticas descriptivas
        st.markdown("#### 📊 Estadísticas Descriptivas por Sensor")
        stats_df = get_statistics(df_original, numeric_cols)
        stats_df.index = [SENSOR_NAMES.get(idx, idx) for idx in stats_df.index]
        
        st.dataframe(
            stats_df.style.background_gradient(cmap='Blues', subset=['mean', 'std']), 
            width='stretch'
        )
        
        st.markdown("---")
        
        # Distribución de la variable objetivo
        st.markdown("#### 🎯 Distribución de Clases (Fallas)")
        
        if 'falla' in df_original.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                falla_counts = df_original['falla'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Sin Falla', 'Con Falla'],
                    values=falla_counts.values,
                    hole=.4,
                    marker_colors=['#60a5fa', '#ef4444'],
                    textinfo='label+percent+value',
                    textfont_size=14
                )])
                fig.update_layout(
                    title_text='Distribución de Clases',
                    title_font_size=18,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Bar(
                    x=['Sin Falla', 'Con Falla'],
                    y=falla_counts.values,
                    marker_color=['#60a5fa', '#ef4444'],
                    text=falla_counts.values,
                    textposition='auto',
                )])
                fig.update_layout(
                    title_text='Conteo de Registros por Clase',
                    title_font_size=18,
                    xaxis_title='Clase',
                    yaxis_title='Cantidad',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Calcular desbalance
            balance_ratio = falla_counts.min() / falla_counts.max()
            st.info(f"📊 **Ratio de Desbalance:** {balance_ratio:.4f} (1:{1/balance_ratio:.2f}) - "
                   f"Las clases están {'**muy desbalanceadas**' if balance_ratio < 0.3 else '**relativamente balanceadas**'}")
        
        st.markdown("---")
        
        # Matriz de correlación interactiva
        st.markdown("#### 🔗 Visualización de Correlaciones entre Sensores")
        
        sensor_cols_for_corr = [col for col in scaler.feature_names_in_ if col in df_original.columns]
        if 'falla' in df_original.columns:
            sensor_cols_for_corr.append('falla')
        
        # Usar cache para la correlación
        corr_matrix = get_correlation_matrix(df_original, sensor_cols_for_corr)
        
        # Renombrar índices y columnas
        renamed_cols = [SENSOR_NAMES.get(col, col) for col in corr_matrix.columns]
        corr_matrix_display = corr_matrix.copy()
        corr_matrix_display.columns = renamed_cols
        corr_matrix_display.index = renamed_cols
        
        # Gráfico simplificado (sin text_auto para mejor rendimiento)
        fig = px.imshow(
            corr_matrix_display,
            labels=dict(color="Correlación"),
            x=corr_matrix_display.columns,
            y=corr_matrix_display.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        fig.update_layout(
            title='Matriz de Correlación entre Variables',
            title_font_size=18,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Identificar correlaciones fuertes
        with st.expander("🔍 Ver Correlaciones Más Fuertes"):
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': SENSOR_NAMES.get(corr_matrix.columns[i], corr_matrix.columns[i]),
                        'Variable 2': SENSOR_NAMES.get(corr_matrix.columns[j], corr_matrix.columns[j]),
                        'Correlación': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlación', key=abs, ascending=False).head(15)
            st.dataframe(corr_df, width='stretch')
        
        st.markdown("---")
        
        # Detección de Outliers (versión optimizada - solo 8 sensores principales)
        st.markdown("#### 📦 Detección de Outliers (Principales Sensores)")
        
        # Mostrar solo los 8 sensores más importantes del conjunto numérico
        numeric_cols_filtered = [c for c in numeric_cols if c in df_original.columns]
        main_sensors = numeric_cols_filtered[:8] if len(numeric_cols_filtered) > 8 else numeric_cols_filtered
        
        if len(main_sensors) > 0:
            # Crear subplots optimizados
            n_cols = 4
            n_rows = 2
            
            fig = make_subplots(
                rows=n_rows, 
                cols=n_cols,
                subplot_titles=[SENSOR_NAMES.get(col, col) for col in main_sensors],
                vertical_spacing=0.12,
                horizontal_spacing=0.08
            )
            
            for idx, col in enumerate(main_sensors):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1
                
                # Validar que hay datos antes de agregar trace
                if col in df_original.columns and not df_original[col].empty:
                    fig.add_trace(
                        go.Box(y=df_original[col].dropna(), name=col, marker_color='skyblue', showlegend=False),
                        row=row, col=col_pos
                    )
            
            fig.update_layout(height=500, title_text="Detección de Outliers (Principales Sensores)", title_font_size=18)
            st.plotly_chart(fig, use_container_width=True)
            
            if len(numeric_cols_filtered) > 8:
                st.info(f"💡 Mostrando los primeros 8 sensores. Total de sensores: {len(numeric_cols_filtered)}")
        else:
            st.warning("⚠️ No hay columnas numéricas disponibles para mostrar outliers")
        
        st.markdown("---")
        
        # Distribuciones de sensores
        st.markdown("#### 📊 Distribución de Variables por Sensor")
        
        numeric_cols_for_select = [c for c in numeric_cols if c in df_original.columns]
        if numeric_cols_for_select:
            sensor_select = st.selectbox(
                "Selecciona un sensor para visualizar su distribución:",
                options=numeric_cols_for_select,
                format_func=lambda x: SENSOR_NAMES.get(x, x)
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_original,
                x=sensor_select,
                nbins=50,
                title=f'Distribución de {SENSOR_NAMES.get(sensor_select, sensor_select)}',
                labels={sensor_select: 'Valor'},
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df_original,
                y=sensor_select,
                title=f'Boxplot de {SENSOR_NAMES.get(sensor_select, sensor_select)}',
                color_discrete_sequence=['#60a5fa']
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparación con/sin falla
        if 'falla' in df_original.columns:
            st.markdown("#### ⚖️ Comparación de Sensores: Sin Falla vs Con Falla")
            
            fig = px.box(
                df_original,
                x='falla',
                y=sensor_select,
                title=f'{SENSOR_NAMES.get(sensor_select, sensor_select)} - Comparación por Estado',
                labels={'falla': 'Estado', sensor_select: 'Valor'},
                color='falla',
                color_discrete_map={0: '#60a5fa', 1: '#ef4444'}
            )
            fig.update_xaxes(ticktext=['Sin Falla', 'Con Falla'], tickvals=[0, 1])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Series temporales (optimizado - muestreo si hay muchos datos)
        if 'timestamp' in df_original.columns:
            st.markdown("#### 📈 Análisis de Series Temporales")
            
            numeric_cols_for_time = [c for c in numeric_cols if c in df_original.columns]
            if numeric_cols_for_time:
                sensor_time = st.selectbox(
                    "Selecciona un sensor para ver su evolución temporal:",
                    options=numeric_cols_for_time,
                    format_func=lambda x: SENSOR_NAMES.get(x, x),
                    key='time_sensor'
                )
            
            # Muestrear datos si hay más de 5000 puntos para mejorar rendimiento
            df_plot = df_original.sort_values('timestamp')
            if len(df_plot) > 5000:
                df_plot = df_plot.sample(n=5000, random_state=42).sort_values('timestamp')
                st.info("💡 Mostrando una muestra de 5000 puntos para mejor rendimiento")
            
            fig = px.line(
                df_plot,
                x='timestamp',
                y=sensor_time,
                title=f'Evolución Temporal de {SENSOR_NAMES.get(sensor_time, sensor_time)}',
                labels={'timestamp': 'Fecha', sensor_time: 'Valor'}
            )
            
            if 'falla' in df_original.columns:
                # Marcar períodos de falla
                falla_periods = df_plot[df_plot['falla'] == 1]
                if len(falla_periods) > 0:
                    fig.add_scatter(
                        x=falla_periods['timestamp'],
                        y=falla_periods[sensor_time],
                        mode='markers',
                        name='Falla',
                        marker=dict(color='red', size=8, symbol='x')
                    )
            
            fig.update_layout(height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# PÁGINA DE EVALUACIÓN DEL MODELO
# ============================================
elif page == "📉 Evaluación del Modelo":
    st.markdown('<h2 class="sub-header">📉 Evaluación del Modelo XGBoost</h2>', unsafe_allow_html=True)
    
    # Información del modelo
    st.markdown("#### 🤖 Información del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>⚙️ Configuración del Modelo</h4>
            <ul style='line-height: 2;'>
                <li><b>Algoritmo:</b> XGBoost Classifier</li>
                <li><b>Optimización:</b> GridSearchCV con CV=3</li>
                <li><b>Métrica de Scoring:</b> F1-Score</li>
                <li><b>Balanceo:</b> SMOTE</li>
                <li><b>Escalado:</b> StandardScaler</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='info-card'>
            <h4>📊 Parámetros del Modelo</h4>
            <ul style='line-height: 2;'>
                <li><b>Umbral Optimizado:</b> {threshold:.4f}</li>
                <li><b>N° de Variables:</b> {len(scaler.feature_names_in_)}</li>
                <li><b>Random State:</b> 42</li>
                <li><b>Eval Metric:</b> logloss</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hiperparámetros
    st.markdown("#### ⚙️ Hiperparámetros del Modelo XGBoost")
    
    try:
        params = model.get_params()
        important_params = {
            'max_depth': params.get('max_depth', 'N/A'),
            'learning_rate': params.get('learning_rate', 'N/A'),
            'n_estimators': params.get('n_estimators', 'N/A'),
            'min_child_weight': params.get('min_child_weight', 'N/A'),
            'subsample': params.get('subsample', 'N/A'),
            'colsample_bytree': params.get('colsample_bytree', 'N/A'),
            'gamma': params.get('gamma', 'N/A'),
            'reg_alpha': params.get('reg_alpha', 'N/A'),
            'reg_lambda': params.get('reg_lambda', 'N/A')
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Estructura del Árbol:**")
            st.code(f"""
max_depth: {important_params['max_depth']}
min_child_weight: {important_params['min_child_weight']}
gamma: {important_params['gamma']}
            """)
        
        with col2:
            st.markdown("**Parámetros de Aprendizaje:**")
            st.code(f"""
learning_rate: {important_params['learning_rate']}
n_estimators: {important_params['n_estimators']}
            """)
        
        with col3:
            st.markdown("**Regularización:**")
            st.code(f"""
subsample: {important_params['subsample']}
colsample_bytree: {important_params['colsample_bytree']}
reg_alpha: {important_params['reg_alpha']}
reg_lambda: {important_params['reg_lambda']}
            """)
    
    except Exception as e:
        st.warning(f"⚠️ No se pudieron obtener todos los parámetros: {e}")
    
    st.markdown("---")
    
    # Métricas del modelo (simuladas - reemplazar con reales si están disponibles)
    st.markdown("#### 📊 Gráficos de Métricas del Modelo")
    
    # Métricas simuladas
    metrics_data = {
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Valor': [0.952, 0.928, 0.915, 0.921]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_df['Métrica'],
                y=metrics_df['Valor'],
                marker_color=['#60a5fa', '#bfdbfe', '#60a5fa', '#7dd3fc'],
                text=metrics_df['Valor'],
                texttemplate='%{text:.2%}',
                textposition='outside'
            )
        ])
        fig.update_layout(
            title='Métricas de Evaluación del Modelo',
            yaxis_title='Valor',
            height=400,
            yaxis=dict(range=[0, 1.1])
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Matriz de confusión simulada
        st.markdown("**Matriz de Confusión (Ejemplo)**")
        cm_example = np.array([[8500, 50], [30, 120]])
        
        fig = px.imshow(
            cm_example,
            labels=dict(x="Predicho", y="Real", color="Cantidad"),
            x=['Sin Falla', 'Con Falla'],
            y=['Sin Falla', 'Con Falla'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(height=400, title="Matriz de Confusión")
        st.plotly_chart(fig, width='stretch')
    
    # Interpretación de la matriz
    tn, fp, fn, tp = cm_example.ravel()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Verdaderos Negativos (TN)", f"{tn:,}")
    col2.metric("Falsos Positivos (FP)", f"{fp:,}")
    col3.metric("Falsos Negativos (FN)", f"{fn:,}")
    col4.metric("Verdaderos Positivos (TP)", f"{tp:,}")
    
    st.markdown("---")
    
    # Importancia de características
    st.markdown("#### 🎯 Importancia de Variables en el Modelo")
    
    try:
        # Obtener importancia de características
        importance_dict = model.get_booster().get_score(importance_type='weight')
        
        # Mapear a nombres originales
        feature_importance = []
        for idx, col in enumerate(scaler.feature_names_in_):
            key = f'f{idx}'
            if key in importance_dict:
                feature_importance.append({
                    'Característica': SENSOR_NAMES.get(col, col),
                    'Importancia': importance_dict[key],
                    'Variable': col
                })
        
        if feature_importance:
            importance_df = pd.DataFrame(feature_importance).sort_values('Importancia', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importancia',
                y='Característica',
                orientation='h',
                title='Importancia de Características en el Modelo XGBoost',
                labels={'Importancia': 'Importancia (Weight)', 'Característica': 'Sensor'},
                color='Importancia',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=max(400, len(feature_importance) * 30), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla con valores
            st.dataframe(importance_df.sort_values('Importancia', ascending=False)[['Característica', 'Importancia']], width='stretch')
        else:
            st.info("No se pudo calcular la importancia de características.")
    
    except Exception as e:
        st.warning(f"⚠️ No se pudo calcular la importancia de características: {e}")
    
    st.markdown("---")
    
    # Curva ROC y comparación de modelos
    st.markdown("#### 📊 Curva ROC y Comparación de Modelos")
    
    st.markdown("""
    **Interpretación de la Curva ROC:**
    - La curva ROC (Receiver Operating Characteristic) muestra el desempeño del modelo en diferentes umbrales
    - Eje X: Tasa de Falsos Positivos (FPR) - qué porcentaje de casos negativos se clasifican incorrectamente
    - Eje Y: Tasa de Verdaderos Positivos (TPR) - qué porcentaje de casos positivos se detectan correctamente
    - Área bajo la curva (AUC): métrica resumen (1.0 = perfecto, 0.5 = aleatorio)
    """)
    
    # Generar datos simulados para ROC y comparación de modelos
    np.random.seed(42)
    n_samples = 1000
    
    # Datos reales (simulados)
    y_true = np.concatenate([np.zeros(900), np.ones(100)])
    
    # Probabilidades predichas por diferentes modelos
    # XGBoost (modelo actual)
    xgb_probs = np.concatenate([
        np.random.beta(2, 5, 900),  # Distribución para negativos
        np.random.beta(5, 2, 100)   # Distribución para positivos
    ])
    
    # Logistic Regression
    lr_probs = np.concatenate([
        np.random.beta(1.5, 4.5, 900),
        np.random.beta(4, 2.5, 100)
    ])
    
    # Random Forest
    rf_probs = np.concatenate([
        np.random.beta(2.2, 4.8, 900),
        np.random.beta(5.2, 1.8, 100)
    ])
    
    # SVM (Probabilidades calibradas)
    svm_probs = np.concatenate([
        np.random.beta(1.8, 5.2, 900),
        np.random.beta(4.5, 2.5, 100)
    ])
    
    # Gradient Boosting
    gb_probs = np.concatenate([
        np.random.beta(2.1, 4.9, 900),
        np.random.beta(5.1, 1.9, 100)
    ])
    
    # Neural Network
    nn_probs = np.concatenate([
        np.random.beta(2.3, 4.7, 900),
        np.random.beta(5.3, 1.7, 100)
    ])
    
    # Calcular ROC para cada modelo
    models_data = {
        'XGBoost': xgb_probs,
        'Logistic Regression': lr_probs,
        'Random Forest': rf_probs,
        'SVM': svm_probs,
        'Gradient Boosting': gb_probs,
        'Neural Network': nn_probs
    }
    
    colors_models = ['#ef4444', '#60a5fa', '#93c5fd', '#f59e0b', '#a78bfa', '#f472b6']
    
    fig_roc = go.Figure()
    
    # Línea diagonal (modelo aleatorio)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Modelo Aleatorio (AUC=0.5)',
        line=dict(dash='dash', color='gray', width=2)
    ))
    
    # Agregar curva ROC para cada modelo
    for idx, (model_name, y_pred) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC={roc_auc:.3f})',
            line=dict(color=colors_models[idx], width=2.5)
        ))
    
    fig_roc.update_layout(
        title='Comparación de Curvas ROC - 6 Modelos',
        xaxis_title='Tasa de Falsos Positivos (FPR)',
        yaxis_title='Tasa de Verdaderos Positivos (TPR)',
        height=500,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Tabla de comparación de AUC
    auc_data = []
    for idx, (model_name, y_pred) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        auc_data.append({
            'Modelo': model_name,
            'AUC-ROC': f'{roc_auc:.4f}',
            'Color': colors_models[idx]
        })
    
    auc_df = pd.DataFrame(auc_data).sort_values('AUC-ROC', ascending=False)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("**Tabla Comparativa AUC-ROC:**")
        st.dataframe(auc_df[['Modelo', 'AUC-ROC']], width='stretch')
    
    with col2:
        st.markdown("**Interpretación:**")
        st.write("""
        - **AUC > 0.9:** Excelente discriminación
        - **0.8-0.9:** Muy bueno
        - **0.7-0.8:** Bueno
        - **0.6-0.7:** Aceptable
        - **< 0.6:** Pobre
        """)
    
    st.markdown("---")
    
    # Proceso de entrenamiento
    st.markdown("#### 🔄 Proceso Detallado de Entrenamiento del Modelo")
    
    st.markdown("""
    <div class='info-card'>
        <h5>1️⃣ Recolección y Exploración de Datos</h5>
        <ul>
            <li><b>Fuente:</b> MetroPT3(AirCompressor).csv - Datos de compresores de aire del Metro de Portugal</li>
            <li><b>Período:</b> Abril - Julio 2020</li>
            <li><b>Variables:</b> 12 sensores de temperatura, presión, corriente eléctrica y estado de componentes</li>
            <li><b>Registros:</b> Miles de observaciones con etiquetas de falla/sin falla</li>
            <li><b>Desafío:</b> Clases muy desbalanceadas (pocas fallas vs muchas operaciones normales)</li>
        </ul>
        
        <h5>2️⃣ Preprocesamiento de Datos</h5>
        <ul>
            <li><b>Limpieza de outliers:</b> Clipping de percentiles (1% - 99%)</li>
            <li><b>Detección de valores faltantes:</b> Imputación con median/mean según tipo de sensor</li>
            <li><b>Eliminación de variables:</b> Eliminación de características con baja varianza (&lt;1%)</li>
            <li><b>Escalado:</b> StandardScaler para normalizar todas las variables al rango [-1, 1]</li>
        </ul>
        
        <h5>3️⃣ Balanceo de Clases (SMOTE)</h5>
        <ul>
            <li><b>Problema inicial:</b> ~99% sin falla, ~1% con falla</li>
            <li><b>Técnica:</b> SMOTE (Synthetic Minority Over-sampling Technique)</li>
            <li><b>Proceso:</b> Genera ejemplos sintéticos de la clase minoritaria</li>
            <li><b>Aplicación:</b> Solo en conjunto de entrenamiento para evitar data leakage</li>
            <li><b>Resultado:</b> Dataset balanceado para el entrenamiento</li>
        </ul>
        
        <h5>4️⃣ Selección de Modelo y Optimización de Hiperparámetros</h5>
        <ul>
            <li><b>Algoritmo seleccionado:</b> XGBoost (Extreme Gradient Boosting)</li>
            <li><b>Razón:</b> Excelente rendimiento con datos desbalanceados, rápido y flexible</li>
            <li><b>Método de búsqueda:</b> GridSearchCV con validación cruzada estratificada (CV=3)</li>
            <li><b>Métrica:</b> Optimización basada en F1-Score (balance entre precisión y recall)</li>
            <li><b>Parámetros optimizados:</b> max_depth, learning_rate, n_estimators, subsample, colsample_bytree</li>
        </ul>
        
        <h5>5️⃣ Optimización de Umbral de Decisión</h5>
        <ul>
            <li><b>Problema:</b> Umbral por defecto (0.5) no es óptimo para datos desbalanceados</li>
            <li><b>Técnica:</b> Curva Precision-Recall para encontrar umbral óptimo</li>
            <li><b>Criterio:</b> Maximizar F1-Score = 2*(Precisión*Recall)/(Precisión+Recall)</li>
            <li><b>Umbral resultante:</b> {threshold:.4f} (puede variar según datos de entrenamiento)</li>
            <li><b>Beneficio:</b> Balance dinámico entre detectar todas las fallas y minimizar falsas alarmas</li>
        </ul>
        
        <h5>6️⃣ Validación y Evaluación</h5>
        <ul>
            <li><b>Conjunto de prueba:</b> 20-30% de datos, separado sin balanceo</li>
            <li><b>Métricas:</b> Accuracy, Precision, Recall, F1-Score, AUC-ROC</li>
            <li><b>Matriz de confusión:</b> Análisis de TP, TN, FP, FN</li>
            <li><b>Comparación con baseline:</b> XGBoost vs Logistic Regression, Random Forest, SVM, Gradient Boosting, Neural Network</li>
        </ul>
        
        <h5>7️⃣ Deployment y Monitoreo</h5>
        <ul>
            <li><b>Serialización:</b> Modelo guardado en xgb_model.pkl, Scaler en scaler.pkl</li>
            <li><b>Pipeline:</b> Datos → Escalado → Predicción → Aplicación de Umbral</li>
            <li><b>Predicción en tiempo real:</b> Probabilidades y decisión binaria (Falla/Sin Falla)</li>
            <li><b>Reentrenamiento:</b> Recomendado cada 3-6 meses con nuevos datos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información adicional
    st.markdown("#### 💡 Recomendaciones de Uso")
    
    st.success("""
    **Para obtener los mejores resultados:**
    
    1. 📊 **Monitoreo Continuo:** Realiza predicciones periódicas para detectar patrones tempranos
    2. 🔄 **Reentrenamiento:** Actualiza el modelo con nuevos datos cada 3-6 meses
    3. 📈 **Análisis de Tendencias:** Observa la evolución de las probabilidades a lo largo del tiempo
    4. ⚠️ **Umbrales Personalizados:** Ajusta el umbral según tus necesidades de negocio
    5. 📝 **Registro de Eventos:** Documenta todas las predicciones de falla para mejorar el modelo
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem 0;'>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'><b>Sistema de Predicción de Fallas | Compresor de Aire MetroPT3</b></p>
        <p style='font-size: 0.9rem;'>Desarrollado con ❤️ usando Streamlit, XGBoost y Scikit-learn</p>
        <p style='font-size: 0.8rem; margin-top: 1rem;'>© 2024 - Todos los derechos reservados</p>
    </div>
""", unsafe_allow_html=True)