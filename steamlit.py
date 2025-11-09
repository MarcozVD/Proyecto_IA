import streamlit as st
import joblib
import numpy as np

# ==========================
# Configuración inicial
# ==========================
st.set_page_config(
    page_title="Predicción de Fallas", 
    page_icon="⚙️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para mejorar la estética
st.markdown("""
    <style>
    /* Mejorar el título principal */
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Mejorar la descripción */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    
    /* Estilo para las tarjetas de entrada */
    .stSlider, .stNumberInput {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    
    /* Mejorar el botón de predicción */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Estilo para alertas de éxito */
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Separadores más elegantes */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Mejorar secciones */
    .section-header {
        color: #667eea;
        font-size: 1.5em;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Footer personalizado */
    .footer {
        text-align: center;
        color: #999;
        padding: 2rem 0 1rem 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Título con HTML personalizado
st.markdown('<h1 class="main-title">🔍 Predicción de Tipo de Falla</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ingrese los valores de los sensores para obtener una predicción precisa del tipo de falla</p>', unsafe_allow_html=True)

# ==========================
# Cargar modelo y encoder
# ==========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgboost_optimizado.joblib")
        encoder = joblib.load("encoder.joblib")
        return model, encoder
    except:
        st.warning("⚠️ No se encontró el modelo o el encoder. Asegúrate de tener los archivos 'modelo.pkl' y 'encoder.pkl'.")
        return None, None

model, encoder = load_model()

# ==========================
# Rango de sliders (min y max del dataset)
# ==========================
ranges = {
    "TP2": (-0.032, 10.676),
    "TP3": (0.73, 10.302),
    "DV_pressure": (-0.032, 9.844),
    "Oil_temperature": (15.4, 89.05),
    "Motor_current": (0.02, 9.295),
    "Towers": (0.0, 1.0),
    "LPS": (0.0, 1.0),
    "Pressure_switch": (0.0, 1.0),
    "Oil_level": (0.0, 1.0),
    "Caudal_impulses": (0.0, 1.0),
}

# ==========================
# Entradas del usuario
# ==========================
st.markdown('<div class="section-header">📊 Parámetros de Entrada</div>', unsafe_allow_html=True)

# Organizar en columnas para mejor visualización
col1, col2 = st.columns(2)

inputs = {}
features_list = list(ranges.items())

# Dividir las características en dos columnas
for idx, (feature, (min_val, max_val)) in enumerate(features_list):
    with col1 if idx % 2 == 0 else col2:
        # Usar slider si el rango es pequeño, input numérico si es amplio
        if max_val - min_val > 2:
            value = st.slider(
                feature, 
                float(min_val), 
                float(max_val), 
                float((min_val + max_val) / 2),
                help=f"Rango: {min_val:.3f} - {max_val:.3f}"
            )
        else:
            value = st.number_input(
                feature, 
                float(min_val), 
                float(max_val), 
                float((min_val + max_val) / 2),
                help=f"Rango: {min_val:.3f} - {max_val:.3f}"
            )
        inputs[feature] = value

# Espaciado antes del botón
st.markdown("<br>", unsafe_allow_html=True)

# ==========================
# Botón de predicción
# ==========================
if st.button("🔮 Realizar Predicción"):
    if model is None:
        st.error("❌ No se pudo cargar el modelo. Verifica los archivos.")
    else:
        with st.spinner('Analizando datos...'):
            # Convertir a array para el modelo
            X_input = np.array([list(inputs.values())])
            pred = model.predict(X_input)
            
            # Decodificar si el encoder existe
            try:
                pred_label = encoder.inverse_transform(pred)[0]
            except:
                pred_label = str(pred[0])

            # Mostrar resultado con estilo personalizado
            st.markdown(f"""
                <div class="success-box">
                    ✅ Tipo de falla predicho: <br><strong>{pred_label}</strong>
                </div>
            """, unsafe_allow_html=True)

# ==========================
# Pie de página
# ==========================
st.markdown("---")
st.markdown(
    '<div class="footer">Aplicación creada con ❤️ usando Streamlit y Python | © 2025</div>', 
    unsafe_allow_html=True
)