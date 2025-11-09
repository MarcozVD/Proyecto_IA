import streamlit as st
import joblib
import numpy as np

# ==========================
# Configuración inicial
# ==========================
st.set_page_config(page_title="Predicción de Fallas", page_icon="⚙️", layout="centered")
st.title("🔍 Predicción de Tipo de Falla")
st.write("Ingrese los valores de los sensores para predecir el tipo de falla.")

# ==========================
# Cargar modelo y encoder
# ==========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("modelo.pkl")
        encoder = joblib.load("encoder.pkl")
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
    "H1": (-0.036, 10.288),
    "DV_pressure": (-0.032, 9.844),
    "Reservoirs": (0.712, 10.300),
    "Oil_temperature": (15.4, 89.05),
    "Motor_current": (0.02, 9.295),
    "COMP": (0.0, 1.0),
    "DV_eletric": (0.0, 1.0),
    "Towers": (0.0, 1.0),
    "MPG": (0.0, 1.0),
    "LPS": (0.0, 1.0),
    "Pressure_switch": (0.0, 1.0),
    "Oil_level": (0.0, 1.0),
    "Caudal_impulses": (0.0, 1.0),
}

# ==========================
# Entradas del usuario
# ==========================
st.subheader("📊 Entradas del modelo")

inputs = {}
for feature, (min_val, max_val) in ranges.items():
    # Se usa slider si el rango es pequeño, input numérico si es amplio
    if max_val - min_val > 2:
        value = st.slider(feature, float(min_val), float(max_val), float((min_val + max_val) / 2))
    else:
        value = st.number_input(feature, float(min_val), float(max_val), float((min_val + max_val) / 2))
    inputs[feature] = value

# ==========================
# Botón de predicción
# ==========================
if st.button("🔮 Predecir tipo de falla"):
    if model is None:
        st.error("❌ No se pudo cargar el modelo. Verifica los archivos.")
    else:
        # Convertir a array para el modelo
        X_input = np.array([list(inputs.values())])
        pred = model.predict(X_input)
        
        # Decodificar si el encoder existe
        try:
            pred_label = encoder.inverse_transform(pred)[0]
        except:
            pred_label = str(pred[0])

        st.success(f"✅ **Tipo de falla predicho:** {pred_label}")

# ==========================
# Pie de página
# ==========================
st.markdown("---")
st.caption("Aplicación creada con ❤️ usando Streamlit y Python")
