import streamlit as st
import joblib
import pandas as pd

# ================================
# CARGA DEL MODELO Y SCALER
# ================================
model = joblib.load("modelo_entrenado.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = ['DV_pressure', 'LPS', 'Motor_current', 'Oil_temperature']

st.title("Predicción de Fallas con XGBoost 🔧⚙️")
st.write("Modelo cargado sin necesidad de subir CSV.")

# ================================
# FORMULARIO DE ENTRADA
# ================================
st.subheader("Ingresa los valores para la predicción:")

dv_pressure = st.number_input("DV_pressure", min_value=0.0, value=1.0)
lps = st.number_input("LPS", min_value=0.0, value=1.0)
motor_current = st.number_input("Motor_current", min_value=0.0, value=1.0)
oil_temperature = st.number_input("Oil_temperature", min_value=0.0, value=40.0)

# ================================
# BOTÓN PARA PREDICCIÓN
# ================================
if st.button("Predecir"):
    try:
        # Construir dataframe EXACTO como el usado en entrenamiento
        data = pd.DataFrame([{
            'DV_pressure': dv_pressure,
            'LPS': lps,
            'Motor_current': motor_current,
            'Oil_temperature': oil_temperature
        }])[FEATURES]  # Garantiza orden correcto

        # Escalar con el mismo scaler del entrenamiento
        data_scaled = scaler.transform(data)

        # Predicción
        prediction = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0][prediction]

        if prediction == 1:
            st.error(f"⚠️ FALLA DETECTADA — Probabilidad: {proba:.4f}")
        else:
            st.success(f"✔️ Sistema Operando — Probabilidad: {proba:.4f}")

    except Exception as e:
        st.error("❌ Error inesperado durante la predicción:")
        st.code(str(e))
