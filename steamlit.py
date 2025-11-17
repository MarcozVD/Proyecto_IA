# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

st.title("🔧 Predicción de Fallas — MetroPT3 AirCompressor")

# ================================
# Cargar modelo y scaler
# ================================
@st.cache_resource
def load_model():
    scaler = joblib.load("models/scaler_knn.pkl")
    model = joblib.load("models/modelo_knn.pkl")
    return scaler, model

scaler, model = load_model()

# ================================
# Lista EXACTA de columnas usadas en el entrenamiento
# Debe coincidir 1:1
# ================================
FEATURES = [
    'H1', 'Towers', 'DV_eletric', 'COMP', 'MPG',
    'Reservoirs', 'TP3', 'TP2'
]

st.subheader("Ingrese los valores de los sensores:")

inputs = {}
for col in FEATURES:
    inputs[col] = st.number_input(col, value=0.0, step=0.1)

# ================================
# Botón de predicción
# ================================
if st.button("🔍 Predecir falla"):

    # Se construye DataFrame con EXACTA estructura
    df_input = pd.DataFrame([[inputs[col] for col in FEATURES]], columns=FEATURES)

    # Escalado
    try:
        df_scaled = scaler.transform(df_input)
    except Exception as e:
        st.error("❌ Error al transformar datos. Revisa nombres y orden de las columnas.")
        st.code(str(e))
        st.stop()

    # Predicción
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    st.write("---")
    st.subheader("📌 Resultado:")

    if pred == 1:
        st.error(f"⚠ FALLA DETECTADA — Probabilidad: {prob*100:.2f}%")
    else:
        st.success(f"✔ Estado NORMAL — Probabilidad de falla: {prob*100:.2f}%")

    st.write("---")
    st.info("Modelo KNN optimizado con SMOTE + GridSearchCV.")
