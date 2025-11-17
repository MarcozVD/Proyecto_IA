# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# Cargar modelo y scaler
# ==========================================
st.title("🔧 Predicción de Fallas — MetroPT3 AirCompressor")

@st.cache_resource
def load_model():
    scaler = joblib.load("models/scaler_knn.pkl")
    model = joblib.load("models/modelo_knn.pkl")
    return scaler, model

try:
    scaler, model = load_model()
    st.success("Modelo cargado correctamente.")
except:
    st.error("❌ No se pudo cargar el modelo. Verifica la ruta /models/")
    st.stop()

# ==========================================
# Campos que el modelo necesita
# ==========================================
FEATURES = [
    'H1', 'Towers', 'DV_eletric', 'COMP', 'MPG',
    'Reservoirs', 'TP3', 'TP2'
]

st.subheader("Ingresa los valores del sensor")

inputs = {}
for feat in FEATURES:
    inputs[feat] = st.number_input(feat, value=0.0, step=0.1)

# ==========================================
# Botón de predicción
# ==========================================
if st.button("🔍 Predecir falla"):
    data = pd.DataFrame([inputs])  # convertir a dataframe
    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    st.write("---")
    st.subheader("📌 Resultado:")

    if pred == 1:
        st.error(f"⚠ **FALLA DETECTADA** — Probabilidad: {prob*100:.2f}%")
    else:
        st.success(f"✔ **Estado normal** — Probabilidad de falla: {prob*100:.2f}%")

    st.write("---")
    st.info("El modelo está basado en KNN optimizado con SMOTE y GridSearchCV.")

# ==========================================
# Cargar datos desde archivo CSV
# ==========================================
st.subheader("📁 Predicción desde archivo CSV")

csv_file = st.file_uploader("Sube un archivo con las columnas requeridas", type=['csv'])

if csv_file is not None:
    df = pd.read_csv(csv_file)

    # Verificación de columnas
    if not all(col in df.columns for col in FEATURES):
        st.error(f"❌ El archivo debe contener estas columnas:\n{FEATURES}")
    else:
        st.success("Archivo válido. Generando predicciones...")
        df_scaled = scaler.transform(df[FEATURES])
        df["pred_falla"] = model.predict(df_scaled)
        df["prob_falla"] = model.predict_proba(df_scaled)[:,1]

        st.write(df.head())

        # Descargar resultados
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Descargar resultados CSV",
            csv_out,
            "predicciones_fallas.csv",
            "text/csv"
        )
