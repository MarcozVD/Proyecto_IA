# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🔧 Predicción de Falla - Compresor Metro")

# ================================
# Mostrar archivos existentes (debug)
# ================================
st.write("📂 Archivos disponibles en el proyecto:")
for f in os.listdir("."):
    st.write(" - ", f)

if os.path.exists("models"):
    st.write("📁 Archivos dentro de /models:")
    for f in os.listdir("models"):
        st.write(" - ", f)
else:
    st.error("❌ La carpeta 'models' no existe")
    st.stop()

# ================================
# Cargar modelo y scaler
# ================================
try:
    model = joblib.load("models/modelo_knn.pkl")
    scaler = joblib.load("models/scaler_knn.pkl")
    st.success("✔ Modelo y scaler cargados correctamente")
except Exception as e:
    st.error(f"❌ Error cargando modelo/scaler: {e}")
    st.stop()

# ================================
# FEATURES EXACTAS DEL ENTRENAMIENTO
# ================================
FEATURES = [
    "H1",
    "Towers",
    "DV_eletric",
    "COMP",
    "MPG",
    "Reservoirs",
    "TP3",
    "TP2",
    "LPS",
    "DV_pressure",
    "Motor_current",
    "Oil_temperature"
]

st.subheader("📥 Ingrese los valores de sensores")

inputs = {}

# Crear inputs numéricos
for col in FEATURES:
    inputs[col] = st.number_input(col, value=0.0)

# Convertir a DataFrame
data_input = pd.DataFrame([inputs])

# ================================
# BOTÓN DE PREDICCIÓN
# ================================
if st.button("🔍 Predecir Falla"):
    try:
        # Escalar
        data_scaled = scaler.transform(data_input)

        # Predicción
        pred = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0][1]

        st.write("---")
        if pred == 1:
            st.error(f"⚠ FALLA DETECTADA (probabilidad {proba:.2f})")
        else:
            st.success(f"✔ SIN Falla (probabilidad de falla {proba:.2f})")

    except Exception as e:
        st.error(f"❌ Error al procesar predicción: {e}")
