# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==============================
# Cargar modelo y scaler
# ==============================
model = joblib.load("models/modelo_entrenado.pkl")
scaler = joblib.load("models/scaler_entrenado.pkl")

st.title("🔧 Predicción de fallas — Compresor Metro")

# ==============================
# FEATURES EXACTAS DEL MODELO
# ==============================
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

st.subheader("Ingrese valores de sensores")

user_input = {}

for col in FEATURES:
    user_input[col] = st.number_input(col, value=0.0)

# ==============================
# PREDICCIÓN
# ==============================
if st.button("Predecir falla"):
    data = pd.DataFrame([user_input])[FEATURES]

    # Escalar
    data_scaled = scaler.transform(data)

    # Predicción
    pred = model.predict(data_scaled)[0]

    if pred == 1:
        st.error("⚠ FALLA DETECTADA EN EL COMPRESOR")
    else:
        st.success("✔ Sin falla detectada")
