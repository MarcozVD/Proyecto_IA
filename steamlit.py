import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------------------
# Cargar modelo y orden real
# ---------------------------
try:
    model = joblib.load("models/modelo_entrenado.pkl")
    column_order = joblib.load("models/column_order.pkl")
except:
    st.error("No se pudo cargar el modelo o el archivo column_order.pkl")
    st.stop()

st.title("Predicción de fallas")

# Ejemplo: genera inputs automáticos
inputs = {}

for col in column_order:
    inputs[col] = st.number_input(f"Ingrese {col}", value=0.0)

# Convertir a DataFrame
df = pd.DataFrame([inputs])

# Asegurar orden correcto
df = df[column_order]

# ---------------------------
# Predicción
# ---------------------------
try:
    pred = model.predict(df)
    st.success(f"Predicción: {pred[0]}")
except Exception as e:
    st.error(f"❌ Error al procesar predicción: {e}")
