# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Predicción de Fallas – Compresor", layout="wide")

# ===============================
#   CARGAR MODELO Y SCALER
# ===============================
scaler = joblib.load("models/scaler_knn.pkl")
model = joblib.load("models/modelo_knn.pkl")

st.title("🔧 Predicción de Fallas en Compresor – IA MetroPT3")
st.write("Ingrese los valores de los sensores para predecir si existe falla (1) o no (0).")


# ===============================
#   CAMPOS NECESARIOS SEGÚN X
# ===============================
features = ['H1','Towers','DV_eletric','COMP','MPG','Reservoirs','TP3','TP2']

form = st.form("input_form")

inputs = {}
col1, col2, col3 = st.columns(3)

for i, col in enumerate(features):
    if i % 3 == 0:
        inputs[col] = col1.number_input(col, value=0.0, format="%.5f")
    elif i % 3 == 1:
        inputs[col] = col2.number_input(col, value=0.0, format="%.5f")
    else:
        inputs[col] = col3.number_input(col, value=0.0, format="%.5f")

submit = form.form_submit_button("🔍 Predecir")

# ===============================
#   PROCESAR Y PREDICCIÓN
# ===============================
if submit:
    # Crear dataframe con una sola fila
    df_input = pd.DataFrame([inputs])

    # Escalar igual que en el entrenamiento
    df_scaled = scaler.transform(df_input)

    # Predicción
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    st.subheader("📌 Resultado de la predicción")
    if pred == 1:
        st.error(f"⚠️ **Posible Falla Detectada** (probabilidad: {prob:.2f})")
    else:
        st.success(f"✔️ **Sistema en estado normal** (probabilidad de falla: {prob:.2f})")

    st.write("Valores utilizados en la predicción:")
    st.dataframe(df_input)
