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

# Columnas que necesita el scaler (¡muy importante!)
FEATURES = scaler.feature_names_in_

st.title("🔧 Predicción de Fallas en Compresor – IA MetroPT3")
st.write("Ingrese los valores de los sensores para predecir si existe falla (1) o no (0).")


# ===============================
#   FORMULARIO DINÁMICO
# ===============================
form = st.form("input_form")

inputs = {}
cols = st.columns(3)

for i, col in enumerate(FEATURES):
    inputs[col] = cols[i % 3].number_input(col, value=0.0)

submit = form.form_submit_button("🔍 Predecir")


# ===============================
#   PROCESAMIENTO Y PREDICCIÓN
# ===============================
if submit:
    # Crear dataframe EXACTO con las columnas correctas
    df_input = pd.DataFrame([inputs], columns=FEATURES)

    # Escalar
    df_scaled = scaler.transform(df_input)

    # Predicción
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    st.subheader("📌 Resultado de la predicción")
    if pred == 1:
        st.error(f"⚠️ **Posible Falla Detectada** (probabilidad: {prob:.2f})")
    else:
        st.success(f"✔️ **Sistema en estado normal** (probabilidad de falla: {prob:.2f})")

    st.write("Valores usados:")
    st.dataframe(df_input)
