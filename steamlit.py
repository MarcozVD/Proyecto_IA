# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Predicción de Fallas – Compresor", layout="wide")

# ===============================
#   CARGAR MODELO Y SCALER
# ===============================
try:
    scaler = joblib.load("models/scaler_knn.pkl")
    model = joblib.load("models/modelo_knn.pkl")
except:
    st.error("❌ No se encontró el modelo o el scaler en la carpeta 'models/'.")
    st.stop()

# Columnas EXACTAS usadas en el entrenamiento del scaler
FEATURES = list(scaler.feature_names_in_)

st.title("🔧 Predicción de Fallas – Compresor MetroPT3")
st.write("Ingrese los valores de los sensores para realizar una predicción basada en el modelo entrenado.")

# ===============================
#   FORMULARIO DE ENTRADA
# ===============================
st.subheader("📝 Ingrese los valores de los sensores:")

form = st.form("sensor_form")

inputs = {}
cols = form.columns(3)

# Crear campos de entrada dinámicamente con los nombres EXACTOS del scaler
for i, col in enumerate(FEATURES):
    inputs[col] = cols[i % 3].number_input(
        label=col,
        value=0.0,
        format="%.5f"
    )

submit = form.form_submit_button("🔍 Predecir")

# ===============================
#   PROCESAR Y PREDECIR
# ===============================
if submit:
    # Construir DataFrame con columnas correctas y mismo orden
    df_input = pd.DataFrame([inputs], columns=FEATURES)

    # Escalado
    try:
        df_scaled = scaler.transform(df_input)
    except Exception as e:
        st.error("❌ Error al escalar los datos. Revisa que los nombres de columnas coincidan.")
        st.code(str(e))
        st.stop()

    # Predicción
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    st.subheader("📌 Resultado de la predicción")

    if pred == 1:
        st.error(f"⚠️ **Posible falla detectada** – Probabilidad: **{prob:.2f}**")
    else:
        st.success(f"✔️ **Sistema estable** – Probabilidad de falla: **{prob:.2f}**")

    st.write("Valores usados en la predicción:")
    st.dataframe(df_input)
