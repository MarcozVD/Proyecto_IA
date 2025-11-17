# app_streamlit.py
import streamlit as st
import joblib
import pandas as pd

# ==========================================================
# CARGA MODELO Y SCALER
# ==========================================================
scaler = joblib.load("models/scaler_knn.pkl")
model = joblib.load("models/modelo_knn.pkl")

# ==========================================================
# TÍTULO
# ==========================================================
st.title("Predicción de Fallas del Compresor de Aire")
st.write("Introduce los valores de los sensores para predecir la falla.")

# ==========================================================
# ENTRADA DE DATOS
# ==========================================================
# Lista completa de columnas que el modelo espera
columns = [
    'H1','Towers','DV_eletric','COMP','MPG','Reservoirs','TP3','TP2',
    'DV_pressure','LPS','Motor_current','Oil_temperature'
]

input_data = {}
for col in columns:
    input_data[col] = st.number_input(f"Ingrese valor para {col}", value=0.0, format="%.3f")

# Convertir a DataFrame con columnas exactas en el mismo orden
input_df = pd.DataFrame([input_data], columns=columns)

# ==========================================================
# BOTÓN DE PREDICCIÓN
# ==========================================================
if st.button("Predecir Falla"):
    try:
        # Escalar datos
        input_scaled = scaler.transform(input_df)
        
        # Predicción
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Mostrar resultados
        if prediction == 1:
            st.error(f"⚠️ Falla detectada con probabilidad {prob:.2%}")
        else:
            st.success(f"✅ Sin falla detectada. Probabilidad de falla: {prob:.2%}")
    except ValueError as e:
        st.error(f"Error en la predicción: {e}")
