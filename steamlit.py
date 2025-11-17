import streamlit as st
import joblib
import pandas as pd

# Carga modelo y scaler
scaler = joblib.load("models/scaler_knn.pkl")
model = joblib.load("models/modelo_knn.pkl")

st.title("Predicción de Fallas del Compresor de Aire")
st.write("Introduce los valores de los sensores para predecir la falla.")

# Extraer columnas exactas del scaler
columns = list(scaler.feature_names_in_)
input_data = {}

for col in columns:
    input_data[col] = st.number_input(f"Ingrese valor para {col}", value=0.0, format="%.3f")

# Crear DataFrame con el mismo orden de columnas que el scaler/modelo
input_df = pd.DataFrame([input_data], columns=columns)

if st.button("Predecir Falla"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Falla detectada con probabilidad {prob:.2%}")
        else:
            st.success(f"✅ Sin falla detectada. Probabilidad de falla: {prob:.2%}")
    except ValueError as e:
        st.error(f"Error en la predicción: {e}")
