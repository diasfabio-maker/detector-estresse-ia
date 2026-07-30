
import streamlit as st
import numpy as np
import joblib

# Carregamento dos artefatos
model = joblib.load('modelo_estresse.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Classificador de Nivel de Estresse")
st.write("Insira os parametros para predicao.")

humidity = st.slider("Umidade Corporal", 10.0, 30.0, 20.0)
temperature = st.slider("Temperatura (F)", 75.0, 100.0, 88.0)
step_count = st.slider("Passos", 0, 200, 100)

if st.button("Classificar"):
    input_data = np.array([[humidity, temperature, step_count]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    classes = {0: "Baixo", 1: "Normal", 2: "Alto"}

    st.subheader("Resultado:")
    st.success(f"Nivel detectado: {classes[prediction]}")
