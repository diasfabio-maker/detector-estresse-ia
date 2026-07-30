import streamlit as st
import numpy as np
import joblib

# Configuração da página
st.set_page_config(
    page_title="Classificador de Estresse",
    page_icon="🧠",
    layout="centered"
)

# Carregamento do modelo
model = joblib.load("modelo_estresse.pkl")
scaler = joblib.load("scaler.pkl")

# Título
st.title("🧠 Classificador de Nível de Estresse")
st.write("Preencha os parâmetros abaixo e clique em **Classificar**.")

# Formulário
with st.form("form_predicao"):

    humidity = st.slider(
        "Umidade Corporal",
        min_value=10.0,
        max_value=30.0,
        value=20.0,
        step=0.1
    )

    temperature = st.slider(
        "Temperatura (°F)",
        min_value=75.0,
        max_value=100.0,
        value=88.0,
        step=0.1
    )

    step_count = st.slider(
        "Passos",
        min_value=0,
        max_value=200,
        value=100,
        step=1
    )

    submitted = st.form_submit_button("🔍 Classificar")

# Executa somente quando clicar no botão
if submitted:

    input_data = np.array([[humidity, temperature, step_count]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    classes = {
        0: "Baixo",
        1: "Normal",
        2: "Alto"
    }

    st.divider()
    st.subheader("Resultado da Classificação")

    if prediction == 0:
        st.success(f"✅ Nível detectado: **{classes[prediction]}**")

    elif prediction == 1:
        st.warning(f"⚠️ Nível detectado: **{classes[prediction]}**")

    else:
        st.error(f"🚨 Nível detectado: **{classes[prediction]}**")
