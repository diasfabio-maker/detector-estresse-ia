import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===============================
# Configuração da página
# ===============================
st.set_page_config(
    page_title="Detector de Estresse",
    page_icon="🧠",
    layout="centered"
)

# ===============================
# Carregamento do modelo
# ===============================
model = joblib.load("modelo_estresse.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Barra lateral
# ===============================
st.sidebar.title("ℹ️ Informações")

st.sidebar.info("""
### Detector de Nível de Estresse

Preencha os parâmetros abaixo e clique em **Classificar**.

O sistema utiliza um modelo de Machine Learning treinado para classificar o nível de estresse.

Projeto desenvolvido para a disciplina de Aprendizado de Máquina.
""")

# ===============================
# Título
# ===============================
st.title("🧠 Classificador de Nível de Estresse")

st.write(
    "Informe os valores abaixo para realizar a classificação do nível de estresse."
)

st.divider()

# ===============================
# Formulário
# ===============================
with st.form("predicao"):

    col1, col2 = st.columns(2)

    with col1:

        humidity = st.number_input(
    "💧 Umidade Corporal",
    value=20.0,
    step=0.1
  )

        step_count = st.number_input(
    "👣 Passos",
    value=100,
    step=1
  )

    with col2:

      temperature_c = st.number_input(
    "🌡 Temperatura (°C)",
    value=31.0,
    step=0.1
)
    st.divider()

    submitted = st.form_submit_button(
        "🔍 Classificar",
        use_container_width=True
    )

# ===============================
# Predição
# ===============================
if submitted:

    # Conversão de Celsius para Fahrenheit
   temperature = (temperature_c * 9/5) + 32

    entrada = np.array([[humidity, temperature, step_count]])
    entrada = scaler.transform(entrada)

    prediction = model.predict(entrada)[0]

    classes = {
        0: "Baixo",
        1: "Normal",
        2: "Alto"
    }

    st.divider()

    st.subheader("🎯 Resultado da Classificação")

    if prediction == 0:
        st.success("✅ Nível detectado: **BAIXO**")

    elif prediction == 1:
        st.warning("⚠️ Nível detectado: **NORMAL**")

    else:
        st.error("🚨 Nível detectado: **ALTO**")

    # ===============================
    # Probabilidades
    # ===============================

    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(entrada)[0]

        st.divider()

        st.subheader("📊 Confiança do Modelo")

        nomes = ["Baixo", "Normal", "Alto"]

        for nome, p in zip(nomes, prob):

            st.write(f"**{nome}** — {p*100:.2f}%")

            st.progress(float(p))

# ===============================
# Rodapé
# ===============================
st.divider()

st.caption("Projeto desenvolvido utilizando Python, Scikit-Learn e Streamlit.")
