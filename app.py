import streamlit as st
import numpy as np
import joblib

# ==========================================
# Configuração da página
# ==========================================
st.set_page_config(
    page_title="Detector de Estresse",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# Carregar modelo e scaler
# ==========================================
model = joblib.load("modelo_estresse.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================
# Barra lateral
# ==========================================
st.sidebar.title("ℹ️ Informações")

st.sidebar.info("""
### Detector de Nível de Estresse

Esta aplicação utiliza um modelo de Machine Learning treinado para prever o nível de estresse de uma pessoa.

Preencha os dados e clique em **Classificar**.

**Observação:** A temperatura é informada em **°C**, mas o sistema faz automaticamente a conversão para **°F**, pois o modelo foi treinado nessa unidade.
""")

# ==========================================
# Título
# ==========================================
st.title("🧠 Classificador de Nível de Estresse")

st.write(
    "Informe os valores abaixo para realizar a classificação."
)

st.divider()

# ==========================================
# Formulário
# ==========================================
with st.form("formulario"):

    col1, col2 = st.columns(2)

    with col1:

        humidity = st.number_input(
            "💧 Umidade Corporal (%)",
            value=20.0,
            step=0.1
        )

        step_count = st.number_input(
            "👣 Quantidade de Passos",
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

# ==========================================
# Predição
# ==========================================
if submitted:

    # Conversão de Celsius para Fahrenheit
    temperature_f = (temperature_c * 9 / 5) + 32

    entrada = np.array([[humidity, temperature_f, step_count]])
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

    st.caption(
        f"Temperatura informada: {temperature_c:.1f} °C "
        f"(convertida automaticamente para {temperature_f:.1f} °F)"
    )

    # ==========================================
    # Probabilidades
    # ==========================================
    if hasattr(model, "predict_proba"):

        prob = model.predict_proba(entrada)[0]

        st.divider()
        st.subheader("📊 Confiança do Modelo")

        nomes = ["🟢 Baixo", "🟡 Normal", "🔴 Alto"]

        for nome, p in zip(nomes, prob):
            st.write(f"**{nome}** — {p*100:.2f}%")
            st.progress(float(p))

# ==========================================
# Rodapé
# ==========================================
st.divider()

st.caption("Projeto desenvolvido com Python, Scikit-Learn e Streamlit.")
