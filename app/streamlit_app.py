"""Streamlit dashboard for near-real-time air quality visualization and prediction.

Run with:
    streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import os
import time
from src import collector, predict

MODEL_DIR = os.path.join("models")


st.set_page_config(page_title="Qualidade do Ar", layout="wide")

st.title("Análise e Previsão de Qualidade do Ar")

with st.sidebar:
    st.header("Controles")
    city = st.text_input("Cidade (opcional)")
    country = st.text_input("País - ISO2 (opcional)")
    pollutant = st.selectbox("Poluente", ["pm25"], index=0)
    model_files = [f for f in os.listdir(MODEL_DIR)] if os.path.exists(MODEL_DIR) else []
    model_choice = st.selectbox("Modelo salvo", ["(nenhum)"] + model_files)
    refresh = st.button("Atualizar agora")
    st.markdown("---")
    st.markdown("Desenvolvido para demonstração acadêmica.")


@st.cache_data(ttl=60)
def get_latest(city, country, limit=200):
    try:
        df = collector.fetch_latest_measurements(city=city or None, country=country or None, limit=limit)
        return df
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        return pd.DataFrame()


def load_selected_model(choice):
    if not choice or choice == "(nenhum)":
        return None, None
    path = os.path.join(MODEL_DIR, choice)
    try:
        model, scaler = predict.load_model(path)
        return model, scaler
    except Exception as e:
        st.warning(f"Não foi possível carregar modelo: {e}")
        return None, None


df = get_latest(city, country)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Leituras recentes")
    if df.empty:
        st.info("Nenhuma leitura disponível — tente ampliar o limite ou verifique a conexão com a API.")
    else:
        # show most recent for the selected pollutant
        st.dataframe(df.sort_values("datetime", ascending=False).head(50))

with col2:
    st.subheader("Mapa rápido")
    if not df.empty and {"latitude", "longitude"}.issubset(df.columns):
        map_df = df.dropna(subset=["latitude", "longitude"]) [["latitude", "longitude"]]
        # streamlit's st.map expects lat/lon named 'lat' and 'lon'
        map_df = map_df.rename(columns={"latitude":"lat","longitude":"lon"})
        st.map(map_df)
    else:
        st.info("Sem coordenadas para mostrar no mapa.")


st.markdown("---")

st.subheader("Previsão (modelo)")
model, scaler = load_selected_model(model_choice)

if model is None:
    st.info("Nenhum modelo carregado. Treine um modelo com `python -m src.train` e coloque o .pkl na pasta models/.")
else:
    # pick the most recent measurement for the pollutant to form an input
    df_poll = df[df["parameter"].str.lower() == pollutant.lower()] if not df.empty else pd.DataFrame()
    if df_poll.empty:
        st.warning("Nenhuma leitura do poluente selecionado disponível para previsão.")
    else:
        latest = df_poll.sort_values("datetime", ascending=False).iloc[0]
        reading = {"hour": int(latest["datetime"].hour), "latitude": float(latest["latitude"]), "longitude": float(latest["longitude"]), "city": latest.get("city")}
        try:
            pred = predict.predict_from_reading(model, scaler, reading)
            st.metric(label=f"Previsão {pollutant} (próxima janela)", value=f"{pred:.2f} {latest.get('unit','')}")
            st.write("Entrada usada para previsão:")
            st.json(reading)
        except Exception as e:
            st.error(f"Erro ao prever: {e}")

st.markdown("---")
st.write("Dicas: treine um modelo com `src/train.py` ou use um CSV histórico para melhores previsões.")
