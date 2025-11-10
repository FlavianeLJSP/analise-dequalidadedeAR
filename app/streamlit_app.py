"""Streamlit dashboard for near-real-time air quality visualization and prediction.

Run with:
    streamlit run app/streamlit_app.py 
"""
import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
# Adiciona o diretório raiz do projeto ao sys.path para encontrar o módulo 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import collector, predict # noqa: E402
import plotly.express as px
import pydeck as pdk

# Constrói o caminho absoluto para a pasta 'models' na raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

st.set_page_config(page_title="Qualidade do Ar", layout="wide")
st.markdown("# Análise e Previsão de Qualidade do Ar")
st.caption("Interface interativa — últimas leituras e previsões por localidade")

# --- Funções de Utilidade e Contexto ---

def get_pollutant_info(value, pollutant="pm25"):
    """Retorna uma classificação e cor baseada no valor do poluente (PM2.5)."""
    if pollutant == "pm25":
        if value <= 12:
            return "Bom", "green"
        elif value <= 35.4:
            return "Moderado", "orange"
        elif value <= 55.4:
            return "Ruim", "red"
        elif value <= 150.4:
            return "Muito Ruim", "purple"
        else:
            return "Perigoso", "maroon"
    return "N/A", "gray"

with st.expander("ℹ️ O que é PM2.5 e como interpretar os valores?"):
    st.markdown("""
        **PM2.5** refere-se a partículas finas inaláveis com diâmetros de 2.5 micrômetros ou menos. Essas partículas são um grande problema de saúde pública, pois podem penetrar profundamente nos pulmões e entrar na corrente sanguínea.
        
        **Níveis de Qualidade do Ar (baseado na EPA - Agência de Proteção Ambiental dos EUA):**
        - **<span style="color:green">Bom (0-12.0 µg/m³):</span>** A qualidade do ar é satisfatória e a poluição do ar representa pouco ou nenhum risco.
        - **<span style="color:orange">Moderado (12.1-35.4 µg/m³):</span>** A qualidade do ar é aceitável. No entanto, pode haver um risco para algumas pessoas, particularmente aquelas que são extraordinariamente sensíveis à poluição do ar.
        - **<span style="color:red">Ruim (35.5-55.4 µg/m³):</span>** Pessoas de grupos sensíveis podem apresentar efeitos na saúde. O público em geral provavelmente não será afetado.
        - **<span style="color:purple">Muito Ruim (55.5-150.4 µg/m³):</span>** Alerta de saúde: todos podem começar a sentir os efeitos na saúde; membros de grupos sensíveis podem ter efeitos mais sérios.
        - **<span style="color:maroon">Perigoso (>150.5 µg/m³):</span>** Alerta de saúde de emergência. Toda a população tem maior probabilidade de ser afetada.
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Controles")
    city_input = st.text_input("Cidade (opcional)")
    country_input = st.text_input("País - ISO2 (opcional)")
    pollutant = st.selectbox("Poluente", ["pm25"], index=0)
    model_files = [f for f in os.listdir(MODEL_DIR)] if os.path.exists(MODEL_DIR) else []
    model_choice = st.selectbox("Modelo salvo", ["(nenhum)"] + model_files)
    st.markdown("---")
    st.write("Última atualização:")
    st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if st.button("Atualizar agora"):
        st.rerun()

@st.cache_data(ttl=30)
def get_latest(city, country, limit=300):
    df = collector.fetch_latest_measurements(city=city or None, country=country or None, limit=limit)
    return df


def load_selected_model(choice):
    if not choice or choice == "(nenhum)":
        return None, None, None
    path = os.path.join(MODEL_DIR, choice)
    try:
        model, scaler, columns = predict.load_model(path)
        return model, scaler, columns
    except Exception:
        return None, None, None


df = get_latest(city_input, country_input)

if df.empty:
    st.info("Nenhuma leitura disponível no momento. Use o botão Atualizar para tentar novamente.")
else:
    # Prepare basic KPIs
    df_poll = df[df["parameter"].str.lower() == pollutant.lower()]
    latest_row = df_poll.sort_values("datetime", ascending=False).iloc[0] if not df_poll.empty else None

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        if latest_row is not None:
            val = latest_row['value']
            level, color = get_pollutant_info(val, pollutant)
            st.metric(label=f"Último {pollutant} registrado", value=f"{val:.1f} {latest_row.get('unit','')}", help=f"Nível: {level}")
            st.markdown(f"**Qualidade do Ar: <span style='color:{color};'>{level}</span>**", unsafe_allow_html=True)
        else:
            st.metric(label=f"Último {pollutant} registrado", value="—")
    with kpi2:        
        # load model and show predicted KPI
        model, scaler, columns = load_selected_model(model_choice)
        if model is not None and latest_row is not None:
            try:
                reading = {"hour": int(latest_row["datetime"].hour), "latitude": float(latest_row["latitude"]), "longitude": float(latest_row["longitude"]), "city": latest_row.get("city")}
                pred = predict.predict_from_reading(model, scaler, columns, reading)
                st.metric(label=f"Previsão {pollutant}", value=f"{pred:.1f} {latest_row.get('unit','')}")
            except Exception as e:
                st.error(f"Erro na previsão: {e}", icon="🚨")
                st.metric(label=f"Previsão {pollutant}", value="Erro")
        else:
            st.metric(label=f"Previsão {pollutant}", value="(nenhum modelo)")
    with kpi3:
        st.metric(label="Total de Leituras na Área", value=str(len(df)))

    st.markdown("---")

    # --- Abas para organizar o conteúdo ---
    tab1, tab2, tab3 = st.tabs(["Visão Geral (Mapa e Gráfico)", "Dados Detalhados (Tabela)", "Sobre o Modelo"])

    with tab1:
        st.subheader("Distribuição Geográfica e Histórico Recente")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Mapa de Estações**")
            if {"latitude", "longitude"}.issubset(df.columns):
                map_df = df.dropna(subset=["latitude", "longitude"]).copy()
                if "value" in map_df.columns:
                    # Corrigido: Acessa a lista de cores e converte o formato 'rgb(r,g,b)' para [r, g, b]
                    def value_to_color(v):
                        color_scale = px.colors.sequential.OrRd
                        idx = min(len(color_scale)-1, int(get_pollutant_info(v)[0] != 'Bom') + int(v > 35.4) + int(v > 55.4))
                        return [int(c) for c in color_scale[idx].replace("rgb(", "").replace(")", "").split(",")]
                    map_df["color"] = map_df["value"].apply(value_to_color)
                else:
                    map_df["color"] = [[0, 120, 200]] * len(map_df)

                deck = pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(latitude=map_df["latitude"].mean(), longitude=map_df["longitude"].mean(), zoom=5, pitch=30),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=map_df,
                            get_position='[longitude, latitude]',
                            get_fill_color='color',
                            get_radius=5000,
                            pickable=True,
                            auto_highlight=True,
                        )
                    ],
                    tooltip={"html": "<b>Cidade:</b> {city} <br/> <b>Valor:</b> {value} {unit}", "style": {"color": "black"}}
                )
                st.pydeck_chart(deck)
            else:
                st.info("Sem coordenadas para mostrar no mapa.")

        with col2:
            st.markdown(f"**{pollutant.upper()} — Leituras recentes (média de 15 min)**")
            if df_poll.empty:
                st.info("Nenhuma leitura do poluente selecionado disponível.")
            else:
                chart_df = df_poll.copy().sort_values("datetime")
                chart_df = chart_df.set_index("datetime")["value"].resample("15T").mean().ffill().reset_index()
                fig = px.line(chart_df, x="datetime", y="value", labels={"datetime":"Data e Hora","value":f"Valor {pollutant}"})
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Tabela de Leituras Detalhadas")
        st.dataframe(df.sort_values("datetime", ascending=False), use_container_width=True)

    with tab3:
        st.subheader("Informações e Teste do Modelo Preditivo")
        if model is None:
            st.warning("Nenhum modelo carregado. Treine um modelo (`python -m src.train`) e selecione-o na barra lateral para ativar as previsões.")
        else:
            st.success(f"Modelo carregado: **{model_choice}**")
            st.write("Este modelo foi treinado para prever o valor do poluente com base na hora, latitude e longitude.")
            
            st.markdown("##### Testar Previsão com a Última Leitura")
            if st.button("Executar Previsão de Teste"):
                if latest_row is not None:
                    try:
                        reading = {"hour": int(latest_row["datetime"].hour), "latitude": float(latest_row["latitude"]), "longitude": float(latest_row["longitude"]), "city": latest_row.get("city")}
                        pred = predict.predict_from_reading(model, scaler, columns, reading)
                        pred_level, pred_color = get_pollutant_info(pred, pollutant)
                        st.success(f"**Previsão:** {pred:.2f} {latest_row.get('unit','')} (Nível: {pred_level})")
                    except Exception as e:
                        st.error(f"Erro ao prever: {e}")
                else:
                    st.warning("Sem leitura recente para usar como entrada de teste.")

st.markdown("---")
st.caption("Sugestão: use um CSV histórico em `src.train --csv` para treinar modelos mais precisos e persistir métricas de avaliação.")
