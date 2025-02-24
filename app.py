import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, time
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid

data_hoje = datetime.today();

def build_sidebar():
    st.image("imagens/Logo_B3.png")
    ticker_list = pd.read_csv("ativos/tickers.csv", index_col=0)
    tickers = st.multiselect(label="Selecione o par de moedas", options=ticker_list, placeholder='Ativos')
    interval = st.selectbox('Selecione o intervalo de tempo', ('1m', '5m', '15m', '30m', '1h', '1d'), key='interval')

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data Inicial", value=data_hoje, key='start_date')
    with col2:
        start_time = st.time_input("Hora Inicial", time(5, 0), key='start_time')
        
    col3, col4 = st.columns(2)
    with col3:
        end_date = st.date_input("Data Final", value=data_hoje, key='final_date')
    with col4:
        end_time = st.time_input("Hora Final", time(18, 0), key='final_time')

    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    if tickers:
        dados_ativo = yf.download(tickers=tickers, start=start_datetime, end=end_datetime, interval=interval)["Close"]
        return tickers, dados_ativo
    return None, None

def build_main(tickers, dados_ativo):
    st.dataframe(dados_ativo)

st.set_page_config(layout="wide")

with st.sidebar:
    tickers, dados_ativo = build_sidebar()

st.title('Identificador de tendências')

if tickers:
    build_main(tickers, dados_ativo)    