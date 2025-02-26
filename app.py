import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, time
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid


data_hoje = datetime.today()
periodo_RSI = 14
colunas_desejadas = ['Fechamento', 'Índice de Força Relativa (RSI)', 'Sinal de Compra']

def formatar_dataframe(dados_ativo):
    dados_ativo.rename(columns={
        'Close': 'Fechamento',
        'Retornos': 'Variação',
        'Ganhos': 'Ganhos (+)',
        'Perdas': 'Perdas (-)',
        'Media_Ganhos': 'Média de Ganhos',
        'Media_Perdas': 'Média de Perdas',
        'RS': 'Força Relativa',
        'RSI': 'Índice de Força Relativa (RSI)',
        'Compra': 'Sinal de Compra'
    }, inplace=True)
    dados_ativo.index.name = 'Data/Hora'  # Alterar o nome da coluna DateTime
    return dados_ativo

def build_sidebar():
    st.image("imagens/Logo_B3.png")
    ticker_list = pd.read_csv("ativos/tickers.csv", index_col=0)

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

    interval = st.selectbox('Selecione o intervalo de tempo', ('1m', '5m', '15m', '30m', '1h', '1d'), key='interval')

    tickers = st.multiselect(label="Selecione o par de moedas", options=ticker_list, placeholder='Ativos')

    if tickers:
        
        data_compra = []
        data_venda = []

        #try:
        dados_ativo = yf.download(tickers=tickers, start=start_datetime, end=end_datetime, interval=interval)[['Close']]

        if dados_ativo.empty:
            st.error("Não foram encontrados dados para os parâmetros fornecidos.")
            return None, None, None, None         

        # Calcular as variações dos preços
        dados_ativo['Retornos'] = dados_ativo['Close'].diff()

        # Separar as variações em ganhos e perdas
        dados_ativo['Ganhos'] = np.where(dados_ativo['Retornos'] > 0, dados_ativo['Retornos'], 0)
        dados_ativo['Perdas'] = np.where(dados_ativo['Retornos'] < 0, abs(dados_ativo['Retornos']), 0)

        # Calcular a media móvel dos ganhos e perdas
        dados_ativo['Media_Ganhos'] = dados_ativo['Ganhos'].rolling(window=periodo_RSI).mean()
        dados_ativo['Media_Perdas'] = dados_ativo['Perdas'].rolling(window=periodo_RSI).mean()

        # Calcular o RS (Relative Strength)
        dados_ativo['RS'] = dados_ativo['Media_Ganhos'] / dados_ativo['Media_Perdas']

        # Calcular o RSI
        dados_ativo['RSI'] = 100 - (100 / (1 + dados_ativo['RS']))

        # Verificar sinais de compra
        dados_ativo.loc[dados_ativo['RSI'] < 30, 'Compra'] = 'S' 
        dados_ativo.loc[dados_ativo['RSI'] > 30, 'Compra'] = 'N'

        for i in range(len(dados_ativo) - 1):
            if pd.notna(dados_ativo['Compra'].iloc[i]) and "S" in dados_ativo['Compra'].iloc[i]:
                if i + 1 < len(dados_ativo):
                    data_compra.append(dados_ativo.iloc[i+1].name) # +1 porque compramos no preço de abertura do dia seguinte.
            
                for j in range(1, 11):
                    if i + j + 1 < len(dados_ativo):
                        if dados_ativo['RSI'].iloc[i + j] > 40: #vendo se nos próximos 10 dias o RSI passa de 40
                            data_venda.append(dados_ativo.iloc[i + j + 1].name) #vende no dia seguinte que bater 40
                            break
                        elif j == 10:
                            data_venda.append(dados_ativo.iloc[i + j + 1].name)

        #except Exception as e:
            #st.error(f"Ocorreu um erro ao processar os dados: {e}")        

        dados_ativo = formatar_dataframe(dados_ativo)

        return tickers, dados_ativo, data_compra, data_venda
    return None, None, None, None

def build_main(tickers, dados_ativo, data_compra, data_venda):
    colunas_disponiveis = [col for col in colunas_desejadas if col in dados_ativo.columns]
    dados_ativo = dados_ativo[colunas_disponiveis]   

    dados_ativo_display = dados_ativo.copy()
    dados_ativo_display.index = dados_ativo_display.index.strftime('%d/%m/%Y %H:%M')

    # Exibir o DataFrame com formatação
    st.dataframe(
        dados_ativo_display.style.set_properties(**{'text-align': 'right'})
        .set_table_styles([{'selector': 'th', 'props': [('max-width', '200px')]}]),
        use_container_width=True
    )

    plt.figure(figsize = (12, 5))
    plt.scatter(dados_ativo.loc[data_compra].index, dados_ativo.loc[data_compra]['Fechamento'], marker = '^',
    c = 'g')
    plt.plot(dados_ativo['Fechamento'], alpha = 0.7)   

    st.pyplot(plt)

    # Qual a média de lucros?
    # Qual a média de perdas?
    # Qual a % de operações vencedoras?
    # Qual expectativa matemática do modelo?
    # Qual retorno acumulado?
    # O retorno acumulado venceu o Buy and Hold na ação?

    # lucros = dados_ativo.loc[data_venda]['Open'].values/dados_ativo.loc[data_compra]['Open'].values - 1

    # operacoes_vencedoras = len(lucros[lucros > 0])/len(lucros)
    # media_ganhos = np.mean(lucros[lucros > 0])
    # media_perdas = abs(np.mean(lucros[lucros < 0]))
    # expectativa_matematica_modelo = (operacoes_vencedoras * media_ganhos) - ((1 - operacoes_vencedoras) * media_perdas)
    # performance_acumulada = (np.cumprod((1 + lucros)) - 1)

    # plt.figure(figsize = (12, 5))
    # plt.plot(data_compra, performance_acumulada)    

st.set_page_config(layout="wide")

with st.sidebar:
    tickers, dados_ativo, data_compra, data_venda = build_sidebar()

st.title('Identificador de tendências')

if tickers:
    build_main(tickers, dados_ativo, data_compra, data_venda)    