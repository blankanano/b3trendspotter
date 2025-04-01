import numpy as np
import pandas as pd
import warnings
import requests
import streamlit as st
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

data_hoje = datetime.today()
periodo_RSI = 14
periodo_BB = 20
colunas_desejadas = ['Fechamento', 'Índice de Força Relativa (RSI)', 'Sinal de Compra']

def get_b3_tickers():
    url = "https://brapi.dev/api/quote/list"
    params = {"country": "brazil"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "stocks" in data:
            return [stock["stock"] + ".SA" for stock in data["stocks"]]
        return []
    except Exception as e:
        st.sidebar.error(f"Erro ao buscar ativos da B3: {e}")
        return []
    
# Função para obter os tickers de contratos futuros da B3 automaticamente
# Função para obter os tickers de contratos futuros da B3 automaticamente
def get_futuros_b3_tickers():
    url = "https://www.b3.com.br/pt_br/market-data-e-indices/precadores/cotacoes/"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    tickers = []
    
    # Procurando ativos futuros na tabela da B3
    for row in soup.find_all("tr")[1:]:  # Pulando o cabeçalho
        cols = row.find_all("td")
        if len(cols) > 1:
            ticker = cols[0].text.strip()
            if ticker and "FUT" in ticker:
                tickers.append(ticker + ".SA")  # Adiciona sufixo do Yahoo Finance
    
    # Validar quais tickers realmente existem no Yahoo Finance
    tickers_validos = []
    for ticker in tickers:
        try:
            dados = yf.download(ticker, period="1d")  # Verifica se há dados disponíveis
            if not dados.empty:
                tickers_validos.append(ticker)
        except:
            continue
    
    return tickers_validos

# Verificar se hoje é um dia útil de mercado
def get_last_trading_day():
    today = datetime.today()
    while today.weekday() in [5, 6]:  # Sábado ou Domingo
        today -= timedelta(days=1)
    return today

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

# 🚀 Cálculo do RSI
def calcular_rsi(series):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=periodo_RSI).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periodo_RSI).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 🚀 Cálculo das Bandas de Bollinger
def calcular_bollinger_bands(series):
    middle_band = series.rolling(window=periodo_BB).mean()
    std_dev = series.rolling(window=periodo_BB).std()
    upper_band = middle_band + (std_dev * 2)
    lower_band = middle_band - (std_dev * 2)
    return middle_band, upper_band, lower_band

# 🚀 Cálculo dos Níveis de Fibonacci
def calcular_fibonacci(dados):
    high_price = dados["High"].max()
    low_price = dados["Low"].min()
    fibo_levels = {
        "Fibo_0": low_price,
        "Fibo_23.6": low_price + 0.236 * (high_price - low_price),
        "Fibo_38.2": low_price + 0.382 * (high_price - low_price),
        "Fibo_50": low_price + 0.5 * (high_price - low_price),
        "Fibo_61.8": low_price + 0.618 * (high_price - low_price),
        "Fibo_100": high_price
    }
    return fibo_levels

# 🚀 Função para Análise de Compra/Venda
def analisar_sinais(dados):
    recomendacoes = []
    
    if "RSI" in dados.columns:
        if dados["RSI"].iloc[-1] < 30:
            recomendacoes.append("RSI indica compra (Sobrevendido)")
        elif dados["RSI"].iloc[-1] > 70:
            recomendacoes.append("RSI indica venda (Sobrecomprado)")
    
    if "EMA_9" in dados.columns and "EMA_21" in dados.columns:
        if dados["EMA_9"].iloc[-1] > dados["EMA_21"].iloc[-1]:
            recomendacoes.append("Cruzamento de Médias indica tendência de alta")
        elif dados["EMA_9"].iloc[-1] < dados["EMA_21"].iloc[-1]:
            recomendacoes.append("Cruzamento de Médias indica tendência de baixa")
    
    if "MACD" in dados.columns and "Signal_Line" in dados.columns:
        if dados["MACD"].iloc[-1] > dados["Signal_Line"].iloc[-1]:
            recomendacoes.append("MACD indica compra")
        elif dados["MACD"].iloc[-1] < dados["Signal_Line"].iloc[-1]:
            recomendacoes.append("MACD indica venda")
    
    return recomendacoes

# 🚀 Cálculo do MACD
def calcular_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# 🚀 Implementação do Ranking Futuro (em progresso)
def calcular_ranking(dados):
    ranking = []
    for ativo, df in dados.items():
        if df is not None and not df.empty:
            ultima_variacao = df["Close"].pct_change().dropna()
            if not ultima_variacao.empty:
                variacao_pct = float(ultima_variacao.iloc[-1] * 100)
            else:
                variacao_pct = 0.0
            
            volume = df["Volume"].iloc[-1] if "Volume" in df.columns else 0

            ranking.append({"Ativo": ativo, "Variação %": variacao_pct, "Volume": volume})

    df_ranking = pd.DataFrame(ranking)
    if not df_ranking.empty:
        df_ranking.sort_values(by=["Variação %", "Volume"], ascending=[False, False], inplace=True)

    return df_ranking

def calcular_estocastico(dados, periodo_k=14, periodo_d=3):
    """
    Calcula o indicador Estocástico (%K e %D).

    Parâmetros:
        - dados: DataFrame do ativo com colunas ['High', 'Low', 'Close']
        - periodo_k: Período de cálculo para %K (padrão: 14)
        - periodo_d: Período de cálculo para %D (padrão: 3)

    Retorna:
        - Duas colunas: %K e %D
    """
    try:
        if len(dados) < periodo_k:
            return None, None  # Se não houver dados suficientes, retorna None

        # Cálculo de %K
        highest_high = dados["High"].rolling(window=periodo_k).max()
        lowest_low = dados["Low"].rolling(window=periodo_k).min()
        dados["%K"] = 100 * ((dados["Close"] - lowest_low) / (highest_high - lowest_low))

        # Cálculo de %D (Média Móvel Simples de %K)
        dados["%D"] = dados["%K"].rolling(window=periodo_d).mean()

        return dados["%K"], dados["%D"]

    except Exception as e:
        print(f"Erro ao calcular Estocástico: {e}")
        return None, None

def get_latest_price(ticker):
    try:
        dados = yf.download(ticker, period="1d", interval="1m")
        if not dados.empty:
            return dados["Close"].iloc[-1]
        return None
    except Exception as e:
        st.error(f"Erro ao buscar dados de {ticker}: {e}")
        return None
    

def gerar_insights(dados):
    insights = []

    if "RSI" in dados.columns:
        if dados["RSI"].iloc[-1] < 30:
            insights.append(("🟢 RSI indica COMPRA (Sobrevendido)", "green"))
        elif dados["RSI"].iloc[-1] > 70:
            insights.append(("🔴 RSI indica VENDA (Sobrecomprado)", "red"))

    if "EMA_9" in dados.columns and "EMA_21" in dados.columns:
        if dados["EMA_9"].iloc[-1] > dados["EMA_21"].iloc[-1]:
            insights.append(("🟢 Médias Móveis indicam tendência de ALTA", "green"))
        elif dados["EMA_9"].iloc[-1] < dados["EMA_21"].iloc[-1]:
            insights.append(("🔴 Médias Móveis indicam tendência de BAIXA", "red"))

    if "MACD" in dados.columns and "Signal_Line" in dados.columns:
        if dados["MACD"].iloc[-1] > dados["Signal_Line"].iloc[-1]:
            insights.append(("🟢 MACD indica COMPRA", "green"))
        elif dados["MACD"].iloc[-1] < dados["Signal_Line"].iloc[-1]:
            insights.append(("🔴 MACD indica VENDA", "red"))

    return insights    