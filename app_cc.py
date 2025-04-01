import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import utils as utils
import time
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
from prophet import Prophet

# Configuração da Página
st.set_page_config(layout="wide", page_title="Identificador de Tendências")

# 🚀 Sidebar - Configuração do Usuário
with st.sidebar:
    st.image("imagens/Logo_B3.png")
    ticker_list = st.session_state.get("ticker_list", utils.get_b3_tickers())

    if "PETR4.SA" not in ticker_list:
        ticker_list.append("PETR4.SA")

    # Seleção do período de análise
    st.markdown("### 📅 Período da Análise")
    start_date = st.date_input("Data Inicial", value=utils.data_hoje - timedelta(days=30), format="DD/MM/YYYY")
    end_date = st.date_input("Data Final", value=utils.get_last_trading_day(), format="DD/MM/YYYY")
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Intervalo de tempo para consulta
    interval = st.selectbox('⏳ Intervalo', ('1m', '5m', '15m', '30m', '1h', '1d'), index=5)
    tickers = st.multiselect("📈 Selecione os ativos", ticker_list, default=["PETR4.SA"])

    # Indicadores técnicos disponíveis
    indicadores = st.multiselect("📊 Escolha os Indicadores:", 
                                 ["Fibonacci", "Média Móvel", "RSI", "Bandas de Bollinger", "MACD", "Volume", "Estocástico"],
                                 default=["RSI"])
    
    # Configuração de alertas de preços
    st.markdown("### 🚨 Configurar Alertas de Preços")
    alertas = {ticker: st.number_input(f"🔔 Alerta para {ticker} (Preço Alvo)", min_value=0.0, step=0.1, format="%.2f") for ticker in tickers}

    # Tempo de atualização automática
    tempo_atualizacao = st.slider("⏳ Tempo de Atualização (segundos)", 10, 120, 30, 10)

# 🚀 Cache de Dados para Evitar Requisições Repetitivas
@st.cache_data(ttl=600)
def get_cached_dados_yfinance(ticker, start_datetime, end_datetime, interval):
    return yf.download(ticker, start=start_datetime.strftime('%Y-%m-%d'), 
                        end=end_datetime.strftime('%Y-%m-%d'), interval=interval)

# 🚀 Função para Baixar, Processar Dados e calcular os indicadores
def get_dados_yfinance(ticker, start_datetime, end_datetime, interval, indicadores):
    try:
        st.info(f"🔄 Buscando dados para {ticker} - Intervalo ({interval}) de {start_datetime.strftime('%d/%m/%Y')} até {end_datetime.strftime('%d/%m/%Y')}...")

        time.sleep(2) # Acrescentado, para evitar várias requisições na B3
        dados = get_cached_dados_yfinance(ticker, start_datetime, end_datetime, interval)

        if dados is None or dados.empty:
            st.error(f"❌ Nenhum dado encontrado para {ticker}.")
            return None, []            

        dados.dropna(inplace=True)
        colunas_para_plot = ["Close"]
        sinais = []

        # Calcula os indicaores, conforme seleção do usuário
        if "Média Móvel" in indicadores:
            dados["EMA_9"] = dados["Close"].ewm(span=9, adjust=False).mean()
            dados["EMA_21"] = dados["Close"].ewm(span=21, adjust=False).mean()
            colunas_para_plot.extend(["EMA_9", "EMA_21"])

            # Detectando sinais de compra e venda (cruzamento das médias)
            for i in range(1, len(dados)):
                if "EMA_9" in dados.columns and "EMA_21" in dados.columns:
                    # Cruzamento de alta
                    if dados["EMA_9"].iloc[i-1] < dados["EMA_21"].iloc[i-1] and dados["EMA_9"].iloc[i] > dados["EMA_21"].iloc[i]:
                        sinais.append({"tipo": "compra", "data": dados.index[i], "preco": dados["Close"].iloc[i]})
                    # Cruzamento de baixa
                    elif dados["EMA_9"].iloc[i-1] > dados["EMA_21"].iloc[i-1] and dados["EMA_9"].iloc[i] < dados["EMA_21"].iloc[i]:
                        sinais.append({"tipo": "venda", "data": dados.index[i], "preco": dados["Close"].iloc[i]})

        if "RSI" in indicadores:
            dados["RSI"] = utils.calcular_rsi(dados["Close"])
            colunas_para_plot.append("RSI")

        if "Bandas de Bollinger" in indicadores:
            dados["BB_Middle"], dados["BB_Upper"], dados["BB_Lower"] = utils.calcular_bollinger_bands(dados["Close"])
            colunas_para_plot.extend(["BB_Upper", "BB_Lower", "BB_Middle"])

        if "Fibonacci" in indicadores:
            fibo_levels = utils.calcular_fibonacci(dados)
            for key, value in fibo_levels.items():
                dados[key] = value
            colunas_para_plot.extend(fibo_levels.keys())

        if "MACD" in indicadores:
            dados["MACD"], dados["Signal_Line"] = utils.calcular_macd(dados["Close"])
            colunas_para_plot.extend(["MACD", "Signal_Line"])

        if "Volume" in indicadores:
            colunas_para_plot.append("Volume")

        if "Estocástico" in indicadores:
            dados["%K"], dados["%D"] = utils.calcular_estocastico(dados)
            colunas_para_plot.extend(["%K", "%D"])

        # Previsão de tendência baseada na regressão linear
        dados["Dias"] = np.arange(len(dados))
        X = dados[["Dias"]].values
        y = dados["Close"].values
        modelo = LinearRegression()
        modelo.fit(X, y)
        dados["Previsão"] = modelo.predict(X)
        colunas_para_plot.append("Previsão")            

        return dados, colunas_para_plot, sinais
    except Exception as e:
        st.error(f"❌ Erro ao obter dados para {ticker}: {e}")
        return None, [], []

st.sidebar.write("🔄 Monitorando preços em tempo real...")

# 🚀 Monitoramento de Preços
st.sidebar.write("🔄 Monitorando preços em tempo real...")

if "historico_alertas" not in st.session_state:
    st.session_state.historico_alertas = []

for ticker in tickers:
    preco_atual = utils.get_latest_price(ticker)
    
    if isinstance(preco_atual, pd.Series):
        preco_atual = preco_atual.iloc[-1]  # Pega o último valor da série
    
    if isinstance(alertas[ticker], pd.Series):
        alerta_ticker = alertas[ticker].iloc[0]  # Pega o primeiro valor da série
    else:
        alerta_ticker = float(alertas[ticker])
    
    if pd.notna(preco_atual) and preco_atual >= alerta_ticker:
        alerta_msg = f"🚨 {ticker} atingiu {preco_atual:.2f} (Alvo: {alerta_ticker:.2f})"
        if alerta_msg not in st.session_state.historico_alertas:
            st.session_state.historico_alertas.append(alerta_msg)
            st.sidebar.warning(alerta_msg)

if st.session_state.historico_alertas:
    st.subheader("📜 Histórico de Alertas")
    for alerta in st.session_state.historico_alertas[-5:]:
        st.write(alerta)

# 🚀 Criando Abas para Cada Ativo Selecionado
abas = st.tabs(tickers)
for i, ticker in enumerate(tickers):
    with abas[i]:
        dados, colunas_para_plot, sinais = get_dados_yfinance(ticker, start_datetime, end_datetime, interval, indicadores)

        if dados is not None:
            df_tabela = dados.copy()

            if interval in ["1m", "5m", "15m", "30m", "1h"]:
                df_tabela.index = df_tabela.index.strftime('%d/%m/%Y %H:%M')
            else:
                df_tabela.index = df_tabela.index.strftime('%d/%m/%Y')

            st.subheader(f"📄 Dados Brutos - {ticker}")
            st.dataframe(df_tabela, use_container_width=True)

            # 📊 **Gráfico Técnico**
            # st.subheader(f"📊 Gráfico Técnico - {ticker}")
            # fig = go.Figure()

            # for coluna in colunas_para_plot:
            #     if coluna in dados.columns:
            #         fig.add_trace(go.Scatter(
            #             x=dados.index,
            #             y=dados[coluna],
            #             mode="lines",
            #             name=coluna
            #         ))

            # fig.update_layout(
            #     title=f"{ticker} - Análise Técnica",
            #     xaxis_title="Data",
            #     yaxis_title="Preço",
            #     template="plotly_dark",
            #     height=600
            # )
            # st.plotly_chart(fig, use_container_width=True)

            # Novo Gráfico Técnico: Candlestick + Médias + RSI
            st.subheader(f"📊 Gráfico Técnico Avançado - {ticker}")

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"Preço + Médias Móveis ({ticker})", "RSI (Índice de Força Relativa)")
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=dados.index,
                open=dados['Open'],
                high=dados['High'],
                low=dados['Low'],
                close=dados['Close'],
                name='Candlestick',
                increasing_line_color='green',
                decreasing_line_color='red'
            ), row=1, col=1)

            # Médias móveis
            if 'EMA_9' in dados.columns:
                fig.add_trace(go.Scatter(x=dados.index, y=dados['EMA_9'], name='EMA 9', line=dict(color='blue')), row=1, col=1)
            if 'EMA_21' in dados.columns:
                fig.add_trace(go.Scatter(x=dados.index, y=dados['EMA_21'], name='EMA 21', line=dict(color='orange')), row=1, col=1)

            # Anotações de sinais (setas de compra/venda)
            for sinal in sinais:
                cor = 'green' if sinal['tipo'] == 'compra' else 'red'
                texto = '📈 Compra' if sinal['tipo'] == 'compra' else '📉 Venda'
                simbolo = 'triangle-up' if sinal['tipo'] == 'compra' else 'triangle-down'

                fig.add_trace(go.Scatter(
                    x=[sinal["data"]],
                    y=[sinal["preco"]],
                    mode='markers+text',
                    marker=dict(color=cor, size=12, symbol=simbolo),
                    text=[texto],
                    textposition='top center',
                    name=texto,
                    showlegend=False,
                    legendgroup='sinais'
                ), row=1, col=1)

            # RSI
            if 'RSI' in dados.columns:
                fig.add_trace(go.Scatter(x=dados.index, y=dados['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.update_yaxes(range=[0, 100], row=2, col=1)
                fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
                fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)

            # Layout
            fig.update_layout(
                height=700,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)            

            # 📊 **Previsão da Tendência Futura**
            # df_tabela["Dias"] = np.arange(len(df_tabela))
            # X = df_tabela[["Dias"]].values
            # y = df_tabela["Close"].values

            # modelo = LinearRegression()
            # modelo.fit(X, y)
            # df_tabela["Previsão"] = modelo.predict(X)
            
            # fig_pred = go.Figure()
            # fig_pred.add_trace(go.Scatter(x=dados.index, y=dados["Close"], mode="lines", name="Preço Real", line=dict(color="blue")))
            # fig_pred.add_trace(go.Scatter(x=dados.index, y=dados["Previsão"], mode="lines", name="Tendência Prevista", line=dict(color="orange", dash="dot")))
            # fig_pred.update_layout(title=f"{ticker} - Previsão de Tendência", template="plotly_white")
            # st.plotly_chart(fig_pred, use_container_width=True)

# 📊 Abas de Modelos de Previsão
aba_modelos = st.tabs(["📈 Linear", "🔮 Prophet", "📊 ARIMA", "🤖 LSTM"])

# Previsão Linear (já estava pronto)
with aba_modelos[0]:
    st.markdown("#### Previsão com Regressão Linear")
    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=dados.index, y=dados["Close"],
        mode="lines", name="Preço Real", line=dict(color="blue")
    ))
    fig_pred.add_trace(go.Scatter(
        x=dados.index, y=dados["Previsão"],
        mode="lines", name="Tendência Prevista",
        line=dict(color="orange", dash="dash"),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.1)'
    ))
    fig_pred.update_layout(
        xaxis_title="Data", yaxis_title="Preço (R$)",
        height=500, template="plotly_white",
        title=f"{ticker} - Regressão Linear"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# Prophet
with aba_modelos[1]:
    st.markdown("#### Previsão com Facebook Prophet")

    df_prophet = pd.DataFrame()
    df_prophet["ds"] = dados.index
    df_prophet["y"] = dados["Close"].values.ravel()

    modelo_prophet = Prophet(daily_seasonality=True)
    modelo_prophet.fit(df_prophet)

    periodos = 7
    futuro = modelo_prophet.make_future_dataframe(periods=periodos)
    forecast = modelo_prophet.predict(futuro)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Histórico", line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Previsão", line=dict(color='orange')))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Limite Superior", line=dict(width=0), showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Limite Inferior", fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(width=0), showlegend=False))

    fig_forecast.update_layout(
        title=f"{ticker} - Previsão Prophet (Próximos {periodos} dias)",
        xaxis_title="Data", yaxis_title="Preço (R$)",
        template="plotly_white", height=500
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Insight
    ultima_previsao = forecast.iloc[-1]["yhat"]
    ultimo_preco = df_prophet["y"].iloc[-1]
    if ultima_previsao > ultimo_preco:
        st.success(f"📈 Alta prevista: R${ultimo_preco:.2f} → R${ultima_previsao:.2f}")
    else:
        st.error(f"📉 Queda prevista: R${ultimo_preco:.2f} → R${ultima_previsao:.2f}")

# ARIMA
with aba_modelos[2]:
    st.markdown("#### Previsão com ARIMA")
    from statsmodels.tsa.arima.model import ARIMA

    try:
        serie_arima = dados["Close"].dropna()

        # Treina modelo ARIMA simples (auto ajuste seria ideal)
        modelo_arima = ARIMA(serie_arima, order=(2, 1, 2))  # p, d, q
        modelo_arima_fitted = modelo_arima.fit()

        # Previsão
        previsao_arima = modelo_arima_fitted.forecast(steps=7)
        datas_futuras = pd.date_range(start=dados.index[-1] + timedelta(days=1), periods=7)

        # Gráfico ARIMA
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=dados.index, y=dados["Close"], name="Histórico", line=dict(color='blue')))
        fig_arima.add_trace(go.Scatter(x=datas_futuras, y=previsao_arima, name="Previsão ARIMA", line=dict(color='orange', dash='dot')))

        fig_arima.update_layout(
            title=f"{ticker} - Previsão ARIMA (Próximos 7 dias)",
            xaxis_title="Data", yaxis_title="Preço (R$)",
            template="plotly_white", height=500
        )
        st.plotly_chart(fig_arima, use_container_width=True)

        # Insight
        # ultimo_preco = dados["Close"].iloc[-1]
        # ultima_previsao = previsao_arima.iloc[-1] if hasattr(previsao_arima, "iloc") else previsao_arima[-1]
        # ultima_previsao = float(previsao_arima.values[-1])

        ultimo_preco = float(dados["Close"].iloc[-1])

        # Pega o último valor escalar de qualquer estrutura
        if isinstance(previsao_arima, (pd.Series, np.ndarray, list)):
            ultima_previsao = float(previsao_arima[-1])
        else:
            ultima_previsao = float(previsao_arima)

        if float(ultima_previsao) > float(ultimo_preco):
            st.success(f"📈 ARIMA prevê alta: R${ultimo_preco:.2f} → R${ultima_previsao:.2f}")
        else:
            st.error(f"📉 ARIMA prevê queda: R${ultimo_preco:.2f} → R${ultima_previsao:.2f}")

    except Exception as e:
        st.warning(f"⚠️ Não foi possível gerar a previsão ARIMA: {e}")

with aba_modelos[3]:
    st.markdown("#### 🤖 Previsão com LSTM (Deep Learning)")

    st.info("🚧 Este modelo ainda está em desenvolvimento e será disponibilizado em breve. Fique ligado para atualizações futuras!")

# with aba_modelos[3]:
#     st.markdown("#### Previsão com LSTM (Rede Neural Recorrente)")

#     try:
#         from tensorflow.keras.models import Sequential
#         from tensorflow.keras.layers import Dense, LSTM
#         from sklearn.preprocessing import MinMaxScaler

#         # Prepara dados
#         serie = dados["Close"].values.reshape(-1, 1)
#         scaler = MinMaxScaler()
#         serie_normalizada = scaler.fit_transform(serie)

#         # Parâmetros
#         janela = 10
#         X_lstm, y_lstm = [], []
#         for i in range(janela, len(serie_normalizada)):
#             X_lstm.append(serie_normalizada[i-janela:i])
#             y_lstm.append(serie_normalizada[i])
#         X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

#         # Modelo LSTM
#         model = Sequential()
#         model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
#         model.add(LSTM(units=50))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=0)

#         # Previsão dos próximos 7 pontos
#         entrada = serie_normalizada[-janela:]
#         previsoes_lstm = []
#         for _ in range(7):
#             entrada_reshaped = entrada.reshape(1, janela, 1)
#             pred = model.predict(entrada_reshaped, verbose=0)
#             previsoes_lstm.append(pred[0][0])
#             entrada = np.append(entrada[1:], pred, axis=0)

#         # Converte previsões para escala original
#         previsoes_lstm = scaler.inverse_transform(np.array(previsoes_lstm).reshape(-1, 1)).flatten()

#         datas_futuras_lstm = pd.date_range(start=dados.index[-1] + timedelta(days=1), periods=7)

#         # Gráfico
#         fig_lstm = go.Figure()
#         fig_lstm.add_trace(go.Scatter(x=dados.index, y=dados["Close"], name="Histórico", line=dict(color='blue')))
#         fig_lstm.add_trace(go.Scatter(x=datas_futuras_lstm, y=previsoes_lstm, name="Previsão LSTM", line=dict(color='orange', dash='dot')))

#         fig_lstm.update_layout(
#             title=f"{ticker} - Previsão LSTM (Próximos 7 dias)",
#             xaxis_title="Data", yaxis_title="Preço (R$)",
#             template="plotly_white", height=500
#         )

#         st.plotly_chart(fig_lstm, use_container_width=True)

#         # Insight
#         ultima_previsao = previsoes_lstm[-1]
#         ultimo_preco = dados["Close"].iloc[-1]
#         if ultima_previsao > ultimo_preco:
#             st.success(f"📈 LSTM prevê alta: R${ultimo_preco:.2f} → R${ultima_previsao:.2f}")
#         else:
#             st.error(f"📉 LSTM prevê queda: R${ultimo_preco:.2f} → R${ultima_previsao:.2f}")

#     except Exception as e:
#         st.warning(f"⚠️ Erro ao rodar o modelo LSTM: {e}")

st.write(f"🔄 Atualizando automaticamente a cada {tempo_atualizacao} segundos...")
time.sleep(tempo_atualizacao)
st.rerun()