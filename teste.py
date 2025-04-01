import yfinance as yf


ticker = "PETR4.SA"
start_date = "2025-01-01"
end_date = "2025-03-14"
interval = "1d"

# Baixando os dados do Yahoo Finance
dados = yf.download(ticker, start=start_date, end=end_date, interval=interval)
print(dados)