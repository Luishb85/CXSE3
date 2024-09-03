import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go
from datetime import timedelta

# ticker a ser analisado
tkr = yf.Ticker('CXSE3.SA')
hist = tkr.history(period="1y")

# benchmark BOVA11
indice = yf.Ticker('BOVA11.SA')
index_data = indice.history(period='1y')

df = hist.join(index_data, rsuffix='_idx')

# Calculando a mudança percentual diária
df['priceRise'] = np.log(df['Close'] / df['Close'].shift(1))
df['volumeRise'] = np.log(df['Volume'] / df['Volume'].shift(1))
df['priceRise_idx'] = np.log(df['Close_idx'] / df['Close_idx'].shift(1))
df['volumeRise_idx'] = np.log(df['Volume_idx'] / df['Volume_idx'].shift(1))

# Substituir valores infinitos por NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remover linhas que contenham NaN
df.dropna(inplace=True)

# Aplicando uma média móvel para suavizar a série de mudanças percentuais
df['priceRise_smooth'] = df['priceRise'].rolling(window=10).mean().fillna(df['priceRise'])

# Incluindo as colunas novas no DataFrame para o modelo
df_model = df[['priceRise_smooth', 'volumeRise', 'priceRise_idx', 'volumeRise_idx']]

# Gerando a variável target
conditions = [
    (df_model['priceRise_smooth'].shift(-1) > 0.01),
    (df_model['priceRise_smooth'].shift(-1) < -0.01)
]
choices = [1, -1]

# Usar loc para evitar SettingWithCopyWarning
df_model.loc[:, 'Pred'] = np.select(conditions, choices, default=0)

# Preparando os dados para o modelo
features = df_model[['priceRise_smooth', 'volumeRise', 'priceRise_idx', 'volumeRise_idx']].to_numpy()
target = df_model['Pred'].to_numpy()

# Dividindo os dados em treino e teste
rows_train, rows_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Treinando o modelo de regressão logística
clf = LogisticRegression()
clf.fit(rows_train, y_train)

# Exibindo a acurácia do modelo
st.write(f"Acurácia do modelo no conjunto de teste: {clf.score(rows_test, y_test):.2f}")

# Função para prever os próximos 5 dias
def prever_proximos_precos(clf, df, dias=5):
    previsoes = []
    ultima_linha = df[['priceRise_smooth', 'volumeRise', 'priceRise_idx', 'volumeRise_idx']].iloc[-1].to_numpy().reshape(1, -1)
    ultimo_preco = hist['Close'].iloc[-1]  # Último preço disponível
    for i in range(dias):
        previsao = clf.predict(ultima_linha)
        preco_alvo = ultimo_preco * np.exp(ultima_linha[0, 0])  # Calcula o preço alvo com a média móvel
        previsoes.append({
            "Previsao": previsao[0],
            "Preco_Alvo": preco_alvo
        })
        # Atualizar a linha para a próxima previsão
        ultima_linha = np.roll(ultima_linha, -1)
        ultima_linha[0, 0] = np.log(preco_alvo / ultimo_preco)  # Atualiza com base no novo preço alvo
        ultimo_preco = preco_alvo  # Atualiza o último preço
    return previsoes

# Função para plotar os gráficos usando Plotly
def plotar_graficos(hist, index_data):
    st.subheader("Gráficos de Preços e Volumes")

    # Gráfico de preços CXSE3 e benchmark
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='CXSE3'))
    fig.add_trace(go.Scatter(x=index_data.index, y=index_data['Close'], mode='lines', name='BOVA11'))
    fig.update_layout(title='Preços de Fechamento', xaxis_title='Data', yaxis_title='Preço de Fechamento')
    st.plotly_chart(fig)

    # Gráfico de volumes CXSE3 e benchmark
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Volume'], mode='lines', name='CXSE3'))
    fig.add_trace(go.Scatter(x=index_data.index, y=index_data['Volume'], mode='lines', name='BOVA11'))
    fig.update_layout(title='Volume Negociado', xaxis_title='Data', yaxis_title='Volume')
    st.plotly_chart(fig)

# Aplicação Streamlit
st.title("Previsão de Cotações da CXSE3 para a Próxima Semana")

# Prever para os próximos 5 dias
previsoes = prever_proximos_precos(clf, df)
datas_previstas = [df.index[-1] + timedelta(days=i+1) for i in range(5)]
tabela_prevista = pd.DataFrame({
    "Data": [data.strftime('%Y-%m-%d') for data in datas_previstas],
    "Previsão de Movimento": ['Alta' if p["Previsao"] == 1 else 'Baixa' if p["Previsao"] == -1 else 'Estável' for p in previsoes],
    "Preço Alvo (R$)": [f'R$ {p["Preco_Alvo"]:.2f}' for p in previsoes]
})

# Exibir a tabela de previsões
st.write("Previsões para os próximos 5 dias:")
st.table(tabela_prevista)

# Mostrar gráficos usando o DataFrame original
plotar_graficos(hist, index_data)
