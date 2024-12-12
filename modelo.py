import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Forçar uso de CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Carregar o arquivo Excel
dados = pd.read_excel('Mega-Sena.xlsx')  # Substitua pelo caminho correto

# Selecionar apenas os números sorteados
numeros = dados[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']]

# Função para preparar os dados
def preparar_dados(numeros, n_passos=10):
    X, y = [], []
    for i in range(n_passos, len(numeros)):
        X.append(numeros.iloc[i-n_passos:i].values.flatten())
        y.append(numeros.iloc[i].values.flatten())  # Prever todos os 6 números
    return np.array(X), np.array(y)

# Preparar os dados para entrada (X) e saída (y)
X, y = preparar_dados(numeros)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
X_train = X_train / 60
X_test = X_test / 60

# Ajustar rótulos para a faixa de 0 a 59
y_train = y_train - 1  # Transformar para categorias 0-59
y_test = y_test - 1

# Criar o modelo de rede neural
modelo = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(6 * 60, activation='softmax')  # Saída para 6 números com 60 possibilidades cada
])

# Compilar o modelo
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
modelo.fit(X_train, y_train.flatten(), epochs=100, batch_size=32, validation_split=0.2)

# Fazer previsões no conjunto de teste
predicoes = modelo.predict(X_test)

# Converter as probabilidades em categorias
numeros_previstos = np.argmax(predicoes, axis=1)

# Transformar a previsão de volta para o intervalo 1-60 e dividir em 6 números
numeros_previstos = numeros_previstos[:len(numeros_previstos) // 6 * 6].reshape(-1, 6) + 1

# Exibir os números previstos
print("Números sugeridos:", numeros_previstos[:10])  # Exibindo os primeiros 10 conjuntos de números sugeridos
