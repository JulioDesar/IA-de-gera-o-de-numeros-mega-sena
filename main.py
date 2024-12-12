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
def preparar_dados(numeros, n_passos=5):
    X, y = [], []
    for i in range(n_passos, len(numeros)):
        X.append(numeros.iloc[i-n_passos:i].values.flatten())
        y.append(numeros.iloc[i].values)
    return np.array(X), np.array(y)

# Preparar os dados para entrada (X) e saída (y)
X, y = preparar_dados(numeros, n_passos=5)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
X_train = X_train / 60
X_test = X_test / 60

# Ajustar rótulos
y_train = y_train.flatten() - 1  # Transformar para categorias 0-59
y_test = y_test.flatten() - 1

# Criar o modelo de rede neural
modelo = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(60, activation='softmax')  # Saída para 60 categorias
])

# Compilar o modelo
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
modelo.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Fazer previsões no conjunto de teste
predicoes = modelo.predict(X_test)

# Converter probabilidades em categorias
numeros_previstos = np.argmax(predicoes, axis=1)

print("Números sugeridos:", numeros_previstos[:6] + 1)  # Voltar para valores no intervalo 1-60
