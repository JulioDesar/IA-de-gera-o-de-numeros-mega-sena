import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scipy.stats import entropy
import random
from collections import deque

# Carregar dados da Mega-Sena
dados = pd.read_excel('Mega-Sena.xlsx')
numeros = dados[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].values

# Ambiente personalizado para aprendizado por reforço
class MegaSenaEnv(gym.Env):
    def __init__(self, numeros):
        super(MegaSenaEnv, self).__init__()
        self.numeros = numeros
        self.action_space = spaces.MultiDiscrete([60] * 6)  # 6 números entre 1 e 60
        self.observation_space = spaces.Box(low=1, high=60, shape=(6,), dtype=np.int32)
        self.current_index = 0
        self.history = []  # Histórico de ações

    def reset(self):
        """Reinicia o ambiente."""
        self.current_index = np.random.randint(0, len(self.numeros) - 1)
        return self.numeros[self.current_index]

    def step(self, action):
        """Passo no ambiente."""
        true_numbers = self.numeros[self.current_index + 1]
        reward = 0

        # Recompensa por números corretos
        acertos = [pred for pred in action if pred in true_numbers]
        reward += len(acertos)

        # Penalizar sequências fixas
        if list(action) == sorted(action) or list(action) == sorted(action, reverse=True):
            reward -= 5

        # Padrões históricos
        soma_acao = sum(action)
        media_acao = np.mean(action)
        soma_historica = np.mean([sum(x) for x in self.numeros])
        media_historica = np.mean(self.numeros)

        if abs(soma_acao - soma_historica) < 10:
            reward += 1
        if abs(media_acao - media_historica) < 2:
            reward += 1

        # Penalizar números repetidos no histórico
        if self.history:
            frequencias = np.bincount(np.concatenate(self.history), minlength=61)
            for num in action:
                reward -= frequencias[num] * 0.1

        # Recompensa por diversidade (entropia)
        probs = np.histogram(action, bins=np.arange(1, 62))[0] / len(action)
        reward += entropy(probs)

        done = True  # Cada episódio é um passo único
        self.history.append(action)
        self.current_index += 1
        return self.numeros[self.current_index], reward, done, {"acertos": acertos, "numeros_reais": true_numbers}

# Rede neural para DQN
def criar_modelo(estado_shape, acao_size):
    modelo = Sequential([
        Dense(128, activation='relu', input_shape=estado_shape),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(acao_size, activation='linear')
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return modelo

# Agente DQN
class DQNAgent:
    def __init__(self, estado_shape, acao_size):
        self.estado_shape = estado_shape
        self.acao_size = acao_size
        self.model = criar_modelo(estado_shape, acao_size)
        
        # Carregar modelo salvo, se existir
        if os.path.exists("mega_sena_dqn_model.h5"):
            self.model.load_weights("mega_sena_dqn_model.h5")
            print("Pesos carregados do modelo salvo.")
        
        self.target_model = criar_modelo(estado_shape, acao_size)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 32

    def lembrar(self, estado, acao, recompensa, proximo_estado, done):
        self.memory.append((estado, acao, recompensa, proximo_estado, done))

    def agir(self, estado):
        if np.random.rand() <= self.epsilon:
            return self._gerar_acao_aleatoria()

        q_valores = self.model.predict(np.array([estado]), verbose=0)[0]

        indices_ordenados = np.argsort(-q_valores)
        numeros_selecionados = []
        for idx in indices_ordenados:
            numero = (idx % 60) + 1
            if numero not in numeros_selecionados:
                numeros_selecionados.append(numero)
            if len(numeros_selecionados) == 6:
                break

        while self._verificar_sequencial(numeros_selecionados):
            numeros_selecionados = self._gerar_acao_aleatoria()

        return np.array(numeros_selecionados)

    def _gerar_acao_aleatoria(self):
        return np.random.choice(range(1, 61), size=6, replace=False)

    def _verificar_sequencial(self, numeros):
        numeros = sorted(numeros)
        for i in range(len(numeros) - 1):
            if numeros[i] + 1 == numeros[i + 1]:
                return True
        return False

    def treinar(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for estado, acao, recompensa, proximo_estado, done in minibatch:
            target = recompensa
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(np.array([proximo_estado]), verbose=0))

            q_valores = self.model.predict(np.array([estado]), verbose=0)
            q_valores[0][np.argmax(acao)] = target
            self.model.fit(np.array([estado]), q_valores, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def atualizar_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# Criar ambiente e agente
env = MegaSenaEnv(numeros)
estado_shape = env.observation_space.shape
acao_size = 6
agente = DQNAgent(estado_shape, acao_size)

# Treinamento
episodios = 500
total_acertos = 0

for episodio in range(episodios):
    estado = env.reset()
    total_recompensa = 0

    for _ in range(1):
        acao = agente.agir(estado)
        while len(set(acao)) < len(acao):
            acao = agente.agir(estado)

        proximo_estado, recompensa, done, info = env.step(acao)
        agente.lembrar(estado, acao, recompensa, proximo_estado, done)
        estado = proximo_estado
        total_recompensa += recompensa

        if done:
            acertos = len(info["acertos"])
            total_acertos += acertos
            numeros_reais = info["numeros_reais"]
            print(f"Episódio {episodio + 1}: Números jogados: {acao}, Números reais: {numeros_reais}, Acertos: {acertos}")
            break

    agente.treinar()
    agente.atualizar_target_model()

    print(f"Episódio {episodio + 1}/{episodios}, Recompensa: {total_recompensa:.2f}, Epsilon: {agente.epsilon:.2f}")

# Salvar modelo após o treinamento
agente.model.save("mega_sena_dqn_model.h5")
print("Modelo salvo como 'mega_sena_dqn_model.h5'.")

# Estatísticas finais
porcentagem_acertos = (total_acertos / (episodios * 6)) * 100
print(f"Total de acertos: {total_acertos}/{episodios * 6} ({porcentagem_acertos:.2f}%)")
