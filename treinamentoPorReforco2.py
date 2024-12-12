import numpy as np
import pandas as pd
import gym
from gym import spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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

    def reset(self):
        """Reinicia o ambiente."""
        self.current_index = np.random.randint(0, len(self.numeros) - 1)
        return self.numeros[self.current_index]

    def step(self, action):
        """Passo no ambiente."""
        true_numbers = self.numeros[self.current_index + 1]
        reward = 0

        # Sistema de recompensa
        for pred, true in zip(action, true_numbers):
            if pred == true:
                reward += 0.5  # Recompensa por número correto
            else:
                reward -= 2  # Penalidade por número errado

        done = True  # Cada episódio é um passo único
        self.current_index += 1
        return self.numeros[self.current_index], reward, done, {}

# Rede neural para DQN
def criar_modelo(estado_shape, acao_size):
    modelo = Sequential([
        Dense(128, activation='relu', input_shape=estado_shape),
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
        self.target_model = criar_modelo(estado_shape, acao_size)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Fator de desconto
        self.epsilon = 1.0  # Taxa de exploração inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def lembrar(self, estado, acao, recompensa, proximo_estado, done):
        self.memory.append((estado, acao, recompensa, proximo_estado, done))

    def agir(self, estado):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(1, 61), size=6, replace=False)  # Gera números únicos aleatórios
        
        # Previsão baseada no modelo
        q_valores = self.model.predict(np.array([estado]), verbose=0)[0]
        numeros_selecionados = []

        # Selecionar 6 números únicos baseados nos valores Q
        for _ in range(6):
            melhor_indice = np.argmax(q_valores) + 1
            while melhor_indice in numeros_selecionados:
                q_valores[np.argmax(q_valores)] = -np.inf  # Ignorar número repetido
                melhor_indice = np.argmax(q_valores) + 1
            numeros_selecionados.append(melhor_indice)

        return np.array(numeros_selecionados)

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
acao_size = 6  # 6 números na saída
agente = DQNAgent(estado_shape, acao_size)

# Treinamento
episodios = 20

for episodio in range(episodios):
    estado = env.reset()
    total_recompensa = 0

    for _ in range(1):  # Um passo por episódio
        acao = agente.agir(estado)
        proximo_estado, recompensa, done, _ = env.step(acao)
        agente.lembrar(estado, acao, recompensa, proximo_estado, done)
        estado = proximo_estado
        total_recompensa += recompensa

        if done:
            break

    agente.treinar()
    agente.atualizar_target_model()

    print(f"Episódio {episodio + 1}/{episodios}, Recompensa: {total_recompensa:.2f}, Epsilon: {agente.epsilon:.2f}")

# Testar o modelo treinado
estado = env.reset()
predicao = agente.agir(estado)
print("Predição de números únicos:", predicao)