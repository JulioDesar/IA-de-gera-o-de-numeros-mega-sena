import numpy as np
import random

# Parâmetros do algoritmo genético
TAMANHO_POPULACAO = 100
GERACOES = 500
TAXA_CRUZAMENTO = 0.8
TAXA_MUTACAO = 0.1
NUMEROS_MEGA_SENA = 6
NUMERO_MAXIMO = 60

# Gerar sorteio aleatório
def gerar_sorteio_aleatorio():
    return np.random.choice(range(1, NUMERO_MAXIMO + 1), size=NUMEROS_MEGA_SENA, replace=False)

# Função de avaliação (fitness)
def calcular_fitness(individuo, sorteio_aleatorio):
    acertos = len(set(individuo).intersection(set(sorteio_aleatorio)))
    recompensa = acertos  # Melhor recompensa para mais acertos
    return recompensa

# Geração inicial da população
def gerar_populacao(tamanho):
    return [np.random.choice(range(1, NUMERO_MAXIMO + 1), size=NUMEROS_MEGA_SENA, replace=False) for _ in range(tamanho)]

# Seleção por torneio
def selecionar_torneio(populacao, fitnesses, k=3):
    selecionados = np.random.choice(len(populacao), k)
    melhor_indice = selecionados[np.argmax([fitnesses[i] for i in selecionados])]
    return populacao[melhor_indice]

# Cruzamento
def cruzar(pai1, pai2):
    ponto_corte = random.randint(1, NUMEROS_MEGA_SENA - 1)
    filho1 = np.concatenate((pai1[:ponto_corte], [num for num in pai2 if num not in pai1[:ponto_corte]]))
    filho2 = np.concatenate((pai2[:ponto_corte], [num for num in pai1 if num not in pai2[:ponto_corte]]))

    # Garantir tamanho fixo de NUMEROS_MEGA_SENA
    filho1 = np.random.choice(filho1, size=NUMEROS_MEGA_SENA, replace=False)
    filho2 = np.random.choice(filho2, size=NUMEROS_MEGA_SENA, replace=False)
    return filho1, filho2

# Mutação
def mutar(individuo):
    if np.random.rand() < TAXA_MUTACAO:
        idx1, idx2 = np.random.choice(range(NUMEROS_MEGA_SENA), size=2, replace=False)
        individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
    return individuo

# Algoritmo Genético
def algoritmo_genetico():
    sorteio_aleatorio = gerar_sorteio_aleatorio()
    print(f"Sorteio Aleatório: {sorteio_aleatorio}")

    populacao = gerar_populacao(TAMANHO_POPULACAO)

    for geracao in range(GERACOES):
        fitnesses = [calcular_fitness(individuo, sorteio_aleatorio) for individuo in populacao]
        nova_populacao = []

        while len(nova_populacao) < TAMANHO_POPULACAO:
            # Seleção
            pai1 = selecionar_torneio(populacao, fitnesses)
            pai2 = selecionar_torneio(populacao, fitnesses)

            # Cruzamento
            if np.random.rand() < TAXA_CRUZAMENTO:
                filho1, filho2 = cruzar(pai1, pai2)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()

            # Mutação
            filho1 = mutar(filho1)
            filho2 = mutar(filho2)

            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:TAMANHO_POPULACAO]

        # Estatísticas da geração
        melhor_fitness = max(fitnesses)
        media_fitness = np.mean(fitnesses)
        melhor_individuo = populacao[np.argmax(fitnesses)]
        print(f"Geração {geracao + 1}: Melhor Fitness: {melhor_fitness:.2f}, Fitness Médio: {media_fitness:.2f}, Melhor Individuo: {melhor_individuo}")

    return populacao[np.argmax(fitnesses)]

# Executar o AG
melhor_solucao = algoritmo_genetico()
melhor_solucao_ordenada = np.sort(melhor_solucao)
print(f"Melhor solução encontrada: {melhor_solucao_ordenada}")
