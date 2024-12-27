import numpy as np
import random
from collections import defaultdict

# Parâmetros do algoritmo genético
TAMANHO_POPULACAO = 100
GERACOES = 100
TAXA_CRUZAMENTO = 0.8
TAXA_MUTACAO = 0.1
NUMEROS_MEGA_SENA = 6
NUMERO_MAXIMO = 60

# Função para gerar um sorteio aleatório
def gerar_sorteio():
    return np.random.choice(range(1, NUMERO_MAXIMO + 1), size=NUMEROS_MEGA_SENA, replace=False)

# Função de avaliação (fitness)
def calcular_fitness(individuo, sorteio_atual):
    return len(set(individuo).intersection(set(sorteio_atual)))

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
    populacao = gerar_populacao(TAMANHO_POPULACAO)
    resultados = defaultdict(list)  # Para armazenar as melhores combinações por número de acertos

    for geracao in range(GERACOES):
        sorteio_atual = gerar_sorteio()
        fitnesses = [calcular_fitness(individuo, sorteio_atual) for individuo in populacao]
        nova_populacao = []

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecionar_torneio(populacao, fitnesses)
            pai2 = selecionar_torneio(populacao, fitnesses)

            if np.random.rand() < TAXA_CRUZAMENTO:
                filho1, filho2 = cruzar(pai1, pai2)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()

            filho1 = mutar(filho1)
            filho2 = mutar(filho2)

            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:TAMANHO_POPULACAO]

        melhor_fitness = max(fitnesses)
        melhor_individuo = populacao[np.argmax(fitnesses)]
        resultados[melhor_fitness].append(np.sort(melhor_individuo))

        if melhor_fitness > 2:
            print(f"Geração {geracao + 1}: Melhor Fitness: {melhor_fitness}, Melhor Individuo: {np.sort(melhor_individuo)}")

        if melhor_fitness > 4:
            break

    return resultados

# Executar o AG
resultados = algoritmo_genetico()

# Exibir resultados
print("\nResumo dos resultados:")
for acertos, combinacoes in sorted(resultados.items(), reverse=True):
    if acertos > 2:
        print(f"{acertos} acertos: {len(combinacoes)}")
    for combinacao in combinacoes:
        if acertos > 2:
            print(f"Combinacao: {combinacao}")
