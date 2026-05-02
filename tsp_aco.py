"""
===========================================================
TSP com Ant Colony Optimization (ACO)
===========================================================

Uso:
- Certifique-se de ter o arquivo: distancia_matrix.csv
- Execute: python tsp_aco.py

Requisitos:
pip install numpy pandas matplotlib networkx imageio

===========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import imageio
import os
import random

# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# parâmetros padrão
DEFAULT_PARAMS = {
    "n_ants": 20,
    "n_iterations": 2000,
    "rho": 0.5,
    "alpha": 1,
    "beta": 5,
    "Q": 100,
    "convergence_limit": 50
}

# ==========================
# CLASSE ACO
# ==========================

class ACO:
    def __init__(self, dist_matrix, params):
        self.dist = dist_matrix
        self.n = len(dist_matrix)

        self.n_ants = params["n_ants"]
        self.n_iterations = params["n_iterations"]
        self.rho = params["rho"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.Q = params["Q"]
        self.convergence_limit = params["convergence_limit"]

        self.pheromone = np.ones((self.n, self.n))
        self.heuristic = 1 / (self.dist + 1e-10)

        self.best_distance = float("inf")
        self.best_route = None

        self.history = []
        self.frames = []

    def _probability(self, i, visited):
        prob = np.zeros(self.n)

        for j in range(self.n):
            if j not in visited:
                tau = self.pheromone[i][j] ** self.alpha
                eta = self.heuristic[i][j] ** self.beta
                prob[j] = tau * eta

        total = prob.sum()

        # 🚨 CASO CRÍTICO: soma zero ou inválida
        if total <= 0 or np.isnan(total) or np.isinf(total):
            # fallback: escolha uniforme entre não visitados
            choices = [j for j in range(self.n) if j not in visited]
            prob = np.zeros(self.n)
            for j in choices:
                prob[j] = 1.0 / len(choices)
            return prob

        return prob / total

    def _construct_solution(self):
        route = [np.random.randint(self.n)]
        visited = set(route)

        while len(route) < self.n:
            current = route[-1]
            probs = self._probability(current, visited)
            next_city = np.random.choice(range(self.n), p=probs)
            route.append(next_city)
            visited.add(next_city)

        route.append(route[0])
        return route

    def _route_distance(self, route):
        return sum(self.dist[route[i]][route[i+1]] for i in range(len(route)-1))

    def _update_pheromone(self, all_routes):
        # evaporação
        self.pheromone *= (1 - self.rho)

        # depósito
        for route, dist in all_routes:
            for i in range(len(route)-1):
                a, b = route[i], route[i+1]
                self.pheromone[a][b] += self.Q / dist
                self.pheromone[b][a] += self.Q / dist

    def _save_frame(self, iteration):
        G = nx.Graph()

        for i in range(self.n):
            G.add_node(i)

        for i in range(self.n):
            for j in range(i+1, self.n):
                weight = self.pheromone[i][j]
                G.add_edge(i, j, weight=weight)

        pos = nx.spring_layout(G, seed=SEED)

        edges = list(G.edges())
        weights = [G[u][v]['weight'] for u, v in edges]

        max_w = max(weights) if max(weights) > 0 else 1

        plt.figure(figsize=(6, 6))

        # nós
        nx.draw_networkx_nodes(G, pos, node_color='red')

        # arestas (CORRETO)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=[(w / max_w) * 5 for w in weights],
            edge_color=weights,
            edge_cmap=plt.cm.Greys
        )

        # labels
        nx.draw_networkx_labels(G, pos)

        filename = f"frame_{iteration}.png"
        plt.savefig(filename)
        plt.close()

        self.frames.append(filename)
        
    def run(self, make_gif=True):
        no_improve = 0
        convergence_iter = None

        for it in range(self.n_iterations):
            all_routes = []

            improved = False  # 👈 novo

            for _ in range(self.n_ants):
                route = self._construct_solution()
                dist = self._route_distance(route)
                all_routes.append((route, dist))

                if dist < self.best_distance:
                    self.best_distance = dist
                    self.best_route = route
                    improved = True  # 👈 marcou melhora

            # 👇 depois de todas as formigas
            if improved:
                no_improve = 0
            else:
                no_improve += 1

            self._update_pheromone(all_routes)
            self.history.append(self.best_distance)

            # salvar frame
            if make_gif and it % 10 == 0:
                self._save_frame(it)

            if no_improve >= self.convergence_limit:
                convergence_iter = it
                break

        # gerar GIF
        if make_gif:
            with imageio.get_writer("aco_evolution.gif", mode='I', duration=2.0) as writer:
                for filename in self.frames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

            for f in self.frames:
                os.remove(f)

        return self.best_distance, self.best_route, convergence_iter

# ==========================
# EXECUÇÃO ÚNICA
# ==========================

def run_single(params):
    dist_matrix = pd.read_csv("distancia_matrix.csv", header=None).values
    aco = ACO(dist_matrix, params)

    best_dist, best_route, conv_iter = aco.run()

    print("\n=== RESULTADO ===")
    print("Melhor distância:", best_dist)
    print("Melhor rota:", best_route)
    print("Convergiu em:", conv_iter)

    plt.plot(aco.history)
    plt.title("Convergência")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor distância")
    plt.show()

    return best_dist, conv_iter

# ==========================
# EXPERIMENTO: NÚMERO DE FORMIGAS
# ==========================

def experiment_ants():
    ants_list = [3, 5, 10, 20, 40, 80]
    results = []

    for ants in ants_list:
        distances = []
        iterations = []

        for _ in range(10):
            params = DEFAULT_PARAMS.copy()
            params["n_ants"] = ants

            dist, conv = run_single_no_plot(params)
            distances.append(dist)
            iterations.append(conv if conv else params["n_iterations"])

        results.append({
            "ants": ants,
            "mean_dist": np.mean(distances),
            "std_dist": np.std(distances),
            "mean_iter": np.mean(iterations)
        })

    df = pd.DataFrame(results)
    print("\n=== EXPERIMENTO FORMIGAS ===")
    print(df)

    plt.figure(figsize=(12, 5))

    x = np.arange(len(df["ants"]))

    # ---- Gráfico 1: distância média ----
    plt.subplot(1, 2, 1)
    plt.bar(x, df["mean_dist"], yerr=df["std_dist"], capsize=5)
    plt.xticks(x, df["ants"])
    plt.xlabel("Número de Formigas")
    plt.ylabel("Distância Média")
    plt.title("Distância Média")
    plt.ylim(bottom=1300)

    # ---- Gráfico 2: iterações ----
    plt.subplot(1, 2, 2)
    plt.bar(x, df["mean_iter"])
    plt.xticks(x, df["ants"])
    plt.xlabel("Número de Formigas")
    plt.ylabel("Iterações até Convergência")
    plt.title("Convergência")

    plt.tight_layout()
    plt.show()

# ==========================
# EXPERIMENTO: EVAPORAÇÃO
# ==========================

def experiment_rho():
    rho_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for rho in rho_list:
        distances = []
        iterations = []

        for _ in range(10):
            params = DEFAULT_PARAMS.copy()
            params["rho"] = rho

            dist, conv = run_single_no_plot(params)
            distances.append(dist)
            iterations.append(conv if conv else params["n_iterations"])

        results.append({
            "rho": rho,
            "mean_dist": np.mean(distances),
            "std_dist": np.std(distances),
            "mean_iter": np.mean(iterations)
        })

    df = pd.DataFrame(results)
    print("\n=== EXPERIMENTO RHO ===")
    print(df)

    plt.figure(figsize=(12, 5))

    x = np.arange(len(df["rho"]))

    # ---- Gráfico 1 ----
    plt.subplot(1, 2, 1)
    plt.bar(x, df["mean_dist"], yerr=df["std_dist"], capsize=5)
    plt.xticks(x, df["rho"])
    plt.xlabel("Taxa de Evaporação")
    plt.ylabel("Distância Média")
    plt.title("Distância Média")
    plt.ylim(bottom=1300)

    # ---- Gráfico 2 ----
    plt.subplot(1, 2, 2)
    plt.bar(x, df["mean_iter"])
    plt.xticks(x, df["rho"])
    plt.xlabel("Taxa de Evaporação")
    plt.ylabel("Iterações até Convergência")
    plt.title("Convergência")

    plt.tight_layout()
    plt.show()

# ==========================
# EXECUÇÃO SEM PLOT (para experimentos)
# ==========================

def run_single_no_plot(params):
    dist_matrix = pd.read_csv("distancia_matrix.csv", header=None).values
    aco = ACO(dist_matrix, params)
    best_dist, _, conv_iter = aco.run(make_gif=False)
    return best_dist, conv_iter

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    # Execução padrão
    run_single(DEFAULT_PARAMS)

    # Experimentos
    #experiment_ants()
    #experiment_rho()
