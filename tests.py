import time
import os
import numpy as np
import requests
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import csv
import random

#########################
# toujours la classe Graph et méthodes de saisie/chargement
#########################
class Graph:
    def __init__(self, N) -> None:
        self.N = N
        self.MGraph = np.zeros((N, N), dtype=int)      # matrice d'adjacence
        self.MDist = np.full((N, N), float('inf'))       # matrice des distances entre les nœuds
        self.MCap = np.zeros((N, N))                     # matrice des capacités des arcs
        self.MSat = np.zeros((N, N))                     # matrice de saturation des arcs

    def remp_mat_adj(self):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    value = simpledialog.askinteger("Matrice d'adjacence",
                                                    f"Connexion entre le nœud {i} et {j} (1/0):",
                                                    minvalue=0, maxvalue=1)
                    self.MGraph[i][j] = value

    def remp_mat_dist(self):
        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.MGraph[i][j] == 1:
                    value = simpledialog.askfloat("Matrice des distances",
                                                  f"Veuillez entrer la distance entre le nœud {i} et {j}:")
                    self.MDist[i][j] = value
                elif i == j:
                    self.MDist[i][j] = 0
                else:
                    self.MDist[i][j] = float('inf')

    def remp_mat_cap(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    value = simpledialog.askinteger("Matrice des capacités",
                                                    f"Capacité entre le nœud {i} et {j}:")
                    self.MCap[i][j] = value

    def remp_mat_sat(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    value = simpledialog.askinteger("Matrice de saturation",
                                                    f"Saturation entre le nœud {i} et {j} (Max {self.MCap[i][j]}):",
                                                    minvalue=0, maxvalue=self.MCap[i][j])
                    self.MSat[i][j] = value

    def charger_données_fichier(self, fichier):
        try:
            with open(fichier, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # ignore la première ligne d'en-têtes
                for row in reader:
                    if len(row) != 5:
                        raise ValueError("Format incorrect dans le fichier")
                    node1, node2, dist, cap, sat = map(float, row)
                    node1, node2 = int(node1), int(node2)
                    self.MGraph[node1][node2] = 1
                    self.MDist[node1][node2] = dist
                    self.MCap[node1][node2] = cap
                    self.MSat[node1][node2] = sat
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")

    def verifier_et_charger_fichier(self):
        fichier = simpledialog.askstring("Fichier", "Veuillez entrer le nom du fichier (avec extension) contenant les matrices :")
        if fichier and os.path.exists(fichier):
            self.charger_données_fichier(fichier)
            return True
        return False

    def update(self):
        """
        mise à jour simulée du réseau.
        Ici, nous modifions aléatoirement la saturation des arcs pour imiter des données GPS.
        """
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    variation = random.uniform(-0.1, 0.1)
                    nouvelle_sat = self.MSat[i][j] * (1 + variation)
                    nouvelle_sat = max(0, min(nouvelle_sat, self.MCap[i][j]))
                    self.MSat[i][j] = nouvelle_sat

    def adjust_flow(self, node, flow):
        for prev_node in range(self.N):
            if self.MGraph[prev_node][node] == 1:
                residual_capacity = self.MCap[prev_node][node]
                if residual_capacity > 0:
                    adjusted_flow = min(flow, residual_capacity)
                    self.MCap[prev_node][node] -= adjusted_flow
                    flow -= adjusted_flow
                    self.adjust_flow(prev_node, adjusted_flow)
                    if flow == 0:
                        return

    def display_matrices(self):
        print("\nMatrice d'adjacence MGraph:")
        print(self.MGraph)
        print("\nMatrice des distances MDist:")
        print(self.MDist)
        print("\nMatrice des capacités MCap:")
        print(self.MCap)

    def visualize_graph(self, best_path=None):
        G = nx.DiGraph()
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    G.add_edge(i, j, capacity=self.MCap[i][j], weight=self.MDist[i][j], saturation=self.MSat[i][j])
        pos = nx.spring_layout(G)
        capacities = nx.get_edge_attributes(G, 'capacity')
        weights = nx.get_edge_attributes(G, 'weight')
        saturations = nx.get_edge_attributes(G, 'saturation')
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
        if best_path:
            best_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
            other_edges = [edge for edge in G.edges() if edge not in best_edges]
        else:
            best_edges = []
            other_edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='blue', width=1, alpha=0.6, zorder=1)
        if best_path:
            nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color='red', width=2, zorder=2)
        edge_labels = {(u, v): f"{weights[(u, v)]} km, {saturations[(u, v)]}/{capacities[(u, v)]} veh" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')
        plt.title("Visualisation du Graphe avec le Meilleur Chemin")
        plt.show()

#########################
# fonctions utilitaires
#########################
def identifier_goulets(Graph):
    goulets = []
    for i in range(Graph.N):
        for j in range(Graph.N):
            if Graph.MGraph[i][j] == 1:
                capacity = Graph.MCap[i][j]
                current_flow = Graph.MSat[i][j]
                if capacity > 0 and (current_flow / capacity) >= 0.8:
                    goulets.append((i, j, capacity, current_flow))
    return goulets

def dfs_all_paths(graph, source, sink, path, all_paths):
    path.append(source)
    if source == sink:
        all_paths.append(list(path))
    else:
        for i in range(graph.N):
            if graph.MGraph[source][i] == 1 and i not in path:
                dfs_all_paths(graph, i, sink, path, all_paths)
    path.pop()

def find_all_paths(Graph, source, sink):
    all_paths = []
    dfs_all_paths(Graph, source, sink, [], all_paths)
    return all_paths

def score_path(Graph, path, goulets):
    sum_distance = 0
    goulets_penalty = 0  
    sum_ratio_sat = 0
    for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]
        sum_distance += Graph.MDist[u][v]
        for g in goulets:
            if g[0] == u and g[1] == v:
                saturation_level = Graph.MSat[u][v] / Graph.MCap[u][v]
                goulets_penalty += saturation_level
                break
        if Graph.MCap[u][v] != 0:
            sum_ratio_sat += (Graph.MSat[u][v] / Graph.MCap[u][v])
        else:
            sum_ratio_sat += float('inf')
    distance_score = 1 / (1 + sum_distance)
    goulets_penalty_score = 1 / (1 + goulets_penalty)
    ratio_sat_score = 1 / (1 + sum_ratio_sat)
    final_score = (distance_score * 0.5 + ratio_sat_score * 0.3 + goulets_penalty_score * 0.2)
    return final_score

#########################
# fonction de calcul de l'heuristique (pré-calculée via Dijkstra inversé)
#########################
def compute_heuristic_array(Graph, goal):
    N = Graph.N
    dist = [float('inf')] * N
    dist[goal] = 0
    visited = [False] * N
    for _ in range(N):
        u = None
        min_val = float('inf')
        for i in range(N):
            if not visited[i] and dist[i] < min_val:
                min_val = dist[i]
                u = i
        if u is None:
            break
        visited[u] = True
        for v in range(N):
            if Graph.MGraph[v][u] == 1:
                if dist[v] > dist[u] + Graph.MDist[v][u]:
                    dist[v] = dist[u] + Graph.MDist[v][u]
    return dist

def heuristic(Graph, node, goal, heuristic_array):
    return heuristic_array[node]

#########################
# fonction hybride ACO_A_star (remplace A_star classique)
#########################
def ACO_A_star(Graph, start, goal, numAnts=10, numIterations=50, alpha=1.0, beta=2.0, evaporationRate=0.5, Q=100):
    pheromone = np.zeros((Graph.N, Graph.N))
    for i in range(Graph.N):
        for j in range(Graph.N):
            if Graph.MGraph[i][j] == 1:
                pheromone[i][j] = 1.0
            else:
                pheromone[i][j] = 0.0

    bestPath = None
    bestCost = float('inf')
    heuristic_array = compute_heuristic_array(Graph, goal)
    
    for iteration in range(numIterations):
        ant_paths = []
        ant_costs = []
        for ant in range(numAnts):
            current = start
            path = [current]
            costSoFar = 0.0
            while current != goal:
                candidates = []
                probabilities = []
                total_prob = 0.0
                for neighbor in range(Graph.N):
                    if Graph.MGraph[current][neighbor] == 1 and neighbor not in path:
                        if Graph.MDist[current][neighbor] == float('inf'):
                            continue
                        g = costSoFar + Graph.MDist[current][neighbor]
                        h = heuristic(Graph, neighbor, goal, heuristic_array)
                        if h == float('inf'):
                            continue
                        f = g + h
                        if f <= 0:
                            f = 1e-6
                        prob = (pheromone[current][neighbor] ** alpha) * ((1.0 / f) ** beta)
                        if prob > 0:
                            candidates.append(neighbor)
                            probabilities.append(prob)
                            total_prob += prob
                if not candidates or total_prob == 0:
                    break
                probabilities = [p / total_prob for p in probabilities]
                next_node = random.choices(candidates, weights=probabilities, k=1)[0]
                costSoFar += Graph.MDist[current][next_node]
                path.append(next_node)
                current = next_node
            if current == goal:
                ant_paths.append(path)
                ant_costs.append(costSoFar)
                if costSoFar < bestCost:
                    bestCost = costSoFar
                    bestPath = path
        for i in range(Graph.N):
            for j in range(Graph.N):
                pheromone[i][j] = (1 - evaporationRate) * pheromone[i][j]
        for idx, path in enumerate(ant_paths):
            cost_path_val = ant_costs[idx]
            delta = Q / cost_path_val if cost_path_val > 0 else Q
            for k in range(len(path) - 1):
                u = path[k]
                v = path[k+1]
                pheromone[u][v] += delta
    return bestPath

#########################
# fonction de fusion des matrices en une matrice de coût
#########################
def compute_cost_matrix(Graph, w_distance=0.4, w_sat=0.6):
    """
    Calcule une matrice de coût en combinant la distance et le ratio saturation/capacité.
    Pour chaque arc (i,j) existant, le coût est défini par :
       coût = w_distance * Graph.MDist[i][j] + w_sat * (Graph.MSat[i][j] / Graph.MCap[i][j])
    Si la capacité est nulle, le coût devient infini.
    """
    cost_matrix = np.full((Graph.N, Graph.N), float('inf'))
    for i in range(Graph.N):
        for j in range(Graph.N):
            if Graph.MGraph[i][j] == 1:
                if Graph.MCap[i][j] != 0:
                    ratio = Graph.MSat[i][j] / Graph.MCap[i][j]
                else:
                    ratio = float('inf')
                cost_matrix[i][j] = w_distance * Graph.MDist[i][j] + w_sat * ratio
    return cost_matrix

#########################
# implémentation de l'algorithme de Dijkstra
#########################
def dijkstra(cost_matrix, start, goal):
    N = cost_matrix.shape[0]
    dist = [float('inf')] * N
    prev = [None] * N
    dist[start] = 0
    visited = [False] * N

    for _ in range(N):
        u = None
        min_val = float('inf')
        for i in range(N):
            if not visited[i] and dist[i] < min_val:
                min_val = dist[i]
                u = i
        if u is None:
            break
        visited[u] = True
        for v in range(N):
            if cost_matrix[u][v] < float('inf'):
                alt = dist[u] + cost_matrix[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

    path = []
    u = goal
    if dist[u] < float('inf'):
        while u is not None:
            path.insert(0, u)
            u = prev[u]
    return path, dist[goal]

def best_path_dijkstra(Graph, start, goal, w_distance=0.4, w_sat=0.6):
    """
    Calcule le chemin optimal entre start et goal en fusionnant les matrices 
    de distance et de saturation en une matrice de coût, puis en appliquant Dijkstra.
    """
    cost_matrix = compute_cost_matrix(Graph, w_distance, w_sat)
    path, cost = dijkstra(cost_matrix, start, goal)
    return path, cost

#########################
# fonctions de gestion d'imprévus et sabotage
#########################
def mise_à_jour(Graph):
    """
    Simule une mise à jour aléatoire, imitant par exemple des données GPS.
    """
    for i in range(Graph.N):
        for j in range(Graph.N):
            if Graph.MGraph[i][j] == 1:
                variation = random.uniform(-0.1, 0.1)
                nouvelle_sat = Graph.MSat[i][j] * (1 + variation)
                nouvelle_sat = max(0, min(nouvelle_sat, Graph.MCap[i][j]))
                Graph.MSat[i][j] = nouvelle_sat

def gestion_imprevu(Graph, bestPath, start, goal, k):
    """
    Gère un imprévu : si sur le meilleur chemin actuel (bestPath)
    un arc est saturé à plus de 70% de sa capacité, on cherche
    K nouveaux chemins alternatifs via ACO_A_star et on pénalise
    les arcs utilisés pour favoriser l'exploration d'autres routes.
    """
    mise_à_jour(Graph)
    trigger_imprevu = False
    for idx in range(len(bestPath) - 1):
        u = bestPath[idx]
        v = bestPath[idx+1]
        if Graph.MCap[u][v] > 0 and (Graph.MSat[u][v] / Graph.MCap[u][v]) >= 0.7:
            trigger_imprevu = True
            break
    if not trigger_imprevu:
        return bestPath
    chemins = []
    for i in range(k):
        newPath = ACO_A_star(Graph, start, goal)
        if newPath is None:
            continue
        chemins.append(newPath)
        for idx in range(len(newPath) - 1):
            u = newPath[idx]
            v = newPath[idx+1]
            Graph.MSat[u][v] = Graph.MCap[u][v]   # forcer la saturation
            Graph.MDist[u][v] = float('inf')        # rendre cet arc très coûteux
    if chemins:
        return random.choice(chemins)
    else:
        return None

def sabotage(Graph, bestPath):
    """
    Fonction de sabotage utilisée pour tester la gestion d'imprévus.
    Choisit un arc aléatoire (sauf le dernier) dans bestPath et force sa saturation.
    """
    if len(bestPath) < 2:
        return
    idx = random.randint(0, len(bestPath) - 2)
    u = bestPath[idx]
    v = bestPath[idx + 1]
    Graph.MSat[u][v] = Graph.MCap[u][v]

#########################
# fonctions de test de performance
#########################
def test_performance(N):
    start_time = time.time()
    graph = Graph(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                graph.MGraph[i][j] = 1
                graph.MDist[i][j] = np.random.randint(1, 10)
                graph.MCap[i][j] = np.random.randint(1, 100)
                graph.MSat[i][j] = np.random.randint(0, int(graph.MCap[i][j]) + 1)
    source = 0
    sink = N - 1
    find_best_path(graph, source, sink)
    end_time = time.time()
    return end_time - start_time

def plot_performance():
    node_counts = range(1, 13)
    times = []
    for N in node_counts:
        print(f"Testing graph with {N} nodes...")
        t = test_performance(N)
        times.append(t)
    plt.figure(figsize=(10, 6))
    plt.plot(list(node_counts), times, marker='o')
    plt.xlabel('Nombre de nœuds')
    plt.ylabel('Temps (secondes)')
    plt.title('Performance du calcul de chemin')
    plt.grid(True)
    plt.show()

#########################
# fonctions principales pour mode interactif et comparaison
#########################
def main_interactive():
    root = tk.Tk()
    root.withdraw()

    N = simpledialog.askinteger("Graph", "Veuillez entrer le nombre de nœuds du graph:")
    if N is None or N <= 0:
        messagebox.showerror("Erreur", "Le nombre de nœuds doit être positif.")
        return

    graph = Graph(N)
    if not graph.verifier_et_charger_fichier():
        graph.remp_mat_adj()
        graph.remp_mat_dist()
        graph.remp_mat_cap()
        graph.remp_mat_sat()

    source = simpledialog.askinteger("Source", "Veuillez entrer le nœud de départ (0 à N-1):")
    if source is None or not 0 <= source < N:
        messagebox.showerror("Erreur", "Le nœud de départ doit être compris entre 0 et N-1.")
        return

    sink = simpledialog.askinteger("Sink", "Veuillez entrer le nœud d'arrivée (0 à N-1):")
    if sink is None or not 0 <= sink < N:
        messagebox.showerror("Erreur", "Le nœud d'arrivée doit être compris entre 0 et N-1.")
        return

    best_path_aco = ACO_A_star(graph, source, sink)
    if best_path_aco is None:
        messagebox.showinfo("Aucun Chemin", "Aucun chemin n'a été trouvé par ACO_A_star.")
    else:
        messagebox.showinfo("Chemin ACO_A_star", f"Chemin ACO_A_star: {' -> '.join(map(str, best_path_aco))}")

    # calcul du chemin de référence avec dijkstra sur la matrice de coût fusionnée
    dijkstra_path, dijkstra_cost = best_path_dijkstra(graph, source, sink)
    if dijkstra_path:
        messagebox.showinfo("Chemin Dijkstra", f"Chemin Dijkstra: {' -> '.join(map(str, dijkstra_path))}\nCoût: {dijkstra_cost}")

    # gestion d'imprévus
    sabotage(graph, best_path_aco)
    alt_path = gestion_imprevu(graph, best_path_aco, source, sink, k=3)
    if alt_path is not None:
        best_path_final = alt_path
    else:
        best_path_final = best_path_aco

    root.destroy()  # c'est pour fermer la fenêtre Tkinter
    graph.visualize_graph(best_path=best_path_final)

def main():
    MODE = "interactive"  # choix entre le mode "interactive" et "performance"
    if MODE == "performance":
        plot_performance()
    elif MODE == "interactive":
        main_interactive()

if __name__ == "__main__":
    main()
