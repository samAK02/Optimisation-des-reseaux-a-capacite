import time
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
import csv
import random

#########################
# Classe Graph avec chargement CSV ou saisie manuelle
#########################
class Graph:
    def __init__(self):
        self.N = None
        self.MGraph = None
        self.MDist  = None
        self.MCap   = None
        self.MSat   = None

    def _init_matrices(self, N):
        self.N = N
        self.MGraph = np.zeros((N, N), dtype=int)
        self.MDist  = np.full((N, N), float('inf'))
        self.MCap   = np.zeros((N, N))
        self.MSat   = np.zeros((N, N))

    def remp_mat_adj(self):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    val = simpledialog.askinteger(
                        "Matrice d'adjacence",
                        f"Connexion entre le nœud {i} et {j} (1/0):",
                        minvalue=0, maxvalue=1)
                    self.MGraph[i][j] = val

    def remp_mat_dist(self):
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.MDist[i][j] = 0
                elif self.MGraph[i][j] == 1:
                    val = simpledialog.askfloat(
                        "Matrice des distances",
                        f"Distance entre {i} et {j}:" )
                    self.MDist[i][j] = val
                else:
                    self.MDist[i][j] = float('inf')

    def remp_mat_cap(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    val = simpledialog.askinteger(
                        "Matrice des capacités",
                        f"Capacité entre {i} et {j}:" )
                    self.MCap[i][j] = val

    def remp_mat_sat(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    val = simpledialog.askinteger(
                        "Matrice de saturation",
                        f"Saturation entre {i} et {j} (Max {self.MCap[i][j]}):",
                        minvalue=0, maxvalue=int(self.MCap[i][j]))
                    self.MSat[i][j] = val

    def charger_donnees_fichier(self, fichier):
        try:
            with open(fichier, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) != 5:
                        raise ValueError("Format incorrect dans le fichier")
                    u, v, dist, cap, sat = row
                    i, j = int(u), int(v)
                    self.MGraph[i][j] = 1
                    self.MDist[i][j] = float(dist)
                    self.MCap[i][j]  = float(cap)
                    self.MSat[i][j]  = float(sat)
        except Exception as e:
            print(f"Erreur chargement: {e}")

    def verifier_et_charger_fichier(self):
        fichier = simpledialog.askstring(
            "Fichier CSV",
            "Nom du fichier CSV (laisser vide pour saisie manuelle) :")
        if fichier and os.path.exists(fichier):
            max_idx = -1
            with open(fichier, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    u, v = int(row[0]), int(row[1])
                    max_idx = max(max_idx, u, v)
            self._init_matrices(max_idx+1)
            self.charger_donnees_fichier(fichier)
            return True
        return False

    def visualize_graph(self, best_path=None):
        G = nx.DiGraph()
        for u in range(self.N):
            for v in range(self.N):
                if self.MGraph[u][v] == 1:
                    G.add_edge(u, v,
                               weight=self.MDist[u][v],
                               capacity=self.MCap[u][v],
                               saturation=self.MSat[u][v])
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue')
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, alpha=0.6)
        if best_path:
            edges = list(zip(best_path, best_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
        plt.show()

#########################
# Calculs de coût et heuristique
#########################
def compute_cost_matrix(G, w_dist=0.4, w_sat=0.6):
    N = G.N
    cost = np.full((N, N), float('inf'))
    for u in range(N):
        for v in range(N):
            if G.MGraph[u][v] == 1:
                ratio = G.MSat[u][v]/G.MCap[u][v] if G.MCap[u][v]>0 else float('inf')
                cost[u][v] = w_dist*G.MDist[u][v] + w_sat*ratio
    return cost

def compute_heuristic_array(G, goal):
    N = G.N
    dist = [float('inf')]*N
    dist[goal] = 0
    visited = [False]*N
    for _ in range(N):
        u = min((d,i) for i,d in enumerate(dist) if not visited[i])[1]
        visited[u] = True
        for v in range(N):
            if G.MGraph[v][u] == 1:
                alt = dist[u] + G.MDist[v][u]
                if alt < dist[v]: dist[v] = alt
    return dist

#########################
# A* et Dijkstra sur matrice de coûts
#########################
def find_best_path_astar(G, start, goal, w_dist=0.4, w_sat=0.6):
    cost = compute_cost_matrix(G, w_dist, w_sat)
    NXG = nx.DiGraph()
    for u in range(G.N):
        for v in range(G.N):
            if cost[u][v] < float('inf'):
                NXG.add_edge(u, v, weight=cost[u][v])
    h_arr = compute_heuristic_array(G, goal)
    def h(u, v=None): return h_arr[u]
    try:
        return nx.astar_path(NXG, start, goal, heuristic=h, weight='weight')
    except nx.NetworkXNoPath:
        return None

def find_best_path_dijkstra(G, start, goal, w_dist=0.4, w_sat=0.6):
    cost = compute_cost_matrix(G, w_dist, w_sat)
    N = G.N
    dist = [float('inf')]*N
    prev = [None]*N
    dist[start] = 0
    visited = [False]*N
    for _ in range(N):
        u = min((d,i) for i,d in enumerate(dist) if not visited[i])[1]
        visited[u] = True
        for v in range(N):
            if cost[u][v] < float('inf'):
                alt = dist[u] + cost[u][v]
                if alt < dist[v]: dist[v], prev[v] = alt, u
    path = []
    u = goal
    while u is not None:
        path.insert(0, u)
        u = prev[u]
    return path, dist[goal]

#########################
# Benchmark et tests de performance
#########################
def test_performance(N):
    graph = Graph()
    graph._init_matrices(N)
    for u in range(N):
        for v in range(N):
            if u != v:
                graph.MGraph[u][v] = 1
                graph.MDist[u][v] = np.random.randint(1, 10)
                graph.MCap[u][v]  = np.random.randint(1, 100)
                graph.MSat[u][v]  = np.random.randint(0, int(graph.MCap[u][v]) + 1)
    t0 = time.time()
    _ = find_best_path_astar(graph, 0, N-1)
    t_astar = time.time() - t0
    t0 = time.time()
    _ , _ = find_best_path_dijkstra(graph, 0, N-1)
    t_dij = time.time() - t0
    return t_astar, t_dij

def plot_performance():
    Ns = range(2, 13)
    astar_times, dij_times = [], []
    for N in Ns:
        t_ast, t_dij = test_performance(N)
        astar_times.append(t_ast)
        dij_times.append(t_dij)
    plt.plot(list(Ns), astar_times, marker='o', label='A*')
    plt.plot(list(Ns), dij_times, marker='o', linestyle='--', label='Dijkstra')
    plt.xlabel('Nombre de nœuds')
    plt.ylabel('Temps (s)')
    plt.title('Performance A* vs Dijkstra')
    plt.legend()
    plt.grid(True)
    plt.show()

#########################
# Mode interactif
#########################
def main_interactive():
    root = tk.Tk(); root.withdraw()
    graph = Graph()
    if not graph.verifier_et_charger_fichier():
        N = simpledialog.askinteger("Graph", "Nombre de nœuds :")
        if N is None or N <= 0:
            messagebox.showerror("Erreur", "N doit être positif.")
            return
        graph._init_matrices(N)
        graph.remp_mat_adj()
        graph.remp_mat_dist()
        graph.remp_mat_cap()
        graph.remp_mat_sat()
    src = simpledialog.askinteger("Source", "Nœud de départ (0..N-1):")
    snk = simpledialog.askinteger("Sink", "Nœud d'arrivée (0..N-1):")
    if src is None or snk is None or not (0 <= src < graph.N and 0 <= snk < graph.N):
        messagebox.showerror("Erreur", "Indices invalides.")
        return
    path_ast = find_best_path_astar(graph, src, snk)
    if path_ast:
        messagebox.showinfo("A*", "Chemin A* : " + ' -> '.join(map(str,path_ast)))
    else:
        messagebox.showinfo("A*", "Aucun chemin A* trouvé")
    path_d, cost_d = find_best_path_dijkstra(graph, src, snk)
    if path_d:
        messagebox.showinfo("Dijkstra", f"Chemin Dijkstra : {' -> '.join(map(str,path_d))}\nCoût : {cost_d}")
    else:
        messagebox.showinfo("Dijkstra", "Aucun chemin Dijkstra trouvé")
    root.destroy()
    graph.visualize_graph(best_path=path_ast or path_d)

if __name__ == '__main__':
    # plot_performance() pour benchmark, sinon interactif
    main_interactive()
