import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
import csv

#########################
# Classe Graph avec chargement CSV ou saisie manuelle
#########################
class Graph:
    def __init__(self) -> None:
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
                        f"Distance entre {i} et {j}:")
                    self.MDist[i][j] = val
                else:
                    self.MDist[i][j] = float('inf')

    def remp_mat_cap(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    val = simpledialog.askinteger(
                        "Matrice des capacités",
                        f"Capacité entre {i} et {j}:")
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
                    self.MDist[i][j]  = float(dist)
                    self.MCap[i][j]   = float(cap)
                    self.MSat[i][j]   = float(sat)
        except Exception as e:
            print(f"Erreur chargement: {e}")

    def verifier_et_charger_fichier(self):
        fichier = simpledialog.askstring(
            "Fichier CSV",
            "Nom du fichier CSV (avec extension), ou laisser vide pour saisie manuelle :")
        if fichier and os.path.exists(fichier):
            max_idx = -1
            with open(fichier, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    u, v = int(row[0]), int(row[1])
                    max_idx = max(max_idx, u, v)
            self._init_matrices(max_idx + 1)
            self.charger_donnees_fichier(fichier)
            return True
        return False

    def visualize_graph(self, best_path=None):
        G = nx.DiGraph()
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    G.add_edge(i, j,
                               weight=self.MDist[i][j],
                               capacity=self.MCap[i][j],
                               saturation=self.MSat[i][j])
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue')
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, alpha=0.6)
        if best_path:
            edges = list(zip(best_path, best_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
        plt.title("Graph Visualization")
        plt.show()

#########################
# Fonctions de calcul (coût & heuristique)
#########################
def compute_cost_matrix(graph, w_dist=0.4, w_sat=0.6):
    N = graph.N
    cost = np.full((N, N), float('inf'))
    for u in range(N):
        for v in range(N):
            if graph.MGraph[u][v] == 1:
                ratio = graph.MSat[u][v]/graph.MCap[u][v] if graph.MCap[u][v] > 0 else float('inf')
                cost[u][v] = w_dist*graph.MDist[u][v] + w_sat*ratio
    return cost


def compute_heuristic(graph, goal):
    N = graph.N
    dist = [float('inf')]*N
    dist[goal] = 0
    visited = [False]*N
    for _ in range(N):
        u = min((d,i) for i,d in enumerate(dist) if not visited[i])[1]
        visited[u] = True
        for v in range(N):
            if graph.MGraph[v][u] == 1:
                alt = dist[u] + graph.MDist[v][u]
                if alt < dist[v]: dist[v] = alt
    return dist

#########################
# A* et Dijkstra sur matrice de coûts
#########################
def find_best_path_astar(graph, start, goal, w_dist=0.4, w_sat=0.6):
    cost = compute_cost_matrix(graph, w_dist, w_sat)
    G = nx.DiGraph()
    for u in range(graph.N):
        for v in range(graph.N):
            if cost[u][v] < float('inf'):
                G.add_edge(u, v, weight=cost[u][v])
    h_arr = compute_heuristic(graph, goal)
    def h(n, target=None): return h_arr[n]
    try:
        return nx.astar_path(G, start, goal, heuristic=h, weight='weight')
    except nx.NetworkXNoPath:
        return None


def find_best_path_dijkstra(graph, start, goal, w_dist=0.4, w_sat=0.6):
    cost = compute_cost_matrix(graph, w_dist, w_sat)
    N = graph.N
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
# Mode interactif (CSV ou manuel)
#########################
def main():
    root = tk.Tk(); root.withdraw()
    graph = Graph()
    if not graph.verifier_et_charger_fichier():
        # saisie manuelle
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
    snk = simpledialog.askinteger("Sink"  , "Nœud d'arrivée  (0..N-1):")
    if src is None or snk is None or not (0 <= src < graph.N and 0 <= snk < graph.N):
        messagebox.showerror("Erreur", "Indices invalides.")
        return
    path_a = find_best_path_astar(graph, src, snk)
    if path_a:
        messagebox.showinfo("A*", "Chemin A* : " + ' -> '.join(map(str,path_a)))
    else:
        messagebox.showinfo("A*", "Pas de chemin A*.")
    path_d, cost_d = find_best_path_dijkstra(graph, src, snk)
    #if path_d:
        #messagebox.showinfo("Dijkstra", f"Chemin Dijkstra : {' -> '.join(map(str,path_d))}\nCoût : {cost_d}")
    #else:
        #messagebox.showinfo("Dijkstra", "Pas de chemin Dijkstra.")
    graph.visualize_graph(best_path=path_a )#or path_d)

if __name__ == '__main__':
    main()
