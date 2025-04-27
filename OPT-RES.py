import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
import csv
import random

class Graph:
    """Représente un graphe orienté avec matrices d'adjacence, de distances, de capacités et de saturation."""
    def __init__(self):
        self.N = 0
        self.MGraph = None
        self.MDist = None
        self.MCap = None
        self.MSat = None

    def _init_matrices(self, N):
        """Initialise les matrices N×N pour le graphe."""
        self.N = N
        self.MGraph = np.zeros((N, N), dtype=int)
        self.MDist  = np.full((N, N), np.inf)
        self.MCap   = np.zeros((N, N), dtype=float)
        self.MSat   = np.zeros((N, N), dtype=float)

    def remp_mat_adj(self):
        """Remplit la matrice d'adjacence par saisie utilisateur."""
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    while True:
                        val = simpledialog.askinteger(
                            "Matrice d'adjacence",
                            f"Connexion entre le nœud {i} et {j} (1=oui/0=non):",
                            minvalue=0, maxvalue=1)
                        if val is None:
                            if messagebox.askyesno("Annuler", "Voulez-vous annuler la saisie manuelle?" ):
                                raise KeyboardInterrupt("Saisie annulée")
                            continue
                        self.MGraph[i][j] = val
                        break

    def remp_mat_dist(self):
        """Remplit la matrice des distances par saisie utilisateur."""
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.MDist[i][j] = 0.0
                elif self.MGraph[i][j] == 1:
                    while True:
                        val = simpledialog.askfloat(
                            "Matrice des distances",
                            f"Distance entre {i} et {j}:")
                        if val is None:
                            messagebox.showerror("Erreur", "Distance invalide.")
                            continue
                        self.MDist[i][j] = val
                        break
                else:
                    self.MDist[i][j] = np.inf

    def remp_mat_cap(self):
        """Remplit la matrice des capacités par saisie utilisateur."""
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    while True:
                        val = simpledialog.askinteger(
                            "Matrice des capacités",
                            f"Capacité entre {i} et {j}:",
                            minvalue=0)
                        if val is None:
                            messagebox.showerror("Erreur", "Capacité invalide.")
                            continue
                        self.MCap[i][j] = val
                        break

    def remp_mat_sat(self):
        """Remplit la matrice de saturation par saisie utilisateur."""
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    maxsat = int(self.MCap[i][j])
                    while True:
                        val = simpledialog.askinteger(
                            "Matrice de saturation",
                            f"Saturation entre {i} et {j} (0..{maxsat}):",
                            minvalue=0, maxvalue=maxsat)
                        if val is None:
                            messagebox.showerror("Erreur", "Saturation invalide.")
                            continue
                        self.MSat[i][j] = val
                        break

    def charger_donnees_fichier(self, fichier):
        """Charge les données depuis un fichier CSV déjà validé."""
        try:
            with open(fichier, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) != 5:
                        raise ValueError(f"Ligne mal formée: {row}")
                    u, v = int(row[0]), int(row[1])
                    d, cap, sat = float(row[2]), float(row[3]), float(row[4])
                    self.MGraph[u][v] = 1
                    self.MDist[u][v] = d
                    self.MCap[u][v]  = cap
                    self.MSat[u][v]  = sat
        except Exception as e:
            messagebox.showerror("Erreur chargement", str(e))
            raise

    def verifier_et_charger_fichier(self):
        """Demande le nom de fichier CSV, initialise les matrices et charge les données s'il existe."""
        fichier = simpledialog.askstring(
            "Fichier CSV",
            "Nom du fichier CSV (laisser vide pour saisie manuelle):")
        if fichier:
            if os.path.isfile(fichier):
                max_idx = 0
                with open(fichier, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        u, v = int(row[0]), int(row[1])
                        max_idx = max(max_idx, u, v)
                self._init_matrices(max_idx+1)
                self.charger_donnees_fichier(fichier)
                return True
            else:
                messagebox.showwarning("Attention", f"Fichier '{fichier}' introuvable.")
        return False

def compute_cost_matrix(G, w_dist=0.4, w_sat=0.6):
    """Calcule la matrice des coûts selon distance et saturation."""
    N = G.N
    cost = np.full((N, N), np.inf)
    for u in range(N):
        for v in range(N):
            if G.MGraph[u][v] == 1:
                ratio = (G.MSat[u][v]/G.MCap[u][v]) if G.MCap[u][v]>0 else np.inf
                cost[u][v] = w_dist*G.MDist[u][v] + w_sat*ratio
    return cost


def compute_heuristic_array(G, goal):
    """Calcule l'heuristique (distance minimale inversée) pour A*."""
    N = G.N
    dist = [np.inf]*N
    dist[goal] = 0
    visited = [False]*N
    for _ in range(N):
        u = min((d,i) for i,d in enumerate(dist) if not visited[i])[1]
        visited[u] = True
        for v in range(N):
            if G.MGraph[v][u]==1:
                alt = dist[u]+G.MDist[v][u]
                if alt<dist[v]: dist[v]=alt
    return dist


def find_best_path_astar(G, start, goal, w_dist=0.4, w_sat=0.6):
    """Retourne le chemin et son coût via A* sur la matrice des coûts."""
    cost = compute_cost_matrix(G, w_dist, w_sat)
    NXG = nx.DiGraph()
    NXG.add_nodes_from(range(G.N))
    for u in range(G.N):
        for v in range(G.N):
            if cost[u][v]<np.inf:
                NXG.add_edge(u,v,weight=cost[u][v])
    h_arr = compute_heuristic_array(G, goal)
    def h(u,v): return h_arr[u]
    try:
        path = nx.astar_path(NXG, start, goal, heuristic=h, weight='weight')
        total = sum(cost[u][v] for u,v in zip(path,path[1:]))
        return path, total
    except:
        return None, np.inf


def find_best_path_dijkstra(G, start, goal, w_dist=0.4, w_sat=0.6):
    """Retourne le chemin et coût via Dijkstra sur la matrice des coûts."""
    cost = compute_cost_matrix(G, w_dist, w_sat)
    N = G.N
    dist = [np.inf]*N
    prev = [None]*N
    dist[start] = 0
    visited = [False]*N
    for _ in range(N):
        u = min((d,i) for i,d in enumerate(dist) if not visited[i])[1]
        visited[u] = True
        for v in range(N):
            if cost[u][v]<np.inf:
                alt = dist[u]+cost[u][v]
                if alt<dist[v]:
                    dist[v]=alt; prev[v]=u
    path=[]; u=goal
    while u is not None:
        path.insert(0,u); u=prev[u]
    return path, dist[goal]


def mise_a_jour(G):
    """Mise à jour aléatoire des saturations."""
    for i in range(G.N):
        for j in range(G.N):
            if G.MGraph[i][j]==1:
                delta = random.uniform(-0.1,0.1)
                new_sat = G.MSat[i][j]*(1+delta)
                G.MSat[i][j] = max(0, min(new_sat, G.MCap[i][j]))


def sabotage(G, best_path):
    """Simule un sabotage en saturant et désactivant un arc du chemin."""
    if not best_path or len(best_path)<2: return None,None,None
    idx = random.randrange(len(best_path)-1)
    u,v = best_path[idx], best_path[idx+1]
    G.MSat[u][v] = G.MCap[u][v]
    G.MGraph[u][v] = 0
    G.MDist[u][v] = np.inf
    return idx, u, v


def main_interactive():
    """Lance l'application interactive avec Tkinter."""
    root = tk.Tk(); root.withdraw()
    graph = Graph()
    try:
        if not graph.verifier_et_charger_fichier():
            N = simpledialog.askinteger("Graph", "Nombre de nœuds :")
            if N is None or N<=0:
                messagebox.showerror("Erreur","N doit être un entier positif.")
                return
            graph._init_matrices(N)
            graph.remp_mat_adj()
            graph.remp_mat_dist()
            graph.remp_mat_cap()
            graph.remp_mat_sat()
        src = simpledialog.askinteger("Source","Nœud départ (0..N-1):")
        sink = simpledialog.askinteger("Sink","Nœud arrivée (0..N-1):")
        if src is None or sink is None or not(0<=src<graph.N and 0<=sink<graph.N):
            messagebox.showerror("Erreur","Indices invalides.")
            return
        path, cost = find_best_path_astar(graph, src, sink)
        if not path:
            messagebox.showinfo("A*","Aucun chemin trouvé.")
            return
        messagebox.showinfo("A*",f"Chemin initial A* : {' -> '.join(map(str,path))} (coût={cost:.2f})")
        idx,u,v = sabotage(graph,path)
        final_path = path
        if idx is not None:
            reroute_idx = random.randint(0, idx)
            reroute_node = path[reroute_idx]
            messagebox.showwarning("Imprévu",f"Problème sur l'arc {u}->{v}. Recalcul à partir du nœud {reroute_node}.")
            k=3; alternatives=[]
            for _ in range(k):
                seg,_ = find_best_path_astar(graph, reroute_node, sink)
                if seg:
                    alternatives.append(seg)
                    for a,b in zip(seg,seg[1:]): graph.MDist[a][b]=np.inf
            if alternatives:
                chosen = random.choice(alternatives)
                final_path = path[:reroute_idx] + chosen
                messagebox.showinfo("Chemin recalculé","Nouveau chemin : "+' -> '.join(map(str,final_path)))
            else:
                messagebox.showinfo("Recalcule","Aucune alternative trouvée.")
    except KeyboardInterrupt:
        messagebox.showinfo("Annulé","Exécution interrompue par l'utilisateur.")
    finally:
        root.destroy()



if __name__ == '__main__':
    main_interactive()
