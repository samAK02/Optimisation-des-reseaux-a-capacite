import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
import csv
import random

class Graph:
  
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

    N = G.N
    cost = np.full((N, N), np.inf)
    for u in range(N):
        for v in range(N):
            if G.MGraph[u][v] == 1:
                ratio = (G.MSat[u][v]/G.MCap[u][v]) if G.MCap[u][v]>0 else np.inf
                cost[u][v] = w_dist*G.MDist[u][v] + w_sat*ratio
    return cost


def compute_heuristic_array(G, goal):
   
    cost = compute_cost_matrix(G, 0.4, 0.6)
    N = G.N
    dist = [float('inf')]*N
    dist[goal] = 0
    visited = [False]*N
    for _ in range(N):
        u = min((d,i) for i,d in enumerate(dist) if not visited[i])[1]
        visited[u] = True
        for v in range(N):
            if cost[v][u] < float('inf') :
                alt = dist[u] + cost[v][u]
                if alt < dist[v]: 
                    dist[v] = alt
    return dist


def find_best_path_astar(G, start, goal, w_dist=0.4, w_sat=0.6):
   
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


def activation_imprévu(G, best_path):
    
    if not best_path or len(best_path)<2: return None,None,None
    idx = random.randrange(len(best_path)-1)
    u,v = best_path[idx], best_path[idx+1]
    G.MSat[u][v] = G.MCap[u][v]
    G.MGraph[u][v] = 0
    G.MDist[u][v] = np.inf
    return idx, u, v

import time

def test_temps_calcul():
    
    Ns = [10, 50, 100, 200, 400, 800]
    temps = []
    for N in Ns:
        # Génération aléatoire d'un graphe creux
        G = Graph()
        G._init_matrices(N)
        p = 0.1
        for i in range(N):
            for j in range(N):
                if i != j and random.random() < p:
                    G.MGraph[i][j] = 1
                    G.MDist[i][j] = random.uniform(1, 10)
                    cap = random.randint(1, 20)
                    G.MCap[i][j] = cap
                    G.MSat[i][j] = random.uniform(0, cap)
                else:
                    G.MGraph[i][j] = 0
                    G.MDist[i][j] = np.inf
        start, goal = 0, N - 1
        t0 = time.perf_counter()
        _path, _cost = find_best_path_astar(G, start, goal)
        t1 = time.perf_counter()
        temps.append(t1 - t0)

    plt.figure()
    plt.plot(Ns, temps, marker='o')
    plt.xlabel("Nombre de sommets")
    plt.ylabel("Temps de calcul (s)")
    plt.title("Performance de A* selon N")
    plt.grid(True)
    plt.show()



def main():
    """Lance l'application interactive avec choix du mode."""
    root = tk.Tk()
    root.withdraw()
    try:
        # Proposition de menu 
        choice = simpledialog.askinteger(
            "Choix du mode",
            "Tapez :\n"
            "1 pour test de temps de calcul\n"
            "2 pour A* interactif avec gestion d'imprévu\n"
            "3 pour comparer A* vs Dijkstra",
            minvalue=1, maxvalue=3
        )
        if choice is None:
            messagebox.showinfo("Annulé", "Aucune option sélectionnée.")
            return

        if choice == 1:
            test_temps_calcul()
            return

    
        graph = Graph()
        if not graph.verifier_et_charger_fichier():
            N = simpledialog.askinteger("Graph", "Nombre de nœuds :")
            if N is None or N <= 0:
                messagebox.showerror("Erreur","N doit être un entier positif.")
                return
            graph._init_matrices(N)
            graph.remp_mat_adj()
            graph.remp_mat_dist()
            graph.remp_mat_cap()
            graph.remp_mat_sat()

        src = simpledialog.askinteger("Source","Nœud départ (0..N-1):")
        sink = simpledialog.askinteger("Sink","Nœud arrivée (0..N-1):")
        if src is None or sink is None or not (0 <= src < graph.N and 0 <= sink < graph.N):
            messagebox.showerror("Erreur","Indices invalides.")
            return

        # Option 3 : comparaison entre  A* et Dijkstra
        if choice == 3:
            path_astar, cost_astar = find_best_path_astar(graph, src, sink)
            path_dijk, cost_dijk   = find_best_path_dijkstra(graph, src, sink)
            if not path_astar and not path_dijk:
                messagebox.showinfo("Comparaison",
                                    "Aucun chemin trouvé par A* ni par Dijkstra.")
                return

           
            msg = ""
            if path_astar:
                msg += f"A* → {' -> '.join(map(str, path_astar))} (coût={cost_astar:.2f})\n"
            else:
                msg += "A* → Aucun chemin trouvé\n"
            if path_dijk:
                msg += f"Dijkstra → {' -> '.join(map(str, path_dijk))} (coût={cost_dijk:.2f})"
            else:
                msg += "Dijkstra → Aucun chemin trouvé"

            messagebox.showinfo("Comparaison A* vs Dijkstra", msg)
            return

        # Option 2 : mode A* interactif avec la fonction activation_imprévu
        path, cost = find_best_path_astar(graph, src, sink)
        if not path:
            messagebox.showinfo("A*","Aucun chemin trouvé.")
            return
        messagebox.showinfo(
            "A*",
            f"Chemin initial A* : {' -> '.join(map(str, path))} (coût={cost:.2f})"
        )

        idx, u, v = activation_imprévu(graph, path)
        if idx is not None:
            reroute_idx = random.randint(0, idx)
            reroute_node = path[reroute_idx]
            messagebox.showwarning(
                "Imprévu",
                f"Problème sur l'arc {u}->{v}. Recalcul à partir du nœud {reroute_node}."
            )
            k = 3
            alternatives = []
            for _ in range(k):
                seg, _ = find_best_path_astar(graph, reroute_node, sink)
                if seg:
                    alternatives.append(seg)
                    for a, b in zip(seg, seg[1:]):
                        graph.MDist[a][b] = np.inf
            if alternatives:
                chosen = random.choice(alternatives)
                final_path = path[:reroute_idx] + chosen
                messagebox.showinfo(
                    "Chemin recalculé",
                    "Nouveau chemin : " + ' -> '.join(map(str, final_path))
                )
            else:
                messagebox.showinfo("Recalcule","Aucune alternative trouvée.")

    except KeyboardInterrupt:
        messagebox.showinfo("Annulé","Exécution interrompue par l'utilisateur.")
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
