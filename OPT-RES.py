import os
import numpy as np
import requests
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import csv

class Graph:
    def __init__(self, N) -> None:
        self.N = N
        self.MGraph = np.zeros((N, N), dtype=int)      # matrice d'adjacence 
        self.MDist = np.full((N, N), float('inf'))     # matrice des distances entre les nœuds
        self.MCap = np.zeros((N, N))                   # matrice des capacités des arcs 
        self.MSat = np.zeros((N,N))                    # matrice de saturation des arcs

    def remp_mat_adj(self):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:                             # On ne demande pas de saisie pour les diagonales, les boucles ne sont d'aucune utilité au programme
                    value = simpledialog.askinteger(
                        "Matrice d'adjacence",
                        f"Connexion entre le nœud {i} et {j} (1/0):",
                        minvalue=0, maxvalue=1)
                    self.MGraph[i][j] = value

    def remp_mat_dist(self):
        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.MGraph[i][j] == 1:
                    value = simpledialog.askfloat(
                        "Matrice des distances",
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
                    value = simpledialog.askinteger(
                        "Matrice des capacités",
                        f"Capacité entre le nœud {i} et {j}:")
                    self.MCap[i][j] = value

    def remp_mat_sat(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    value = simpledialog.askinteger(
                        "Matrice de saturation",
                        f"Saturation entre le nœud {i} et {j} (Max {self.MCap[i][j]}):",
                        minvalue=0, maxvalue=self.MCap[i][j])
                    self.MSat[i][j] = value


#############################################################################################################################################################################################

    def charger_données_fichier(self, fichier):
            try:
                with open(fichier, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Ignore la première ligne d'en-têtes
                    
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
                
#######################################################################################################################################################################################
   
    def update(self):  #ce serait bien d'avoir une API pour trouver les données automatiquement 
        url = "exemple d'url récup avec une API"
        try:
            response = requests.get(url)
            data = response.json()

            for i in range(self.N):
                for j in range(self.N):
                    if self.MGraph[i][j] == 1:
                        self.MCap[i][j] += 1
        except Exception as e:
            print(f"Erreur lors de la mise à jour des capacités : {e}")



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

        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=16, font_weight='bold', arrows=True)

        if best_path:
            edgelist = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color='red', width=2)

        nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges() if edge not in edgelist], edge_color='blue', width=1)

        
        edge_labels = {(u, v): f"{weights[(u, v)]} km, {saturations[(u, v)]}/{capacities[(u, v)]} veh" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')

        plt.title("Visualisation du Graphe avec le Meilleur Chemin")
        plt.show()


def identifier_goulets(Graph):
    goulets = []
    for i in range(Graph.N):
        for j in range(Graph.N):
            if Graph.MGraph[i][j] == 1:
                capacity = Graph.MCap[i][j]
                current_flow = Graph.MSat[i][j]
                if current_flow >= capacity * 0.8:  # fication du seuil du goulet d'étranglement
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

        
        for g in goulets:                  # Penalité pour les goulets
            if g[0] == u and g[1] == v:
                saturation_level = Graph.MSat[u][v] / Graph.MCap[u][v]
                goulets_penalty += saturation_level
                break

    
        
        if Graph.MCap[u][v] != 0:
            sum_ratio_sat += (Graph.MSat[u][v] / Graph.MCap[u][v])
        else:
            sum_ratio_sat += float('inf')
                
    # normalisation des scores 
    distance_score = 1 / (1 + sum_distance)  # on inverse parce qu'on veut favoriser le plus petit score
    goulets_penalty_score = 1 / (1 + goulets_penalty)  # pénalité pour la plus grande saturation 
    ratio_sat_score = 1 / (1 + sum_ratio_sat)  # pénalité pour le plus grand ratio saturation/capacité
 

                     # calcul du score final pour le chemin trouvé
    final_score = (
        distance_score * 0.4 + 
        ratio_sat_score * 0.5 + 
        goulets_penalty_score * 0.1 
    )
    return final_score

#####################################################################################################################

def A_star(Graph, start, end):


    # Initialiser les listes de nœuds ouverts et fermés
    open_set = [start]
    came_from = {}

    # Initialiser les scores g et f pour tous les nœuds
    g_score = {}
    f_score = {}

    for i in range(Graph.N):
        g_score[i] = float('inf')
        f_score[i] = float('inf')

    g_score[start] = 0
    f_score[start] = heuristic(Graph, start, end)

    while len(open_set) > 0:
        # Trouver le nœud avec le score f le plus bas
        current = open_set[0]
        for node in open_set:
            if f_score[node] < f_score[current]:
                current = node

        if current == end:
            return reconstruct_path(came_from, current)

        open_set.remove(current)

        for neighbor in range(Graph.N):
            if Graph.MGraph[current][neighbor] == 1:
                tentative_g_score = g_score[current] + Graph.MDist[current][neighbor]

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(Graph, neighbor, end)

                    if neighbor not in open_set:
                        open_set.append(neighbor)

    return None


def heuristic(Graph, node, end):
    distance = Graph.MDist[node][end]

    if distance == float('inf'):
        return float('inf')

    if Graph.MGraph[node][end] == 1 and Graph.MCap[node][end] > 0:
        sat_ratio = Graph.MSat[node][end] / Graph.MCap[node][end]
    else:
        sat_ratio = 0

    penalty_weight = 1.0
    heuristic_cost = distance + penalty_weight * sat_ratio * distance

    return heuristic_cost

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path



# #####################################################################################################################

def find_best_path(Graph, start, end):


    if Graph is None or Graph.N == 0:
        return None
    
    if Graph.N > 10:
        print("Nous allons utiliser la méthode approchée avec A*")
        return A_star(Graph, start, end )

    all_paths = find_all_paths(Graph, start, end)
    if not all_paths:
        return None
    
    goulets = identifier_goulets(Graph)
    best_path = None
    best_score = -1

    for path in all_paths:
        score = score_path(Graph, path, goulets)
        if score > best_score:
            best_score = score
            best_path = path

    return best_path


def main():
    
    root = tk.Tk()
    root.withdraw() 


    N = simpledialog.askinteger("Graph", "Veuillez entrer le nombre de nœuds du graph:")
    if N is None or N <= 0:
        messagebox.showerror("Erreur", "Le nombre de nœuds doit être positif.")
        return
    
    
    graphe = Graph(N)

    if not graphe.verifier_et_charger_fichier():  
        graphe.remp_mat_adj()
        graphe.remp_mat_dist()
        graphe.remp_mat_cap()
        graphe.remp_mat_sat()

    source = simpledialog.askinteger("Source", "Veuillez entrer le nœud de départ (0 à N-1):")
    if source is None or not 0 <= source < N:
        messagebox.showerror("Erreur", "Le nœud de départ doit être compris entre 0 et N-1.")
        return

    sink = simpledialog.askinteger("Sink", "Veuillez entrer le nœud d'arrivée (0 à N-1):")
    if sink is None or not 0 <= sink < N:
        messagebox.showerror("Erreur", "Le nœud d'arrivée doit être compris entre 0 et N-1.")
        return

    best_path = find_best_path(graphe, source, sink)

    if best_path is None:
        messagebox.showinfo("Aucun Chemin", "Aucun chemin n'a été trouvé entre les nœuds spécifiés.")
    else:
        messagebox.showinfo("Meilleur Chemin", f"Le meilleur chemin est: {' -> '.join(map(str, best_path))}")
        graphe.visualize_graph(best_path=best_path)

if __name__ == "__main__":
    main()






