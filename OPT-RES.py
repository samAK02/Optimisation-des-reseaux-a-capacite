"""
Programme d'optimisation en transports


classe:
-Graph: modélisation d'un graph orienté pondéré et à capacité symbolisant : les nœuds, les connexions, la capacité et saturation de chaque arc

-Méthodes:

 . remplissage du graph avec les fonctions remp_mat_adj() , remp_mat_dist() , remp_mat_cap et remp_mat_sat()
 . Mise à jour du jour (il faudrait une API qui nous donne les données en temps réel, à implémenter) avec la fonction update()
 . ajustement du flow en orientant vers un arc pas saturé et gérant la capacité avec la fonction adjust_flow()
 . visualisation du graph en nous montrons le chemin optimal via la méthode visualize_graph()

-Fonctions:
 
 . identifications des goulets d'étranglements avec la fonction identifier_goulet()
 . trouver tous les chemins possibles d'un nœud X et Y via la fonction find_all_path()
 . calcul du score de chaque chemin en fonction de la distance, du ratio saturation/ capacité, si la route contient des goulets avec la fonction score_path() 
 . classe les chemins selon leur score et nous donne le meilleur avec la fonction find_best_path()
 . fonction main() qui permet à l'utilisateur d'entrer les données 



"""

import numpy as np
import requests
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, N) -> None:
        self.N = N
        self.MGraph = np.zeros((N, N), dtype=int)      # matrice d'adjacence 
        self.MDist = np.full((N, N), float('inf'))     # matrice des distances entre les nœuds
        self.MCap = np.zeros((N, N))                   # matrice des capacités des arcs 
        self.MSat = np.zeros((N,N))                    # matrice de saturation des arcs

    def remp_mat_adj(self):
        print("\nRemplissage de la matrice d'adjacence (1 pour une connexion, 0 sinon):")
        for i in range(self.N):
            for j in range(self.N):
                while True:
                    try:
                        value = int(input(f"Connexion entre le node {i} et {j} (1/0): "))
                        if value in [0, 1]:
                            self.MGraph[i][j] = value
                            break
                        else:
                            print("Valeur invalide. Veuillez entrer 0 ou 1.")
                    except ValueError:
                        print("Entrée invalide. Veuillez entrer 0 ou 1.")

    def remp_mat_dist(self):
        print("\nRemplissage de la matrice des distances:")
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.MDist[i][j] = 0
                else:
                    if self.MGraph[i][j] == 1:
                        while True:
                            try:
                                value = float(input(f"Veuillez entrer la distance entre le node {i} et {j}: "))
                                if value > 0 :
                                    self.MDist[i][j] = value
                                    break
                                else:
                                    print("Veuillez entrer une valeur qui soit positive")
                            except ValueError:
                                print("Entrée invalide. Veuillez entrer une valeur qui soit positive")                       
                    else:
                        self.MDist[i][j] = float('inf')

    def remp_mat_cap(self):
        print("\nRemplissage de la matrice des capacités:")
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    while True:
                        try:
                            value = int(input(f"Capacité entre le nœud {i} et le nœud {j}: "))
                            if value > 0 :
                                self.MCap[i][j] = value
                                break
                            else:
                                print("veuillez entrer une valeur de capacité positive")
                        except ValueError:
                            print("Entrée invalide. Veuillez entrer une valeur qui soit positive")
                else:
                    self.MCap[i][j] = 0


    def remp_mat_sat(self):
        print("\nRemplissage de la matrice indiquant la saturation de chaque arc.:")
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    while True:
                        try:
                            value = int(input(f"saturation entre le nœud {i} et le nœud {j}: "))
                            if value >= 0 and value <= self.MCap[i][j]:
                                self.MSat[i][j] = value
                                break
                            else:
                                print("veuillez entrer une valeur de saturation positive ET qui soit inférieur ou égale à la capacité de l'arc associé")
                            
                        except ValueError:
                            print("Entrée invalide. veuillez entrer une valeur de saturation positive ET qui soit inférieur ou égale à la capacité de l'arc associé")
                else:
                    self.MSat[i][j] = 0
                

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
                capacity = Graph.MCap[i][j] + Graph.MCap[j][i]
                current_flow = Graph.MCap[j][i]
                if current_flow >= capacity * 0.8:
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
    sum_capacities = 0
    goulets_penality = 0  
    sum_ratio_sat = 0

    for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]

        sum_distance += Graph.MDist[u][v]

        for g in goulets:
            if g[0] == u and g[1] == v:
                goulets_penality += 1
                break
        
        sum_capacities += Graph.MCap[u][v]
           
        if Graph.MCap[u][v] != 0:
            sum_ratio_sat += (Graph.MSat[u][v] / Graph.MGraph[u][v])
        else:
            sum_ratio_sat += float('inf')
                

    distance_score = 1/(1+sum_distance)
    goulets_penality_score = 1/(1 + goulets_penality)
    ratio_sat_score = 1/(1+sum_ratio_sat)
    

    final_score = (distance_score * 0.4) + (ratio_sat_score * 0.4) + (goulets_penality_score * 0.1) + (sum_capacities * 0.1)
    return final_score

def find_best_path(Graph, start, end):
    if Graph is None or Graph.N == 0:
        return None
    
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

    N = None
    while N is None or N < 0 :
        try:
            N = int(input("Veuillez entrer le nombre de nœuds du associés au graph: "))
            if N < 0 :
                print("Le nombre de nœuds doit être positive ou nulle, veuillez insérer cette valeur: ")
        except ValueError:
            print("Le nombre de nœuds doit être positive ou nulle, veuillez insérer cette valeur:")
            N = None

    graph = Graph(N)
    
    graph.remp_mat_adj()
    graph.remp_mat_dist()
    graph.remp_mat_cap()
    graph.remp_mat_sat()

    source = None
    while source is None or not [0, N-1]:
        try:
            source = int(input("Veuillez entrer la source (0 à N-1): "))
            if not 0<= source <= N-1:
                print(f"Veuillez entrer le nœud associé au départ, tel qu'il soit entre 0 et {N -1}: ")
        except ValueError:
            print(f"Veuillez entrer le nœud associé au départ, tel qu'il soit entre 0 et {N -1}: ")
    

    sink = None
    while sink is None or not [0, N-1]:
        try:
            sink = int(input("Veuillez entrer le sink (0 à N-1): "))
            if not 0<= source <= N-1:
                print(f"Veuillez entrer le nœud associé à l'arrivée , tel qu'il soit entre 0 et {N -1}: ")
        except ValueError:
            print(f"Veuillez entrer le nœud associé à l'arrivée, tel qu'il soit entre 0 et {N -1}: ")


   
    best_path = find_best_path(graph, source, sink)

    if best_path:
        print("Le meilleur chemin trouvé est :", best_path)
       
        graph.visualize_graph(best_path)
    else:
        print("Aucun chemin trouvé entre la source et le sink.")


if __name__ == "__main__":
    main()
