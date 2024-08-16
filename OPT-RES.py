import numpy as np
import requests
from collections import deque

class Graph:
    def __init__(self, N) -> None:
        self.N = N
        self.MGraph = np.zeros((N, N), dtype=int)
        self.MDist = np.full((N, N), float('inf'))
        self.MCap = np.zeros((N, N))

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
                        self.MDist[i][j] = float(input(f"Veuillez entrer la distance entre le node {i} et {j}: "))
                    else:
                        self.MDist[i][j] = float('inf')

    def remp_mat_cap(self):
        print("\nRemplissage de la matrice des capacités:")
        for i in range(self.N):
            for j in range(self.N):
                if self.MGraph[i][j] == 1:
                    self.MCap[i][j] = float(input(f"Capacité entre le nœud {i} et le nœud {j}: "))
                else:
                    self.MCap[i][j] = 0


    def update(self):    #il faut une API elle est optionnelle pour l'instant
        url = "exemple d'url récup avec une API "
        try:
            response = requests.get(url)
            data=response.json()

            for i in rang(self.N):
                for j in rang(self.N):
                    if MGraph[i][j] == 1 :
                        MCap[i][j] += 1 
        except Exception as e:
            print(f"Erreur lors de la mise à jour des capacités : {e}")



    def bfs(self,source, sink, parent):
        visited =[False] * self.N
        queue = deque([source])
        visited[source] = True

        while queue:
            u = queue.popleft()

            for v in range(self.N):
                if visited[v] == False and MCap[u][v] > 0 :
                    queue.append(v)
                    visited[v]=True
                    parent[v] = u

                    if v == sink:
                        return True
        return False 



    def edmonkarp(self, source, sink):
        pass 





    def display_matrices(self):
        print("\nMatrice d'adjacence MGraph:")
        print(self.MGraph)
        print("\nMatrice des distances MDist:")
        print(self.MDist)
        print("\nMatrice des capacités MCap:")
        print(self.MCap)

