# Programme d'Optimisation de Trafic Routier



## Aperçu
Ce programme modélise un réseau routier sous forme de graphe orienté pondéré, en utilisant des matrices pour les connexions, les distances, les capacités, et la saturation des arcs. Il permet d'optimiser la gestion du trafic en trouvant les meilleurs chemins.



## Classes et Fonctions

### `Graph`
Cette classe représente le graphe du réseau routier. Elle utilise des matrices pour stocker les informations liées aux connexions entre les nœuds, les distances, les capacités, et la saturation des arcs.

#### Fonctions :
- **`__init__(N)`** : Initialise un graphe de taille N avec des matrices de connexions (`MGraph`), de distances (`MDist`), de capacités (`MCap`), et de saturation (`MSat`).
- **`remp_mat_adj()`** : Remplit la matrice d'adjacence (`MGraph`) à partir des saisies utilisateur.
- **`remp_mat_dist()`** : Remplit la matrice des distances (`MDist`) pour les arcs existants.
- **`remp_mat_cap()`** : Remplit la matrice des capacités (`MCap`) des arcs.
- **`remp_mat_sat()`** : Remplit la matrice de saturation (`MSat`) des arcs.
- **`charger_données_fichier(fichier)`** : Charge les données des matrices depuis un fichier CSV.
- **`verifier_et_charger_fichier()`** : Vérifie l'existence d'un fichier et charge les données s'il existe.
- **`update()`** : Met à jour dynamiquement les capacités des arcs en récupérant les données d'une API.
- **`adjust_flow(node, flow)`** : Ajuste le flot dans le réseau en réduisant la capacité résiduelle des arcs.
- **`display_matrices()`** : Affiche les matrices `MGraph`, `MDist`, et `MCap` dans la console.
- **`visualize_graph(best_path=None)`** : Visualise le graphe et met en évidence le meilleur chemin si fourni.

### Fonctions Utilitaires
- **`identifier_goulets(Graph)`** : Identifie les goulets d'étranglement, c'est-à-dire les arcs avec un niveau de saturation élevé.
- **`dfs_all_paths(graph, source, sink, path, all_paths)`** : Explore tous les chemins possibles entre deux nœuds dans le graphe.
- **`find_all_paths(Graph, source, sink)`** : Trouve tous les chemins entre deux nœuds donnés.
- **`score_path(Graph, path, goulets)`** : Calcule un score pour un chemin en fonction des distances, des goulets et des ratios saturation/capacité.
- **`A_star(Graph, start, end)`** : Utilise l'algorithme A* pour trouver le meilleur chemin entre deux nœuds.
- **`heuristic(Graph, node, end)`** : Calcule le coût heuristique basé sur la distance et le ratio saturation/capacité.
- **`reconstruct_path(came_from, current)`** : Reconstruit le chemin trouvé par l'algorithme A*.
- **`find_best_path(Graph, start, end)`** : Trouve le meilleur chemin en utilisant soit l'algorithme A* pour les grands graphes, soit une recherche exhaustive pour les plus petits graphes.

#  Programme de test de performances "LP_tests.py" 

## Aperçu

Ce programme optimise un problème de gestion du trafic routier en utilisant la programmation linéaire pour minimiser un coût global, basé sur la distance et le ratio saturation/capacité des routes dans un réseau donné. Le réseau est défini par un fichier CSV d'arêtes entre nœuds, contenant des informations sur les distances, capacités, et saturations des arcs. Le programme trouve une solution qui minimise le coût total du trafic tout en respectant les contraintes de flux entre les nœuds.

---

## Fonctions


- **`read_graph_from_csv(file_name)`** : Cette fonction lit un fichier CSV contenant les arêtes du graphe et extrait les informations de distance, capacité, et saturation pour chaque route. Elle permet de construire la liste des arcs du réseau à partir d'un fichier CSV.

### 
- **`solve_linear_program(edges)`** : Résout un problème de programmation linéaire pour minimiser un coût lié à la distance et à la saturation des routes dans un graphe orienté.Cette fonction trouve la solution optimale en minimisant le coût de trafic tout en respectant les contraintes de flux (entrée et sortie de chaque nœud).


---

## Utilité des fonctions

- **`read_graph_from_csv`** : Importer les données de trafic depuis un fichier CSV.
- **`solve_linear_program`** : Optimiser les flux de trafic en minimisant les coûts liés à la distance et à la saturation des routes.


