# Programme d'Optimisation dans les réseaux à capacité



## Aperçu
Ce programme modélise un réseau routier sous forme de graphe orienté pondéré, en utilisant des matrices pour les connexions, les distances, les capacités, et la saturation des arcs. Il permet d'optimiser la gestion du trafic tout en prennant en compte une gestion d'imprévus.



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
- **`visualize_graph(best_path=None)`** : Visualise le graphe et met en évidence le meilleur chemin si fourni.

### Fonctions d'optimisation

- **`compute_cost_matrix(graph, w_dist=0.4, w_sat=0.6)`** : Cette fonction permet de fusionner toutes les matrices de la classe *Graph* afin de pouvoir utiliser les algorithmes qui prennent en compte qu'un seul paramètre.
- **`compute_heuristic(graph, goal)`** : Sert d’heuristique admissible pour A*, lance l'algorithme de *Dijkstra* à partir du sommet final sur la matrice de coûts et Renvoie un tableau dist[] où dist[i] est la distance minimale de i jusqu’à goal.
- **`find_best_path_astar(graph, start, goal, w_dist=0.4, w_sat=0.6)`** : Calcule l’heuristique via compute_heuristic et renvoie la liste de nœuds du chemin, ou "None" si aucun chemin n'est trouvé.


### Fonctions de gestion d'imprévus
- **`mise_à_jour(Graph)`** : Pour chaque arc, modifie aléatoirement la saturation de +10% ou -10%, chaque ajout est bornée entre 0 et la capacité.
- **`gestion_imprevu(Graph, bestPath, start, goal, k)`** : Appelle la fonction `mise_à_jour()` et si un arc du chemin optimal est saturé à plus de 70% :génère k chemins avec `find_best_path_astar()` et en renvoie un aléatoirement parmis les k. Sinon renvoie *best_path* ou "None".
- **`activation_imprévu(Graph, best_path)`** : Choisit aléatoirement un arc de bestPath (sauf le dernier) et force sa saturation à la capacité maximale.



#  Programme de test de performances "tests.py" 

## Aperçu
Ce programme permet de tester différents aspects du programme tels que l'activation de la gestion d'imprévus et permet de calculer le taux de performances du programme.
(il y aura dans cette partie uniquement les fonctions qui ne sont pas disponibles dans le fichier du code principal).
---

## Fonctions


- **`find_best_path_dijkstra(G, start, goal, w_dist=0.4, w_sat=0.6):`** : Cette fonctionpermet de trouver le chemn optimal en utilisant l'algorithme de *Dijkstra* et nous renvoie le chemin optimal ainsi que le coût calculé.
- **`test_temps_calcul()`** : Crée des graphes aléatoires de tailles (N = 10, 50, 100, …), hronomètre `find_best_path_astar` pour chaque N et Affiche l’évolution du temps en fonction de N.
- **`main()`** : propose à l'utilisateur de choisir le type de test qu'il veut effectuer. 






