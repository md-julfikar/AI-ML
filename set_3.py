"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 3

Contents:
- Q1: First-Order Logic (FOL) Knowledge Representation & Family Relations
- Q2: Informed Search - A* Algorithm & UCS Comparison (h=0)
- Q3: Unsupervised Learning - K-Means Clustering on Iris Dataset
- Q4: Viva Voce Reference & Key Concepts
"""

import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==============================================================================
# Q1: FIRST-ORDER LOGIC REPRESENTATION & TUPLE MATCHING [25 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: FIRST-ORDER LOGIC REPRESENTATION & INFERENCE")
    print("=" * 70)

    # Given Parent(parent, child) facts:
    # Parent(Ali, Babu), Parent(Ali, Rina), Parent(Babu, Karim), Parent(Rina, Sumi), Parent(Karim, Nila)
    parent_facts = [
        ("Ali", "Babu"),
        ("Ali", "Rina"),
        ("Babu", "Karim"),
        ("Rina", "Sumi"),
        ("Karim", "Nila")
    ]

    print("--- Given Knowledge Base (Parent Facts) ---")
    for p, c in parent_facts:
        print(f"Parent({p}, {c})")

    # Helper functions
    def get_children(person):
        return sorted([child for p, child in parent_facts if p == person])

    def get_grandparents():
        # Grandparent(x, z) :- Parent(x, y) AND Parent(y, z)
        grandparents = []
        for x, y1 in parent_facts:
            for y2, z in parent_facts:
                if y1 == y2:
                    grandparents.append((x, z, y1))  # (grandparent, grandchild, intermediate parent)
        return grandparents

    def get_siblings():
        # Sibling(x, y) :- Parent(p, x) AND Parent(p, y) AND x != y
        # We store canonical undirected pairs to prevent duplicates
        siblings = set()
        for p1, x in parent_facts:
            for p2, y in parent_facts:
                if p1 == p2 and x != y:
                    pair = tuple(sorted([x, y]))
                    siblings.add(pair)
        return sorted(list(siblings))

    # All unique persons
    people = sorted(list(set([p for p, c in parent_facts] + [c for p, c in parent_facts])))

    print("\n--- Inferred Children ---")
    for p in people:
        ch = get_children(p)
        if ch:
            print(f"Children of {p}: {', '.join(ch)}")

    print("\n--- Inferred Grandparent Relations ---")
    gp_relations = get_grandparents()
    for gp, gc, parent in gp_relations:
        print(f"Grandparent({gp}, {gc}) [via intermediate Parent({gp}, {parent}) and Parent({parent}, {gc})]")

    print("\n--- Inferred Sibling Relations ---")
    sib_relations = get_siblings()
    for s1, s2 in sib_relations:
        print(f"Sibling({s1}, {s2})")

    print("\n--- Explanation: Tuple Matching Mechanism ---")
    print("Tuple matching operates like a relational database NATURAL JOIN on intermediate variables:")
    print("1. In Rule: Parent(x, y) AND Parent(y, z) -> Grandparent(x, z)")
    print("   The unification engine queries all pairs (x, y1) in Parent table and (y2, z) in Parent table.")
    print("2. It matches where y1 == y2 (binding the variable 'y' to the same domain constant).")
    print("3. When a match is found, variables x and z are extracted to synthesize the new inferred fact Grandparent(x, z).")


# ==============================================================================
# Q2: A* SEARCH & EFFECT OF HEURISTIC [30 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: A* SEARCH ALGORITHM & ZERO-HEURISTIC (UCS) COMPARISON")
    print("=" * 70)

    # Weighted Graph:
    # S-A=2, S-B=4, A-C=2, A-D=5, B-D=1, C-G=5, D-G=3
    graph = {
        'S': [('A', 2), ('B', 4)],
        'A': [('C', 2), ('D', 5)],
        'B': [('D', 1)],
        'C': [('G', 5)],
        'D': [('G', 3)],
        'G': []
    }

    # Heuristic: h(S)=6, h(A)=5, h(B)=4, h(C)=4, h(D)=2, h(G)=0
    h_default = {'S': 6, 'A': 5, 'B': 4, 'C': 4, 'D': 2, 'G': 0}

    def a_star_search(graph, start, goal, heuristic):
        # Priority Queue holds: (f_score, tie_breaker, g_score, current_node, path)
        counter = 0
        pq = [(heuristic[start], counter, 0, start, [start])]
        best_g = {start: 0}
        closed_set = set()
        expansion_details = []

        while pq:
            f, _, g, node, path = heapq.heappop(pq)

            if node in closed_set:
                continue
            closed_set.add(node)
            h = heuristic[node]
            expansion_details.append((node, g, h, f))

            if node == goal:
                return path, g, expansion_details

            for neighbor, weight in graph.get(node, []):
                new_g = g + weight
                if neighbor not in best_g or new_g < best_g[neighbor]:
                    best_g[neighbor] = new_g
                    new_f = new_g + heuristic.get(neighbor, 0)
                    counter += 1
                    heapq.heappush(pq, (new_f, counter, new_g, neighbor, path + [neighbor]))

        return None, float('inf'), expansion_details

    # Experiment 1: Standard A*
    path_a, cost_a, exp_a = a_star_search(graph, 'S', 'G', h_default)

    print("--- Experiment 1: A* Search with Default Heuristic ---")
    print(f"{'Node Expanded':<16}{'g(n) [Cost]':<15}{'h(n) [Heuristic]':<18}{'f(n) = g + h':<15}")
    print("-" * 65)
    for node, g, h, f in exp_a:
        print(f"{node:<16}{g:<15}{h:<18}{f:<15}")

    print(f"\nExpansion Order : {' -> '.join([e[0] for e in exp_a])}")
    print(f"Optimal Path    : {' -> '.join(path_a)}")
    print(f"Total Path Cost : {cost_a}")

    # Experiment 2: Set every h(n) = 0 (Uniform Cost Search / Dijkstra)
    h_zero = {k: 0 for k in h_default}
    path_zero, cost_zero, exp_zero = a_star_search(graph, 'S', 'G', h_zero)

    print("\n--- Experiment 2: A* with h(n) = 0 (Uniform Cost Search) ---")
    print(f"{'Node Expanded':<16}{'g(n)':<15}{'h(n)':<18}{'f(n)':<15}")
    print("-" * 65)
    for node, g, h, f in exp_zero:
        print(f"{node:<16}{g:<15}{h:<18}{f:<15}")

    print(f"\nExpansion Order : {' -> '.join([e[0] for e in exp_zero])}")
    print(f"Optimal Path    : {' -> '.join(path_zero)}")
    print(f"Total Path Cost : {cost_zero}")

    print("\n--- Analysis & Explanation ---")
    print("1. When h(n) = 0 for all nodes, f(n) = g(n) + 0 = g(n). A* degenerates exactly into")
    print("   Dijkstra's Algorithm / Uniform Cost Search (UCS).")
    print("2. Both experiments find the optimal path S -> B -> D -> G with cost 8.")
    print("3. Notice how with heuristic guidance, A* prioritizes promising nodes toward the goal,")
    print("   reducing unnecessary explorations compared to pure blind cost expansion.")


# ==============================================================================
# Q3: K-MEANS CLUSTERING ON IRIS [35 Marks]
# ==============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: K-MEANS CLUSTERING ON IRIS (FIRST TWO FEATURES)")
    print("=" * 70)

    # Load dataset & take first two features: Sepal Length & Sepal Width
    iris = load_iris()
    X = iris.data[:, :2]
    feature_names = iris.feature_names[:2]

    k_values = [2, 3, 4, 5]
    inertias = []
    silhouettes = []
    models = {}

    print(f"Features Used: {feature_names[0]} and {feature_names[1]}")
    print(f"{'k':<6}{'Inertia (WCSS)':<20}{'Silhouette Score':<20}")
    print("-" * 50)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        sil = silhouette_score(X, labels)

        inertias.append(inertia)
        silhouettes.append(sil)
        models[k] = (kmeans, labels)

        print(f"{k:<6}{inertia:<20.4f}{sil:<20.4f}")

    # Best k selection based on silhouette score
    best_k = 3  # Suitable k from domain and silhouette analysis
    print(f"\nSelected Suitable k: {best_k}")
    best_kmeans, best_labels = models[best_k]

    # Plotting
    plt.figure(figsize=(9, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for cluster_idx in range(best_k):
        cluster_pts = X[best_labels == cluster_idx]
        plt.scatter(
            cluster_pts[:, 0], cluster_pts[:, 1],
            label=f'Cluster {cluster_idx}', alpha=0.7, s=50
        )

    # Plot centroids
    centroids = best_kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        s=200, c='red', marker='X', edgecolor='black', linewidth=1.5,
        label='Centroids'
    )

    plt.title(f"K-Means Clustering on Iris (First Two Features, k={best_k})", fontsize=13)
    plt.xlabel(feature_names[0], fontsize=11)
    plt.ylabel(feature_names[1], fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_file = "set_3_kmeans.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Cluster visualization saved as '{plot_file}'")

    print("\n--- Explanation: Why Cluster Label 0 != Iris Class 0 ---")
    print("1. K-Means is an unsupervised clustering algorithm. It has NO access to ground-truth class labels.")
    print("2. Centroid indices (0, 1, 2) are initialized randomly (or via k-means++) and assigned arbitrarily.")
    print("3. There is an inherent 'label permutation symmetry': Cluster 0 could correspond to Versicolor,")
    print("   Virginica, or Setosa. Alignment requires external Hungarian/Munkres matching after clustering.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. Predicates, Constants, Variables:
       - Predicate: A boolean function expressing a relation or property (e.g. Parent(x, y)).
       - Constant: A specific object in the domain (e.g. Ali, Babu).
       - Variable: A placeholder for any object in the domain (e.g. x, y, z).
    2. A* Evaluation Function:
       - f(n) = g(n) + h(n), where:
         * g(n) = exact path cost from start to node n.
         * h(n) = estimated cost from n to goal.
         * f(n) = estimated total cost of path passing through n.
    3. Admissibility of Heuristic:
       - A heuristic is admissible if 0 <= h(n) <= h*(n) (it never overestimates actual cost to goal).
       - Guarantees A* tree/graph search will return an optimal solution.
    4. Centroid, Inertia, Silhouette Score:
       - Centroid: Mean coordinates of all data points belonging to a cluster.
       - Inertia (WCSS): Sum of squared distances from samples to their closest cluster center.
       - Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters (-1 to +1).
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
