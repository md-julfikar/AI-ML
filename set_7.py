"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 7

Contents:
- Q1: First-Order Logic (FOL) Family Tree Reasoning
- Q2: Greedy Best-First Search & Heuristic Sensitivity / Misdirection
- Q3: K-Means Clustering on All 4 Iris Features with Scaling Comparison
- Q4: Viva Voce Reference & Key Concepts
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ==============================================================================
# Q1: FIRST-ORDER LOGIC & FAMILY RELATION INFERENCE [25 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: FOL FAMILY KNOWLEDGE BASE & RELATIONAL QUERIES")
    print("=" * 70)

    # Given Facts:
    # Parent(Rahman, Hasan), Parent(Rahman, Nabila), Parent(Hasan, Rafi),
    # Parent(Nabila, Tania), Parent(Rafi, Samia)
    parent_db = [
        ("Rahman", "Hasan"),
        ("Rahman", "Nabila"),
        ("Hasan", "Rafi"),
        ("Nabila", "Tania"),
        ("Rafi", "Samia")
    ]

    print("--- Stored Parent Facts ---")
    for p, c in parent_db:
        print(f"Parent({p}, {c})")

    def children_of(person):
        return sorted([c for p, c in parent_db if p == person])

    def grandchildren_of(person):
        # Grandchild(z, x) :- Parent(x, y) AND Parent(y, z)
        gc = set()
        for p1, c1 in parent_db:
            if p1 == person:
                for p2, c2 in parent_db:
                    if p2 == c1:
                        gc.add(c2)
        return sorted(list(gc))

    def grandparent_relations():
        gp_list = []
        for p1, c1 in parent_db:
            for p2, c2 in parent_db:
                if c1 == p2:
                    gp_list.append((p1, c2, c1))
        return gp_list

    def siblings():
        # Sibling(x, y) :- Parent(p, x) AND Parent(p, y) AND x != y
        # Avoid self-sibling and duplicate reflections
        sibs = set()
        for p1, x in parent_db:
            for p2, y in parent_db:
                if p1 == p2 and x != y:
                    sibs.add(tuple(sorted([x, y])))
        return sorted(list(sibs))

    all_people = sorted(list(set([p for p, _ in parent_db] + [c for _, c in parent_db])))

    print("\n--- Children of Each Person ---")
    for person in all_people:
        ch = children_of(person)
        if ch:
            print(f"children_of({person}) = {ch}")

    print("\n--- Grandchildren of Each Person ---")
    for person in all_people:
        gc = grandchildren_of(person)
        if gc:
            print(f"grandchildren_of({person}) = {gc}")

    print("\n--- All Inferred Grandparent Relations ---")
    for gp, gc, mid in grandparent_relations():
        print(f"Grandparent({gp}, {gc}) [via Parent({gp}, {mid}) and Parent({mid}, {gc})]")

    print("\n--- All Inferred Sibling Relations (No Duplicates, No Self-Siblings) ---")
    for s1, s2 in siblings():
        print(f"Siblings({s1}, {s2})")


# ==============================================================================
# Q2: GREEDY BEST-FIRST SEARCH & HEURISTIC SENSITIVITY [30 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: GREEDY BEST-FIRST SEARCH & HEURISTIC MODIFICATION")
    print("=" * 70)

    # Graph: S->{A,B,C}, A->{D}, B->{E}, C->{F}, D->{G}, E->{G}, F->{G}
    graph = {
        'S': ['A', 'B', 'C'],
        'A': ['D'],
        'B': ['E'],
        'C': ['F'],
        'D': ['G'],
        'E': ['G'],
        'F': ['G'],
        'G': []
    }

    # Original Heuristics: S=7, A=4, B=5, C=3, D=2, E=2, F=1, G=0
    h_orig = {'S': 7, 'A': 4, 'B': 5, 'C': 3, 'D': 2, 'E': 2, 'F': 1, 'G': 0}

    def greedy_bfs(g, start, goal, h):
        counter = 0
        pq = [(h[start], counter, [start])]
        visited = set()
        expansion_order = []

        while pq:
            _, _, path = heapq.heappop(pq)
            node = path[-1]

            if node in visited:
                continue
            visited.add(node)
            expansion_order.append(node)

            if node == goal:
                return path, expansion_order

            for nxt in g.get(node, []):
                if nxt not in visited:
                    counter += 1
                    heapq.heappush(pq, (h[nxt], counter, path + [nxt]))
        return None, expansion_order

    # Run 1: Original heuristics
    path1, exp1 = greedy_bfs(graph, 'S', 'G', h_orig)

    # Modified Heuristics: Make B artificially appear the best (e.g. h(B) = 1, h(C) = 6)
    # This lures Greedy BFS into exploring branch B -> E -> G instead of branch C -> F -> G
    h_mod = {'S': 7, 'A': 4, 'B': 1, 'C': 6, 'D': 2, 'E': 2, 'F': 1, 'G': 0}
    path2, exp2 = greedy_bfs(graph, 'S', 'G', h_mod)

    print("--- Run 1: Standard Heuristics ---")
    print(f"Heuristics : {h_orig}")
    print(f"Path Found : {' -> '.join(path1)}")
    print(f"Expansion  : {' -> '.join(exp1)}")

    print("\n--- Run 2: Modified Heuristics (Misdirection via h(B)=1, h(C)=6) ---")
    print(f"Heuristics : {h_mod}")
    print(f"Path Found : {' -> '.join(path2)}")
    print(f"Expansion  : {' -> '.join(exp2)}")

    print("\n--- Explanation: How Heuristic Changes Alter Exploration ---")
    print("1. Greedy Best-First relies exclusively on h(n) to select the next node from the priority queue.")
    print("2. In Run 1, at start node S, C has the lowest heuristic (h=3 vs A=4, B=5), so it explores S -> C -> F -> G.")
    print("3. In Run 2, lowering h(B) to 1 causes the algorithm to immediately explore node B instead.")
    print("   This illustrates the fundamental flaw of Greedy Search: misleading or poorly calibrated heuristics")
    print("   divert the search down less efficient or deceptive paths without any cost backtracking.")


# ==============================================================================
# Q3: K-MEANS & FEATURE SCALING COMPARISON [35 Marks]
# ==============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: K-MEANS CLUSTERING ON ALL 4 IRIS FEATURES & SCALING COMPARISON")
    print("=" * 70)

    iris = load_iris()
    X_raw = iris.data
    feature_names = iris.feature_names

    # 1. Without Scaling
    km_raw = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_raw = km_raw.fit_predict(X_raw)
    inertia_raw = km_raw.inertia_
    sil_raw = silhouette_score(X_raw, labels_raw)

    print("--- Configuration 1: Unscaled Features ---")
    print(f"Cluster Centroids (4D):\n{km_raw.cluster_centers_}")
    print(f"Inertia (WCSS)   : {inertia_raw:.4f}")
    print(f"Silhouette Score : {sil_raw:.4f}")

    # 2. With StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    km_scaled = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_scaled = km_scaled.fit_predict(X_scaled)
    inertia_scaled = km_scaled.inertia_
    sil_scaled = silhouette_score(X_scaled, labels_scaled)

    print("\n--- Configuration 2: Standardized Features (StandardScaler) ---")
    print(f"Cluster Centroids (Normalized Space):\n{km_scaled.cluster_centers_}")
    print(f"Inertia (WCSS)   : {inertia_scaled:.4f}")
    print(f"Silhouette Score : {sil_scaled:.4f}")

    # Plot any two features (e.g. Sepal Length vs Petal Length)
    plt.figure(figsize=(12, 5))

    # Subplot 1: Raw
    plt.subplot(1, 2, 1)
    plt.scatter(X_raw[:, 0], X_raw[:, 2], c=labels_raw, cmap='viridis', s=40, alpha=0.8)
    plt.scatter(km_raw.cluster_centers_[:, 0], km_raw.cluster_centers_[:, 2], c='red', marker='X', s=150, label='Centroids')
    plt.title(f"Unscaled K-Means (k=3, Sil={sil_raw:.3f})")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[2])
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Scaled
    plt.subplot(1, 2, 2)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=labels_scaled, cmap='plasma', s=40, alpha=0.8)
    plt.scatter(km_scaled.cluster_centers_[:, 0], km_scaled.cluster_centers_[:, 2], c='red', marker='X', s=150, label='Centroids')
    plt.title(f"Standardized K-Means (k=3, Sil={sil_scaled:.3f})")
    plt.xlabel(f"{feature_names[0]} (z-score)")
    plt.ylabel(f"{feature_names[2]} (z-score)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_file = "set_7_kmeans_scaling.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nVisualization saved as '{plot_file}'")

    print("\n--- Why Centroids and Inertia Cannot Be Compared Naively Across Scaled Spaces ---")
    print("1. Unit of Measurement: Inertia is the sum of squared Euclidean distances in feature units.")
    print(f"   Unscaled inertia ({inertia_raw:.2f}) is in cm^2, whereas standardized inertia ({inertia_scaled:.2f})")
    print("   is in dimensionless standardized z-score variance units.")
    print("2. Centroid Coordinates: Unscaled centroids represent actual feature means in original physical units (cm).")
    print("   Standardized centroids represent distance from zero in units of standard deviations.")
    print("3. Silhouette score, being normalized between -1 and +1, provides a much more meaningful comparative metric.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. First-Order Logic (FOL) vs Propositional Logic:
       - Propositional logic deals with atomic facts (True/False).
       - FOL introduces objects, relations (predicates), functions, and quantifiers (Forall, Exists).
    2. Greedy Best-First Search:
       - Uses heuristic evaluation f(n) = h(n).
       - Incomplete in infinite spaces, not optimal, prone to misleading local minima.
    3. Unsupervised Learning:
       - Machine learning where the model discovers latent structure in unlabelled data (e.g. clustering).
    4. Feature Scaling in Clustering:
       - Without scaling, features with large variance dominate Euclidean distance calculations.
       - Standardization ensures equal geometric weighting across dimensions.
    5. Silhouette Score:
       - s = (b - a) / max(a, b), where a is mean intra-cluster distance and b is mean nearest-cluster distance.
       - Ranges from -1 to +1; values close to 1 indicate well-separated, compact clusters.
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
