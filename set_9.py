"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 9

Contents:
- Q1: BFS vs DFS Adjacency Sensitivity & Shallowest Path Verification
- Q2: A* Search, Admissible Heuristics & Inadmissibility Pitfalls
- Q3: K-Means Model Selection (Elbow Method & Silhouette Analysis on Iris)
- Q4: Viva Voce Reference & Key Concepts
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import collections
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==============================================================================
# Q1: BFS AND DFS COMPARISON & ADJACENCY REORDERING [30 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: BFS AND DFS COMPARISON (ORIGINAL VS REORDERED ADJACENCY)")
    print("=" * 70)

    # Graph: A->{B,C}, B->{D,E}, C->{F}, D->{H}, E->{G}, F->{G}, G:{}, H->{G}
    graph_orig = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': ['H'],
        'E': ['G'],
        'F': ['G'],
        'G': [],
        'H': ['G']
    }

    # Reordered Graph (reversed neighbor order)
    graph_rev = {
        'A': ['C', 'B'],
        'B': ['E', 'D'],
        'C': ['F'],
        'D': ['H'],
        'E': ['G'],
        'F': ['G'],
        'G': [],
        'H': ['G']
    }

    def bfs(g, start, goal):
        q = collections.deque([[start]])
        visited = set()
        exp = []
        while q:
            p = q.popleft()
            node = p[-1]
            if node in visited:
                continue
            visited.add(node)
            exp.append(node)
            if node == goal:
                return p, exp
            for nxt in g.get(node, []):
                if nxt not in visited:
                    q.append(p + [nxt])
        return None, exp

    def dfs(g, start, goal):
        stk = [[start]]
        visited = set()
        exp = []
        while stk:
            p = stk.pop()
            node = p[-1]
            if node in visited:
                continue
            visited.add(node)
            exp.append(node)
            if node == goal:
                return p, exp
            for nxt in reversed(g.get(node, [])):
                if nxt not in visited:
                    stk.append(p + [nxt])
        return None, exp

    # Original Runs
    bfs_p1, bfs_e1 = bfs(graph_orig, 'A', 'G')
    dfs_p1, dfs_e1 = dfs(graph_orig, 'A', 'G')

    # Reordered Runs
    bfs_p2, bfs_e2 = bfs(graph_rev, 'A', 'G')
    dfs_p2, dfs_e2 = dfs(graph_rev, 'A', 'G')

    print("--- 1. Original Adjacency Order ---")
    print(f"BFS Path : {' -> '.join(bfs_p1)} | Trace: {' -> '.join(bfs_e1)}")
    print(f"DFS Path : {' -> '.join(dfs_p1)} | Trace: {' -> '.join(dfs_e1)}")

    print("\n--- 2. Reversed Adjacency Order ---")
    print(f"BFS Path : {' -> '.join(bfs_p2)} | Trace: {' -> '.join(bfs_e2)}")
    print(f"DFS Path : {' -> '.join(dfs_p2)} | Trace: {' -> '.join(dfs_e2)}")

    print("\n--- Analysis of Results ---")
    print("1. Which is affected more visibly?")
    print("   DFS is affected MUCH MORE visibly! In original order, DFS explored through D and H (depth 4: A->B->D->H->G),")
    print("   whereas under reversed order, DFS immediately picked C -> F -> G (depth 3), completely changing path & length.")
    print("2. Does BFS return a shallowest path?")
    print("   YES! In unweighted graphs, BFS explores systematically level-by-level (depth 0, 1, 2, 3...) using a FIFO queue.")
    print("   The first time the goal is dequeued, it is guaranteed to have the minimal number of edges (shortest/shallowest path).")


# ==============================================================================
# Q2: A* SEARCH AND HEURISTIC ADMISSIBILITY [30 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: A* SEARCH & HEURISTIC ADMISSIBILITY")
    print("=" * 70)

    # Edges: S-A=1, S-B=5, A-C=2, A-D=4, B-D=1, C-G=5, D-G=2
    graph = {
        'S': [('A', 1), ('B', 5)],
        'A': [('C', 2), ('D', 4)],
        'B': [('D', 1)],
        'C': [('G', 5)],
        'D': [('G', 2)],
        'G': []
    }

    # True optimal remaining costs to G:
    # h*(G) = 0
    # h*(D) = 2
    # h*(C) = 5
    # h*(B) = 1 + 2 = 3
    # h*(A) = min(2+5, 4+2) = min(7, 6) = 6
    # h*(S) = min(1+6, 5+3) = min(7, 8) = 7 (Optimal path: S -> A -> D -> G = 7)

    # 1. Admissible heuristic: h(n) <= h*(n)
    h_admissible = {'S': 7, 'A': 6, 'B': 3, 'C': 5, 'D': 2, 'G': 0}

    # 2. Inadmissible heuristic: severely overestimate h(A) = 15 (> 6)
    h_inadmissible = {'S': 7, 'A': 15, 'B': 3, 'C': 5, 'D': 2, 'G': 0}

    def a_star(g, start, goal, heuristic):
        counter = 0
        pq = [(heuristic[start], counter, 0, start, [start])]
        best_g = {start: 0}
        closed_set = set()
        trace = []

        while pq:
            f, _, g_val, node, path = heapq.heappop(pq)
            if node in closed_set:
                continue
            closed_set.add(node)
            trace.append((node, g_val, heuristic[node], f))

            if node == goal:
                return path, g_val, trace

            for nxt, weight in g.get(node, []):
                new_g = g_val + weight
                if nxt not in best_g or new_g < best_g[nxt]:
                    best_g[nxt] = new_g
                    new_f = new_g + heuristic[nxt]
                    counter += 1
                    heapq.heappush(pq, (new_f, counter, new_g, nxt, path + [nxt]))
        return None, float('inf'), trace

    p_adm, cost_adm, tr_adm = a_star(graph, 'S', 'G', h_admissible)
    p_inadm, cost_inadm, tr_inadm = a_star(graph, 'S', 'G', h_inadmissible)

    print("--- 1. Admissible Heuristic Experiment (h <= h*) ---")
    print(f"Heuristics : {h_admissible}")
    print(f"{'Node':<8}{'g(n)':<10}{'h(n)':<10}{'f(n)':<10}")
    print("-" * 38)
    for n, gv, hv, fv in tr_adm:
        print(f"{n:<8}{gv:<10}{hv:<10}{fv:<10}")
    print(f"Optimal Path Found : {' -> '.join(p_adm)} (Cost = {cost_adm})")

    print("\n--- 2. Inadmissible Heuristic Experiment (Overestimation: h(A) = 15) ---")
    print(f"Heuristics : {h_inadmissible}")
    print(f"{'Node':<8}{'g(n)':<10}{'h(n)':<10}{'f(n)':<10}")
    print("-" * 38)
    for n, gv, hv, fv in tr_inadm:
        print(f"{n:<8}{gv:<10}{hv:<10}{fv:<10}")
    print(f"Suboptimal Path Found : {' -> '.join(p_inadm)} (Cost = {cost_inadm})")

    print("\n--- Why Admissibility Matters ---")
    print("1. Admissibility guarantees that A* is mathematically guaranteed to find the lowest-cost optimal path.")
    print("2. When h(A) overestimates (15 instead of 6), A* computes f(A) = 1 + 15 = 16. It mistakenly presumes path A")
    print("   is prohibitively expensive, prematurely abandoning it and settling on S -> B -> D -> G (cost = 8).")
    print(f"3. Thus, violating admissibility causes A* to miss the true optimal path (cost = {cost_adm}).")


# ==============================================================================
# Q3: K-MEANS MODEL SELECTION (ELBOW & SILHOUETTE) [30 Marks]
# ==============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: K-MEANS MODEL SELECTION (FIRST TWO IRIS FEATURES)")
    print("=" * 70)

    iris = load_iris()
    X = iris.data[:, :2]
    feature_names = iris.feature_names[:2]

    k_range = [2, 3, 4, 5]
    inertias = []
    silhouettes = []
    models = {}

    print(f"{'k':<6}{'Inertia (WCSS)':<20}{'Silhouette Score':<20}")
    print("-" * 46)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        ine = km.inertia_
        sil = silhouette_score(X, labels)

        inertias.append(ine)
        silhouettes.append(sil)
        models[k] = (km, labels)
        print(f"{k:<6}{ine:<20.4f}{sil:<20.4f}")

    # Plotting Elbow and Silhouette Graphs
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Elbow graph
    axes[0].plot(k_range, inertias, marker='o', color='b', linewidth=2)
    axes[0].set_title("Elbow Method (Inertia vs k)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].grid(True, alpha=0.3)

    # 2. Silhouette graph
    axes[1].plot(k_range, silhouettes, marker='s', color='green', linewidth=2)
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)

    # 3. Final cluster visualization for selected k=3
    chosen_k = 3
    km_chosen, labels_chosen = models[chosen_k]
    scatter = axes[2].scatter(X[:, 0], X[:, 1], c=labels_chosen, cmap='tab10', s=45, alpha=0.8)
    axes[2].scatter(km_chosen.cluster_centers_[:, 0], km_chosen.cluster_centers_[:, 1],
                    color='red', marker='X', s=160, edgecolor='black', label='Centroids')
    axes[2].set_title(f"Final Clusters (k={chosen_k})")
    axes[2].set_xlabel(feature_names[0])
    axes[2].set_ylabel(feature_names[1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "set_9_kmeans_selection.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nModel selection graphs saved to '{plot_path}'")

    print("\n--- Why Inertia Decreases as k Increases ---")
    print("1. Inertia measures the sum of squared Euclidean distances from each point to its assigned centroid.")
    print("2. Adding more centroids increases model degrees of freedom, allowing centroids to sit closer to data points.")
    print("3. In the extreme case where k = N (number of samples), each point is its own centroid, driving inertia to exactly 0.")
    print("   Hence, inertia monotonically decreases, necessitating methods like the Elbow elbow or Silhouette score to prevent overfitting.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. BFS vs DFS Path Properties:
       - BFS always finds the shallowest path (minimum edges) in unweighted graphs.
       - DFS plunges down whichever branch is ordered first; path length depends on tie-breaking/adjacency order.
    2. Admissible Heuristic:
       - A heuristic h(n) that never overestimates the true cost to reach the goal: 0 <= h(n) <= h*(n).
       - Essential for A* tree and graph search optimality.
    3. Consistency (Monotonicity):
       - h(n) <= c(n, a, n') + h(n'). A stronger condition than admissibility, ensuring the first time
         a node is expanded, its path is already optimal.
    4. Inertia vs Silhouette Score:
       - Inertia only measures compactness within clusters (always decreases with k).
       - Silhouette score balances both cluster cohesion (intra) and cluster separation (inter).
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
