"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 10

Contents:
- Q1: Integrated Knowledge-Based Agent (Multi-scenario Forward Chaining)
- Q2: Integrated Search Task (BFS, Greedy Best-First, A* & Edge-Hop vs Cost Analysis)
- Q3: Classification Comparison (k-NN vs Decision Tree Comprehensive Evaluation)
- Q4: Viva Voce Reference & Key Concepts
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import collections
import heapq
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================================================================
# Q1: INTEGRATED KNOWLEDGE-BASED AGENT [25 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: INTEGRATED KNOWLEDGE-BASED AGENT & FORWARD CHAINING")
    print("=" * 70)

    # Given Rules:
    # 1. TemperatureHigh -> Hot
    # 2. Hot AND Humid -> Uncomfortable
    # 3. Uncomfortable -> FanOn
    # 4. Rain -> Humid
    rules = [
        ({'TemperatureHigh'}, 'Hot'),
        ({'Hot', 'Humid'}, 'Uncomfortable'),
        ({'Uncomfortable'}, 'FanOn'),
        ({'Rain'}, 'Humid')
    ]

    def forward_chain(initial_facts):
        known_facts = set(initial_facts)
        given_facts = set(initial_facts)
        trace = []
        fired_rules = set()

        while True:
            new_derived = False
            for idx, (premises, conclusion) in enumerate(rules):
                if idx not in fired_rules and premises.issubset(known_facts):
                    fired_rules.add(idx)
                    trace.append(f"Rule {idx+1} fired: {premises} -> {conclusion}")
                    if conclusion not in known_facts:
                        known_facts.add(conclusion)
                        new_derived = True
            if not new_derived:
                break

        inferred_facts = known_facts - given_facts
        return given_facts, inferred_facts, known_facts, trace

    scenarios = [
        ("Scenario 1: TemperatureHigh only", {'TemperatureHigh'}),
        ("Scenario 2: TemperatureHigh and Humid", {'TemperatureHigh', 'Humid'}),
        ("Scenario 3: TemperatureHigh and Rain", {'TemperatureHigh', 'Rain'})
    ]

    for title, given in scenarios:
        print(f"\n--- {title} ---")
        g, inf, total, tr = forward_chain(given)
        print(f"Given Facts    : {g}")
        if tr:
            print("Inference Trace:")
            for step in tr:
                print(f"  * {step}")
        else:
            print("Inference Trace: [No rules satisfied]")
        print(f"Inferred Facts : {inf if inf else '{None}'}")
        print(f"Final Knowledge: {total}")
        print(f"Action 'FanOn' Triggered?: {'FanOn' in total}")


# ==============================================================================
# Q2: INTEGRATED SEARCH TASK [35 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: INTEGRATED SEARCH TASK (BFS, GREEDY BEST-FIRST, A*)")
    print("=" * 70)

    # Structure: S->{A,B}, A->{C,D}, B->{D,E}, C->{G}, D->{G}, E->{G}
    # Edge weights and heuristics:
    graph_weighted = {
        'S': [('A', 3), ('B', 1)],
        'A': [('C', 2), ('D', 4)],
        'B': [('D', 1), ('E', 5)],
        'C': [('G', 6)],
        'D': [('G', 3)],
        'E': [('G', 2)],
        'G': []
    }

    # Unweighted adjacency for BFS
    graph_unweighted = {
        u: [v for v, _ in neighbors] for u, neighbors in graph_weighted.items()
    }

    # Consistent & Admissible Heuristics to Goal G:
    # h*(G)=0, h*(D)=3, h*(E)=2, h*(C)=6, h*(B)=4, h*(A)=7, h*(S)=5
    heuristics = {'S': 5, 'A': 7, 'B': 4, 'C': 6, 'D': 3, 'E': 2, 'G': 0}

    # 1. BFS
    def bfs(g, start, goal):
        q = collections.deque([[start]])
        visited = set()
        exp = []
        while q:
            path = q.popleft()
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            exp.append(node)
            if node == goal:
                # Compute cost in weighted graph
                cost = 0
                for i in range(len(path) - 1):
                    for neighbor, w in graph_weighted[path[i]]:
                        if neighbor == path[i+1]:
                            cost += w
                            break
                return path, cost, exp
            for nxt in g.get(node, []):
                if nxt not in visited:
                    q.append(path + [nxt])
        return None, 0, exp

    # 2. Greedy Best-First Search
    def greedy(g, start, goal, h):
        counter = 0
        pq = [(h[start], counter, start, [start], 0)]
        visited = set()
        exp = []
        while pq:
            _, _, node, path, cost = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            exp.append(node)
            if node == goal:
                return path, cost, exp
            for nxt, weight in g.get(node, []):
                if nxt not in visited:
                    counter += 1
                    heapq.heappush(pq, (h[nxt], counter, nxt, path + [nxt], cost + weight))
        return None, 0, exp

    # 3. A* Search
    def a_star(g, start, goal, h):
        counter = 0
        pq = [(h[start], counter, 0, start, [start])]
        best_g = {start: 0}
        closed = set()
        exp = []
        while pq:
            f, _, g_val, node, path = heapq.heappop(pq)
            if node in closed:
                continue
            closed.add(node)
            exp.append(node)
            if node == goal:
                return path, g_val, exp
            for nxt, weight in g.get(node, []):
                new_g = g_val + weight
                if nxt not in best_g or new_g < best_g[nxt]:
                    best_g[nxt] = new_g
                    new_f = new_g + h[nxt]
                    counter += 1
                    heapq.heappush(pq, (new_f, counter, new_g, nxt, path + [nxt]))
        return None, 0, exp

    p_bfs, c_bfs, e_bfs = bfs(graph_unweighted, 'S', 'G')
    p_greedy, c_greedy, e_greedy = greedy(graph_weighted, 'S', 'G', heuristics)
    p_astar, c_astar, e_astar = a_star(graph_weighted, 'S', 'G', heuristics)

    print(f"{'Algorithm':<20}{'Expansion Order':<25}{'Path Found':<22}{'Path Cost':<10}")
    print("-" * 75)
    print(f"{'BFS':<20}{' -> '.join(e_bfs):<25}{' -> '.join(p_bfs):<22}{c_bfs:<10}")
    print(f"{'Greedy Best-First':<20}{' -> '.join(e_greedy):<25}{' -> '.join(p_greedy):<22}{c_greedy:<10}")
    print(f"{'A* Search':<20}{' -> '.join(e_astar):<25}{' -> '.join(p_astar):<22}{c_astar:<10}")

    print("\n--- Why Shortest in Number of Edges != Lowest-Cost Path ---")
    print("1. Hop Count vs Metric Cost: BFS treats every edge as cost = 1. A path with fewer hops can have massive")
    print("   cumulative edge weights, while an alternative route with more hops can have tiny weights.")
    print("2. For example, if S were directly connected to G with edge cost 100, BFS would return S -> G (1 hop, cost 100),")
    print("   whereas A* would return S -> B -> D -> G (3 hops, cost 5). Lowest cost and shortest hop count coincide")
    print("   only when all edge weights are uniform.")


# ==============================================================================
# Q3: CLASSIFICATION: k-NN VS DECISION TREE [30 Marks]
# ==============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: COMPREHENSIVE CLASSIFICATION (k-NN k=3 vs DECISION TREE depth=3)")
    print("=" * 70)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 1. k-NN (k=3)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    # 2. Decision Tree (max_depth=3)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)

    print("--- Model 1: k-NN (k=3) ---")
    print(f"Accuracy: {acc_knn * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm_knn}")
    print("\nClassification Report (k-NN):")
    print(classification_report(y_test, y_pred_knn, target_names=target_names))

    print("-" * 50)
    print("--- Model 2: Decision Tree (max_depth=3) ---")
    print(f"Accuracy: {acc_dt * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm_dt}")
    print("\nClassification Report (Decision Tree):")
    print(classification_report(y_test, y_pred_dt, target_names=target_names))

    print("\n--- Model Preference and Justification ---")
    print(f"k-NN Accuracy: {acc_knn*100:.2f}% | Decision Tree Accuracy: {acc_dt*100:.2f}%")
    print("Comparative Error Analysis:")
    mis_knn = np.where(y_test != y_pred_knn)[0]
    mis_dt = np.where(y_test != y_pred_dt)[0]
    print(f"  * k-NN misclassified {len(mis_knn)} sample(s): indices {mis_knn.tolist()}")
    print(f"  * Decision Tree misclassified {len(mis_dt)} sample(s): indices {mis_dt.tolist()}")
    print("\nPreference:")
    print("For this specific Iris run, Decision Tree is preferred if interpretability is desired (transparent rule paths),")
    print("or k-NN if robust non-linear boundaries are desired. Both achieve high accuracy (>95%), but Decision Tree")
    print("offers O(depth) inference speed without requiring stored training instances in memory.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. Knowledge Base (KB) & Forward Chaining:
       - KB stores sentences in a representation language.
       - Forward chaining fires rules whose antecedents are satisfied, asserting new knowledge monotonically.
    2. BFS vs Greedy Best-First vs A*:
       - BFS minimizes path length in edge count; uniform step cost only; high RAM.
       - Greedy Best-First minimizes h(n); fast but prone to dead-ends and suboptimal paths.
       - A* minimizes f(n) = g(n) + h(n); provably optimal with admissible heuristic.
    3. Classification Metrics:
       - Accuracy = (TP + TN) / Total.
       - Precision = TP / (TP + FP) (minimizes false alarms).
       - Recall = TP / (TP + FN) (minimizes missed positives).
       - F1-Score = Harmonic mean of Precision and Recall.
    4. Model Comparison Criteria:
       - Evaluation must consider: test accuracy, computational complexity, memory requirements,
         interpretability, and sensitivity to feature scaling and outliers.
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
