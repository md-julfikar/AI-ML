"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 5

Contents:
- Q1: Three-Location Cleaning Agent with Full State Tracking & PEAS
- Q2: Greedy Best-First vs A* Search (Path Cost vs Heuristic Analysis)
- Q3: Decision Tree Classification on Iris (Depth Tuning, Overfitting & Metrics)
- Q4: Viva Voce Reference & Key Concepts
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import heapq
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================================================================
# Q1: THREE-LOCATION CLEANING AGENT [25 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: THREE-LOCATION CLEANING AGENT (LOCATIONS A, B, C)")
    print("=" * 70)

    # Initial environment setup: A=Dirty, B=Dirty, C=Clean, Agent starts at A
    env_state = {'A': 'Dirty', 'B': 'Dirty', 'C': 'Clean'}
    current_location = 'A'
    locations_order = ['A', 'B', 'C']

    print(f"Initial State: {env_state}, Initial Location: {current_location}\n")
    print(f"{'Step':<6}{'Agent Loc':<12}{'Percept':<18}{'Action Taken':<18}{'New Environment State':<30}")
    print("-" * 80)

    step = 1
    while True:
        # Check if all locations are clean
        all_clean = all(status == 'Clean' for status in env_state.values())

        if all_clean:
            action = 'NoOp'
            percept = (current_location, env_state[current_location])
            print(f"{step:<6}{current_location:<12}{str(percept):<18}{action:<18}{str(env_state):<30}")
            print("\n--> All rooms are Clean! Agent terminates with NoOp.")
            break

        current_status = env_state[current_location]
        percept = (current_location, current_status)

        if current_status == 'Dirty':
            action = 'Clean'
            env_state[current_location] = 'Clean'
        else:
            # Move to next location in the loop A -> B -> C -> A
            curr_idx = locations_order.index(current_location)
            next_loc = locations_order[(curr_idx + 1) % len(locations_order)]
            action = f'MoveTo({next_loc})'
            current_location = next_loc

        print(f"{step:<6}{current_location:<12}{str(percept):<18}{action:<18}{str(env_state):<30}")
        step += 1

    print("\n--- PEAS Description for Three-Location Cleaning Agent ---")
    peas = """
    P - Performance Measure:
        * Cleanliness percentage of all 3 rooms
        * Energy efficiency (minimizing moves and cleaning actions)
        * Time taken to reach 100% clean state
        * Penalties for unnecessary cleaning or moving when already clean
    E - Environment:
        * 3 discrete rooms/tiles: A, B, C
        * Deterministic, fully/partially observable, discrete, static between steps
        * Clean or Dirty status per room
    A - Actuators:
        * Vacuum/Clean mechanism
        * Locomotion motors (Move to adjacent/next location)
        * Halt / Idle (NoOp)
    S - Sensors:
        * Location sensor (Identifies current room: A, B, or C)
        * Dirt detector (Optical/infrared sensor returning Clean or Dirty)
    """
    print(peas)


# ==============================================================================
# Q2: GREEDY BEST-FIRST AND A* SEARCH [30 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: GREEDY BEST-FIRST SEARCH VS A* SEARCH")
    print("=" * 70)

    # Graph representation:
    # Edges: A-B=2, A-C=3, B-D=4, B-E=2, C-E=2, D-G=3, E-G=4
    # Heuristics: h(A)=6, h(B)=5, h(C)=4, h(D)=2, h(E)=3, h(G)=0
    graph = {
        'A': [('B', 2), ('C', 3)],
        'B': [('D', 4), ('E', 2)],
        'C': [('E', 2)],
        'D': [('G', 3)],
        'E': [('G', 4)],
        'G': []
    }

    heuristics = {'A': 6, 'B': 5, 'C': 4, 'D': 2, 'E': 3, 'G': 0}
    start = 'A'
    goal = 'G'

    # 1. Greedy Best-First Search
    def greedy_best_first(g, start_node, goal_node, h):
        counter = 0
        pq = [(h[start_node], counter, start_node, [start_node], 0)]
        visited = set()
        expansion_order = []

        while pq:
            h_val, _, node, path, g_cost = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            expansion_order.append(node)

            if node == goal_node:
                return path, g_cost, expansion_order

            for neighbor, weight in g.get(node, []):
                if neighbor not in visited:
                    counter += 1
                    heapq.heappush(pq, (h[neighbor], counter, neighbor, path + [neighbor], g_cost + weight))
        return None, float('inf'), expansion_order

    # 2. A* Search
    def a_star(g, start_node, goal_node, h):
        counter = 0
        pq = [(h[start_node], counter, 0, start_node, [start_node])]
        best_g = {start_node: 0}
        closed_set = set()
        expansion_order = []

        while pq:
            f, _, g_cost, node, path = heapq.heappop(pq)
            if node in closed_set:
                continue
            closed_set.add(node)
            expansion_order.append(node)

            if node == goal_node:
                return path, g_cost, expansion_order

            for neighbor, weight in g.get(node, []):
                new_g = g_cost + weight
                if neighbor not in best_g or new_g < best_g[neighbor]:
                    best_g[neighbor] = new_g
                    new_f = new_g + h[neighbor]
                    counter += 1
                    heapq.heappush(pq, (new_f, counter, new_g, neighbor, path + [neighbor]))
        return None, float('inf'), expansion_order

    path_greedy, cost_greedy, exp_greedy = greedy_best_first(graph, start, goal, heuristics)
    path_astar, cost_astar, exp_astar = a_star(graph, start, goal, heuristics)

    print(f"Goal: Search from '{start}' to '{goal}'")
    print(f"Heuristics h(n): {heuristics}\n")

    print(f"{'Algorithm':<20}{'Expansion Order':<25}{'Final Path':<22}{'Path Cost':<10}")
    print("-" * 75)
    print(f"{'Greedy Best-First':<20}{' -> '.join(exp_greedy):<25}{' -> '.join(path_greedy):<22}{cost_greedy:<10}")
    print(f"{'A* Search':<20}{' -> '.join(exp_astar):<25}{' -> '.join(path_astar):<22}{cost_astar:<10}")

    print("\n--- Explanation: Why Smallest h(n) Alone May Not Produce Lowest Cost ---")
    print("1. Greedy Best-First evaluates nodes solely on f(n) = h(n), ignoring the accumulated path cost g(n).")
    print("2. In this graph:")
    print("   - From A, Greedy prefers C (h=4) over B (h=5). From C it goes to E (h=3), then G (h=0).")
    print("     Greedy Path: A -> C -> E -> G, with Total Cost = 3 + 2 + 4 = 9.")
    print("   - In contrast, A* considers both g(n) and h(n): Path A -> B -> E -> G has cost 2 + 2 + 4 = 8 (or A -> B -> D -> G = 2 + 4 + 3 = 9).")
    print("     A* selects A -> B -> E -> G with cost 8, which is strictly cheaper!")
    print("3. Myopic heuristic bias: Greedy is easily misled by local optimistic estimates into taking globally expensive detours.")


# ==============================================================================
# Q3: DECISION TREE CLASSIFICATION ON IRIS [35 Marks]
# ==============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: DECISION TREE CLASSIFICATION & DEPTH TUNING ON IRIS")
    print("=" * 70)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # 70/30 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    depth_values = [1, 2, 3, 4, None]
    results = {}

    print(f"{'max_depth':<12}{'Train Accuracy (%)':<22}{'Test Accuracy (%)':<20}")
    print("-" * 55)

    best_test_acc = -1
    best_depth = None
    best_model = None

    for depth in depth_values:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, dt.predict(X_train)) * 100
        test_acc = accuracy_score(y_test, dt.predict(X_test)) * 100

        depth_str = str(depth) if depth is not None else "None"
        print(f"{depth_str:<12}{train_acc:<22.2f}{test_acc:<20.2f}")

        results[depth_str] = (train_acc, test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_depth = depth
            best_model = dt

    print(f"\nBest Model: max_depth = {best_depth} (Test Accuracy = {best_test_acc:.2f}%)")

    # Confusion matrix and detailed metrics for best model
    y_test_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)

    print("\nConfusion Matrix (Best Decision Tree):")
    print(cm)

    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    print("--- Discussion on Underfitting vs Overfitting ---")
    print("1. Underfitting (max_depth = 1): A single split (decision stump) cannot capture multi-class boundaries,")
    print("   leading to low train (~66.7%) and test accuracy.")
    print("2. Optimal Fit (max_depth = 2 or 3): Captures the true underlying data manifold, generalizing well to test data.")
    print("3. Overfitting (max_depth = None): A deep, unconstrained tree memorizes noisy leaf nodes, achieving 100% train")
    print("   accuracy but risking degraded test generalization on noisy or complex datasets.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. Simple Reflex Agent:
       - Selects actions based solely on current percept using condition-action rules. Has no internal state memory.
    2. Model-Based Agent with State:
       - Maintains internal state representation to handle partially observable environments and past history.
    3. Greedy Best-First vs A*:
       - Greedy uses f(n) = h(n); fast but neither complete nor cost-optimal.
       - A* uses f(n) = g(n) + h(n); complete and optimal with an admissible and consistent heuristic.
    4. Decision Tree Splitting Criteria:
       - Gini Impurity: Gini = 1 - sum(p_i^2). Measures probability of misclassifying a random sample.
       - Information Gain / Entropy: H(S) = -sum(p_i * log2(p_i)). Measures information theoretic uncertainty.
    5. Precision, Recall, F1-Score:
       - Precision = TP / (TP + FP): Proportion of predicted positives that were truly positive.
       - Recall = TP / (TP + FN): Proportion of actual positives correctly identified.
       - F1-Score = 2 * (Precision * Recall) / (Precision + Recall): Harmonic mean balancing both.
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
