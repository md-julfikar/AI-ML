"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 1

Contents:
- Q1: Simple Reflex Vacuum Cleaner Agent + PEAS Description
- Q2: Uninformed & Informed Search (BFS, DFS, Greedy Best-First Search)
- Q3: Machine Learning - k-NN Classification on Iris Dataset
- Q4: Viva Voce Reference & Key Concepts
"""

import collections
import heapq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================================================================
# Q1: INTELLIGENT AGENT AND PEAS [25 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: TWO-ROOM SIMPLE REFLEX VACUUM AGENT & PEAS")
    print("=" * 70)

    # Initial Environment: Room A = Dirty, Room B = Clean, Agent location = B
    state = {'A': 'Dirty', 'B': 'Clean'}
    current_loc = 'B'

    print(f"Initial State: {state}, Starting Location: {current_loc}\n")
    print(f"{'Step':<6}{'Percept (Loc, Status)':<25}{'Action':<15}{'New State':<30}")
    print("-" * 75)

    step = 1
    while True:
        status = state[current_loc]
        percept = (current_loc, status)

        # Termination condition: Both rooms are clean
        if state['A'] == 'Clean' and state['B'] == 'Clean':
            action = 'NoOp'
            print(f"{step:<6}{str(percept):<25}{action:<15}{str(state):<30}")
            print("\n--> Goal Reached: Both rooms are Clean. Agent halts with NoOp.")
            break

        # Simple reflex rules
        if status == 'Dirty':
            action = 'Suck'
            state[current_loc] = 'Clean'
        elif current_loc == 'A':
            action = 'Right'
            current_loc = 'B'
        elif current_loc == 'B':
            action = 'Left'
            current_loc = 'A'
        else:
            action = 'NoOp'

        print(f"{step:<6}{str(percept):<25}{action:<15}{str(state):<30}")
        step += 1

    print("\n--- PEAS Description for University Document Delivery Robot ---")
    peas = """
    P - Performance Measure:
        * Delivery punctuality (on-time rate)
        * Document integrity and security (no lost or damaged papers)
        * Safety (zero collisions with students/faculty, obstacle avoidance)
        * Battery efficiency and energy consumption
    E - Environment:
        * University building hallways, corridors, elevators, doorways
        * Static obstacles: desks, pillars, doors, trash bins
        * Dynamic obstacles: walking students, professors, cleaning carts
    A - Actuators:
        * Electric drive wheels / steering motors
        * Secure compartment latch / electronic lock
        * Audio speaker & LCD display (greeting, delivery notifications)
        * Wireless module to call elevators or open smart doors
    S - Sensors:
        * LiDAR / Ultrasonic sensors for depth and collision avoidance
        * RGB-D cameras for door recognition and hallway navigation
        * RFID / Barcode / QR scanner for document tracking & ID badge verification
        * Wheel encoders and IMU for odometry and localization
    """
    print(peas)


# ==============================================================================
# Q2: SEARCH ALGORITHMS (BFS, DFS, GREEDY BEST-FIRST) [30 Marks]
# ==============================================================================
def run_q2():
    print("=" * 70)
    print("Q2: GRAPH SEARCH ALGORITHMS (BFS, DFS, GREEDY BEST-FIRST)")
    print("=" * 70)

    # Graph representation
    # Graph: A:{B,C}, B:{D,E}, C:{F}, D:{G}, E:{G}, F:{G}, G:{}
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': ['G'],
        'E': ['G'],
        'F': ['G'],
        'G': []
    }

    # Heuristic values h(n)
    heuristics = {'A': 6, 'B': 4, 'C': 3, 'D': 2, 'E': 2, 'F': 1, 'G': 0}
    start_node = 'A'
    goal_node = 'G'

    # 1. Breadth-First Search (BFS)
    def bfs(graph, start, goal):
        queue = collections.deque([[start]])
        visited = set()
        expanded_order = []

        while queue:
            path = queue.popleft()
            node = path[-1]

            if node in visited:
                continue
            visited.add(node)
            expanded_order.append(node)

            if node == goal:
                return path, expanded_order

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(path + [neighbor])
        return None, expanded_order

    # 2. Depth-First Search (DFS)
    def dfs(graph, start, goal):
        stack = [[start]]
        visited = set()
        expanded_order = []

        while stack:
            path = stack.pop()
            node = path[-1]

            if node in visited:
                continue
            visited.add(node)
            expanded_order.append(node)

            if node == goal:
                return path, expanded_order

            # Push in reverse order so first child is explored first
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(path + [neighbor])
        return None, expanded_order

    # 3. Greedy Best-First Search
    def greedy_best_first(graph, start, goal, h):
        # Priority queue stores: (heuristic_cost, tie_breaker_counter, path)
        counter = 0
        pq = [(h[start], counter, [start])]
        visited = set()
        expanded_order = []

        while pq:
            cost, _, path = heapq.heappop(pq)
            node = path[-1]

            if node in visited:
                continue
            visited.add(node)
            expanded_order.append(node)

            if node == goal:
                return path, expanded_order

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    counter += 1
                    heapq.heappush(pq, (h[neighbor], counter, path + [neighbor]))
        return None, expanded_order

    bfs_path, bfs_exp = bfs(graph, start_node, goal_node)
    dfs_path, dfs_exp = dfs(graph, start_node, goal_node)
    greedy_path, greedy_exp = greedy_best_first(graph, start_node, goal_node, heuristics)

    print(f"Goal: Find path from '{start_node}' to '{goal_node}'")
    print(f"Heuristics h(n): {heuristics}\n")

    print(f"{'Algorithm':<20}{'Found Path':<25}{'Nodes Expanded (Order)':<30}")
    print("-" * 75)
    print(f"{'BFS':<20}{' -> '.join(bfs_path):<25}{' -> '.join(bfs_exp):<30}")
    print(f"{'DFS':<20}{' -> '.join(dfs_path):<25}{' -> '.join(dfs_exp):<30}")
    print(f"{'Greedy Best-First':<20}{' -> '.join(greedy_path):<25}{' -> '.join(greedy_exp):<30}")

    print("\n--- Comparison & Analysis ---")
    print("1. BFS guarantees the shortest path in terms of number of edges (A -> B -> D -> G or A -> C -> F -> G).")
    print("2. DFS plunges deep along the first branch (A -> B -> D -> G) with minimal memory overhead O(b*d).")
    print("3. Greedy Best-First uses heuristic h(n) to prioritize nodes closest to goal: from A, it chooses C (h=3 vs B h=4),")
    print("   then from C chooses F (h=1), then reaches G. It expands fewer nodes because it is guided by the heuristic.")


# ==============================================================================
# Q3: MACHINE LEARNING - k-NN ON IRIS DATASET [35 Marks]
# ==============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: k-NN CLASSIFICATION ON IRIS DATASET")
    print("=" * 70)

    # 1. Load Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    print(f"Dataset: Iris (Total samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(target_names)})")

    # 2. Stratified train/test split (70/30) with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3. Train k-NN for k = 1, 3, 5, 7
    k_values = [1, 3, 5, 7]
    accuracies = {}
    models = {}

    print("\n--- Model Evaluation (Unscaled Features) ---")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[k] = acc
        models[k] = (knn, y_pred)
        print(f"k = {k:<2} | Test Accuracy: {acc * 100:.2f}%")

    # Identify best k
    best_k = max(accuracies, key=accuracies.get)
    print(f"\nBest k identified: k = {best_k} (Accuracy = {accuracies[best_k]*100:.2f}%)")

    # Display Confusion Matrix for Best k
    best_knn, best_pred = models[best_k]
    cm_unscaled = confusion_matrix(y_test, best_pred)
    print(f"\nConfusion Matrix (k = {best_k}, Unscaled):")
    print(cm_unscaled)
    print("\nClassification Report (Unscaled):")
    print(classification_report(y_test, best_pred, target_names=target_names))

    # 4. Standardize the features and repeat best k experiment
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=best_k)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)
    cm_scaled = confusion_matrix(y_test, y_pred_scaled)

    print("-" * 50)
    print(f"--- Standardized Features Experiment (k = {best_k}) ---")
    print(f"Test Accuracy with StandardScaler: {acc_scaled * 100:.2f}%")
    print(f"Confusion Matrix (Scaled):\n{cm_scaled}")

    print("\n--- Comment on the Effect of Scaling ---")
    print("k-NN relies on distance metrics (e.g., Euclidean distance). Without scaling, features with")
    print("larger magnitudes or variances dominate the distance calculation, distorting neighbor queries.")
    print("Standardization brings all features to zero mean and unit variance, allowing each feature to")
    print("contribute equitably to the distance metric. On Iris, features have comparable scales, but on")
    print("datasets with mismatched units, scaling prevents catastrophic feature dominance.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. Agent & Simple Reflex Agent:
       - An agent perceives its environment through sensors and acts upon it via actuators.
       - A Simple Reflex Agent acts solely on current percept (condition-action rules), ignoring history.
    2. PEAS Framework:
       - Performance measure, Environment, Actuators, Sensors. Defines task environment.
    3. BFS vs DFS vs Greedy Search:
       - BFS: Uses FIFO queue, complete and optimal for unweighted graphs, O(b^d) memory.
       - DFS: Uses LIFO stack, not optimal, O(b*m) memory.
       - Greedy Best-First: Uses priority queue sorted by h(n), incomplete in infinite spaces, not optimal.
    4. Heuristic Function h(n):
       - An estimate of the cheapest path cost from node n to the goal. Must be >= 0, h(Goal) = 0.
    5. k-Nearest Neighbors (k-NN):
       - Non-parametric, lazy-learning, instance-based classifier.
       - Hyperparameter k: Small k -> low bias, high variance (overfitting); Large k -> smoother boundary (underfitting).
    6. Feature Scaling:
       - Essential for distance-based algorithms (k-NN, K-Means, SVM) to equalize feature scales.
    7. Confusion Matrix:
       - Table showing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
