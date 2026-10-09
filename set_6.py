"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 6

Contents:
- Q1: Propositional Logic Forward Chaining & Material Implication Truth Table
- Q2: Uninformed Search (BFS, DFS, IDDFS) & Memory Characteristics
- Q3: k-NN vs Decision Tree Comparison & Scaling Sensitivity Analysis
- Q4: Viva Voce Reference & Key Concepts
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import collections
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# ==============================================================================
# Q1: PROPOSITIONAL LOGIC & FORWARD CHAINING [25 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: PROPOSITIONAL LOGIC FORWARD CHAINING & IMPLICATION TRUTH TABLE")
    print("=" * 70)

    # Given Rules:
    # 1. Registered -> Student
    # 2. Student -> HasID
    # 3. Student AND PaidFee -> ExamEligible
    # 4. ExamEligible -> CanEnterExamHall
    rules = [
        ({'Registered'}, 'Student'),
        ({'Student'}, 'HasID'),
        ({'Student', 'PaidFee'}, 'ExamEligible'),
        ({'ExamEligible'}, 'CanEnterExamHall')
    ]

    def forward_chain(initial_facts):
        known_facts = set(initial_facts)
        inference_trace = []
        fired_rules = set()

        while True:
            new_inference = False
            for idx, (premises, conclusion) in enumerate(rules):
                if idx not in fired_rules and premises.issubset(known_facts):
                    fired_rules.add(idx)
                    inference_trace.append(f"Rule {idx+1}: {set(premises)} -> {conclusion}")
                    if conclusion not in known_facts:
                        known_facts.add(conclusion)
                        new_inference = True
            if not new_inference:
                break
        return known_facts, inference_trace

    # Case 1: Registered only
    facts1 = {'Registered'}
    res1, trace1 = forward_chain(facts1)
    print("--- Test Case 1: Initial Fact = {'Registered'} ---")
    print("Inference Trace:")
    for t in trace1:
        print(f"  * {t}")
    print(f"Final Inferred Knowledge: {res1}\n")

    # Case 2: Registered plus PaidFee
    facts2 = {'Registered', 'PaidFee'}
    res2, trace2 = forward_chain(facts2)
    print("--- Test Case 2: Initial Facts = {'Registered', 'PaidFee'} ---")
    print("Inference Trace:")
    for t in trace2:
        print(f"  * {t}")
    print(f"Final Inferred Knowledge: {res2}\n")

    # Truth Table of P -> Q
    print("--- Truth Table for Material Implication (P -> Q) ---")
    print(f"{'P':<10}{'Q':<10}{'P -> Q':<12}{'Evaluation Note'}")
    print("-" * 55)
    for p in [True, False]:
        for q in [True, False]:
            implies = (not p) or q
            note = "<-- FALSE (Broken Promise)" if (p is True and q is False) else "True (Vacuously or Directly)"
            print(f"{str(p):<10}{str(q):<10}{str(implies):<12}{note}")

    print("\nObservation: P -> Q is FALSE ONLY when P is True and Q is False.")


# ==============================================================================
# Q2: BFS, DFS, AND IDDFS COMPARISON [30 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: BFS, DFS, AND IDDFS SEARCH COMPARISON")
    print("=" * 70)

    # Graph: S->{A,B}, A->{C,D}, B->{E,F}, C->{G}, D->{H}, E->{I}, F->{J}
    graph = {
        'S': ['A', 'B'],
        'A': ['C', 'D'],
        'B': ['E', 'F'],
        'C': ['G'],
        'D': ['H'],
        'E': ['I'],
        'F': ['J'],
        'G': [],
        'H': [],
        'I': [],
        'J': []
    }

    start = 'S'
    goal = 'G'

    # 1. BFS
    def bfs(g, s, target):
        q = collections.deque([[s]])
        visited = set()
        exp = []
        while q:
            p = q.popleft()
            node = p[-1]
            if node in visited:
                continue
            visited.add(node)
            exp.append(node)
            if node == target:
                return p, exp
            for nxt in g.get(node, []):
                if nxt not in visited:
                    q.append(p + [nxt])
        return None, exp

    # 2. DFS
    def dfs(g, s, target):
        stk = [[s]]
        visited = set()
        exp = []
        while stk:
            p = stk.pop()
            node = p[-1]
            if node in visited:
                continue
            visited.add(node)
            exp.append(node)
            if node == target:
                return p, exp
            for nxt in reversed(g.get(node, [])):
                if nxt not in visited:
                    stk.append(p + [nxt])
        return None, exp

    # 3. IDDFS helper
    def dls(node, target, limit, path, visit_trace):
        visit_trace.append(node)
        if node == target:
            return path, "FOUND"
        if limit <= 0:
            return None, "CUTOFF"

        cutoff = False
        for nxt in graph.get(node, []):
            res, status = dls(nxt, target, limit - 1, path + [nxt], visit_trace)
            if status == "FOUND":
                return res, "FOUND"
            if status == "CUTOFF":
                cutoff = True
        return (None, "CUTOFF") if cutoff else (None, "FAILED")

    def iddfs(s, target, max_depth=5):
        all_exp = []
        for d in range(max_depth + 1):
            curr_trace = []
            path, status = dls(s, target, d, [s], curr_trace)
            all_exp.extend(curr_trace)
            if status == "FOUND":
                return path, curr_trace, all_exp, d
        return None, [], all_exp, -1

    p_bfs, exp_bfs = bfs(graph, start, goal)
    p_dfs, exp_dfs = dfs(graph, start, goal)
    p_iddfs, last_trace, total_exp_iddfs, found_d = iddfs(start, goal)

    print(f"Goal: Find path from '{start}' to '{goal}'\n")
    print(f"{'Algorithm':<15}{'Found Path':<22}{'Expanded Nodes':<30}")
    print("-" * 70)
    print(f"{'BFS':<15}{' -> '.join(p_bfs):<22}{' -> '.join(exp_bfs):<30}")
    print(f"{'DFS':<15}{' -> '.join(p_dfs):<22}{' -> '.join(exp_dfs):<30}")
    print(f"{'IDDFS':<15}{' -> '.join(p_iddfs):<22}{' -> '.join(last_trace):<30}")

    print("\n--- Memory and Search Characteristics Comparison ---")
    print("1. BFS: Level-by-level exploration. Memory complexity is O(b^d), requiring exponential queue storage.")
    print("2. DFS: Explores branch down to deepest leaf. Memory is linear O(b*m), but not optimal.")
    print("3. IDDFS: Combines BFS optimality/completeness with DFS O(b*d) linear memory. It regenerates shallow nodes,")
    print("   which incurs negligible ~O(b^d) runtime overhead while providing optimal paths with minimal RAM usage.")


# ==============================================================================
# Q3: k-NN VS DECISION TREE & SCALING SENSITIVITY [35 Marks]
# ==============================================================================
def run_q3():
    print("\n" + "=" * 70)
    print("Q3: k-NN VS DECISION TREE CLASSIFICATION ON IRIS")
    print("=" * 70)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 1. k-NN (k=5) unscaled
    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(X_train, y_train)
    y_pred_knn = knn_raw.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    # 2. Decision Tree (max_depth=3)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)

    print(f"--- Model 1: k-NN (k=5, Unscaled) ---")
    print(f"Accuracy: {acc_knn * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm_knn}")

    # Misclassified samples in k-NN
    mis_knn = np.where(y_test != y_pred_knn)[0]
    print(f"Misclassified samples count: {len(mis_knn)}")
    for idx in mis_knn:
        print(f"  Test Index {idx}: True={target_names[y_test[idx]]}, Predicted={target_names[y_pred_knn[idx]]}")

    print(f"\n--- Model 2: Decision Tree (max_depth=3) ---")
    print(f"Accuracy: {acc_dt * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm_dt}")

    # Misclassified samples in Decision Tree
    mis_dt = np.where(y_test != y_pred_dt)[0]
    print(f"Misclassified samples count: {len(mis_dt)}")
    for idx in mis_dt:
        print(f"  Test Index {idx}: True={target_names[y_test[idx]]}, Predicted={target_names[y_pred_dt[idx]]}")

    # 3. Standardize and rerun k-NN
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_s, y_train)
    y_pred_knn_s = knn_scaled.predict(X_test_s)
    acc_knn_s = accuracy_score(y_test, y_pred_knn_s)
    cm_knn_s = confusion_matrix(y_test, y_pred_knn_s)

    print(f"\n--- Model 3: k-NN (k=5, Standardized Features) ---")
    print(f"Accuracy: {acc_knn_s * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm_knn_s}")

    print("\n--- Why Scaling Affects k-NN More Than a Decision Tree ---")
    print("1. k-NN calculates Euclidean distance: d(x, y) = sqrt(sum((x_i - y_i)^2)).")
    print("   Differences across features are pooled into a single scalar distance. If features have unequal")
    print("   scales or variances, the feature with larger magnitudes completely dominates the neighbor selection.")
    print("2. Decision Trees evaluate one individual feature at a time via orthogonal axis splits (e.g. x_i <= theta).")
    print("   A monotonic transformation or rescaling of x_i simply scales the threshold theta identically,")
    print("   leaving the ordering and splitting decisions completely invariant. Hence, Decision Trees are scale-invariant.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. Forward Chaining:
       - Inference starting with known premises, iteratively applying rules to assert conclusions.
       - Suitable for planning, monitoring, and state verification.
    2. Material Implication (P -> Q):
       - Equivalent to (NOT P OR Q). Only False when P is True and Q is False.
    3. BFS vs DFS vs IDDFS:
       - BFS: Complete, optimal for uniform cost, high memory O(b^d).
       - DFS: Incomplete in infinite paths, suboptimal, low memory O(b*m).
       - IDDFS: Best of both worlds: complete, optimal, low memory O(b*d).
    4. k-NN vs Decision Tree:
       - k-NN is instance-based, non-parametric, lazy learner; expensive at test time O(N*D); sensitive to scaling.
       - Decision Tree is eager learner, builds explicit interpretable rules; O(depth) inference time; scale-invariant.
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
