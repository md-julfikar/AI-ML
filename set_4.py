"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 4

Contents:
- Q1: Search Comparison (BFS, DFS, Bidirectional Search & Order Sensitivity)
- Q2: WEKA Classification Simulation & Step-by-Step Guide (J48 vs IBk with 10-Fold CV)
- Q3: Linear Regression & Perturbation Analysis on R^2
- Q4: Viva Voce Reference & Key Concepts
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==============================================================================
# Q1: SEARCH COMPARISON & ADJACENCY ORDER SENSITIVITY [30 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: BFS, DFS, AND BIDIRECTIONAL SEARCH COMPARISON")
    print("=" * 70)

    # Graph: P->{Q,R}, Q->{S,T}, R->{U}, S->{V}, T->{V}, U->{V}, V:{}
    graph = {
        'P': ['Q', 'R'],
        'Q': ['S', 'T'],
        'R': ['U'],
        'S': ['V'],
        'T': ['V'],
        'U': ['V'],
        'V': []
    }

    start_node = 'P'
    goal_node = 'V'

    # 1. BFS
    def bfs(g, start, goal):
        queue = collections.deque([[start]])
        visited = set()
        exp_order = []
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            exp_order.append(node)
            if node == goal:
                return path, exp_order
            for nxt in g.get(node, []):
                if nxt not in visited:
                    queue.append(path + [nxt])
        return None, exp_order

    # 2. DFS
    def dfs(g, start, goal):
        stack = [[start]]
        visited = set()
        exp_order = []
        while stack:
            path = stack.pop()
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            exp_order.append(node)
            if node == goal:
                return path, exp_order
            # Push in reverse order so first child in list is processed first
            for nxt in reversed(g.get(node, [])):
                if nxt not in visited:
                    stack.append(path + [nxt])
        return None, exp_order

    # 3. Bidirectional Search
    def bidirectional_search(g, start, goal):
        # Build reverse graph for backward search
        rev_g = {k: [] for k in g}
        for u, neighbors in g.items():
            for v in neighbors:
                if v not in rev_g:
                    rev_g[v] = []
                rev_g[v].append(u)

        q_f = collections.deque([[start]])
        q_b = collections.deque([[goal]])
        visited_f = {start: [start]}
        visited_b = {goal: [goal]}
        exp_order = []

        while q_f and q_b:
            # Forward step
            path_f = q_f.popleft()
            node_f = path_f[-1]
            exp_order.append(f"{node_f}(fwd)")
            if node_f in visited_b:
                full_path = path_f[:-1] + visited_b[node_f][::-1]
                return full_path, exp_order

            for nxt in g.get(node_f, []):
                if nxt not in visited_f:
                    new_p = path_f + [nxt]
                    visited_f[nxt] = new_p
                    q_f.append(new_p)

            # Backward step
            path_b = q_b.popleft()
            node_b = path_b[-1]
            exp_order.append(f"{node_b}(bwd)")
            if node_b in visited_f:
                full_path = visited_f[node_b][:-1] + path_b[::-1]
                return full_path, exp_order

            for prev in rev_g.get(node_b, []):
                if prev not in visited_b:
                    new_p = path_b + [prev]
                    visited_b[prev] = new_p
                    q_b.append(new_p)

        return None, exp_order

    path_bfs, exp_bfs = bfs(graph, start_node, goal_node)
    path_dfs, exp_dfs = dfs(graph, start_node, goal_node)
    path_bi, exp_bi = bidirectional_search(graph, start_node, goal_node)

    print(f"{'Algorithm':<22}{'Path Found':<25}{'Expanded Nodes Count':<22}{'Expansion Sequence':<35}")
    print("-" * 105)
    print(f"{'BFS':<22}{' -> '.join(path_bfs):<25}{len(exp_bfs):<22}{' -> '.join(exp_bfs):<35}")
    print(f"{'DFS (Original Q)':<22}{' -> '.join(path_dfs):<25}{len(exp_dfs):<22}{' -> '.join(exp_dfs):<35}")
    print(f"{'Bidirectional':<22}{' -> '.join(path_bi):<25}{len(exp_bi):<22}{' -> '.join(exp_bi):<35}")

    # Reversing Q's children: Q->{T, S}
    graph_rev_q = {
        'P': ['Q', 'R'],
        'Q': ['T', 'S'],  # Reversed order
        'R': ['U'],
        'S': ['V'],
        'T': ['V'],
        'U': ['V'],
        'V': []
    }
    path_dfs_rev, exp_dfs_rev = dfs(graph_rev_q, start_node, goal_node)
    print(f"{'DFS (Reversed Q)':<22}{' -> '.join(path_dfs_rev):<25}{len(exp_dfs_rev):<22}{' -> '.join(exp_dfs_rev):<35}")

    print("\n--- Adjacency Order Effect on DFS ---")
    print(f"Original Q children [S, T] -> DFS Path: {' -> '.join(path_dfs)}")
    print(f"Reversed Q children [T, S] -> DFS Path: {' -> '.join(path_dfs_rev)}")
    print("Conclusion: YES, reversing the adjacency order of Q's children changes DFS path and expansion order!")
    print("DFS strictly commits to the first branch in the adjacency list. Since T is explored before S,")
    print("DFS directly discovers V through T rather than S.")


# ==============================================================================
# Q2: WEKA CLASSIFICATION EXPERIMENT SIMULATION & BUFFER [30 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: WEKA CLASSIFICATION (J48 vs IBk on Iris, 10-Fold CV)")
    print("=" * 70)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 1. J48 equivalent (Decision Tree with entropy criterion)
    j48 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    j48_pred = cross_val_predict(j48, X, y, cv=cv)
    j48_acc = accuracy_score(y, j48_pred)
    j48_cm = confusion_matrix(y, j48_pred)

    print(f"--- J48 (C4.5 Decision Tree) 10-Fold Cross-Validation ---")
    print(f"Accuracy: {j48_acc * 100:.2f}% ({np.sum(j48_pred == y)} / {len(y)} correctly classified)")
    print(f"Confusion Matrix:\n{j48_cm}")

    # 2. IBk (k-NN) for k = 1, 3, 5
    print("\n--- IBk (Instance-Based / k-NN) 10-Fold Cross-Validation ---")
    best_ibk_k = None
    best_ibk_acc = -1
    best_ibk_cm = None

    for k in [1, 3, 5]:
        ibk = KNeighborsClassifier(n_neighbors=k)
        ibk_pred = cross_val_predict(ibk, X, y, cv=cv)
        ibk_acc = accuracy_score(y, ibk_pred)
        cm = confusion_matrix(y, ibk_pred)
        print(f"IBk (k={k}): Accuracy = {ibk_acc * 100:.2f}% | Correct: {np.sum(ibk_pred == y)}/{len(y)}")
        if ibk_acc > best_ibk_acc:
            best_ibk_acc = ibk_acc
            best_ibk_k = k
            best_ibk_cm = cm

    print(f"\nBest IBk Configuration: k = {best_ibk_k} (Accuracy = {best_ibk_acc * 100:.2f}%)")
    print(f"Confusion Matrix (Best IBk):\n{best_ibk_cm}")

    print("\n--- WEKA Result Comparison & Summary ---")
    if best_ibk_acc > j48_acc:
        print(f"IBk (k={best_ibk_k}) outperforms J48 ({best_ibk_acc*100:.2f}% vs {j48_acc*100:.2f}%).")
    else:
        print(f"J48 and IBk show competitive performance on the Iris dataset.")

    print("""
----------------------------------------------------------------------
WEKA EXPLORER STEP-BY-STEP PROCEDURE FOR LAB EXAM SUBMISSION:
1. Open WEKA GUI Chooser -> Click 'Explorer'.
2. In 'Preprocess' tab: Click 'Open file...' -> Navigate to 'data/iris.arff'.
3. Inspect attributes: sepallength, sepalwidth, petallength, petalwidth, class.
4. Go to 'Classify' tab:
   - Select Test Options: 'Cross-validation', Folds = 10.
   - For J48: Click 'Choose' -> trees -> J48 -> Click 'Start'.
   - For IBk: Click 'Choose' -> lazy -> IBk. Click on IBk text to set KNN=1, 3, 5 -> Click 'Start'.
5. Right click on result history -> 'Save result buffer' or take screenshot.
----------------------------------------------------------------------
    """)


# ==============================================================================
# Q3: LINEAR REGRESSION & R^2 SENSITIVITY [30 Marks]
# ==============================================================================
def run_q3():
    print("=" * 70)
    print("Q3: LINEAR REGRESSION & EFFECT OF TARGET MODIFICATION ON R^2")
    print("=" * 70)

    # Given data:
    # X = [2, 4, 6, 8, 10, 12]
    # Y = [5, 9, 13, 17, 21, 25]
    X = np.array([2, 4, 6, 8, 10, 12]).reshape(-1, 1)
    Y = np.array([5, 9, 13, 17, 21, 25], dtype=float)

    # 1. Fit original model
    model1 = LinearRegression()
    model1.fit(X, Y)
    slope1 = model1.coef_[0]
    intercept1 = model1.intercept_
    pred_15_1 = model1.predict([[15]])[0]
    preds1 = model1.predict(X)
    r2_1 = r2_score(Y, preds1)

    print("--- Original Model ---")
    print(f"Fitted Line: Y = {slope1:.4f} * X + {intercept1:.4f}")
    print(f"Slope (m)       : {slope1:.4f}")
    print(f"Intercept (c)   : {intercept1:.4f}")
    print(f"Prediction at X=15 : {pred_15_1:.4f}")
    print(f"R^2 Score       : {r2_1:.6f} (Notice R^2 = 1.0, perfect collinear relationship: Y = 2X + 1)")

    # 2. Modify one target value: change Y[5] (at X=12) from 25 to 35
    Y_modified = Y.copy()
    Y_modified[5] = 35.0  # Introduce leverage outlier

    model2 = LinearRegression()
    model2.fit(X, Y_modified)
    slope2 = model2.coef_[0]
    intercept2 = model2.intercept_
    pred_15_2 = model2.predict([[15]])[0]
    preds2 = model2.predict(X)
    r2_2 = r2_score(Y_modified, preds2)

    print("\n--- Modified Model (Changed Y[5] from 25 to 35) ---")
    print(f"Fitted Line: Y = {slope2:.4f} * X + {intercept2:.4f}")
    print(f"Slope (m)       : {slope2:.4f}")
    print(f"Intercept (c)   : {intercept2:.4f}")
    print(f"Prediction at X=15 : {pred_15_2:.4f}")
    print(f"R^2 Score       : {r2_2:.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_grid = np.linspace(1, 16, 100).reshape(-1, 1)

    # Original plot
    axes[0].scatter(X, Y, color='blue', s=60, label='Original Data')
    axes[0].plot(x_grid, model1.predict(x_grid), color='red', linestyle='--', label=f'Fit: Y={slope1:.2f}X+{intercept1:.2f}')
    axes[0].scatter([15], [pred_15_1], color='green', marker='X', s=120, label=f'Pred(15)={pred_15_1:.1f}')
    axes[0].set_title(f"Original Perfect Line (R^2 = {r2_1:.4f})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Modified plot
    axes[1].scatter(X, Y_modified, color='purple', s=60, label='Modified Data (pt 12->35)')
    axes[1].plot(x_grid, model2.predict(x_grid), color='crimson', linestyle='--', label=f'Fit: Y={slope2:.2f}X+{intercept2:.2f}')
    axes[1].scatter([15], [pred_15_2], color='darkorange', marker='X', s=120, label=f'Pred(15)={pred_15_2:.1f}')
    axes[1].set_title(f"Modified Outlier Data (R^2 = {r2_2:.4f})")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_path = "set_4_regression.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved as '{plot_path}'")

    print("\n--- Discussion on the Effect on R^2 ---")
    print(f"1. Originally, the data satisfied Y = 2X + 1 perfectly, yielding residual sum of squares SS_res = 0 and R^2 = 1.0.")
    print(f"2. When the single target point at X=12 is changed from 25 to 35, non-zero residuals are introduced.")
    print(f"3. Consequently, R^2 dropped from 1.0000 to {r2_2:.4f}, demonstrating that linear regression and R^2 are")
    print("   highly sensitive to outliers, especially at boundary/high-leverage points.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. BFS vs DFS:
       - BFS explores level by level using a FIFO queue; optimal for unit costs; high memory O(b^d).
       - DFS plunges deep along branches using a LIFO stack; not optimal; memory efficient O(b*m).
    2. Bidirectional Search:
       - Simultaneously searches forward from start and backward from goal until the two frontiers meet.
       - Reduces time complexity from O(b^d) to O(2 * b^(d/2)), a massive reduction in search space.
    3. 10-Fold Cross-Validation:
       - Divides data into 10 equal folds. Trains on 9 folds and tests on the remaining 1 fold,
         repeating 10 times so every sample is tested once. Avoids train/test split bias.
    4. J48 Algorithm:
       - WEKA's implementation of Ross Quinlan's C4.5 algorithm. Builds decision trees using
         Gain Ratio (entropy-based) and includes subtree pruning.
    5. IBk Algorithm:
       - WEKA's Instance-Based Learner (k-NN). Stores training instances and classifies test queries
         using distance-weighted nearest neighbors.
    6. Regression vs Classification:
       - Regression predicts continuous numerical quantities (e.g. price, temperature).
       - Classification maps features to discrete categorical class labels.
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
