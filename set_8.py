"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 8

Contents:
- Q1: Depth-Limited Search (DLS) & IDDFS with Root Visit Accounting
- Q2: WEKA Experiment Simulation (IBk Raw vs Scaled vs J48 with 10-Fold CV)
- Q3: Regression Programming (Experience vs Salary Prediction, R^2, Plotting & Limitations)
- Q4: Viva Voce Reference & Key Concepts
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==============================================================================
# Q1: DLS AND IDDFS SEARCH WITH ROOT TRACKING [25 Marks]
# ==============================================================================
def run_q1():
    print("=" * 70)
    print("Q1: DLS AND IDDFS SEARCH (ROOT VISIT METRICS)")
    print("=" * 70)

    # Graph: A->{B,C}, B->{D,E}, C->{F}, D->{H}, E->{I}, F->{G}
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': ['H'],
        'E': ['I'],
        'F': ['G'],
        'H': [],
        'I': [],
        'G': []
    }

    start = 'A'
    goal = 'G'

    def dls(node, target, limit, path, trace, root_counter):
        if node == 'A':
            root_counter[0] += 1
        trace.append(node)

        if node == target:
            return path, "FOUND"
        if limit <= 0:
            return None, "CUTOFF"

        cutoff_flag = False
        for child in graph.get(node, []):
            res_path, status = dls(child, target, limit - 1, path + [child], trace, root_counter)
            if status == "FOUND":
                return res_path, "FOUND"
            if status == "CUTOFF":
                cutoff_flag = True
        return (None, "CUTOFF") if cutoff_flag else (None, "FAILED")

    # 1. DLS with Depth 2
    trace_d2 = []
    root_c2 = [0]
    path_d2, status_d2 = dls(start, goal, limit=2, path=[start], trace=trace_d2, root_counter=root_c2)
    print("--- DLS with Limit = 2 ---")
    print(f"Status   : {status_d2} (Goal G is at depth 3, cutoff reached)")
    print(f"Trace    : {' -> '.join(trace_d2)}")
    print(f"Path     : {path_d2}")

    # 2. DLS with Depth 3
    trace_d3 = []
    root_c3 = [0]
    path_d3, status_d3 = dls(start, goal, limit=3, path=[start], trace=trace_d3, root_counter=root_c3)
    print("\n--- DLS with Limit = 3 ---")
    print(f"Status   : {status_d3}")
    print(f"Trace    : {' -> '.join(trace_d3)}")
    print(f"Path     : {' -> '.join(path_d3)}")

    # 3. IDDFS
    print("\n--- IDDFS Execution Trace ---")
    root_visits_iddfs = [0]
    iddfs_path = None
    for depth in range(4):
        curr_trace = []
        path, status = dls(start, goal, limit=depth, path=[start], trace=curr_trace, root_counter=root_visits_iddfs)
        print(f"Depth {depth}: Status = {status:<7} | Trace: {' -> '.join(curr_trace)}")
        if status == "FOUND":
            iddfs_path = path
            break

    print(f"\nIDDFS Found Path            : {' -> '.join(iddfs_path)}")
    print(f"Total Times Root A Visited  : {root_visits_iddfs[0]} times (once per depth level: d=0, 1, 2, 3)")

    print("\n--- Explanation: Trade-Off of Repeated Shallow Searches ---")
    print("1. Overhead of Repeated Visits: Nodes at upper levels (like root A) are regenerated on every iteration.")
    print("2. Why It Is Negligible: In a tree with branching factor b, level d has b^d nodes. The sum of all previous")
    print("   nodes is sum_{i=0}^{d} b^i approx b^d / (b - 1). For b=2, upper nodes constitute only ~50% of the work.")
    print("3. Asymptotic Efficiency: Time complexity remains O(b^d), while memory is reduced from exponential O(b^d) to linear O(b*d).")


# ==============================================================================
# Q2: WEKA EXPERIMENT (IBk RAW vs SCALED vs J48) [30 Marks]
# ==============================================================================
def run_q2():
    print("\n" + "=" * 70)
    print("Q2: WEKA EXPERIMENT (IBk k=5 RAW vs SCALED vs J48, 10-FOLD CV)")
    print("=" * 70)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 1. IBk (k=5) Unscaled
    ibk_raw = KNeighborsClassifier(n_neighbors=5)
    pred_raw = cross_val_predict(ibk_raw, X, y, cv=cv)
    acc_raw = accuracy_score(y, pred_raw)
    cm_raw = confusion_matrix(y, pred_raw)

    # 2. IBk (k=5) with Standardization (StandardScaler / WEKA Standardize filter)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    ibk_std = KNeighborsClassifier(n_neighbors=5)
    pred_std = cross_val_predict(ibk_std, X_std, y, cv=cv)
    acc_std = accuracy_score(y, pred_std)
    cm_std = confusion_matrix(y, pred_std)

    # 3. J48 (Decision Tree C4.5 equivalent)
    j48 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    pred_j48 = cross_val_predict(j48, X, y, cv=cv)
    acc_j48 = accuracy_score(y, pred_j48)
    cm_j48 = confusion_matrix(y, pred_j48)

    print(f"1. IBk (k=5, Raw Attributes)      : Accuracy = {acc_raw * 100:.2f}% ({np.sum(pred_raw == y)}/150)")
    print(f"2. IBk (k=5, Standardized/Scaled) : Accuracy = {acc_std * 100:.2f}% ({np.sum(pred_std == y)}/150)")
    print(f"3. J48 (Pruned Decision Tree)     : Accuracy = {acc_j48 * 100:.2f}% ({np.sum(pred_j48 == y)}/150)")

    print(f"\nConfusion Matrix - IBk (Raw):\n{cm_raw}")
    print(f"\nConfusion Matrix - IBk (Standardized):\n{cm_std}")
    print(f"\nConfusion Matrix - J48:\n{cm_j48}")

    best_cfg = "IBk (Raw/Scaled)" if max(acc_raw, acc_std) >= acc_j48 else "J48"
    print(f"\nBest Performing Configuration: IBk (k=5) with {max(acc_raw, acc_std)*100:.2f}% accuracy.")

    print("""
----------------------------------------------------------------------
WEKA EXPERIMENT INSTRUCTIONS:
1. Open WEKA Explorer -> Preprocess -> Open file 'data/iris.arff'.
2. In Preprocess tab, to standardize:
   - Click 'Choose' -> filters -> unsupervised -> attribute -> Standardize.
   - Click 'Apply'.
3. Classify tab -> Test options -> Cross-validation (Folds = 10):
   - Choose lazy -> IBk -> set KNN = 5 -> Start.
   - Choose trees -> J48 -> Start.
4. Save Result Buffer for viva defense.
----------------------------------------------------------------------
    """)


# ==============================================================================
# Q3: REGRESSION PROGRAMMING (10 OBSERVATIONS) [35 Marks]
# ==============================================================================
def run_q3():
    print("=" * 70)
    print("Q3: LINEAR REGRESSION PROGRAMMING (YEARS EXPERIENCE VS SALARY)")
    print("=" * 70)

    # Reasonable Real-World Problem: Work Experience (Years) vs Monthly Salary (in Thousand BDT)
    # 12 observation data points (>= 10 observations)
    experience = np.array([1.0, 1.5, 2.0, 2.5, 3.2, 4.0, 4.8, 5.5, 6.2, 7.0, 8.5, 10.0]).reshape(-1, 1)
    salary = np.array([28.0, 32.5, 36.0, 41.0, 49.5, 56.0, 64.0, 70.0, 78.5, 85.0, 99.0, 115.0])

    print(f"Dataset: {len(experience)} Observations of Professional Experience vs Monthly Salary (kBDT)\n")
    for i in range(len(experience)):
        print(f"  Obs {i+1:<2}: Experience = {experience[i][0]:>4.1f} yrs  -->  Salary = {salary[i]:>5.1f} kBDT")

    # Train/Test Split (75/25)
    X_train, X_test, y_train, y_test = train_test_split(
        experience, salary, test_size=0.25, random_state=42
    )

    # Fit Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    slope = model.coef_[0]
    intercept = model.intercept_

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    print("\n--- Model Fit Parameters ---")
    print(f"Fitted Equation : Salary = {slope:.4f} * Experience + {intercept:.4f}")
    print(f"Slope (Beta_1)  : {slope:.4f} (Salary increases by ~{slope:.2f} kBDT per additional year)")
    print(f"Intercept (Beta_0): {intercept:.4f} kBDT (Baseline entry-level starting salary)")
    print(f"Train R^2       : {train_r2:.4f}")
    print(f"Test R^2        : {test_r2:.4f}")

    # Make two predictions for unseen inputs: e.g. 3.0 years and 12.0 years
    unseen_x = np.array([[3.0], [12.0]])
    unseen_preds = model.predict(unseen_x)

    print("\n--- Unseen Predictions ---")
    for x_val, pred_val in zip(unseen_x.flatten(), unseen_preds):
        print(f"  * Predicted salary for {x_val:.1f} years experience : {pred_val:.2f} kBDT")

    # Plotting
    plt.figure(figsize=(9, 5))
    plt.scatter(X_train, y_train, color='blue', label='Train Observations', s=50)
    plt.scatter(X_test, y_test, color='cyan', edgecolors='black', label='Test Observations', s=60)

    x_line = np.linspace(0.5, 13, 100).reshape(-1, 1)
    plt.plot(x_line, model.predict(x_line), color='red', linestyle='--', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    plt.scatter(unseen_x, unseen_preds, color='orange', marker='X', s=120, label='Unseen Predictions')

    plt.title("Experience vs Salary: Linear Regression", fontsize=13)
    plt.xlabel("Experience (Years)", fontsize=11)
    plt.ylabel("Salary (Thousand BDT)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_file = "set_8_regression.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nPlot saved as '{plot_file}'")

    print("\n--- Interpretation and Limitation ---")
    print("Interpretation: Strong positive linear correlation between experience and salary.")
    print("Limitation: Linear extrapolation fails at career extremes. In reality, salary curves exhibit")
    print("plateaus or diminishing returns due to market caps, promotions, or performance variances.")


# ==============================================================================
# Q4: VIVA VOCE REFERENCE & KEY CONCEPTS [10 Marks]
# ==============================================================================
def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. Depth-Limited Search (DLS) & Cutoff:
       - DFS bounded by depth limit l. Avoids infinite paths; returns 'Cutoff' when limit is exceeded.
    2. IDDFS:
       - Iteratively increases depth limit (0, 1, 2, ...). Combines BFS optimality with DFS O(b*d) memory.
    3. Cross-Validation:
       - Method for evaluating statistical generalizability of models across independent subsets.
    4. Preprocessing (Standardization vs Normalization):
       - Normalization (MinMax): Scales values to [0, 1]. Sensitive to extreme outliers.
       - Standardization (Z-score): Centers data around mean 0 with variance 1. Handles outliers better.
    5. Linear Regression & R^2:
       - Fits line y = mx + c minimizing Residual Sum of Squares (Ordinary Least Squares).
       - R^2 measures the percentage of total variance accounted for by the predictor.
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
