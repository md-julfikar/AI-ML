"""
ICT-4202: Artificial Intelligence and Machine Learning Laboratory
Islamic University, Kushtia, Bangladesh
FINAL LABORATORY EXAMINATION - SET 2

Contents:
- Q1: Knowledge-Based Agent (Forward Chaining with Propositional Logic)
- Q2: Depth-Limited Search (DLS) and Iterative Deepening DFS (IDDFS)
- Q3: Simple Linear Regression & Sensitivity Analysis (R^2 impact)
- Q4: Viva Voce Reference & Key Concepts
"""

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score





def run_q1():
    print("=" * 70)
    print("Q1: KNOWLEDGE-BASED AGENT & FORWARD CHAINING")
    print("=" * 70)

    
    rules = [
        ({'Cloudy'}, 'Rain'),
        ({'Rain'}, 'WetGround'),
        ({'WetGround'}, 'Slippery'),
        ({'Slippery'}, 'DriveSlowly'),
        ({'Rain', 'Cold'}, 'WearJacket')
    ]

    def forward_chain(initial_facts):
        known_facts = set(initial_facts)
        firing_sequence = []
        fired_rules = set()

        while True:
            new_inference_in_round = False
            for idx, (premises, conclusion) in enumerate(rules):
                if idx not in fired_rules and premises.issubset(known_facts):
                    fired_rules.add(idx)
                    firing_sequence.append(f"Rule {idx+1} fired: {premises} -> {conclusion}")
                    if conclusion not in known_facts:
                        known_facts.add(conclusion)
                        new_inference_in_round = True
            if not new_inference_in_round:
                break

        return known_facts, firing_sequence

    
    facts_case1 = {'Cloudy'}
    final_facts_1, sequence_1 = forward_chain(facts_case1)

    print("--- Experiment 1: Without 'Cold' ---")
    print(f"Initial Facts: {facts_case1}")
    print("Rule Firing Sequence:")
    for seq in sequence_1:
        print(f"  * {seq}")
    print(f"Final Inferred Knowledge: {final_facts_1}")
    print("Status of 'WearJacket':", "WearJacket" in final_facts_1)

    
    facts_case2 = {'Cloudy', 'Cold'}
    final_facts_2, sequence_2 = forward_chain(facts_case2)

    print("\n--- Experiment 2: With 'Cold' ---")
    print(f"Initial Facts: {facts_case2}")
    print("Rule Firing Sequence:")
    for seq in sequence_2:
        print(f"  * {seq}")
    print(f"Final Inferred Knowledge: {final_facts_2}")
    print("Status of 'WearJacket':", "WearJacket" in final_facts_2)

    print("\n--- Explanation: Why WearJacket does or does not fire ---")
    print("1. Rule 5 states: (Rain AND Cold) -> WearJacket. Both antecedents MUST be simultaneously true.")
    print("2. In Case 1, 'Rain' is derived from 'Cloudy', but 'Cold' is neither given nor derivable.")
    print("   Since the conjunction condition (Cold) is False, Rule 5 never fires.")
    print("3. In Case 2, 'Cold' is provided as an initial fact. Once 'Rain' is inferred, all premises of")
    print("   Rule 5 are satisfied, causing 'WearJacket' to fire and enter the knowledge base.")





def run_q2():
    print("\n" + "=" * 70)
    print("Q2: DEPTH-LIMITED SEARCH (DLS) & ITERATIVE DEEPENING DFS (IDDFS)")
    print("=" * 70)

    
    
    graph = {
        'S': ['A', 'B'],
        'A': ['C', 'D'],
        'B': ['E', 'F'],
        'C': ['H'],
        'D': ['G'],
        'E': ['I'],
        'F': ['J'],
        'H': [],
        'G': [],
        'I': [],
        'J': []
    }

    start_node = 'S'
    goal_node = 'G'

    def dls(node, goal, limit, path, visit_trace):
        visit_trace.append(node)
        if node == goal:
            return path, "FOUND"
        if limit <= 0:
            return None, "CUTOFF"

        cutoff_occurred = False
        for neighbor in graph.get(node, []):
            res_path, status = dls(neighbor, goal, limit - 1, path + [neighbor], visit_trace)
            if status == "FOUND":
                return res_path, "FOUND"
            if status == "CUTOFF":
                cutoff_occurred = True

        return (None, "CUTOFF") if cutoff_occurred else (None, "FAILED")

    
    trace_dls2 = []
    path_dls2, status_dls2 = dls(start_node, goal_node, limit=2, path=[start_node], visit_trace=trace_dls2)
    print(f"--- DLS with limit = 2 ---")
    print(f"Status: {status_dls2}")
    print(f"Exploration Trace: {' -> '.join(trace_dls2)}")
    print(f"Result Path: {path_dls2} (Goal 'G' is at depth 3, so limit 2 hits Cutoff before reaching G)")

    
    trace_dls3 = []
    path_dls3, status_dls3 = dls(start_node, goal_node, limit=3, path=[start_node], visit_trace=trace_dls3)
    print(f"\n--- DLS with limit = 3 ---")
    print(f"Status: {status_dls3}")
    print(f"Exploration Trace: {' -> '.join(trace_dls3)}")
    print(f"Result Path: {' -> '.join(path_dls3)}")

    
    print(f"\n--- IDDFS (Iterative Deepening Depth-First Search) ---")
    iddfs_overall_trace = []
    root_visits = 0

    max_search_depth = 4
    iddfs_found_path = None

    for depth in range(max_search_depth + 1):
        trace_at_d = []
        root_visits += 1
        path, status = dls(start_node, goal_node, limit=depth, path=[start_node], visit_trace=trace_at_d)
        iddfs_overall_trace.append((depth, status, trace_at_d, path))
        print(f"Iteration limit = {depth}: Status = {status:<7} | Trace: {' -> '.join(trace_at_d)}")
        if status == "FOUND":
            iddfs_found_path = path
            print(f"--> IDDFS found goal at depth limit {depth}! Path: {' -> '.join(path)}")
            break

    print("\n--- Explanation: Cutoff and Repeated Shallow Search ---")
    print("1. Cutoff: In DLS, when the search reaches the depth limit without finding the goal or exhausting")
    print("   branches, a 'Cutoff' signal indicates that the search was terminated artificially due to depth constraints.")
    print("2. Repeated Shallow Search: IDDFS repeats exploration of upper levels (e.g. root S is visited every iteration).")
    print("   Why it is acceptable: In trees with branching factor b >= 2, the bottom level b^d contains the vast majority")
    print("   of nodes (~(b/(b-1)) * b^d). The asymptotic time complexity remains O(b^d), while memory is strictly O(b*d).")





def run_q3():
    print("\n" + "=" * 70)
    print("Q3: LINEAR REGRESSION & R^2 SCORE SENSITIVITY")
    print("=" * 70)

    
    hours = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    scores_original = np.array([4, 5, 8, 9, 11, 14])

    
    reg1 = LinearRegression()
    reg1.fit(hours, scores_original)
    slope1 = reg1.coef_[0]
    intercept1 = reg1.intercept_
    pred_7_1 = reg1.predict([[7]])[0]
    preds1 = reg1.predict(hours)
    r2_1 = r2_score(scores_original, preds1)

    print("--- Model 1: Original Observations ---")
    print(f"Fitted Equation : Score = {slope1:.4f} * Hours + {intercept1:.4f}")
    print(f"Slope (m)       : {slope1:.4f}")
    print(f"Intercept (c)   : {intercept1:.4f}")
    print(f"Predicted Score for 7 study hours: {pred_7_1:.4f}")
    print(f"Coefficient of Determination (R^2): {r2_1:.4f} ({r2_1*100:.2f}% variance explained)")

    
    scores_modified = np.array([4, 5, 8, 9, 11, 18])
    reg2 = LinearRegression()
    reg2.fit(hours, scores_modified)
    slope2 = reg2.coef_[0]
    intercept2 = reg2.intercept_
    pred_7_2 = reg2.predict([[7]])[0]
    preds2 = reg2.predict(hours)
    r2_2 = r2_score(scores_modified, preds2)

    print("\n--- Model 2: Modified Score at 6 Hours (14 -> 18) ---")
    print(f"Fitted Equation : Score = {slope2:.4f} * Hours + {intercept2:.4f}")
    print(f"Slope (m)       : {slope2:.4f}")
    print(f"Intercept (c)   : {intercept2:.4f}")
    print(f"Predicted Score for 7 study hours: {pred_7_2:.4f}")
    print(f"Coefficient of Determination (R^2): {r2_2:.4f} ({r2_2*100:.2f}% variance explained)")

    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    
    x_line = np.linspace(0.5, 7.5, 100).reshape(-1, 1)
    axes[0].scatter(hours, scores_original, color='blue', label='Observed Data', s=60)
    axes[0].plot(x_line, reg1.predict(x_line), color='red', linestyle='--', label=f'Fit: y={slope1:.2f}x+{intercept1:.2f}')
    axes[0].scatter([7], [pred_7_1], color='green', marker='X', s=100, label=f'Pred(7)={pred_7_1:.2f}')
    axes[0].set_title(f"Original Data (R^2 = {r2_1:.4f})")
    axes[0].set_xlabel("Study Hours")
    axes[0].set_ylabel("Exam Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    
    axes[1].scatter(hours, scores_modified, color='purple', label='Modified Data (pt 6 -> 18)', s=60)
    axes[1].plot(x_line, reg2.predict(x_line), color='crimson', linestyle='--', label=f'Fit: y={slope2:.2f}x+{intercept2:.2f}')
    axes[1].scatter([7], [pred_7_2], color='darkorange', marker='X', s=100, label=f'Pred(7)={pred_7_2:.2f}')
    axes[1].set_title(f"Modified Data (R^2 = {r2_2:.4f})")
    axes[1].set_xlabel("Study Hours")
    axes[1].set_ylabel("Exam Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = "set_2_linear_regression.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"\nPlot saved successfully as '{plot_filename}'")

    print("\n--- Explanation of Change in R^2 & Slope ---")
    print(f"1. Slope increased from {slope1:.4f} to {slope2:.4f} because the point (6, 18) acts as a high-leverage point")
    print("   pulling the regression line upward at the right extreme.")
    print(f"2. R^2 changed from {r2_1:.4f} to {r2_2:.4f}. While both are strong linear relationships, moving the endpoint")
    print("   from 14 to 18 alters the variance and residuals. R^2 quantifies the proportion of variation explained;")
    print("   large deviations at extreme predictor values exert strong influence on both slope and R^2.")





def run_q4():
    print("\n" + "=" * 70)
    print("Q4: VIVA VOCE KEY CONCEPTS & SUMMARY")
    print("=" * 70)
    notes = """
    1. Forward Chaining:
       - Data-driven reasoning starting from known facts, iteratively applying Modus Ponens
         to infer new facts until the goal is proved or no new conclusions can be drawn.
    2. Facts vs Rules:
       - Fact: An assertion known to be unconditionally true (e.g., Cloudy).
       - Rule: A conditional statement (Antecedent -> Consequent) defining logical dependencies.
    3. Depth-Limited Search (DLS):
       - DFS restricted to a fixed maximum depth limit l. Solves infinite path issues in DFS
         but is incomplete if goal depth d > l, and suboptimal if d < l.
    4. Iterative Deepening DFS (IDDFS):
       - Combines BFS completeness and optimality (for uniform step costs) with DFS space efficiency O(b*d).
    5. Slope (m) and Intercept (c):
       - In y = mx + c, slope represents the rate of change in y per unit change in x.
       - Intercept represents the baseline expected value of y when x = 0.
    6. Coefficient of Determination (R^2):
       - R^2 = 1 - (SS_res / SS_tot). Measures the proportion of variance in target explained by features.
       - R^2 = 1 denotes perfect prediction; R^2 = 0 denotes performance equivalent to the mean model.
    """
    print(notes)


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()
