# Project Architecture Specification — Steel Plate Fault Classification

This document specifies the software architecture, data pipelines, feature selection mechanisms, model hierarchies, and leakage prevention protocols for the **Steel Plate Fault Type Classification** project (Team 3).

---

## 1. High-Level Architecture & End-to-End Data Flow

The project adheres to a strict unidirectional data flow where each stage has clearly delineated responsibilities. Learned transformations are fitted only on training data/folds. Feature candidates produced by Train-only Filter/GA selection are ranked on the held-out Validation split; Test remains untouched until final evaluation.

```text
+-----------------------------------------------------------------------------------+
|                                1. DATA INGESTION                                  |
|   data/raw/Faults.NNA (1,941 samples x 34 columns) + data/raw/Faults27x7_var       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          2. VALIDATION & QUALITY AUDIT                            |
|   - Verify: 0 missing values, 0 duplicate records                                 |
|   - One-Hot Audit: exactly one positive label per row (0 multi-label, 0 zero-label)|
|   - Output: results/target_quality_report.csv, data/data_dictionary.csv           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        3. TARGET TRANSFORMATION & SPLIT                           |
|   - Map 7 One-Hot columns to multiclass 'Class' & 'Class_ID'                      |
|   - 70 / 15 / 15 Stratified Split: Train (1,358), Val (291), Test (292)           |
|   - Output: results/split_distribution.csv, results/split_indices.csv             |
|   - CRITICAL: Test partition is ISOLATED and FROZEN until Step 9                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       4. EXPLORATORY DATA ANALYSIS (EDA)                          |
|   - Executed strictly on Training Partition (1,358 samples)                       |
|   - 7 representative feature percentiles (Min, Q25, Q50, Q75, Q90, Q95, Max)       |
|   - Correlation matrix, redundancy detection, colored 2D scatter plots            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        5. OUTLIER & ROBUSTNESS ANALYSIS                           |
|   - Domain principle: NO automated sample deletion (outliers = rare defects)      |
|   - IQR & Isolation Forest diagnostics (global & within-class)                    |
|   - Scaler comparison: StandardScaler vs RobustScaler vs TrainWinsorizer          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     6. BASELINE & HYPERPARAMETER OPTIMIZATION                     |
|   - Majority-Class Baseline (proof of utility)                                    |
|   - Interpretable: Logistic Regression (L2, class_weight='balanced')              |
|   - Nonlinear: Random Forest Classifier (balanced_subsample)                      |
|   - 5-Fold Stratified CV on Train with RandomizedSearchCV (seed 2026)             |
|   - Logistic: C={0.1,0.5,1,2,5}; budget=5; RF: 2x3x2 space; budget=3       |
|   - Overfitting check: Train-CV generalization gap calculation                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            7. FEATURE SELECTION LAYER                             |
|   A. All Features: Full 27-gene baseline                                          |
|   B. Filter Method: Pairwise correlation |r| >= 0.90 on Train (21 features)        |
|   C. Genetic Algorithm (GA): Binary chromosome (27 genes), complexity penalty:    |
|      Fitness = CV_Macro_F1 - 0.01 * (N_selected / 27)                             |
|      Evaluated over 3 independent seeds (11, 22, 33) on Train CV                  |
|      Stability metrics: Jaccard similarity, selection frequencies                 |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    8. UNIFIED CANDIDATE SELECTION & FREEZING                      |
|   - Benchmark all combinations (Models x Subsets) via 5-Fold Stratified Train CV  |
|   - Objective metric: CV Macro F1 + Balanced Accuracy                             |
|   - WINNER SELECTED & FROZEN: RandomForest_Nonlinear + All_27 (27 features)        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      9. SINGLE FINAL TEST EVALUATION (FROZEN)                     |
|   - Fit frozen pipeline on Train + Validation (1,649 samples)                     |
|   - Evaluate EXACTLY ONCE on untouched Test set (292 samples)                     |
|   - Full 7-class Confusion Matrix, Per-class Precision / Recall / F1              |
|   - Macro F1, Weighted F1, Accuracy vs Balanced Accuracy comparison               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                   10. INTERPRETATION, ROBUSTNESS & REPORTING                      |
|   - Controlled Gaussian noise sensitivity (Validation set perturbation)           |
|   - High-cost defect tracking (Stains, Dirtiness Recall and False Negatives)      |
|   - Top bidirectional error analysis (Bumps <-> Other_Faults, Pastry <-> Other)   |
|   - Automated generation of Persian PDF deliverables & experiment catalogs        |
+-----------------------------------------------------------------------------------+
```

---

## 2. Directory Layout & Module Responsibilities

The codebase is organized into modular layers ensuring high maintainability, reusability, and separation of concerns:

```text
d:\data-analysis_project4\
│
├── README.md                          # Defense-ready documentation (team table at top)
├── ARCHITECTURE.md                    # System architecture and theoretical justifications
├── FINAL_PROJECT_AUDIT.md             # Formal verification audit against project specification
├── requirements.txt                   # Exact verified Python environment dependencies
│
├── data/
│   ├── raw/
│   │   ├── Faults.NNA                 # Untouched official raw dataset (1,941 x 34)
│   │   └── Faults27x7_var             # Official variable order and column names
│   ├── processed/
│   │   └── processed_dataset.csv      # 1,941 samples with standardized multiclass 'Class'
│   └── data_dictionary.csv            # Comprehensive metadata, percentiles, and descriptions
│
├── notebooks/
│   ├── 01_analyst_full_analysis.ipynb               # Preserved analyst analysis notebook (executed)
│   ├── 02_final_model_pipeline.ipynb                # End-to-end production pipeline notebook (executed)
│   └── final_steel_plate_fault_classification.ipynb # Standalone final-submission notebook (executed)
│
├── src/
│   ├── utils.py                       # Global paths, seeds (42), class and feature lists
│   ├── data_loader.py                 # Clean raw and processed ingestion loaders
│   ├── data_validation.py             # Target One-Hot checks, geometry checks, data dictionary
│   ├── preprocessing.py               # Target conversion, 70/15/15 stratified split, Winsorizer
│   ├── feature_selection.py           # Correlation filter (|r| >= 0.90) on Train
│   ├── genetic_feature_selection.py   # Multi-seed GA with complexity penalty & Jaccard metrics
│   ├── models.py                      # Model factories (Dummy, LogisticRegression, RandomForest)
│   ├── tuning.py                      # 5-fold RandomizedSearchCV and train-CV gap calculation
│   ├── noise_analysis.py              # Outlier detection, robust scalers, Gaussian noise testing
│   ├── evaluation.py                  # Metrics, 7-class CM, misclassification & high-cost tracking
│   ├── reporting.py                   # ReportLab Persian RTL PDF generator (8-page & defense)
│   ├── build_report.py                # Standalone script to compile both PDF reports
│   └── ga_selector.py                 # Preserved analyst GA interface (backward compatibility)
│
├── results/
│   ├── experiments.csv                # Authoritative experimental comparison table
│   ├── genetic_selection.csv          # Multi-seed GA subset and fitness tracking
│   ├── final_test_metrics.json        # Frozen single test evaluation results
│   ├── per_class_metrics.csv          # Per-class precision, recall, f1, support, false negatives
│   ├── confusion_matrix.csv           # 7x7 Test confusion matrix
│   ├── cost_sensitive_classes.csv     # Recall & FN for Stains and Dirtiness
│   ├── class_overlap_analysis.csv     # Differentiating features for top error pairs
│   ├── noise_sensitivity.csv          # Perturbation results on validation set
│   ├── hyperparameter_default_vs_tuned.csv # Train vs CV scores and generalization gap
│   ├── ga_feature_selection_results.csv # Multi-seed GA performance
│   ├── ga_subset_overlap.csv          # Pairwise Jaccard similarities
│   ├── ga_feature_frequency.csv       # Feature selection frequency across seeds
│   ├── confusion_matrix/              # Dedicated directory with CM CSV and PNG
│   ├── figures/                       # Complete set of generated publication figures
│   └── final/                         # Frozen final artifacts
│
├── reports/
│   ├── analyst_report.pdf             # 8-page comprehensive Persian RTL report
│   └── final_defense_answers.pdf      # Persian answers to all 6 defense questions
│
└── presentation/
    └── presentation_structure.md      # 10-minute presentation guide for team members
```

---

## 3. Critical Architectural Decision: Genetic Algorithm Evaluation Strategy

### The Architectural Dilemma
In feature selection using Genetic Algorithms, two wrapper paradigms exist:
1. **Surrogate Model Wrapper**: The GA evaluates chromosome fitness using a fast, regularized linear surrogate (e.g., Logistic Regression with L2 regularization).
2. **Target Model Wrapper**: The current GA implementation evaluates chromosome fitness with the regularized Logistic Regression surrogate on Train-only cross-validation. The resulting subsets are then compared with Logistic Regression and Random Forest.
3. **Leakage-safe candidate ranking**:
   - Filter and GA subsets are generated using Train only.
   - After feature selection, the candidates are fitted on Train and ranked once on the held-out Validation partition.
   - Re-running Cross-Validation on the same Train data for final candidate ranking is intentionally avoided because the selected feature subsets have already seen that Train data.
   - Test is not used for any ranking or tuning decision.
4. **Final result**:
   - The best Validation candidate is `RandomForest_Nonlinear + All_27` with Validation Macro F1 = **0.8288**.
   - The final model is fitted on Train+Validation and evaluated once on Test.
   - Final Test Macro F1 = **0.8298** and Balanced Accuracy = **0.8197**.
5. **Scientific conclusion**:
   - GA produced stable 18–19 feature candidates across three seeds, but none beat All_27 on the independent Validation split.
   - Therefore the project does not claim a successful dimensionality reduction; all 27 features are retained in the final model.
---

## 4. Leakage Prevention Protocol

Data leakage is the most critical failure mode in applied machine learning. The architecture enforces five strict boundaries:

1. **Stratified Partitioning First**:
   Data splitting into Train (70%), Validation (15%), and Test (15%) is performed immediately after target transformation. All subsequent operations (EDA, scaling, filter selection, GA optimization, hyperparameter tuning) are restricted strictly to Train.
2. **Encapsulated Preprocessing**:
   - Standard scaling (`StandardScaler`) and robust scaling (`RobustScaler`) are never fitted on the entire dataset. They are encapsulated inside scikit-learn `Pipeline` objects and fitted only on the training fold during cross-validation.
   - Outlier clipping uses `TrainWinsorizer`, a custom scikit-learn transformer that computes the 1st and 99th percentiles strictly from the training partition and clips test/validation data without updating thresholds.
3. **Filter Selection on Train Only**:
   The Pearson correlation matrix is computed exclusively on `X_train`. Features identified as redundant ($|r| \ge 0.90$) are dropped based only on training statistics.
4. **GA Optimization Within Train CV**:
   The GA fitness function computes Macro F1 using a 3-fold Stratified CV internal to `X_train`. Neither Validation nor Test samples are ever seen by the GA.
5. **Frozen Test Evaluation**:
   Hyperparameter configurations, feature subsets, and scaler choices are locked during Stage 8. The chosen pipeline is fitted on `Train + Validation` and evaluated **exactly once** on `Test`.

---

## 5. Genetic Algorithm Architectural Details

```text
Chromosome (Length = 27 bits):
[ 0,  1,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0,  1,  0,  1,  1,  0 ]
  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
  f1  f2  f3  f4  f5  f6  f7  f8  f9  f10 ...                                                             f27
  (0 = Feature Excluded, 1 = Feature Selected; Minimum Active Genes = 3; Target Columns NEVER Included)
```

- **Fitness Formulation**:
  $$\text{Fitness}(\mathbf{c}) = \overline{\text{Macro F1}}_{\text{3-fold CV}}(\mathbf{c}) - \lambda \times \left(\frac{\sum_{i=1}^{27} c_i}{27}\right)$$
  where $\lambda = 0.01$. This penalty penalizes excessive feature counts, favoring compact and stable subsets.
- **Population Size**: 6 individuals per generation.
- **Generations**: 3 generations with convergence tracking.
- **Selection**: Tournament Selection with size $k = 3$.
- **Crossover**: Single-Point Crossover with probability $p_c = 0.80$.
- **Mutation**: Bit-flip mutation with probability $p_m = 0.04$ per gene, enforcing the minimum 3-feature constraint.
- **Elitism**: Top 2 chromosomes are carried over directly to the next generation without modification.
- **Seeds**: Executed across independent seeds `11`, `22`, `33`.

### Stability Measures
- **Selection Frequency**: Counts how many seeds chose each feature (100% = Stable, 33–66% = Moderately Stable, 0% = Excluded).
- **Pairwise Jaccard Similarity**:
  $$J(S_a, S_b) = \frac{|S_a \cap S_b|}{|S_a \cup S_b|}$$
  Values range between 0.4800 and 0.6087 across seeds, with 10 features permanently selected in 100% of runs.

---

## 6. Industrial Deployment & Human-in-the-Loop Framework

The classification model is architected as an **inspection triage system** rather than an autonomous decision maker:
1. **Confidence-Based Routing**:
   - Defect predictions with predicted-class probability $\ge 0.85$ are automatically routed to routine sorting.
   - Predictions with confidence $< 0.85$ or classifications belonging to high-cost categories (`Stains`, `Dirtiness`) are routed to human visual inspectors for secondary verification.
2. **Cost-Sensitive Monitoring**:
   - Missed defects (False Negatives) in `Dirtiness` and `Stains` can result in ruined rolled sheets or customer rejections.
   - The evaluation framework explicitly tracks Recall and False Negative counts for these classes independently of global accuracy.
3. **Domain Shift & Drift Mitigation**:
   - Sensor k-factor shifts, illumination changes, and camera lens degradation are monitored via daily Kolmogorov-Smirnov (KS) tests on continuous feature distributions.
   - Significant covariate shift triggers human re-calibration and active learning annotation.
