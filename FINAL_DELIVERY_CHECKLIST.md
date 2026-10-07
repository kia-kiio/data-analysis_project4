# Final Delivery Checklist — Corrected

## Data & EDA
- [x] Raw UCI files preserved.
- [x] One-Hot target integrity verified: exactly one positive label per row.
- [x] 27-feature data dictionary and class distribution documented.
- [x] Required statistics and percentiles produced.
- [x] Required class-wise Boxplot produced.
- [x] Two scatter plots and correlation analysis produced.

## Modeling
- [x] Majority baseline.
- [x] Interpretable Logistic Regression.
- [x] Nonlinear Random Forest.
- [x] Stratified Train/Validation/Test split.
- [x] Hyperparameter search uses Train-only CV.
- [x] Robust scaling / winsorization comparison.
- [x] Global and within-class outlier diagnostics without automatic deletion.
- [x] Controlled noise sensitivity on Validation.
- [x] Class-weight impact on Macro F1 and per-class Recall reported.

## Genetic Feature Selection
- [x] 27-bit chromosome.
- [x] Complexity-penalized Macro F1 fitness.
- [x] Three independent seeds: 11, 22, 33.
- [x] Feature frequency and Jaccard stability reported.
- [x] GA compared against Filter and All_27.
- [x] **Corrected:** GA/Filter candidates are not re-ranked by CV on the same Train data after feature selection; final candidate ranking uses held-out Validation.

- [x] Canonical Train/Validation/Test row ordering used for reproducible shuffled CV across execution paths.
- [x] Notebook outputs re-executed and synchronized with authoritative results.

## Final Evaluation
- [x] Test remains untouched during candidate selection.
- [x] Final model selected before Test access.
- [x] Final model: `RandomForest_Nonlinear + All_27`.
- [x] Test Accuracy: 0.8116.
- [x] Test Balanced Accuracy: 0.8197.
- [x] Test Macro F1: 0.8298.
- [x] Per-class Precision/Recall/F1 and FN reported.
- [x] 7×7 confusion matrix reported.
- [x] Top two reciprocal error pairs analyzed.
- [x] Cost-sensitive classes reported.

## Important scientific conclusion
The corrected leakage-safe selection does **not** support claiming that GA can reduce the 27 features without meaningful performance loss on this dataset. All 27 features therefore remain in the final model. GA remains a required experimental component and its stability/results are fully reported.

## Packaging note
The 8-page Persian PDF report is synchronized with the corrected results and is ready for submission.
