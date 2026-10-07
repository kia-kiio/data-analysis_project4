# Leakage Audit

- Raw data files were not modified.
- Stratified Train/Validation/Test indices were fixed before Stage 3 modeling.
- Test rows were excluded from baseline, robustness, noise, filter, GA, hyperparameter tuning, and candidate selection.
- Scaling was inside Pipelines.
- Winsorization learned its percentiles from training data inside a Pipeline.
- Filter selection was computed from X_train only.
- GA fitness used only 3-fold CV on Train.
- Hyperparameter search used Train-only 5-fold CV with seed 2026.
- Unified model/feature candidate selection used the held-out Validation split after Train-only feature/model preparation.
- Noise sensitivity fits the final model on Train only and evaluates perturbed Validation rows; Test is excluded.
- Final Test was evaluated once after fitting the selected pipeline on Train+Validation.
- Outliers were not automatically deleted.

Final selected model: **RandomForest_Nonlinear**
Final selected candidate: **All_27**
Feature count: **27**
