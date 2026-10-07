# Robustness Analysis Interpretation

Robustness experiments were conducted to evaluate preprocessing sensitivity. The main comparison was performed using Logistic Regression because scaling and feature distribution have a direct impact on linear models. Tree-based Random Forest models are less sensitive to monotonic scaling transformations. Therefore, this analysis is interpreted as a preprocessing robustness study rather than a final model selection experiment.

The results show that RobustScaler and Winsorization provide small improvements in Macro F1 compared with StandardScaler. The improvement is limited, therefore the original preprocessing pipeline remains appropriate.
