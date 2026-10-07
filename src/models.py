"""
Project: Steel Plate Fault Type Classification
Module: src/models.py
Description: Model factory definitions for baseline, interpretable (Logistic Regression),
and nonlinear (Random Forest) classifiers with class weighting support.
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

def get_majority_baseline():
    """
    Trivial majority-class baseline classifier.
    Serves as proof of value beyond zero-rule classification.
    """
    return DummyClassifier(strategy="most_frequent")

def get_interpretable_model(C=1.0, solver="lbfgs", class_weight="balanced", scaler="standard", random_state=42):
    """
    Interpretable linear multiclass model.
    Encapsulated inside a Pipeline with feature scaling to prevent leakage.
    """
    scaler_step = StandardScaler() if scaler == "standard" else RobustScaler()
    return Pipeline([
        ("scale", scaler_step),
        ("model", LogisticRegression(
            C=C,
            solver=solver,
            max_iter=4000,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])

def get_nonlinear_model(n_estimators=80, min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample", random_state=42):
    """
    Nonlinear multiclass Random Forest classifier.
    Inherently models feature interactions and non-monotonic decision boundaries.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=1
    )
