"""
Project: Steel Plate Fault Type Classification
Module: src/tuning.py
Description: Hyperparameter tuning on Training data using 5-fold Stratified CV
and evaluation of Train-Validation generalization gaps.
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import json
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils import RESULTS_DIR

def run_hyperparameter_search(X_train, y_train, cv_splits=5, search_seed=2026):
    """
    Executes RandomizedSearchCV on X_train only using 5-fold Stratified CV.
    Target metric: Macro F1.
    """
    search_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    # 1. Logistic Regression search space
    logit_base = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=4000, class_weight='balanced', random_state=42))
    ])
    logit_grid = {
        'model__C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'model__solver': ['lbfgs']
    }
    logit_search = RandomizedSearchCV(
        logit_base, logit_grid, cv=search_cv, scoring='f1_macro',
        n_jobs=1, refit=True, return_train_score=True, n_iter=5, random_state=search_seed
    )
    t0 = time.perf_counter()
    logit_search.fit(X_train, y_train)
    logit_runtime = time.perf_counter() - t0
    
    # 2. Random Forest search space
    rf_base = RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=1)
    rf_grid = {
        'n_estimators': [50, 80],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.5]
    }
    rf_search = RandomizedSearchCV(
        rf_base, rf_grid, cv=search_cv, scoring='f1_macro',
        n_jobs=1, refit=True, return_train_score=True, n_iter=3, random_state=search_seed
    )
    t0 = time.perf_counter()
    rf_search.fit(X_train, y_train)
    rf_runtime = time.perf_counter() - t0
    
    # Save search summary table
    hp_rows = [
        {
            'model': 'Logistic_Interpretable',
            'best_cv_macro_f1': float(logit_search.best_score_),
            'best_params': json.dumps(logit_search.best_params_, sort_keys=True),
            'n_candidates': len(logit_search.cv_results_['params']),
            'search_type': f'RandomizedSearchCV_{cv_splits}fold',
            'search_seed': search_seed,
            'runtime_sec': logit_runtime
        },
        {
            'model': 'RandomForest_Nonlinear',
            'best_cv_macro_f1': float(rf_search.best_score_),
            'best_params': json.dumps(rf_search.best_params_, sort_keys=True),
            'n_candidates': len(rf_search.cv_results_['params']),
            'search_type': f'RandomizedSearchCV_{cv_splits}fold',
            'search_seed': search_seed,
            'runtime_sec': rf_runtime
        }
    ]
    pd.DataFrame(hp_rows).to_csv(RESULTS_DIR / "hyperparameter_search.csv", index=False)
    
    # Default vs tuned comparison table
    cv_eval = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    default_gap_rows = []
    
    models_to_compare = [
        (
            'Logistic_Interpretable',
            Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(max_iter=4000, class_weight='balanced', random_state=42))]),
            logit_search.best_estimator_
        ),
        (
            'RandomForest_Nonlinear',
            RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=1),
            rf_search.best_estimator_
        )
    ]
    
    for label, default_est, tuned_est in models_to_compare:
        for status, est in [('default', default_est), ('tuned', tuned_est)]:
            sc = cross_validate(est, X_train, y_train, cv=cv_eval, scoring='f1_macro', return_train_score=True, n_jobs=1)
            train_m = float(sc['train_score'].mean())
            cv_m = float(sc['test_score'].mean())
            default_gap_rows.append({
                'model': label,
                'status': status,
                'train_macro_f1_mean': train_m,
                'train_macro_f1_std': float(sc['train_score'].std(ddof=1)),
                'cv_macro_f1_mean': cv_m,
                'cv_macro_f1_std': float(sc['test_score'].std(ddof=1)),
                'train_cv_gap': float(train_m - cv_m)
            })
            
    gap_df = pd.DataFrame(default_gap_rows)
    gap_df.to_csv(RESULTS_DIR / "hyperparameter_default_vs_tuned.csv", index=False)
    gap_df.to_csv(RESULTS_DIR / "hyperparameter_train_validation_gap.csv", index=False)
    
    return logit_search, rf_search, gap_df
