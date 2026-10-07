"""
Project: Steel Plate Fault Type Classification
Module: src/feature_selection.py
Description: Filter-based feature selection based on correlation redundancy on Train data only.
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score

from utils import RESULTS_DIR, FEATURE_NAMES

def run_correlation_filter(X_train, threshold=0.90):
    """
    Train-only correlation filter.
    Identifies feature pairs with |r| >= threshold and removes redundant features deterministically.
    Returns:
        filter_features: list of kept feature names
        pairs_df: DataFrame recording redundant pairs, correlation, kept, and removed features
    """
    corr = X_train.corr(numeric_only=True)
    features = [f for f in FEATURE_NAMES if f in X_train.columns]
    
    removed = set()
    pairs = []
    
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            r = float(corr.loc[a, b])
            if abs(r) >= threshold:
                keep = a if a not in removed else (b if b not in removed else a)
                drop = b if keep == a else a
                if keep not in removed and drop not in removed:
                    removed.add(drop)
                pairs.append({
                    'feature_1': a,
                    'feature_2': b,
                    'correlation': r,
                    'abs_correlation': abs(r),
                    'kept': keep,
                    'removed': drop,
                    'redundancy_flag': 'high' if abs(r) >= 0.90 else 'moderate-high'
                })
                
    filter_features = [f for f in features if f not in removed]
    pairs_df = pd.DataFrame(pairs).sort_values('abs_correlation', ascending=False)
    
    # Save redundancy report
    pairs_df.to_csv(RESULTS_DIR / "feature_redundancy.csv", index=False)
    return filter_features, pairs_df

def evaluate_filter_selection(X_train, y_train, X_val, y_val, threshold=0.90):
    """
    Fit pipeline on Train with filtered features and evaluate on Validation.
    """
    filter_features, pairs_df = run_correlation_filter(X_train, threshold=threshold)
    
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42))
    ])
    
    t0 = time.perf_counter()
    pipe.fit(X_train[filter_features], y_train)
    val_pred = pipe.predict(X_val[filter_features])
    runtime = time.perf_counter() - t0
    
    val_macro_f1 = float(f1_score(y_val, val_pred, average='macro'))
    val_bal_acc = float(balanced_accuracy_score(y_val, val_pred))
    
    res_df = pd.DataFrame([{
        'selection_method': f'correlation_filter_abs_{threshold:.2f}',
        'threshold': threshold,
        'selected_features': '|'.join(filter_features),
        'n_selected': len(filter_features),
        'validation_macro_f1': val_macro_f1,
        'validation_balanced_accuracy': val_bal_acc,
        'runtime_sec': runtime
    }])
    res_df.to_csv(RESULTS_DIR / "filter_feature_selection.csv", index=False)
    return filter_features, res_df
