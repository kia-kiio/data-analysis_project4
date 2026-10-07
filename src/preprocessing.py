"""
Project: Steel Plate Fault Type Classification
Module: src/preprocessing.py
Description: Target conversion (One-Hot to multiclass), stratified splitting (70/15/15),
and leakage-safe transformer definitions (StandardScaler, RobustScaler, Winsorizer).
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

from utils import (
    ROOT_DIR, DATA_DIR, RESULTS_DIR, FEATURE_NAMES, RAW_TARGET_COLS,
    CLASSES, TARGET_MAPPING
)

class TrainWinsorizer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe Winsorization transformer.
    Learns clipping thresholds (lower and upper percentiles) on the training set/fold only.
    """
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.lo_ = None
        self.hi_ = None

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(A, self.lower * 100, axis=0)
        self.hi_ = np.nanpercentile(A, self.upper * 100, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

def validate_one_hot(df, targets=None):
    """
    Validate that target columns represent valid mutually exclusive One-Hot encodings.
    """
    if targets is None:
        targets = [c for c in RAW_TARGET_COLS if c in df.columns]
    sums = df[targets].sum(axis=1)
    return {
        "valid": bool((sums == 1).all()),
        "zero_label_rows": int((sums == 0).sum()),
        "multi_label_rows": int((sums > 1).sum()),
        "total_rows": len(df)
    }

def convert_one_hot_to_multiclass(df, target_cols=None):
    """
    Convert 7 binary One-Hot columns into single 'Class' and 'Class_ID' columns.
    Enforces standardized class spelling (e.g. K_Scatch).
    """
    if target_cols is None:
        target_cols = [c for c in RAW_TARGET_COLS if c in df.columns]
        
    val = validate_one_hot(df, target_cols)
    if not val["valid"]:
        raise ValueError(f"One-Hot validation failed: {val}")
        
    out = df.copy()
    raw_class = out[target_cols].idxmax(axis=1)
    out["Class"] = raw_class.map(TARGET_MAPPING)
    
    # Class_ID as integer codes according to CLASSES order
    class_to_id = {c: i for i, c in enumerate(CLASSES)}
    out["Class_ID"] = out["Class"].map(class_to_id)
    return out

# Backward-compatible alias
add_class = convert_one_hot_to_multiclass

def stratified_split(df, target="Class", test_size=0.15, val_size=0.15, random_state=42):
    """
    Perform a strict 70 / 15 / 15 stratified split.
    70% Train (1358 samples), 15% Validation (291 samples), 15% Test (292 samples).
    """
    # 70% train, 30% temporary
    tr, tmp = train_test_split(df, test_size=(test_size + val_size), stratify=df[target], random_state=random_state)
    # 15% val, 15% test (50% of the 30% temp set)
    va, te = train_test_split(tmp, test_size=(test_size / (test_size + val_size)), stratify=tmp[target], random_state=random_state)
    return tr, va, te

def get_split_indices_and_distribution(df, random_state=42):
    """
    Computes split indices and records class distribution across splits.
    Saves split_distribution.csv and split_indices.csv.
    """
    X = df[FEATURE_NAMES]
    y = df['Class']
    
    X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = train_test_split(
        X, y, df.index, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_tmp, y_tmp, idx_tmp, test_size=0.50, stratify=y_tmp, random_state=random_state
    )
    
    # Class distribution per split
    records = []
    for split_name, split_y in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
        vc = split_y.value_counts().reindex(CLASSES, fill_value=0)
        for c, count in vc.items():
            records.append({
                'split': split_name,
                'class': c,
                'count': int(count),
                'percentage': float(count / len(split_y) * 100)
            })
    dist_df = pd.DataFrame(records)
    dist_df.to_csv(RESULTS_DIR / "split_distribution.csv", index=False)
    
    # Save split indices
    idx_df = pd.DataFrame({
        'row_index': list(idx_train) + list(idx_val) + list(idx_test),
        'split': ['Train'] * len(idx_train) + ['Validation'] * len(idx_val) + ['Test'] * len(idx_test)
    }).sort_values('row_index')
    idx_df.to_csv(RESULTS_DIR / "split_indices.csv", index=False)
    
    return (X_train, y_train, idx_train), (X_val, y_val, idx_val), (X_test, y_test, idx_test), dist_df
