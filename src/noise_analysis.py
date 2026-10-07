"""
Project: Steel Plate Fault Type Classification
Module: src/noise_analysis.py
Description: Outlier diagnostic analysis (IQR, IsolationForest), robustness comparisons,
and controlled Gaussian noise sensitivity experimentation on Validation.
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from utils import RESULTS_DIR, FEATURE_NAMES, CLASSES
from preprocessing import TrainWinsorizer

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    }

def run_outlier_analysis(X_train, y_train, random_state=42):
    """
    Performs global and within-class outlier diagnostics on X_train.
    Distinguishes statistical anomalies without automated record deletion.
    """
    q1 = X_train.quantile(0.25)
    q3 = X_train.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    
    flag_iqr_global = (X_train.lt(lo) | X_train.gt(hi))
    row_iqr_global = flag_iqr_global.any(axis=1)
    
    # Global Isolation Forest on robustly scaled data
    Xr = (X_train - X_train.median()) / (iqr.replace(0, np.nan)).fillna(1)
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=random_state, n_jobs=-1)
    iso_pred = iso.fit_predict(Xr.fillna(0))
    row_iso_global = (iso_pred == -1)
    
    out_rows = [
        {
            'scope': 'global_train', 'method': 'IQR', 'class': 'ALL',
            'flagged_count': int(row_iqr_global.sum()),
            'flagged_percentage': float(row_iqr_global.mean() * 100),
            'feature_flag_count': int(flag_iqr_global.sum().sum())
        },
        {
            'scope': 'global_train', 'method': 'IsolationForest', 'class': 'ALL',
            'flagged_count': int(row_iso_global.sum()),
            'flagged_percentage': float(row_iso_global.mean() * 100),
            'feature_flag_count': np.nan
        }
    ]
    
    # Per-class outlier analysis
    for cls in CLASSES:
        mask = (y_train.values == cls)
        Xi = X_train.loc[mask]
        q1c = Xi.quantile(0.25)
        q3c = Xi.quantile(0.75)
        iqrc = q3c - q1c
        flags_c = (Xi.lt(q1c - 1.5 * iqrc) | Xi.gt(q3c + 1.5 * iqrc))
        row_iqr_c = flags_c.any(axis=1)
        
        Xrc = (Xi - Xi.median()) / (iqrc.replace(0, np.nan)).fillna(1)
        iso_c = IsolationForest(n_estimators=200, contamination=0.05, random_state=random_state, n_jobs=-1)
        iso_pred_c = iso_c.fit_predict(Xrc.fillna(0))
        row_iso_c = (iso_pred_c == -1)
        
        out_rows.extend([
            {
                'scope': 'within_class_train', 'method': 'IQR', 'class': cls,
                'flagged_count': int(row_iqr_c.sum()),
                'flagged_percentage': float(row_iqr_c.mean() * 100),
                'feature_flag_count': int(flags_c.sum().sum())
            },
            {
                'scope': 'within_class_train', 'method': 'IsolationForest', 'class': cls,
                'flagged_count': int(row_iso_c.sum()),
                'flagged_percentage': float(row_iso_c.mean() * 100),
                'feature_flag_count': np.nan
            }
        ])
        
    for f in FEATURE_NAMES:
        out_rows.append({
            'scope': 'feature_global_train', 'method': 'IQR', 'class': 'ALL',
            'flagged_count': int(flag_iqr_global[f].sum()),
            'flagged_percentage': float(flag_iqr_global[f].mean() * 100),
            'feature_flag_count': np.nan,
            'feature': f
        })
        
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS_DIR / "outlier_analysis.csv", index=False)
    return out_df

def evaluate_robustness_strategies(X_train, y_train, X_val, y_val):
    """
    Compares Standard Scaling vs Robust Scaling vs Winsorization+RobustScaling on Validation.
    All transformers fitted strictly on Train.
    """
    clipper = TrainWinsorizer(0.01, 0.99)
    robust_pipes = {
        'StandardScaler_Logistic': Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42))
        ]),
        'RobustScaler_Logistic': Pipeline([
            ('scale', RobustScaler()),
            ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42))
        ]),
        'Winsorize1to99_RobustScaler_Logistic': Pipeline([
            ('clip', clipper),
            ('scale', RobustScaler()),
            ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42))
        ])
    }
    
    rows = []
    for name, pipe in robust_pipes.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        dt = time.perf_counter() - t0
        
        m = compute_metrics(y_val, pred)
        rec = recall_score(y_val, pred, labels=CLASSES, average=None, zero_division=0)
        prec = precision_score(y_val, pred, labels=CLASSES, average=None, zero_division=0)
        f1 = f1_score(y_val, pred, labels=CLASSES, average=None, zero_division=0)
        
        row = {'strategy': name, **m, 'runtime_sec': dt}
        for i, c in enumerate(CLASSES):
            row[f'recall_{c}'] = float(rec[i])
            row[f'precision_{c}'] = float(prec[i])
            row[f'f1_{c}'] = float(f1[i])
        rows.append(row)
        
    rob_df = pd.DataFrame(rows)
    rob_df.to_csv(RESULTS_DIR / "robustness_comparison.csv", index=False)
    return rob_df

def run_noise_sensitivity_experiment(model, chosen_features, X_train, y_train, X_val, y_val, candidate="GA_11", noise_seed=2026):
    """
    Controlled noise perturbation experiment on continuous features of Validation set.
    Model is trained strictly on X_train. Test set is completely untouched.
    Levels: 0.0, 0.01, 0.05, 0.10.
    """
    continuous_feats = [f for f in chosen_features if X_train[f].nunique() > 2 and not set(X_train[f].dropna().unique()).issubset({0, 1})]
    noise_std = X_train[continuous_feats].std(ddof=0).replace(0, 1.0)
    
    model.fit(X_train[chosen_features], y_train)
    rng = np.random.default_rng(noise_seed)
    
    rows = []
    for level in [0.0, 0.01, 0.05, 0.10]:
        Xn = X_val[chosen_features].astype(float).copy()
        if level > 0.0:
            eps = rng.normal(0, level, size=(len(Xn), len(continuous_feats))) * noise_std.to_numpy()
            Xn.loc[:, continuous_feats] = Xn[continuous_feats].to_numpy() + eps
            
        pred = model.predict(Xn)
        m = compute_metrics(y_val, pred)
        rec = recall_score(y_val, pred, labels=CLASSES, average=None, zero_division=0)
        prec = precision_score(y_val, pred, labels=CLASSES, average=None, zero_division=0)
        f1 = f1_score(y_val, pred, labels=CLASSES, average=None, zero_division=0)
        
        row = {
            'model': type(model.named_steps['model'] if hasattr(model, 'named_steps') else model).__name__,
            'candidate': candidate,
            'feature_count': len(chosen_features),
            'n_features': len(chosen_features),
            'evaluation_split': 'Validation',
            'noise_level': level,
            'noise_definition': 'Gaussian perturbation on continuous selected features; sigma = noise_level * Train feature std; binary features unchanged',
            **m
        }
        for i, c in enumerate(CLASSES):
            row[f'precision_{c}'] = float(prec[i])
            row[f'recall_{c}'] = float(rec[i])
            row[f'f1_{c}'] = float(f1[i])
        rows.append(row)
        
    noise_df = pd.DataFrame(rows)
    noise_df.to_csv(RESULTS_DIR / "noise_sensitivity.csv", index=False)
    
    method_text = (
        f"# Noise Sensitivity Methodology\n\n"
        f"- **Model**: {row['model']}\n"
        f"- **Candidate**: {candidate}\n"
        f"- **Selected features**: {len(chosen_features)}\n"
        f"- **Evaluation split**: Validation (Test excluded)\n"
        f"- **Perturbed features**: {len(continuous_feats)} continuous features\n"
        f"- **Noise levels**: 0.0, 0.01, 0.05, 0.10 relative to Train standard deviation\n"
        f"- **All 7 classes monitored**: Precision, Recall, and F1 computed for all defect types\n"
    )
    (RESULTS_DIR / "noise_sensitivity_method.md").write_text(method_text, encoding="utf-8")
    return noise_df
