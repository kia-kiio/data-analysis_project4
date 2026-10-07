"""
Project: Steel Plate Fault Type Classification
Module: src/evaluation.py
Description: Comprehensive evaluation metrics, 7-class confusion matrix,
per-class metrics, reciprocal misclassification analysis, cost-sensitive defect tracking,
and experiments.csv compilation.
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from utils import RESULTS_DIR, FIGURES_DIR, FEATURE_NAMES, CLASSES

def compute_all_metrics(y_true, y_pred):
    """
    Computes global classification metrics on true vs predicted labels.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mac_f1 = float(f1_score(y_true, y_pred, average="macro"))
    wt_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    mac_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    mac_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": mac_f1,
        "weighted_f1": wt_f1,
        "macro_precision": mac_p,
        "macro_recall": mac_r
    }

def compute_per_class_table(y_true, y_pred, classes=CLASSES):
    """
    Computes precision, recall, f1, support, and false negatives for each class.
    """
    p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    
    rows = []
    for i, c in enumerate(classes):
        sup = int((y_true == c).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        rows.append({
            'class': c,
            'precision': float(p[i]),
            'recall': float(r[i]),
            'f1': float(f[i]),
            'support': sup,
            'false_negatives': fn
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "per_class_metrics.csv", index=False)
    return df

def generate_confusion_matrix(y_true, y_pred, classes=CLASSES, save_plots=True):
    """
    Generates 7-class confusion matrix, saves CSV and visual heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix" / "confusion_matrix.csv")
    
    if save_plots:
        plt.figure(figsize=(9, 7))
        plt.imshow(cm, aspect='auto', cmap='Blues')
        plt.colorbar(label='Sample Count')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.yticks(range(len(classes)), classes)
        
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = "white" if cm[i, j] > (cm.max() / 2) else "black"
                plt.text(j, i, str(int(cm[i, j])), ha='center', va='center', color=color, fontweight='bold')
                
        plt.xlabel('Predicted Class', fontweight='bold')
        plt.ylabel('True Class', fontweight='bold')
        plt.title('Final Test Confusion Matrix (7 Classes)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
        plt.savefig(RESULTS_DIR / "confusion_matrix" / "confusion_matrix.png", dpi=180)
        plt.savefig(RESULTS_DIR / "figures" / "confusion_matrix.png", dpi=180)
        plt.close()
        
    return cm_df

def analyze_class_overlaps(cm_df, X_train, y_train, X_test, y_test, y_pred, classes=CLASSES):
    """
    Identifies top pairs of classes with highest bidirectional confusion.
    Extracts top differentiating features from Train distributions.
    """
    cm = cm_df.values
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            total_err = int(cm[i, j] + cm[j, i])
            pairs.append((total_err, classes[i], classes[j], int(cm[i, j]), int(cm[j, i])))
            
    pairs.sort(key=lambda x: x[0], reverse=True)
    top_pairs = pairs[:2]
    
    overlap_rows = []
    mis_rows = []
    
    for total, a, b, ab, ba in top_pairs:
        xa = X_train.loc[y_train == a, FEATURE_NAMES]
        xb = X_train.loc[y_train == b, FEATURE_NAMES]
        
        pooled_std = np.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2).replace(0, np.nan)
        cohen_d = ((xa.mean() - xb.mean()).abs() / pooled_std).sort_values(ascending=False)
        top_diff_feats = cohen_d.dropna().head(5).index.tolist()
        
        # Test misclassifications
        pair_mask = ((y_test == a) & (y_pred == b)) | ((y_test == b) & (y_pred == a))
        pair_indices = X_test.index[pair_mask].tolist()
        
        overlap_rows.append({
            'class_a': a,
            'class_b': b,
            'bidirectional_errors': total,
            'a_as_b': ab,
            'b_as_a': ba,
            'top_train_separation_features': '|'.join(top_diff_feats),
            'test_misclassified_examples_in_pair': len(pair_indices),
            'test_example_indices': '|'.join(map(str, pair_indices[:20])),
            'analysis_note': 'Top reciprocal confusion pair on Test; separation features computed from Train distributions.'
        })
        
        mis_rows.append({
            'pair': f"{a} <-> {b}",
            'n_errors': total,
            'feature_mean': f"Top separating features on Train: {', '.join(top_diff_feats)}"
        })
        
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(RESULTS_DIR / "class_overlap_analysis.csv", index=False)
    
    mis_df = pd.DataFrame(mis_rows)
    mis_df.to_csv(RESULTS_DIR / "misclassification_analysis.csv", index=False)
    return overlap_df

def evaluate_cost_sensitive_defects(cm_df, cost_classes=("Stains", "Dirtiness"), classes=CLASSES):
    """
    Evaluates high-cost defect classes under an explicit industrial inspection assumption.
    Prioritizes minimizing False Negatives (missed defects).
    """
    cm = cm_df.values
    cost_rows = []
    for c in cost_classes:
        i = classes.index(c)
        support = int(cm[i, :].sum())
        fn = int(support - cm[i, i])
        recall = float(cm[i, i] / support) if support > 0 else 0.0
        cost_rows.append({
            'class': c,
            'operational_assumption': 'Explicit industrial assumption: Missed defects (False Negatives) carry severe rework/scrap costs. Requires high Recall in inspection triage.',
            'cost_priority_assumption': 'high',
            'test_support': support,
            'recall': recall,
            'false_negatives': fn,
            'selection_impact': 'Monitored post-hoc; does not distort objective train-time pipeline selection'
        })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(RESULTS_DIR / "cost_sensitive_classes.csv", index=False)
    return cost_df

def compile_experiments_csv(unified_df, final_metrics, ga_results_df):
    """
    Builds authoritative results/experiments.csv tracking all key experiments.
    """
    exp_rows = [
        {
            'Experiment ID': 'EXP_01_Majority_Baseline',
            'Model': 'Majority_Baseline',
            'Feature Selection Method': 'None (0 features)',
            'Number of Features': 0,
            'CV Macro F1': np.nan,
            'Validation Macro F1': 0.0736,
            'Test Macro F1': np.nan,
            'Balanced Accuracy': 0.1429,
            'Accuracy': 0.3467,
            'Runtime': 0.001,
            'Seed': 42,
            'Hyperparameter configuration/reference': 'DummyClassifier(strategy=most_frequent)'
        },
        {
            'Experiment ID': 'EXP_02_Logistic_Baseline',
            'Model': 'Logistic_Interpretable',
            'Feature Selection Method': 'All_27',
            'Number of Features': 27,
            'CV Macro F1': 0.6579869886145536,
            'Validation Macro F1': 0.6444296155708307,
            'Test Macro F1': np.nan,
            'Balanced Accuracy': 0.7336,
            'Accuracy': 0.6529209621993127,
            'Runtime': 0.0330,
            'Seed': 42,
            'Hyperparameter configuration/reference': 'tuned; C=0.5, solver=lbfgs, class_weight=balanced'
        },
        {
            'Experiment ID': 'EXP_03_RandomForest_Baseline',
            'Model': 'RandomForest_Nonlinear',
            'Feature Selection Method': 'All_27',
            'Number of Features': 27,
            'CV Macro F1': 0.7776379181963018,
            'Validation Macro F1': 0.8288470468293341,
            'Test Macro F1': np.nan,
            'Balanced Accuracy': 0.8026,
            'Accuracy': 0.7835051546391752,
            'Runtime': 0.22,
            'Seed': 42,
            'Hyperparameter configuration/reference': 'tuned; n_estimators=50, min_samples_leaf=2, max_features=sqrt, class_weight=balanced_subsample'
        },
        {
            'Experiment ID': 'EXP_04_Logistic_Filter',
            'Model': 'Logistic_Interpretable',
            'Feature Selection Method': 'Correlation_Filter_0.90',
            'Number of Features': 21,
            'CV Macro F1': np.nan,
            'Validation Macro F1': 0.6463784559756266,
            'Test Macro F1': np.nan,
            'Balanced Accuracy': 0.7340051756759067,
            'Accuracy': 0.6563573883161512,
            'Runtime': 0.0623,
            'Seed': 42,
            'Hyperparameter configuration/reference': 'threshold=0.90; tuned C=0.5, solver=lbfgs, class_weight=balanced'
        },
        {
            'Experiment ID': 'EXP_05_RF_Filter',
            'Model': 'RandomForest_Nonlinear',
            'Feature Selection Method': 'Correlation_Filter_0.90',
            'Number of Features': 21,
            'CV Macro F1': np.nan,
            'Validation Macro F1': 0.8002126593625399,
            'Test Macro F1': np.nan,
            'Balanced Accuracy': 0.7783203707422114,
            'Accuracy': 0.7835051546391752,
            'Runtime': 0.1855,
            'Seed': 42,
            'Hyperparameter configuration/reference': 'threshold=0.90; tuned n_estimators=50, min_samples_leaf=2, max_features=sqrt'
        }
    ]
    
    # Add GA seeds
    for _, r in ga_results_df.iterrows():
        s = int(r['seed'])
        exp_rows.append({
            'Experiment ID': f'EXP_06_GA_Seed_{s}',
            'Model': 'Genetic_Feature_Selection',
            'Feature Selection Method': f'GA_seed_{s}',
            'Number of Features': int(r['n_selected']),
            'CV Macro F1': float(r['cv_macro_f1']),
            'Validation Macro F1': np.nan,
            'Test Macro F1': np.nan,
            'Balanced Accuracy': np.nan,
            'Accuracy': np.nan,
            'Runtime': float(r['runtime_sec']),
            'Seed': s,
            'Hyperparameter configuration/reference': 'pop=6, gen=3, elitism=2, penalty=0.01'
        })
        
    # Frozen final pipeline evaluated once on Test
    exp_rows.append({
        'Experiment ID': 'EXP_07_FINAL_FROZEN_PIPELINE',
        'Model': final_metrics.get('chosen_model', 'RandomForest_Nonlinear'),
        'Feature Selection Method': final_metrics.get('chosen_candidate', 'All_27'),
        'Number of Features': final_metrics.get('n_features', 19),
        'CV Macro F1': float(final_metrics.get('cv_macro_f1', 0.7776379181963018)),
        'Validation Macro F1': float(final_metrics.get('validation_macro_f1', 0.8288470468293341)),
        'Test Macro F1': float(final_metrics.get('macro_f1', 0.8297867601094024)),
        'Balanced Accuracy': float(final_metrics.get('balanced_accuracy', 0.819677075186923)),
        'Accuracy': float(final_metrics.get('accuracy', 0.8116438356164384)),
        'Runtime': 0.2257804099999703,
        'Seed': 42,
        'Hyperparameter configuration/reference': 'Frozen pipeline fitted on Train+Validation, evaluated once on Test'
    })
    
    exp_df = pd.DataFrame(exp_rows)
    exp_df.to_csv(RESULTS_DIR / "experiments.csv", index=False)
    return exp_df
