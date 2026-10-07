"""
Project: Steel Plate Fault Type Classification
Module: src/data_validation.py
Description: Data quality checks, target one-hot validation, geometric consistency,
and data dictionary generation.
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

from utils import (
    ROOT_DIR, DATA_DIR, RESULTS_DIR, FEATURE_NAMES, RAW_TARGET_COLS,
    CLASSES, TARGET_MAPPING
)
from data_loader import load_raw_data

FEATURE_DESCRIPTIONS = {
    'X_Minimum': 'Minimum X coordinate of defect bounding region',
    'X_Maximum': 'Maximum X coordinate of defect bounding region',
    'Y_Minimum': 'Minimum Y coordinate of defect bounding region',
    'Y_Maximum': 'Maximum Y coordinate of defect bounding region',
    'Pixels_Areas': 'Total area of the defect in pixels',
    'X_Perimeter': 'X-direction perimeter measurement of defect region',
    'Y_Perimeter': 'Y-direction perimeter measurement of defect region',
    'Sum_of_Luminosity': 'Sum of pixel luminosity values in the defect region',
    'Minimum_of_Luminosity': 'Minimum pixel luminosity inside defect',
    'Maximum_of_Luminosity': 'Maximum pixel luminosity inside defect',
    'Length_of_Conveyer': 'Conveyor position/length measurement during scanning',
    'TypeOfSteel_A300': 'Binary indicator for steel plate grade A300',
    'TypeOfSteel_A400': 'Binary indicator for steel plate grade A400',
    'Steel_Plate_Thickness': 'Nominal thickness of the steel plate (mm)',
    'Edges_Index': 'Ratio of boundary/edge pixels to defect area',
    'Empty_Index': 'Ratio of non-defect pixels in defect bounding box',
    'Square_Index': 'Measure of defect squareness (aspect ratio similarity)',
    'Outside_X_Index': 'Fraction of defect outside primary X bounding margin',
    'Edges_X_Index': 'Boundary edge variation along the X direction',
    'Edges_Y_Index': 'Boundary edge variation along the Y direction',
    'Outside_Global_Index': 'Global defect position indicator (0 to 1)',
    'LogOfAreas': 'Logarithmic transformation of Pixels_Areas (base 10)',
    'Log_X_Index': 'Logarithmic transformation of defect X-span',
    'Log_Y_Index': 'Logarithmic transformation of defect Y-span',
    'Orientation_Index': 'Defect spatial orientation / directional index',
    'Luminosity_Index': 'Normalized average luminosity index of defect',
    'SigmoidOfAreas': 'Sigmoid transformation of Pixels_Areas'
}

def validate_raw_data(df=None):
    """
    Validate raw data integrity:
    1. Check missing values
    2. Check duplicate rows
    3. Check One-Hot target mutually exclusive properties
    4. Detect 0-positive or multi-positive label rows
    """
    if df is None:
        df, _, _ = load_raw_data()
        
    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    
    target_cols = [c for c in RAW_TARGET_COLS if c in df.columns]
    target_sums = df[target_cols].sum(axis=1)
    
    zero_label_rows = df[target_sums == 0].index.tolist()
    multi_label_rows = df[target_sums > 1].index.tolist()
    invalid_binary = (~df[target_cols].isin([0, 1]).all(axis=1)).sum()
    
    # Target quality report
    tq_rows = []
    for t in target_cols:
        tq_rows.append({
            'target': t,
            'positive_count': int(df[t].sum()),
            'positive_rate_pct': float(df[t].mean() * 100),
            'unique_values': int(df[t].nunique()),
            'valid_binary': bool(df[t].isin([0, 1]).all())
        })
    tq_rows.extend([
        {'target': 'OneHot_row_sum', 'positive_count': int(target_sums.sum()), 'positive_rate_pct': float(target_sums.mean() * 100), 'unique_values': int(target_sums.nunique()), 'valid_binary': bool((target_sums == 1).all())},
        {'target': 'zero_label_rows', 'positive_count': len(zero_label_rows), 'positive_rate_pct': float(len(zero_label_rows) / n_rows * 100), 'unique_values': len(zero_label_rows), 'valid_binary': True},
        {'target': 'multi_label_rows', 'positive_count': len(multi_label_rows), 'positive_rate_pct': float(len(multi_label_rows) / n_rows * 100), 'unique_values': len(multi_label_rows), 'valid_binary': True},
        {'target': 'invalid_label_rows', 'positive_count': int(invalid_binary), 'positive_rate_pct': float(invalid_binary / n_rows * 100), 'unique_values': int(invalid_binary), 'valid_binary': True}
    ])
    tq_df = pd.DataFrame(tq_rows)
    tq_df.to_csv(RESULTS_DIR / "target_quality_report.csv", index=False)
    
    # Save affected indices
    affected_lines = ["row_index\tissue\n"]
    for idx in zero_label_rows:
        affected_lines.append(f"{idx}\tzero_label\n")
    for idx in multi_label_rows:
        affected_lines.append(f"{idx}\tmulti_label\n")
    (RESULTS_DIR / "target_quality_affected_indices.csv").write_text("".join(affected_lines), encoding="utf-8")
    
    # Summary JSON
    summary = {
        'rows': n_rows,
        'feature_count': len(FEATURE_NAMES),
        'target_count': len(target_cols),
        'missing_values': missing_total,
        'duplicate_rows': duplicate_total,
        'onehot_invalid_rows': len(zero_label_rows) + len(multi_label_rows),
        'classes': target_cols,
        'class_counts': {t: int(df[t].sum()) for t in target_cols}
    }
    (RESULTS_DIR / "stage1_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary, tq_df

def validate_geometry(df):
    """
    Check geometric consistency rules across features.
    These are domain consistency diagnostics, NOT automated deletion filters.
    """
    rules = [
        ('X_Minimum_le_X_Maximum', (df['X_Minimum'] <= df['X_Maximum'])),
        ('Y_Minimum_le_Y_Maximum', (df['Y_Minimum'] <= df['Y_Maximum'])),
        ('Pixels_Areas_positive', (df['Pixels_Areas'] > 0)),
        ('X_Perimeter_positive', (df['X_Perimeter'] > 0)),
        ('Y_Perimeter_positive', (df['Y_Perimeter'] > 0)),
        ('Steel_Plate_Thickness_positive', (df['Steel_Plate_Thickness'] > 0)),
        ('Edges_Index_in_[0,1]', df['Edges_Index'].between(0, 1)),
        ('Empty_Index_in_[0,1]', df['Empty_Index'].between(0, 1)),
        ('Square_Index_in_[0,1]', df['Square_Index'].between(0, 1)),
        ('Outside_X_Index_in_[0,1]', df['Outside_X_Index'].between(0, 1)),
        ('Edges_X_Index_in_[0,1]', df['Edges_X_Index'].between(0, 1)),
        ('Edges_Y_Index_in_[0,1]', df['Edges_Y_Index'].between(0, 1)),
        ('Outside_Global_Index_in_[0,1]', df['Outside_Global_Index'].between(0, 1)),
        ('SigmoidOfAreas_in_[0,1]', df['SigmoidOfAreas'].between(0, 1)),
    ]
    rows = []
    for rule, mask in rules:
        rows.append({
            'rule': rule,
            'valid_count': int(mask.sum()),
            'invalid_count': int((~mask).sum()),
            'invalid_percentage': float((~mask).mean() * 100)
        })
    geom_df = pd.DataFrame(rows)
    geom_df.to_csv(RESULTS_DIR / "geometry_quality.csv", index=False)
    return geom_df

def generate_data_dictionary(df):
    """
    Generate the authoritative Data Dictionary covering all 27 features and target columns.
    Saves to both data/data_dictionary.csv and results/data_dictionary.csv.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        is_feat = col in FEATURE_NAMES
        role = 'feature' if is_feat else ('target_multiclass' if col in ['Class', 'Class_ID'] else 'target_one_hot')
        desc = FEATURE_DESCRIPTIONS.get(col, f"Fault class indicator: {col}")
        
        dtype_str = str(s.dtype)
        if s.nunique() == 2 and set(s.dropna().unique()).issubset({0, 1}):
            feature_type = 'Binary'
        elif np.issubdtype(s.dtype, np.integer):
            feature_type = 'Integer'
        elif np.issubdtype(s.dtype, np.floating):
            feature_type = 'Continuous'
        else:
            feature_type = 'Categorical'
            
        rows.append({
            'column': col,
            'role': role,
            'dtype': dtype_str,
            'feature_type': feature_type,
            'description': desc,
            'min': float(s.min()) if np.issubdtype(s.dtype, np.number) else np.nan,
            'max': float(s.max()) if np.issubdtype(s.dtype, np.number) else np.nan,
            'mean': float(s.mean()) if np.issubdtype(s.dtype, np.number) else np.nan,
            'median': float(s.median()) if np.issubdtype(s.dtype, np.number) else np.nan,
            'std': float(s.std()) if np.issubdtype(s.dtype, np.number) else np.nan,
            'q25': float(s.quantile(0.25)) if np.issubdtype(s.dtype, np.number) else np.nan,
            'q50': float(s.quantile(0.50)) if np.issubdtype(s.dtype, np.number) else np.nan,
            'q75': float(s.quantile(0.75)) if np.issubdtype(s.dtype, np.number) else np.nan,
            'q90': float(s.quantile(0.90)) if np.issubdtype(s.dtype, np.number) else np.nan,
            'q95': float(s.quantile(0.95)) if np.issubdtype(s.dtype, np.number) else np.nan,
            'missing_count': int(s.isna().sum()),
            'unique_count': int(s.nunique())
        })
    dict_df = pd.DataFrame(rows)
    dict_df.to_csv(RESULTS_DIR / "data_dictionary.csv", index=False)
    dict_df.to_csv(DATA_DIR / "data_dictionary.csv", index=False)
    return dict_df

if __name__ == '__main__':
    raw_df, _, _ = load_raw_data()
    summary, tq = validate_raw_data(raw_df)
    print("Validation Summary:", summary)
    geom = validate_geometry(raw_df)
    print("Geometry Quality Checks completed.")
    d_dict = generate_data_dictionary(raw_df)
    print("Data Dictionary generated for", len(d_dict), "columns.")
