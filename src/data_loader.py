"""
Project: Steel Plate Fault Type Classification
Module: src/data_loader.py
Description: Data loading utilities for raw and processed datasets.
Preserves analyst's load_raw and load_processed interfaces.
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from pathlib import Path
import pandas as pd
from utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_NAMES, RAW_TARGET_COLS, CLASSES, TARGET_MAPPING

TARGETS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

def load_raw(root="."):
    """
    Load the raw UCI dataset and column names.
    Preserved for backward compatibility with analyst code.
    """
    root = Path(root)
    raw = root / "data" / "raw" if (root / "data" / "raw").exists() else RAW_DATA_DIR
    var_file = raw / "Faults27x7_var"
    nna_file = raw / "Faults.NNA"
    
    names = [x.strip() for x in var_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.read_csv(nna_file, sep="\t", header=None, names=names)
    return df, names[:27], names[27:]

def load_processed(root="."):
    """
    Load the processed dataset with multiclass target.
    Preserved for backward compatibility with analyst code.
    """
    root = Path(root)
    path = root / "data" / "processed" / "processed_dataset.csv" if (root / "data" / "processed" / "processed_dataset.csv").exists() else PROCESSED_DATA_DIR / "processed_dataset.csv"
    return pd.read_csv(path)

def load_raw_data():
    """
    Authoritative raw loader returning (DataFrame, feature_names, target_names).
    """
    return load_raw(RAW_DATA_DIR.parents[1])

def load_processed_data():
    """
    Authoritative processed loader returning (DataFrame, feature_names, target_column).
    """
    df = load_processed(PROCESSED_DATA_DIR.parents[1])
    return df, FEATURE_NAMES, "Class"
