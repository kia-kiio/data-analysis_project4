"""
Project: Steel Plate Fault Type Classification
Module: src/utils.py
Description: Shared utility functions, path configurations, seed control, and dataset constants.
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import os
import random
from pathlib import Path
import numpy as np

# Base directory paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
PRESENTATION_DIR = ROOT_DIR / "presentation"

# Ensure essential output directories exist
for d in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, RESULTS_DIR / "confusion_matrix", RESULTS_DIR / "figures", RESULTS_DIR / "final"]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset constants
FEATURE_NAMES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
    "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
    "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
    "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400",
    "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index",
    "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas",
    "Log_X_Index", "Log_Y_Index", "Orientation_Index",
    "Luminosity_Index", "SigmoidOfAreas"
]

# Official 7 fault classes (standardized spelling)
CLASSES = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

# Raw UCI target column names (with K_Scatch as in UCI Faults27x7_var)
RAW_TARGET_COLS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

# Mapping between raw target names and standardized class names
TARGET_MAPPING = {
    "Pastry": "Pastry",
    "Z_Scratch": "Z_Scratch",
    "K_Scatch": "K_Scatch",
    "K_Scratch": "K_Scatch",
    "Stains": "Stains",
    "Dirtiness": "Dirtiness",
    "Bumps": "Bumps",
    "Other_Faults": "Other_Faults"
}

def set_seed(seed=42):
    """
    Ensure reproducibility across Python, NumPy, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
