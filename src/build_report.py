"""
Project: Steel Plate Fault Type Classification
Module: src/build_report.py
Description: Wrapper script to build reports/analyst_report.pdf and reports/final_defense_answers.pdf.
Cross-platform compatible across Windows and Linux.
Preserves analyst's entry point.
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from pathlib import Path
import shutil
from reporting import generate_analyst_report, generate_defense_answers_report

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    rep1 = generate_analyst_report(root / "reports/analyst_report.pdf")
    shutil.copy(rep1, root / "analyst_report.pdf")
    rep2 = generate_defense_answers_report(root / "reports/final_defense_answers.pdf")
    print(f"Reports successfully generated:\n- {rep1}\n- {root / 'analyst_report.pdf'}\n- {rep2}")
