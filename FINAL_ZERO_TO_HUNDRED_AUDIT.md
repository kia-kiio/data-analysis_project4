# Final Zero-to-Hundred Audit — Steel Plate Fault Project V8

## Basis
- Official project specification: `پروژه_تیم_۳_تشخیص_نوع_عیب_ورق_فولادی.pdf`
- Audited package: this project directory / V8 ZIP, with the latest V9 strict corrections
- No previous audit claim was treated as authoritative; results were checked against code, CSV/JSON outputs, notebooks, report, README, and the official specification.

## Execution verification
- `python src/data_validation.py`: PASS
- `python src/stage2_analysis.py`: PASS
- `python src/stage3_analysis.py`: PASS
- Reproduced final selection: `RandomForest_Nonlinear + All_27`
- Reproduced Test Macro F1: `0.8297867601`
- Reproduced Test Balanced Accuracy: `0.8196770752`
- Stored notebooks contain no error outputs and no unexecuted code cells.

## Requirement scorecard

| Requirement | Status | Evidence |
|---|---|---|
| 1941 samples / 27 features / 7 targets | PASS | raw data, validation outputs |
| One-hot target integrity | PASS | `target_quality_report.csv` |
| Data Dictionary | PASS | `data/data_dictionary.csv` |
| Stratified 70/15/15 split | PASS | `split_distribution.csv`, `split_indices.csv` |
| Test isolation | PASS | code path + `leakage_audit.md` |
| 7+ feature statistics and required percentiles | PASS | EDA outputs/report |
| Class-wise Boxplot | PASS | `boxplot_by_class.png` |
| Two colored scatter plots | PASS | `scatter_plot_1.png`, `scatter_plot_2.png` |
| Correlation/redundancy analysis | PASS | correlation outputs |
| Majority baseline | PASS | `experiments.csv` |
| Interpretable model | PASS | Logistic Regression |
| Nonlinear multiclass model | PASS | Random Forest |
| Train-only hyperparameter search | PASS | `hyperparameter_search.csv` |
| Robust/outlier analysis without automatic deletion | PASS | IQR + Isolation Forest + robustness comparison |
| Controlled noise sensitivity | PASS | `noise_sensitivity.csv` |
| 27-gene GA | PASS | `ga_configuration.json`, GA outputs |
| Macro F1 + feature-count penalty | PASS | `ga_selector.py`, GA outputs |
| 3 GA seeds | PASS | 11, 22, 33 |
| Feature frequency / Jaccard / score spread | PASS | GA result files |
| All vs Filter vs GA comparison | PASS | `unified_model_selection_validation.csv` |
| Final candidate ranking on held-out Validation | PASS | authoritative unified table |
| Final Test after selection | PASS | `final_test_metrics.json` |
| 7x7 confusion matrix | PASS | `confusion_matrix.csv` |
| Per-class Precision/Recall/F1/FN | PASS | `per_class_metrics.csv` |
| Two reciprocal error pairs | PASS | `class_overlap_analysis.csv` |
| Cost-sensitive classes + FN | PASS | `cost_sensitive_classes.csv` |
| Human-in-the-loop | PASS | README/report |
| Domain-shift limitation | PASS | README/report |
| 6–8 page report | PASS | 8-page PDF |
| README execution/versions/seeds/runtime/responsibilities | PASS | README |

## Findings

### Fixed during this audit
1. `results/final_qc_report.md` contained stale reciprocal-error counts (25/14 instead of the authoritative 22/15).
2. `results/DATASET_CHALLENGES.md` contained stale noise metrics and stale reciprocal-error counts.
3. `presentation/presentation_structure.md` contained stale 10% noise recalls, stale reciprocal-error counts, an incorrect Accuracy-vs-Balanced-Accuracy gap, and an overstated Train-CV gap reduction.

### Methodological conclusion
The final scientific conclusion is supported: GA produced 18–19 feature subsets across three seeds, but no GA or Filter candidate exceeded the 27-feature Random Forest on the held-out Validation split. Therefore the final model correctly retains all 27 features.

### Important limitation
GA fitness is based on 3-fold CV inside Train; it is not a fully nested outer-CV estimate of feature-selection performance. The project does not claim it is. Final candidate ranking is instead performed on the held-out Validation split, followed by one final Test evaluation.

## Final verdict
**READY FOR SUBMISSION, WITH THE ABOVE DOCUMENTATION SYNCHRONIZED.**

The core code, data split, model-selection logic, final metrics, report, and official-project requirements are consistent after the stale documentation fixes above.


## Additional audit pass — October 2026

The following issues were found and corrected after a second source-level audit:
1. The defense-report Q4 text contained stale Random Forest hyperparameters and a stale validation score; it now reflects the authoritative tuned configuration (`n_estimators=50`, `min_samples_leaf=2`, `max_features=sqrt`), the Train-CV gap change (0.2181 → 0.2115), best CV Macro F1 (0.7776), and the held-out Validation Macro F1 (0.8288).
2. GA tables in the report were relabeled so their 0.63–0.65 values are explicitly identified as **Train 3-Fold CV Macro F1**, not held-out Validation scores.
3. The supplementary class-weight comparison script was changed from Test evaluation to Validation evaluation and no longer writes/overwrites the final Test predictions. This preserves the project's one-final-Test-evaluation rule.
4. Random Forest deployment documentation now refers to predicted-class probability rather than Softmax, which is not the output mechanism of Random Forest.
5. Notebook 01 no longer preserves a stale displayed `runpy` return object containing an old experiment table; its Stage-3 execution output is kept as the visible result.


## Latest strict pass — V9 strict corrections
1. Fixed an objectively incorrect presentation sentence that said the GA reduced 27 features to 27 features; it now correctly states 18–19 selected features (about 30% reduction).
2. Updated this audit file's stale V3 title/reference so the audit metadata matches the current package lineage.
3. Synchronized the team-role table with the team-role mapping specified for the deliverable: Melina Salemi = Analyst, Kiana Sarkari = Architect, Marvel Keshtkar = Manager. This is a deliverable consistency correction, not a change to the scientific methodology.

No new scientific leakage, metric, split, GA, or final-Test error was found in this pass.


## Latest strict pass — V9
- Added the complete hyperparameter search space, search budget, seeds, metric, CV protocol, and runtime to the project documentation.
- Synchronized the authoritative final-fit runtime across `final_test_metrics.json`, `experiments.csv`, `results_summary.csv`, and `src/evaluation.py`.
- Added an explicit status field clarifying why non-final candidates have no separate Test score.
- Patched the visible report metadata to identify the current final-audit package and added the missing hyperparameter search-space evidence to the PDF report.
