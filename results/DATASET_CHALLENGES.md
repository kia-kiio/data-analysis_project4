# DATASET CHALLENGES

## Challenge: Class imbalance
**Evidence:** 7 classes with substantially different frequencies.  
**Measured Result:** Largest class `Other_Faults` = 673 (34.67%); smallest `Dirtiness` = 55 (2.83%); ratio = 12.24:1.  
**Potential Impact:** Accuracy can hide weak performance on minority faults.  
**Recommended Solution:** Use Macro F1, Balanced Accuracy, and per-class Recall/Precision/F1 alongside Accuracy.  
**How It Was Tested:** Stratified Train/Validation/Test split and class-sensitive metrics.  
**Observed Result:** Final Test Macro F1 = 0.8298 and Balanced Accuracy = 0.8197; minority-class recalls are reported separately.  
**Remaining Risk:** Small support makes minority metrics more variable.

## Challenge: Feature redundancy and high correlation
**Evidence:** Train-only correlation analysis.  
**Measured Result:** Strong relationships include TypeOfSteel_A300/A400, Y_Minimum/Y_Maximum, and Pixels_Areas/Sum_of_Luminosity.  
**Potential Impact:** Redundant measurements can increase complexity and make subsets unstable.  
**Recommended Solution:** Document redundancy and compare all features with filter and GA subsets rather than deleting features manually.  
**How It Was Tested:** Full Train correlation matrix and feature-pair ranking.  
**Observed Result:** Filter subset used 21 features; GA subsets used 18–19 features across seeds (Seeds 11, 22, 33).  
**Remaining Risk:** Correlation does not prove identical predictive information.

## Challenge: Binary steel-type complementarity
**Evidence:** The two TypeOfSteel columns are mutually exclusive in the raw data.  
**Measured Result:** A300=777 and A400=1164; both are not simultaneously 1.  
**Potential Impact:** Keeping both can encode one categorical state with complementary indicators.  
**Recommended Solution:** Preserve them for the baseline and let feature selection determine whether both are useful.  
**How It Was Tested:** Value counts and row-wise joint-state check on Train.  
**Observed Result:** GA subsets differ in whether both indicators are retained.  
**Remaining Risk:** Manual removal could discard information useful to a nonlinear model.

## Challenge: Scale differences
**Evidence:** Feature ranges vary by orders of magnitude.  
**Measured Result:** Coordinate, area, perimeter and luminosity variables have much larger ranges than index variables near [0,1].  
**Potential Impact:** Distance/linear methods can be sensitive to scale.  
**Recommended Solution:** Fit scaling only inside training/CV pipelines.  
**How It Was Tested:** Train-only scale analysis and pipeline-based Logistic Regression.  
**Observed Result:** Logistic models were evaluated with StandardScaler inside Pipeline; Random Forest was evaluated without mandatory scaling.  
**Remaining Risk:** Distribution shift can change the appropriate preprocessing.

## Challenge: Outliers and heavy tails
**Evidence:** Global IQR flags are much more numerous than Isolation Forest flags.  
**Measured Result:** Global IQR flagged 582 rows (42.86%); Isolation Forest flagged 68 rows (5.01%).  
**Potential Impact:** Mechanical deletion would risk removing valid fault observations.  
**Recommended Solution:** Treat outliers as diagnostics; compare robust strategies without automatic deletion.  
**How It Was Tested:** Global and per-class IQR plus Isolation Forest.  
**Observed Result:** Per-class diagnostics were also produced and no automatic deletion was performed.  
**Remaining Risk:** Statistical detectors cannot determine physical validity by themselves.

## Challenge: Rare valid observations
**Evidence:** Small classes can contain observations that are global outliers but normal within class.  
**Measured Result:** Per-class outlier diagnostics were calculated separately for all seven classes.  
**Potential Impact:** Removing such observations can damage recall for rare faults.  
**Recommended Solution:** Require evidence of recording error before deletion.  
**How It Was Tested:** Class-conditional IQR and Isolation Forest diagnostics.  
**Observed Result:** No automatic outlier removal was applied.  
**Remaining Risk:** Physical inspection metadata were not available.

## Challenge: Noise sensitivity
**Evidence:** Controlled perturbation experiment on continuous selected features of Validation set.  
**Measured Result:** Under the final `RandomForest_Nonlinear` on `All_27` (27 features), Validation Macro F1 was 0.8288 at 0% noise, 0.8030 at 1%, 0.7450 at 5%, and 0.7145 at 10% noise; all 7 classes were evaluated.  
**Potential Impact:** Measurement noise can materially reduce balanced performance.  
**Recommended Solution:** Monitor sensor/camera quality and consider robust preprocessing where justified.  
**How It Was Tested:** Gaussian noise scaled by Train feature standard deviation; binary steel-type features were unchanged.  
**Observed Result:** Tree ensemble thresholding maintains resilience at low noise (<=1%), with degradation at 5% and 10% noise.  
**Remaining Risk:** Synthetic Gaussian noise is only one noise model.

## Challenge: Class overlap
**Evidence:** Final Test confusion matrix and bidirectional error analysis.  
**Measured Result:** Top reciprocal pairs were Bumps↔Other_Faults (22 bidirectional errors: 10 Bumps as Other, 12 Other as Bumps) and Pastry↔Other_Faults (15 bidirectional errors: 8 Pastry as Other, 7 Other as Pastry).  
**Potential Impact:** Similar defect geometry/appearance can limit separability.  
**Recommended Solution:** Inspect the identified feature groups and monitor per-class errors.  
**How It Was Tested:** Final Test confusion matrix plus Train-based feature separation diagnostics.  
**Observed Result:** Pair-specific features were recorded in `class_overlap_analysis.csv`.  
**Remaining Risk:** Two-dimensional plots alone cannot establish causal overlap.

## Challenge: Leakage risk
**Evidence:** Project requires Test protection and fold-local learned transformations.  
**Measured Result:** Split fixed as Train=1358, Validation=291, Test=292; scaling and winsorization are pipeline/fold-local; GA and hyperparameter search use Train only.  
**Potential Impact:** Leakage would produce optimistic estimates.  
**Recommended Solution:** Keep Test untouched until final evaluation and audit every learned transformation.  
**How It Was Tested:** `leakage_audit.md` and execution logs.  
**Observed Result:** Final Test was evaluated only after model/subset selection.  
**Remaining Risk:** External deployment shift is not represented by this dataset.

## Challenge: Feature-selection instability
**Evidence:** Three independent GA seeds.  
**Measured Result:** Selected feature counts = 18–19 across seeds (Seeds 11, 22, 33); 3-Fold CV Macro F1 = 0.6322–0.6515 (Fitness: 0.6256–0.6444).  
**Potential Impact:** A single GA run could overstate confidence in a subset.  
**Recommended Solution:** Report selection frequency and subset overlap across seeds.  
**How It Was Tested:** Seeds 11, 22 and 33 with 3-fold Train CV and global best-so-far preservation.  
**Observed Result:** Frequency and pairwise overlap are saved in results (Jaccard similarity: 0.4800 to 0.6087; 10 core features).  
**Remaining Risk:** Three seeds provide evidence of stability but not a proof of global stability.

## Challenge: Hyperparameter overfitting
**Evidence:** Search is performed with bounded RandomizedSearchCV budgets on Train.  
**Measured Result:** Logistic: 5 candidates; Random Forest: 3 candidates; both 5-fold Stratified CV with seed 2026.  
**Potential Impact:** Excessive search can overfit CV noise.  
**Recommended Solution:** Keep search budget explicit and use Test only once.  
**How It Was Tested:** `hyperparameter_search.csv` and `unified_model_selection_validation.csv`.  
**Observed Result:** Feature candidates were generated using Train-only procedures, then ranked once on the held-out Validation split. The best Validation candidate was `RandomForest_Nonlinear` + `All_27` with Macro F1 = 0.8288.  
**Remaining Risk:** CV selection uncertainty remains, especially for minority classes.

## Challenge: Domain-shift/generalization risk
**Evidence:** The project specification explicitly identifies transfer to another line/camera as a risk.  
**Measured Result:** No external factory/camera dataset was supplied, so domain shift was not experimentally quantified.  
**Potential Impact:** Performance may fall when measurement distributions change.  
**Recommended Solution:** Monitor feature distributions and minority-class Recall after deployment.  
**How It Was Tested:** No external-domain experiment was possible with supplied data.  
**Observed Result:** Not completed due to unavailable external-domain data.  
**Remaining Risk:** External generalization remains unverified.
