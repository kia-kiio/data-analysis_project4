# Final Project Specification Verification Audit — Corrected

این ممیزی بر اساس صورت پروژه رسمی و فایل‌های نهایی پروژه انجام شده است. دو ایراد روش‌شناختی شناسایی و اصلاح شدند:  
1) رتبه‌بندی Filter/GA با Cross-Validation مجدد روی همان Train پس از انتخاب ویژگی؛  
2) Boxplot نوت‌بوک مرجع که به تفکیک کلاس نبود.

## نتیجه کلی
**وضعیت: اصلاح‌شده و قابل بازبینی نهایی.**

در ممیزی تکمیلی، خروجی‌های قدیمی داخل Notebook دوم و همچنین اختلاف ترتیب نمونه‌ها بین اجرای Notebook و Stage 3 شناسایی شد. Notebook دوم با ترتیب canonical بر اساس index دوباره اجرا شد تا Cross-Validation و GA در هر دو مسیر دقیقاً قابل‌بازسازی باشند. جدول `experiments.csv` و ثابت‌های قابل‌بازسازی در `src/evaluation.py` نیز با نتایج نهایی همگام شدند.  
Test فقط پس از تثبیت انتخاب‌ها استفاده شده است. مدل نهایی فعلی `RandomForest_Nonlinear + All_27` است.

| مورد | وضعیت | شواهد |
|---|---|---|
| دیتاست 1941×27 + 7 هدف | PASS | `data/raw`, `results/target_quality_report.csv` |
| One-Hot دقیقاً یک برچسب برای هر ردیف | PASS | `results/target_quality_report.csv` |
| فرهنگ داده 27 ویژگی | PASS | `data/data_dictionary.csv` |
| عدم توازن و معیارهای متوازن | PASS | `results/class_distribution.csv` |
| Train/Validation/Test لایه‌بندی‌شده | PASS | `results/split_distribution.csv` |
| ایزوله‌بودن Test | PASS | `results/leakage_audit.md` |
| آمار حداقل 7 ویژگی + صدک‌ها | PASS | `results/eda_summary.csv` |
| Boxplot به تفکیک کلاس | PASS | `figures/boxplot_by_class.png`, notebook نهایی |
| دو Scatter + Correlation | PASS | `figures/` |
| مدل پایه + تفسیرپذیر + غیرخطی | PASS | `results/experiments.csv` |
| اثر وزن کلاس و Recall هر کلاس | PASS | `results/class_weight_comparison.csv` |
| Robust/Outlier analysis بدون حذف کورکورانه | PASS | `results/outlier_analysis.csv`, `results/robustness_comparison.csv` |
| Hyperparameter search فقط Train/CV | PASS | `results/hyperparameter_search.csv` |
| GA با 27 ژن، penalty و حداقل 3 seed | PASS | `results/genetic_selection.csv`, `results/ga_subset_overlap.csv` |
| مقایسه All/Filter/GA | PASS | `results/unified_model_selection_validation.csv` |
| **رتبه‌بندی نهایی کاندیداها روی Validation مستقل** | **PASS — اصلاح شد** | notebook نهایی + `results/unified_model_selection_validation.csv` |
| ارزیابی نهایی Test فقط یک بار | PASS | `results/final_test_metrics.json` |
| Confusion Matrix هفت‌کلاسه + per-class metrics | PASS | `results/confusion_matrix.csv`, `results/per_class_metrics.csv` |
| دو جفت خطای اصلی | PASS | `results/misclassification_analysis.csv` |
| کلاس‌های پرهزینه و FN | PASS | `results/cost_sensitive_classes.csv` |
| Human-in-the-loop و محدودیت دامنه | PASS | README / ARCHITECTURE |
| سه Notebook اجراشدنی و بدون خطا | PASS | `notebooks/01_analyst_full_analysis.ipynb`, `notebooks/02_final_model_pipeline.ipynb`, `notebooks/final_steel_plate_fault_classification.ipynb` |

## نتیجه مدل نهایی

- Model: `RandomForest_Nonlinear`
- Candidate: `All_27`
- Validation Macro F1: **0.8288**
- Test Accuracy: **0.8116**
- Test Balanced Accuracy: **0.8197**
- Test Macro F1: **0.8298**
- Test Weighted F1: **0.8125**

## نتیجه انتخاب ویژگی

GA در سه seed اجرا شده و زیرمجموعه‌های 18–19 ویژگی تولید کرده است، اما در رتبه‌بندی مستقل Validation هیچ زیرمجموعه GA/Filter از All_27 بهتر نبود. بنابراین پروژه نباید ادعا کند که GA کاهش ویژگی بدون افت را اثبات کرده است؛ نتیجه علمی صحیح این است که **برای این داده، نگه‌داشتن همه 27 ویژگی بر اساس Validation انتخاب شده است**.

## نکته اجرایی

گزارش PDF هشت‌صفحه‌ای موجود با نتایج اصلاح‌شده (`RandomForest + All_27`، Test Macro F1 = 0.8298 و Balanced Accuracy = 0.8197) همگام است. سه Notebook نیز بدون خطا اجرا شده‌اند و CSVهای اصلی با آخرین اجرای پروژه همگام هستند.


## Latest strict pass — V8
- Presentation typo fixed: GA subset size is 18–19 features, not “27 to 27”.
- Audit version metadata synchronized from V3 to V8.
- Team role table synchronized with the specified team mapping.
- Scientific checks remain unchanged: final model RandomForest + All_27; Test Macro F1 0.8298; Test Balanced Accuracy 0.8197; Test used only after candidate selection.
