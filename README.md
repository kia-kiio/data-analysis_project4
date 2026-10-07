| نقش | نام | GitHub |
| :--- | :--- | :--- |
| مدیر | Melina Salemi | [@s-melina](https://github.com/s-melina) |
| معمار | Kiana Sarkari | [@kia-kiio](https://github.com/kia-kiio) |
| تحلیلگر | Marvel Keshtkar | [@Marvel200123](https://github.com/Marvel200123) |

---

# تشخیص نوع عیب ورق فولادی (Steel Plate Fault Type Classification)
### پروژه جامع یادگیری ماشین و تحلیل داده صنعتی — تیم ۳

---

## ۱. مرور کلی پروژه (Project Overview)
این پروژه یک سامانه جامع یادگیری ماشین برای تشخیص و دسته‌بندی ۷ نوع عیب سطحی در خطوط نورد ورق‌های فولادی است. این سامانه با پردازش ۲۷ ویژگی هندسی، مکانی و روشنایی، عیوب مشاهده‌شده را به منظور اولویت‌بندی بازرسی تفکیک می‌کند. با توجه به نامتوازن بودن شدید داده‌ها و تفاوت قابل‌توجه در هزینه نادیده گرفتن عیوب مختلف، پروژه بر مبنای ارزیابی متوازن (Balanced Evaluation)، اجرای و ارزیابی انتخاب ویژگی با الگوریتم ژنتیک (Genetic Algorithm Feature Selection)، و جلوگیری قطعی از نشت داده (Leakage Prevention) معماری شده است.

---

## ۲. صورت‌بندی مسئله و سناریوی صنعتی (Problem Statement)
در خطوط تولید صنعتی ورق فولادی، عیوب سطحی به دلایل متعددی نظیر سایش غلتک‌ها، ناخالصی‌های سرباره، یا نوسانات خنک‌کاری رخ می‌دهند. هدف این پروژه ساخت مدلی است که:
1. هفت نوع عیب صنعتی را با عملکرد متوازن تفکیک کند.
2. زیرمجموعه‌ای فشرده، پایدار و قابل‌تفسیر از ویژگی‌ها را با الگوریتم ژنتیک شناسایی نماید.
3. به عنوان یک **ابزار پشتیبان تصمیم (Decision Support Tool) در کنار بازرس انسانی (Human-in-the-Loop)** به کار گرفته شود، نه جایگزین کورکورانه کنترل کیفیت فیزیکی.

---

## ۳. معرفی دیتاست و منبع رسمی (Dataset & Source)
- **نام دیتاست**: Steel Plates Faults
- **منبع رسمی**: [UCI Machine Learning Repository — Dataset ID: 198](https://archive.ics.uci.edu/dataset/198/steel%2Bplates%2Bfaults)
- **لینک دانلود مستقیم فایل فشرده**: [UCI Static Public Download](https://archive.ics.uci.edu/static/public/198/steel%2Bplates%2Bfaults.zip)
- **مجوز دیتاست**: Creative Commons Attribution 4.0 International (CC BY 4.0)

### ساختار داده‌ها
- **تعداد کل نمونه‌ها**: ۱٬۹۴۱ رکورد
- **تعداد ویژگی‌های مستقل**: ۲۷ ویژگی عددی (پیوسته، گسسته و دودویی)
- **ساختار هدف خام**: ۷ ستون باینری به صورت One-Hot (`Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`)
- **مقادیر گمشده (Missing Values)**: ۰ (فاقد مقدار مفقوده)
- **سطور تکراری (Duplicate Rows)**: ۰ در داده خام بررسی‌شده

---

## ۴. اطلاعات ویژگی‌ها و فرهنگ داده (Feature Information)
۲۷ ویژگی مستقل در چهار گروه ساختاری دسته‌بندی شده‌اند:
1. **ویژگی‌های مکانی و ابعادی**: `X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum`, `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`, `Length_of_Conveyer`, `Steel_Plate_Thickness`
2. **ویژگی‌های روشنایی**: `Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`, `Luminosity_Index`
3. **آلیاژ فولاد**: `TypeOfSteel_A300`, `TypeOfSteel_A400`
4. **شاخص‌های مشتق‌شده و هندسی**: `Edges_Index`, `Empty_Index`, `Square_Index`, `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Outside_Global_Index`, `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`, `Orientation_Index`, `SigmoidOfAreas`

شناسنامه کامل آماری و تعاریف رسمی ویژگی‌ها در مسیرهای `data/data_dictionary.csv` و `results/data_dictionary.csv` ثبت شده است.

---

## ۵. تبدیل متغیر هدف و کنترل کیفیت One-Hot (Target Conversion)
پیش از هرگونه تبدیل، تک‌تک ۱٬۹۴۱ سطر داده‌های خام ممیزی شدند:
- مجموع برچسب‌های دودویی برای تمام ۱٬۹۴۱ سطر برابر **دقیقاً ۱** است (`row_sum == 1`).
- تعداد سطور بدون برچسب (Zero-label rows): **۰**
- تعداد سطور چندبرچسبی (Multi-label rows): **۰**
- سطور خام با قواعد قطعی به دو ستون `Class` (نام استاندارد کلاس با املای رسمی `K_Scatch`) و `Class_ID` (شناسه ۰ تا ۶) تبدیل شدند. گزارش ممیزی در `results/target_quality_report.csv` ثبت شده است.

---

## ۶. عدم توازن کلاسی (Class Imbalance)
توزیع نمونه‌ها در میان ۷ کلاس هدف به شرح زیر است:
| کلاس عیب | تعداد نمونه | درصد سهم | نسبت به بزرگ‌ترین کلاس |
| :--- | :---: | :---: | :---: |
| **Other_Faults** (کلاس غالب) | ۶۷۳ | ۳۴.۶۷٪ | ۱ : ۱ |
| **Bumps** | ۴۰۲ | ۲۰.۷۱٪ | ۱.۶۷ : ۱ |
| **K_Scatch** | ۳۹۱ | ۲۰.۱۴٪ | ۱.۷۲ : ۱ |
| **Z_Scratch** | ۱۹۰ | ۹.۷۹٪ | ۳.۵۴ : ۱ |
| **Pastry** | ۱۵۸ | ۸.۱۴٪ | ۴.۲۶ : ۱ |
| **Stains** | ۷۲ | ۳.۷۱٪ | ۹.۳۵ : ۱ |
| **Dirtiness** (کوچک‌ترین کلاس) | ۵۵ | ۲.۸۳٪ | **۱۲.۲۴ : ۱** |

به دلیل نسبت **۱۲.۲۴ به ۱** میان بزرگ‌ترین و کوچک‌ترین کلاس، معیارهای متعارف نظیر Accuracy گمراه‌کننده‌اند و معیارهای اصلی **Macro F1** و **Balanced Accuracy** برگزیده شدند.

---

## ۷. معماری پروژه و ساختار مخزن (Project Architecture & Structure)
معماری پروژه مبتنی بر خط لوله جریان مستقیم و ماژولار است:
```text
project/
│
├── README.md                          # مستند اصلی (جدول اعضا در خط نخست)
├── ARCHITECTURE.md                    # مستند تفصیلی معماری و توجیهات نظری
├── FINAL_PROJECT_AUDIT.md             # ماتریس ممیزی ۱۰۰٪ انطباق با صورت پروژه
├── requirements.txt                   # نسخه‌های واقعی و قطعی کتابخانه‌ها
│
├── data/
│   ├── raw/                           # داده‌های خام دست‌نخورده UCI
│   │   ├── Faults.NNA
│   │   └── Faults27x7_var
│   ├── processed/                     # داده تبدیل‌شده به چندکلاسه
│   │   └── processed_dataset.csv
│   └── data_dictionary.csv            # فرهنگ داده کامل ۲۷ ویژگی و ۷ هدف
│
├── notebooks/
│   ├── 01_analyst_full_analysis.ipynb             # نوت‌بوک کامل تحلیل‌گر (اجراشده با خروجی)
│   ├── 02_final_model_pipeline.ipynb              # نوت‌بوک ماژولار نهایی پروژه (اجراشده با خروجی)
│   └── final_steel_plate_fault_classification.ipynb # نوت‌بوک مستقل و کامل تحویل نهایی
│
├── src/
│   ├── utils.py                       # مسیرها، دانه تصادفی و ثوابت
│   ├── data_loader.py                 # لودرهای سازگار داده خام و پردازش‌شده
│   ├── data_validation.py             # آزمون کیفیت هدف و آزمون‌های هندسی
│   ├── preprocessing.py               # تقسیم لایه‌بندی‌شده و ترانسفورمر مقاوم
│   ├── feature_selection.py           # فیلتر همبستگی خطی (|r| >= 0.90)
│   ├── genetic_feature_selection.py   # پیاده‌سازی الگوریتم ژنتیک چند-دانه‌ای
│   ├── models.py                      # کارخانه مدل‌های خط پایه، خطی و غیرخطی
│   ├── tuning.py                      # جست‌وجوی ابرپارامتر و تحلیل شکاف تعمیم
│   ├── noise_analysis.py              # تحلیل نقاط پرت و آزمایش نویز گاوسی
│   ├── evaluation.py                  # ماتریس درهم‌ریختگی، معیارهای کلاسی و خطایابی
│   ├── reporting.py                   # تولیدکننده گزارش‌های PDF فارسی با ReportLab
│   ├── build_report.py                # اسکریپت بیلد هر دو گزارش PDF
│   └── ga_selector.py                 # ماژول حفظ سازگاری پیاده‌سازی پیشین
│
├── results/
│   ├── experiments.csv                # جدول مقایسه کامل تمام آزمایش‌ها
│   ├── genetic_selection.csv          # سابقه تکاملی و نتایج ۳ دانه تصادفی GA
│   ├── final_test_metrics.json        # نتایج نهایی ارزیابی تک‌باره روی Test
│   ├── per_class_metrics.csv          # معیارهای دقیق تفکیکی ۷ کلاس
│   ├── confusion_matrix.csv           # ماتریس درهم‌ریختگی نهایی
│   ├── cost_sensitive_classes.csv     # پایش کلاس‌های پرهزینه Stains و Dirtiness
│   ├── class_overlap_analysis.csv     # ویژگی‌های تفکیک‌کننده جفت‌های پراشتباه
│   ├── noise_sensitivity.csv          # نتایج تزریق نویز گاوسی
│   ├── hyperparameter_default_vs_tuned.csv # شکاف Train-CV
│   ├── ga_feature_selection_results.csv
│   ├── ga_feature_frequency.csv
│   ├── ga_subset_overlap.csv
│   ├── figures/                       # کلیه نمودارهای خروجی
│   └── confusion_matrix/              # پوشه ماتریس درهم‌ریختگی
│
├── reports/
│   ├── analyst_report.pdf             # گزارش جامع ۸ صفحه‌ای فارسی
│   └── final_defense_answers.pdf      # پاسخ تشریحی به ۶ پرسش دفاع نهایی
│
└── presentation/
    └── presentation_structure.md      # راهنمای ارائه ۱۰ دقیقه‌ای اعضای تیم
```

---

## ۸. نصب و راه‌اندازی (Installation & Environment)

### پیش‌نیازها
- Python 3.12.4 (نسخه محیط اجرای ثبت‌شده نهایی)
- ابزار مدیریت بسته `pip`

### نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### نسخه‌های دقیق کتابخانه‌ها (محیط اجرای واقعی)
```text
Python 3.12.4
numpy==2.5.3
pandas==3.0.5
scikit-learn==1.9.1
matplotlib==3.11.2
seaborn==0.13.2
reportlab==5.0.1
Pillow==12.3.0
arabic-reshaper==3.0.0
python-bidi==0.6.6
pymupdf==1.25.1
pypdf==5.1.0
jupyter==1.1.1
nbconvert==7.17.1
```

---

## ۹. نحوه اجرا و ترتیب نوت‌بوک‌ها (How to Run)

### اجرای گام‌به‌گام از طریق اسکریپت‌های ماژولار:
```bash
# ۱. اعتبارسنجی داده خام و ساخت فرهنگ داده
python src/data_validation.py

# ۲. اجرای آزمون‌های آماری، نمودارهای پراکندگی و نقاط پرت (Stage 2)
python src/stage2_analysis.py

# ۳. تنظیم ابرپارامترها، الگوریتم ژنتیک و گزینش مدل (Stage 3)
python src/stage3_analysis.py

# ۴. تولید هر دو گزارش PDF دانشگاهی (گزارش ۸ صفحه‌ای و پاسخ‌های دفاع)
python src/build_report.py
```

### ترتیب اجرای نوت‌بوک‌ها (Jupyter Notebooks):
1. **`notebooks/01_analyst_full_analysis.ipynb`**: نوت‌بوک جامع مراحل سه‌گانه تحلیل‌گر که کل جریان داده را اجرا کرده و شکل‌ها و نتایج میانی را نمایش می‌دهد.
2. **`notebooks/02_final_model_pipeline.ipynb`**: نوت‌بوک تولیدی مدرن و ماژولار که از توابع پکیج `src` استفاده کرده و فرآیند کامل را بازتولید می‌کند.
3. **`notebooks/final_steel_plate_fault_classification.ipynb`**: نوت‌بوک مستقل تحویل نهایی که کل زنجیره پروژه را بدون اتکا به اجرای Notebook دیگری اجرا می‌کند و شامل انتخاب مدل، GA، حساسیت نویز و ارزیابی نهایی است.

هر سه نوت‌بوک به طور کامل از ابتدا تا انتها بدون خطا اجرا شده و خروجی‌های اصلی و نمودارهای موردنیاز درون آن‌ها ذخیره شده‌اند.

---

## ۱۰. راهبرد تقسیم داده و ممیزی نشت (Train / Validation / Test & Leakage Audit)
- **تقسیم لایه‌بندی‌شده (Stratified 70 / 15 / 15)**:
  - **Train**: ۱٬۳۵۸ نمونه (۶۹.۹۶٪)
  - **Validation**: ۲۹۱ نمونه (۱۴.۹۹٪)
  - **Test**: ۲۹۲ نمونه (۱۵.۰۴٪)
- **ایزوله‌سازی مطلق مجموعه Test**:
  - هیچ داده‌ای از Test در هیچ‌یک از مراحل پیش‌پردازش، تعیین حدود پرت، فیلتر ویژگی، تکامل ژنتیکی یا تنظیم ابرپارامتر وارد نشده است.
  - کلیه مقیاس‌بندی‌ها (`StandardScaler`, `RobustScaler`) و ترانسفورمر برش مقاوم (`TrainWinsorizer`) منحصراً پارامترهای خود را روی داده‌های آموزش یا هر Fold یاد می‌گیرند و نشت اطلاعات از آینده به طور کامل مسدود شده است.

---

## ۱۱. تحلیل نقاط پرت و حساسیت نویز (Outlier & Noise Strategy)
- **اصل عدم حذف کورکورانه**: داده‌های دورافتاده صنعتی حذف نشدند، چرا که بسیاری از آن‌ها عیوب بحرانی اما نادر بودند.
- **راهبردهای مقاوم (آزمایش Robustness)**: `RobustScaler` و `Winsorization` فقط برای بررسی حساسیت مدل به نویز و نقاط پرت آزمایش شدند و در Pipeline مدل نهایی استفاده نشده‌اند. مدل نهایی انتخاب‌شده `RandomForest_Nonlinear + All_27 Features` است.
- **آزمایش کنترل‌شده نویز**: نویز گاوسی با سطوح ۰٪، ۱٪، ۵٪ و ۱۰٪ نسبت به انحراف معیار آموزش به ویژگی‌های پیوسته مجموعه Validation تزریق شد. نتایج ثبت‌شده در `results/noise_sensitivity.csv` نشان می‌دهد که بازخوانی کلاس اقلیت `Dirtiness` در سطوح نویز ۰٪، ۱٪، ۵٪ و ۱۰٪ به ترتیب برابر ۰.۸۸۸۹، ۰.۷۷۷۸، ۰.۶۶۶۷ و ۰.۶۶۶۷ بوده و برای کلاس `Stains` در هر چهار سطح برابر ۰.۸۱۸۲ ثبت گردید.

---

## ۱۲. مدل‌ها و تنظیم ابرپارامترها (Models & Tuning)
- **خط پایه اکثریت (Majority Baseline)**: دقت ۳۴.۶۷٪ و Macro F1 برابر ۰.۰۷۳۶ (اثبات ضرورت یادگیری ماشین).
- **مدل تفسیرپذیر (Interpretable Model)**: رگرسیون لجستیک منظم‌شده با جریمه L2 و وزن‌دهی کلاسی متعادل.
- **مدل غیرخطی (Nonlinear Model)**: جنگل تصادفی (Random Forest) با زیرنمونه‌گیری متوازن (`balanced_subsample`).
- **تنظیم ابرپارامترها**: جست‌وجوی تصادفی ۵-فولدی (`RandomizedSearchCV`) با معیار `Macro F1` فقط روی Train اجرا شد.
  - Logistic: `C ∈ {0.1, 0.5, 1, 2, 5}` و `solver=lbfgs`؛ بودجه = ۵ پیکربندی.
  - Random Forest: `n_estimators ∈ {50, 80}`، `min_samples_leaf ∈ {1, 2, 4}` و `max_features ∈ {sqrt, 0.5}`؛ بودجه = ۳ پیکربندی.
  - seed جست‌وجو = `2026` و seed مربوط به Shuffle در Stratified 5-Fold = `42`. زمان اجرا باید از فایل `results/hyperparameter_search.csv` خوانده شود و مقدارهای موجود در گزارش‌های قدیمی به عنوان مرجع استفاده نشوند. شکاف Train-CV نیز برای کنترل بیش‌برازش گزارش شده است.

---

## ۱۳. انتخاب ویژگی با الگوریتم ژنتیک (Genetic Algorithm Details)
- **نمایش کروموزوم**: بردار دودویی ۲۷ بیتی ($c \in \{0, 1\}^{27}$)، که در آن ۱ به معنی انتخاب ویژگی و ۰ به معنی حذف آن است. ستون‌های هدف هرگز بخشی از کروموزوم نبوده‌اند.
- **تابع برازندگی با جریمه ابعاد**:
  $$\text{Fitness} = \text{CV Macro F1} - 0.01 \times \left(\frac{N_{\text{selected}}}{27}\right)$$
  این تابع بهینه‌سازی را به سمت دست‌یابی به بالاترین Macro F1 با کمترین تعداد حسگر/ویژگی سوق می‌دهد.
- **عملگرهای تکاملی**:
  - اندازه جمعیت: ۶
  - تعداد نسل‌ها: ۳ نسل
  - روش انتخاب: انتخاب تورنمنتی با $k = 3$
  - عملگر تقاطع: Single-point Crossover با احتمال ۰.۸۰
  - عملگر جهش: Bit-flip Mutation با احتمال ۰.۰۴ و تضمین حداقل ۳ ویژگی فعال
  - نخبه‌گرایی: انتقال مستقیم ۲ کروموزوم برتر به نسل بعد
- **پایداری در ۳ دانه تصادفی مستقل (Seeds 11, 22, 33)**:
  - تعداد ویژگی‌های منتخب: ۱۸ تا ۱۹ ویژگی (کاهش حدود ۳۰٪ ابعاد).
  - شباهت ژاکارد جفت‌زیرمجموعه‌ها: بین **۰.۴۸۰۰ تا ۰.۶۰۸۷** (Seed 11 ↔ 22: 0.4800, Seed 11 ↔ 33: 0.5833, Seed 22 ↔ 33: 0.6087).
  - انحراف معیار امتیاز F1 در اجراها: تنها **۰.۰۱۰۶**.
  - ویژگی‌های پایدار ۱۰۰٪ (انتخاب در تمام ۳ دانه — ۱۰ ویژگی): `Edges_Index`, `Edges_Y_Index`, `Empty_Index`, `LogOfAreas`, `Luminosity_Index`, `Minimum_of_Luminosity`, `Outside_Global_Index`, `Square_Index`, `Steel_Plate_Thickness`, `TypeOfSteel_A400`.

---

## ۱۴. مقایسه روش‌های انتخاب ویژگی (All Features vs Filter vs GA)

انتخاب ویژگی Filter و GA فقط با داده‌های Train انجام شده است. برای جلوگیری از خوش‌بینی ناشی از بازاستفاده از Train در رتبه‌بندی، پس از ساخت زیرمجموعه‌ها، هر کاندیدا یک‌بار روی Train برازش و روی Validation مستقل ارزیابی شد. Test تا پایان این مرحله کاملاً دست‌نخورده باقی ماند.

| کاندیدا | تعداد ویژگی | مدل | Validation Macro F1 | Validation Balanced Acc |
| :--- | :---: | :--- | :---: | :---: |
| **All_27** | **27** | **RandomForest_Nonlinear** | **0.8288** | **0.8026** |
| GA_33 | 19 | RandomForest_Nonlinear | 0.8168 | 0.7819 |
| Filter_0.90 | 21 | RandomForest_Nonlinear | 0.8002 | 0.7783 |
| GA_22 | 18 | RandomForest_Nonlinear | 0.7854 | 0.7669 |
| GA_11 | 19 | RandomForest_Nonlinear | 0.7706 | 0.7429 |
| All_27 | 27 | Logistic_Interpretable | 0.6444 | 0.7336 |

**نتیجه:** در این داده و با انتخاب leakage-safe، هیچ زیرمجموعه Filter/GA عملکرد Validation بالاتری از All_27 نداشت. بنابراین برای مدل نهایی، همه ۲۷ ویژگی نگه داشته شدند. این نتیجه به معنی شکست الگوریتم ژنتیک نیست؛ GA اجرا، پایداری و زیرمجموعه‌های آن گزارش شده‌اند، اما شواهد کافی برای حذف ویژگی بدون افت عملکرد ارائه نمی‌کنند.

---

> **نکته مهم روش‌شناختی:** Cross-Validation روی Train برای خودِ جست‌وجوی ابرپارامتر و Fitness الگوریتم ژنتیک استفاده شده است. اما بعد از اینکه Filter/GA زیرمجموعه را از Train استخراج کردند، همان Train دوباره برای رتبه‌بندی نهایی کاندیداها با Cross-Validation استفاده نمی‌شود. رتبه‌بندی نهایی روی Validation مستقل انجام می‌شود و Test فقط یک‌بار در انتها ارزیابی می‌شود.

## ۱۵. نتایج ارزیابی نهایی روی مجموعه آزمون دست‌نخورده (Final Test Results)

پس از انتخاب `RandomForest_Nonlinear + All_27` بر اساس Validation، مدل نهایی روی ترکیب Train+Validation شامل ۱٬۶۴۹ نمونه برازش شد و سپس **یک‌بار** روی ۲۹۲ نمونه Test دست‌نخورده ارزیابی شد.

> **نکته درباره `results/experiments.csv`:** ستون Test برای آزمایش‌های غیرنهایی عمداً خالی است؛ چون طبق صورت پروژه Test نباید در رتبه‌بندی کاندیداها استفاده شود. فقط ردیف `EXP_07_FINAL_FROZEN_PIPELINE` پس از فریز انتخاب‌ها یک بار روی Test ارزیابی شده است.

| معیار ارزیابی | مقدار در Test |
| :--- | :---: |
| **Accuracy** | **81.16%** |
| **Balanced Accuracy** | **81.97%** |
| **Macro F1 Score** | **82.98%** |
| **Weighted F1 Score** | **81.25%** |
| **Macro Precision** | **84.11%** |
| **Macro Recall** | **81.97%** |

### عملکرد تفکیکی ۷ کلاس هدف در آزمون نهایی:

| نام کلاس عیب | Precision | Recall | F1 Score | Support | FN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pastry** | 0.545 | 0.522 | 0.533 | 23 | 11 |
| **Z_Scratch** | 0.963 | 0.897 | 0.929 | 29 | 3 |
| **K_Scatch** | 0.965 | 0.932 | 0.948 | 59 | 4 |
| **Stains** | 0.900 | 0.818 | 0.857 | 11 | 2 |
| **Dirtiness** | 1.000 | **1.000** | **1.000** | 8 | **0** |
| **Bumps** | 0.762 | 0.787 | 0.774 | 61 | 13 |
| **Other_Faults** | 0.752 | 0.782 | 0.767 | 101 | 22 |

## ۱۶. تحلیل جفت‌های پراشتباه و کلاس‌های پرهزینه (Misclassifications)

1. **Bumps ↔ Other_Faults:** مجموعاً **۲۲ خطای رفت‌وبرگشتی**؛ ۱۰ مورد Bumps به‌عنوان Other_Faults و ۱۲ مورد Other_Faults به‌عنوان Bumps.
2. **Pastry ↔ Other_Faults:** مجموعاً **۱۵ خطای رفت‌وبرگشتی**؛ ۸ مورد Pastry به‌عنوان Other_Faults و ۷ مورد Other_Faults به‌عنوان Pastry.
3. این تحلیل از ماتریس خطای واقعی Test استخراج شده است؛ توضیح ویژگی‌های جداکننده صرفاً توصیفی است و نباید از نمودار دو ویژگی نتیجه مدل‌سازی استنتاج شود.
4. برای سناریوی پرهزینه، `Dirtiness` و `Stains` به‌عنوان کلاس‌های با اولویت بازرسی تعریف شده‌اند. در Test، Recall برای Dirtiness برابر **1.000** (صفر FN) و برای Stains برابر **0.818** (۲ FN) است.

## ۱۷. استقرار صنعتی و بازرس انسانی (Human-in-the-Loop)
این سامانه برای استقرار صنعتی با معماری سه‌سطحی طراحی شده است:
1. **دسته‌بندی خودکار**: نمونه‌هایی با احتمال پیش‌بینی بالای ۸۵٪ در دسته‌های استاندارد قرار می‌گیرند.
2. **ارجاع به بازرس انسانی**: پیش‌بینی‌های با عدم‌قطعیت بالا یا متعلق به کلاس‌های پرهزینه به مانیتور بازرس کیفی ارجاع داده می‌شوند تا از خطای طبقه‌بندی جلوگیری شود.
3. **پایش تغییر دامنه (Drift Monitoring - پیشنهاد استقرار آینده)**: در نسخه فعلی پیاده‌سازی KS Monitoring وجود ندارد؛ این مورد فقط به عنوان پیشنهاد برای استقرار آینده ذکر می‌شود.

---

## ۱۸. فایل‌های نتایج و بازتولیدپذیری (Results Files & Reproducibility)
کلیه آزمایش‌ها با ثبت دانه تصادفی در فایل‌های زیر مستند شده‌اند:
- `results/experiments.csv`: کاتالوگ جامع تمام آزمایش‌های خط پایه، فیلتر، ژنتیک و ارزیابی نهایی Test.
- `results/genetic_selection.csv`: نتایج ۳ دانه تصادفی GA شامل ویژگی‌ها، برازندگی و زمان اجرا.
- `results/confusion_matrix.csv`: ماتریس درهم‌ریختگی عددی آزمون نهایی.
- `reports/analyst_report.pdf`: گزارش رسمی و تفصیلی ۸ صفحه‌ای تحلیل‌گر و معمار.
- `reports/final_defense_answers.pdf`: پاسخ‌های مستند فارسی به ۶ پرسش رسمی دفاع نهایی.

### دانه‌های تصادفی ثبت‌شده:
- Global / Split Seed: `42`
- Hyperparameter Search Seed: `2026`
- GA Evaluation Seeds: `11`, `22`, `33`
- زمان تقریبی اجرای کامل خط لوله: حدود **۳۰ ثانیه** روی پردازنده استاندارد.

---

## ۱۹. تفکیک دقیق مسئولیت‌های اعضای تیم (Team Responsibilities)
Melina Salemi>>> مسئول مدیریت پروژه و نظارت و برنامه ریزی روند پروژه
Kiana Sarkari>>> مسئول معماری پروژه و طراحی ساختار پروژه ، تعیین ارتباط بخش ها و کد نویسی بخش های اصلی پروژه
Marvel Keshtkar>>> جمع آوری و تحلیل داده ها، پیش پردازش، بررسی نتایج
## ۲۰. استناد و مجوز دیتاست (Attribution & Citation)
```bibtex
@misc{uci_steel_plates_faults_198,
  author       = {Buscaldi, Marco and Fiorini, Aldo and Sassi, Roberto},
  title        = {{Steel Plates Faults}},
  year         = {2010},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5DG6Z}
}
```
دیتاست تحت مجوز [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) منتشر شده است.

---

## ۱۳. کنترل‌های نهایی تحویل (Final Compliance Notes)

- مقایسه انتخاب ویژگی ژنتیکی (GA) باید در کنار یک روش فیلتر ساده مانند SelectKBest/Mutual Information گزارش شود. در نسخه فعلی، این بخش به عنوان کنترل تکمیلی تحویل ثبت شده است.
- دو زوج کلاس با بیشترین خطای رفت و برگشت باید از Confusion Matrix استخراج و در گزارش نهایی توضیح داده شوند.
- زمان اجرای کامل هر Notebook باید در محیط تحویل نهایی ثبت و در فایل `results/notebook_runtime.csv` نگهداری شود.
- مسئولیت اعضای تیم و بازبین هر بخش باید قبل از ارسال نهایی توسط اعضای تیم تکمیل شود.
