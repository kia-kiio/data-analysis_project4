# گزارش کنترل کیفیت نهایی — نسخه اصلاح‌شده

## اصلاحات ضروری
- جفت‌های `Bumps ↔ Other_Faults` با **۲۲ خطای رفت‌وبرگشتی** (۱۰ مورد Bumps به Other و ۱۲ مورد برعکس) و `Pastry ↔ Other_Faults` با **۱۵ خطای رفت‌وبرگشتی** (۸ مورد Pastry به Other و ۷ مورد برعکس) در ارزیابی نهایی استخراج شد؛ این اعداد با `class_overlap_analysis.csv` و Confusion Matrix نهایی هم‌خوان هستند.
- README تکمیل شد: لینک رسمی UCI، روش اجرا، نسخه دقیق کتابخانه‌ها، seedها، زمان اجرای واقعی Stageها و مسئولیت هر سه عضو اضافه شد.
- Notebook از `subprocess.run` به `runpy.run_path` تغییر کرد تا اجرای End-to-End در همان Kernel انجام شود؛ Notebook با nbconvert از ابتدا تا انتها با موفقیت اجرا و خروجی‌ها ذخیره شد.
- هر دو `scatter_plot_1.png` و `scatter_plot_2.png` داخل PDF قرار گرفتند.

## اصلاحات مهم
- تحلیل دو جفت پراشتباه با تعداد خطاهای رفت‌وبرگشتی و پنج ویژگی جداساز Train در PDF اضافه شد.
- اختلاف Accuracy و Balanced Accuracy در PDF تفسیر شد.
- تنها implementation عملیاتی GA در `src/ga_selector.py` متمرکز شد و Stage 3 از همان Tournament/Crossover/Mutation استفاده می‌کند.
- تنظیمات کامل GA شامل chromosome، population، generations، elitism، tournament، crossover، mutation، minimum features، fitness، seed و CV در README و `results/ga_configuration.json` ثبت شد.
- جدول percentileهای 25/50/75/90/95 برای هشت ویژگی در PDF اضافه شد.
- `feature_distributions.png` به subplotهای مستقل اصلاح شد.
- نمودارهای اصلی در Notebook نیز inline نمایش داده می‌شوند.

## کنترل نهایی
- PDF: **8 صفحه**.
- Notebook: End-to-End با موفقیت اجرا شد و خروجی‌های اصلی باقی مانده‌اند.
- سه Stage در محیط بررسی: حدود 1.2s + 5.5s + 20.8s ≈ **28s**.
- GA: سه seed (11, 22, 33)، 27 ژن، 3-fold CV روی Train و fitness شامل Macro F1 و penalty.
- Test فقط پس از تثبیت انتخاب‌ها استفاده شده است.

- کنترل تکمیلی: مقایسه Default/Tuned با همان 5-fold Train-CV اضافه شد؛ جدول در `results/hyperparameter_default_vs_tuned.csv` است.
- کنترل تکمیلی: Noise Sensitivity و Recall کلاس‌های کوچک در گزارش PDF/Notebook قابل مشاهده‌اند.
- کنترل تکمیلی: False Negative کلاس‌های پرهزینه و پایداری GA (Jaccard/پراکندگی امتیاز) در PDF اضافه شدند.
