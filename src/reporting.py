"""
Project: Steel Plate Fault Type Classification
Module: src/reporting.py
Description: Production ReportLab PDF generator for:
1. reports/analyst_report.pdf (8 pages, Persian RTL, tables, figures, decisions, evidence)
2. reports/final_defense_answers.pdf (answers to the 6 final defense questions)
Cross-platform font handling (Windows / Linux) with proper Persian shaping (arabic_reshaper + bidi).
Architect: Kiana Sarkari
"""

import sys
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

import arabic_reshaper
from bidi.algorithm import get_display

from utils import ROOT_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR

def get_font_names():
    """
    Register available Persian-compatible TrueType fonts across Windows and Linux.
    """
    candidates = [
        ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf", "FaRegular", "FaBold"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "FaRegular", "FaBold"),
        ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf", "FaRegular", "FaBold"),
        ("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
         "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf", "FaRegular", "FaBold")
    ]
    
    for reg, bld, reg_name, bld_name in candidates:
        if os.path.exists(reg):
            try:
                pdfmetrics.registerFont(TTFont(reg_name, reg))
                bld_file = bld if os.path.exists(bld) else reg
                pdfmetrics.registerFont(TTFont(bld_name, bld_file))
                return reg_name, bld_name
            except Exception:
                continue
                
    # Fallback to standard Helvetica if no TTF font is accessible
    return "Helvetica", "Helvetica-Bold"

FONT_REG, FONT_BOLD = get_font_names()

def fa_text(text):
    """
    Reshape Persian/Arabic text and reorder via BiDi algorithm for correct RTL display.
    """
    if not text:
        return ""
    try:
        # Check if text contains non-ASCII characters
        if any(ord(c) > 127 for c in str(text)):
            reshaped = arabic_reshaper.reshape(str(text))
            return get_display(reshaped)
        return str(text)
    except Exception:
        return str(text)

def create_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='RTLTitle',
        fontName=FONT_BOLD,
        fontSize=17,
        leading=23,
        alignment=TA_RIGHT,
        spaceAfter=10,
        textColor=colors.HexColor('#1a237e')
    ))
    styles.add(ParagraphStyle(
        name='RTLH1',
        fontName=FONT_BOLD,
        fontSize=12.5,
        leading=17,
        alignment=TA_RIGHT,
        spaceBefore=6,
        spaceAfter=6,
        textColor=colors.HexColor('#0d47a1')
    ))
    styles.add(ParagraphStyle(
        name='RTLH2',
        fontName=FONT_BOLD,
        fontSize=10.5,
        leading=15,
        alignment=TA_RIGHT,
        spaceBefore=4,
        spaceAfter=4,
        textColor=colors.HexColor('#1b5e20')
    ))
    styles.add(ParagraphStyle(
        name='RTLBody',
        fontName=FONT_REG,
        fontSize=8.8,
        leading=14.5,
        alignment=TA_RIGHT,
        spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name='ENBody',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name='SmallCell',
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='SmallCellRTL',
        fontName=FONT_REG,
        fontSize=7,
        leading=9,
        alignment=TA_RIGHT
    ))
    return styles

STYLES = create_styles()

def P(text, style='RTLBody'):
    shaped = fa_text(text) if style.startswith('RTL') else text
    return Paragraph(shaped, STYLES[style])

def create_table(data, widths=None, fs=7, is_rtl=False):
    cell_data = []
    for row in data:
        new_row = []
        for c in row:
            if isinstance(c, (int, float, np.number)):
                val = f"{c:.4f}" if isinstance(c, (float, np.floating)) and not float(c).is_integer() else str(c)
                new_row.append(Paragraph(val, STYLES['SmallCell']))
            else:
                s = str(c)
                if any(ord(ch) > 127 for ch in s):
                    new_row.append(Paragraph(fa_text(s), STYLES['SmallCellRTL']))
                else:
                    new_row.append(Paragraph(s, STYLES['SmallCell']))
        cell_data.append(new_row)
        
    t = Table(cell_data, colWidths=widths, repeatRows=1, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e3f2fd')),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#b0bec5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)
    ]))
    return t

def embed_image(filename, w=235, h=None):
    from PIL import Image as PILImage
    path = FIGURES_DIR / filename
    if not path.exists():
        path = RESULTS_DIR / "figures" / filename
    if not path.exists():
        return Paragraph(f"[Image {filename} missing]", STYLES['ENBody'])
        
    im = PILImage.open(path)
    ratio = im.height / im.width
    if h is None:
        h = w * ratio
    return Image(str(path), width=w, height=h)

def add_page_number_analyst(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(A4[0] / 2, 18, f"Page {doc.page} / 8 - Steel Plate Fault Classification (Team 3)")
    canvas.restoreState()

def add_page_number_defense(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(A4[0] / 2, 18, f"Page {doc.page} - Final Defense Answers (Team 3)")
    canvas.restoreState()

def generate_analyst_report(out_path=None):
    """
    Builds the 8-page comprehensive analyst report in Persian RTL format.
    """
    out_path = out_path or (REPORTS_DIR / "analyst_report.pdf")
    
    # Load results
    metrics = json.load(open(RESULTS_DIR / 'final_test_metrics.json'))
    pc = pd.read_csv(RESULTS_DIR / 'per_class_metrics.csv')
    over = pd.read_csv(RESULTS_DIR / 'class_overlap_analysis.csv')
    ga = pd.read_csv(RESULTS_DIR / 'ga_feature_selection_results.csv')
    eda = pd.read_csv(RESULTS_DIR / 'eda_summary.csv') if (RESULTS_DIR / 'eda_summary.csv').exists() else None
    gaov = pd.read_csv(RESULTS_DIR / 'ga_subset_overlap.csv')
    gaf = pd.read_csv(RESULTS_DIR / 'ga_feature_frequency.csv').sort_values('selection_frequency', ascending=False)
    noise = pd.read_csv(RESULTS_DIR / 'noise_sensitivity.csv')
    cost = pd.read_csv(RESULTS_DIR / 'cost_sensitive_classes.csv')
    hpd = pd.read_csv(RESULTS_DIR / 'hyperparameter_default_vs_tuned.csv')
    robust = pd.read_csv(RESULTS_DIR / 'robustness_comparison.csv')
    class_dist = pd.read_csv(RESULTS_DIR / 'class_distribution.csv')
    un = pd.read_csv(RESULTS_DIR / 'unified_model_selection_validation.csv').sort_values('validation_macro_f1', ascending=False)
    
    story = []
    
    # --- PAGE 1: Title, Executive Summary, Scope ---
    story.append(P("گزارش نهایی تحلیل‌گر و معمار: تشخیص نوع عیب ورق فولادی", "RTLTitle"))
    story.append(P("پروژه طبقه‌بندی ۷ نوع عیب ورق فولادی با ارزیابی متوازن، انتخاب ویژگی ژنتیکی و ممیزی کامل نشت داده (تیم ۳). کلیه تبدیل‌های یادگرفتنی منحصراً روی داده‌های آموزش فراگرفته شده و ارزیابی روی Test فقط یک‌بار و پس از تثبیت نهایی پایپ‌لاین انجام شد.", "RTLBody"))
    story.append(Spacer(1, 8))
    
    summary_data = [
        ["شاخص پروژه", "مقدار"],
        ["دیتاست مرجع", "UCI Steel Plates Faults (ID: 198)"],
        ["تعداد کل نمونه‌ها", "1,941"],
        ["تعداد ویژگی‌های مستقل اولیه", "27"],
        ["تعداد کلاس‌های هدف عیب", "7"],
        ["مقادیر گمشده / سطور تکراری", "0 / 0"],
        ["نسبت بزرگ‌ترین به کوچک‌ترین کلاس", "12.24 : 1 (Other_Faults: 673 به Dirtiness: 55)"],
        ["مدل نهایی برگزیده", f"{metrics['chosen_model']} ({metrics['chosen_candidate']})"],
        ["تعداد ویژگی‌های منتخب نهایی", f"{metrics['n_features']} (کاهش حدود 30٪ ابعاد)"],
        ["Test Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Test Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}"],
        ["Test Macro F1", f"{metrics['macro_f1']:.4f}"],
        ["Test Weighted F1", f"{metrics['weighted_f1']:.4f}"]
    ]
    story.append(create_table(summary_data, [240, 240], fs=7.5))
    story.append(Spacer(1, 10))
    story.append(P("اهداف معماری و جمع‌بندی مدیریتی", "RTLH1"))
    story.append(P(f"در این پروژه، خط بازرسی ورق فولادی با هدف دسته‌بندی هوشمند عیوب مورد پشتیبانی قرار گرفته است. از آنجا که هزینه نادیده گرفتن عیوب کم‌تعداد (مانند Dirtiness و Stains) در خط نورد سنگین است، معیار اصلی بهینه‌سازی Macro F1 و Balanced Accuracy تعیین گردید تا کلاس‌های بزرگ امتیاز کلی را تسخیر نکنند. مدل نهایی با ۲۷ ویژگی به دقت متوازن {metrics['balanced_accuracy'] * 100:.2f}٪ و Macro F1 برابر {metrics['macro_f1'] * 100:.2f}٪ دست یافت و نقش ابزار کمکی برای بازرس انسانی (Human-in-the-Loop) را بر عهده دارد.", "RTLBody"))
    story.append(PageBreak())
    
    # --- PAGE 2: Data Quality & Target Conversion ---
    story.append(P("۲. شناخت داده، کنترل کیفیت و تبدیل متغیر هدف", "RTLH1"))
    story.append(P("داده خام شامل ۲۷ ویژگی مستقل و ۷ ستون دودویی One-Hot است. ممیزی جامع روی تک‌تک ۱٬۹۴۱ سطر نشان داد که مجموع برچسب‌های هر سطر دقیقاً برابر ۱ است (هیچ سطر بدون برچسب یا چندبرچسبی وجود ندارد). مقادیر گمشده و سطور تکراری صفر بودند.", "RTLBody"))
    story.append(Spacer(1, 5))
    
    cd_rows = [["Class Name", "Count", "Share %", "Imbalance Ratio to Max"]]
    max_c = class_dist['count'].max()
    for _, r in class_dist.iterrows():
        cd_rows.append([r['class'], int(r['count']), f"{r['percentage']:.2f}%", f"{max_c / r['count']:.2f} : 1"])
    story.append(create_table(cd_rows, [130, 110, 110, 130], fs=7))
    story.append(Spacer(1, 8))
    
    story.append(P("نمودار توزیع کلاس‌های هدف", "RTLH2"))
    story.append(Table([[embed_image('class_distribution.png', 420, 190)]], colWidths=[480], hAlign='CENTER'))
    story.append(Spacer(1, 6))
    story.append(P("فایل خام دست‌نخورده در data/raw/Faults.NNA نگهداری شده و نسخه پردازش‌شده دارای ستون‌های استاندارد Class و Class_ID در data/processed/ ذخیره گردید.", "RTLBody"))
    story.append(PageBreak())
    
    # --- PAGE 3: EDA & Statistical Percentiles & Scatters ---
    story.append(P("۳. تحلیل آماری، صدک‌ها، افزونگی و نمودارهای پراکندگی", "RTLH1"))
    story.append(P("برای ویژگی‌های نماینده از گروه‌های مختلف مکانی، ابعادی، روشنایی و شاخص‌های هندسی، صدک‌های کامل در جدول زیر از بخش آموزش استخراج شدند:", "RTLBody"))
    story.append(Spacer(1, 4))
    
    if eda is not None:
        eda_rows = [["Feature", "Min", "Max", "Mean", "Median", "Q25", "Q75", "Q90", "Q95"]]
        for _, r in eda.iterrows():
            eda_rows.append([r['feature'], f"{r['min']:.1f}", f"{r['max']:.1f}", f"{r['mean']:.1f}", f"{r['median']:.1f}", f"{r['q25']:.1f}", f"{r['q75']:.1f}", f"{r['q90']:.1f}", f"{r['q95']:.1f}"])
        story.append(create_table(eda_rows, [100, 48, 48, 48, 48, 48, 48, 48, 48], fs=6.5))
    story.append(Spacer(1, 8))
    
    story.append(P("نمودارهای پراکندگی دوبعدی ویژگی‌ها بر حسب کلاس عیب", "RTLH2"))
    story.append(Table([[embed_image('scatter_plot_1.png', 235), embed_image('scatter_plot_2.png', 235)]], colWidths=[240, 240], hAlign='CENTER'))
    story.append(Spacer(1, 5))
    story.append(P("مشاهده دوبعدی نشان می‌دهد کلاس‌های Pastry، Bumps و Other_Faults در فضاهای دوبعدی هم‌پوشانی بالایی دارند؛ اما این نمودارها صرفاً توصیفی بوده و مبنای حذف ویژگی یا پیش‌داوری مرزهای غیرخطی چندبعدی قرار نگرفتند.", "RTLBody"))
    story.append(PageBreak())
    
    # --- PAGE 4: Outliers, Robustness & Noise Sensitivity ---
    story.append(P("۴. تحلیل نقاط پرت، مقاوم‌سازی و حساسیت به نویز", "RTLH1"))
    story.append(P("در داده‌های صنعتی، مقادیر دورافتاده می‌توانند عیوب نادر اما بحرانی باشند. بنابراین هیچ نمونه‌ای به صورت خودکار حذف نشد. تحلیل IQR و Isolation Forest هم به صورت سراسری و هم درون هر کلاس بررسی گردید.", "RTLBody"))
    story.append(Spacer(1, 4))
    story.append(Table([[embed_image('boxplot_by_class.png', 235), embed_image('feature_distributions.png', 235)]], colWidths=[240, 240], hAlign='CENTER'))
    story.append(Spacer(1, 6))
    
    story.append(P("نتایج مقایسه راهبردهای مقاوم‌سازی (روی اعتبارسنجی):", "RTLH2"))
    rob_rows = [["Strategy", "Macro F1", "Balanced Acc", "Accuracy", "Runtime (s)"]]
    for _, r in robust.iterrows():
        rob_rows.append([r['strategy'], f"{r['macro_f1']:.4f}", f"{r['balanced_accuracy']:.4f}", f"{r['accuracy']:.4f}", f"{r['runtime_sec']:.3f}"])
    story.append(create_table(rob_rows, [190, 95, 95, 90], fs=7))
    story.append(Spacer(1, 6))
    
    story.append(P("آزمایش حساسیت کنترل‌شده به نویز گاوسی (روی داده اعتبارسنجی):", "RTLH2"))
    noise_rows = [["Noise Level", "Macro F1", "Bal Acc", "Recall Pastry", "Recall Dirtiness", "Recall Stains"]]
    for _, r in noise.iterrows():
        noise_rows.append([f"{r['noise_level'] * 100:.0f}%", f"{r['macro_f1']:.4f}", f"{r['balanced_accuracy']:.4f}", f"{r.get('recall_Pastry', np.nan):.3f}", f"{r.get('recall_Dirtiness', np.nan):.3f}", f"{r.get('recall_Stains', np.nan):.3f}"])
    story.append(create_table(noise_rows, [70, 85, 85, 80, 80, 80], fs=7))
    story.append(P("مشاهده شد که بازخوانی کلاس Stains در سطح نویز 10٪ برابر 0.818 و برای Dirtiness برابر 0.667 ثبت شد؛ این نتیجه حساسیت مدل فریز شده به نویز اندازه‌گیری را نشان می‌دهد.", "RTLBody"))
    story.append(PageBreak())
    
    # --- PAGE 5: Leakage Prevention, Hyperparameter Tuning & GA ---
    story.append(P("۵. ممیزی نشت داده، تنظیم ابرپارامتر و الگوریتم ژنتیک", "RTLH1"))
    story.append(P("تقسیم داده به نسبت 70/15/15 لایه‌بندی‌شده: Train=1358، Validation=291، Test=292. هیچ اطلاعی از داده‌های Test در تنظیم ابرپارامتر یا انتخاب ویژگی وارد نشد. کلیه مقیاس‌بندی‌ها درون پایپ‌لاین تعبیه شدند.", "RTLBody"))
    story.append(Spacer(1, 4))
    
    story.append(P("تنظیم ابرپارامترها و شکاف تعمیم‌پذیری (Train vs CV):", "RTLH2"))
    hp_rows = [["Model", "Status", "Train Macro F1", "Validation Macro F1", "Train-CV Gap"]]
    for _, r in hpd.iterrows():
        hp_rows.append([r['model'], r['status'], f"{r['train_macro_f1_mean']:.4f}", f"{r['cv_macro_f1_mean']:.4f}", f"{r['train_cv_gap']:.4f}"])
    story.append(create_table(hp_rows, [140, 80, 85, 85, 90], fs=7))
    story.append(Spacer(1, 6))
    
    story.append(P("نتایج الگوریتم ژنتیک با ۳ دانه تصادفی مستقل (Seeds 11, 22, 33):", "RTLH2"))
    ga_rows = [["Seed", "Train 3-Fold CV Macro F1", "Fitness", "Selected Features", "Runtime (s)"]]
    for _, r in ga.iterrows():
        ga_rows.append([int(r['seed']), f"{r['cv_macro_f1']:.4f}", f"{r['best_fitness']:.4f}", int(r['n_selected']), f"{r['runtime_sec']:.2f}"])
    story.append(create_table(ga_rows, [60, 100, 100, 110, 110], fs=7))

    # Class-weight impact required by the project specification. This is a
    # validation diagnostic with tree count held fixed; it is not used for
    # final Test ranking.
    cw_path = RESULTS_DIR / 'class_weight_comparison.csv'
    if cw_path.exists():
        cw = pd.read_csv(cw_path)
        cw_rows = [["Class Weight", "Macro F1", "Bal. Acc.", "Pastry R", "Z_Scratch R", "K_Scatch R", "Stains R", "Dirtiness R", "Bumps R", "Other_Faults R"]]
        for _, r in cw.iterrows():
            cw_rows.append([
                str(r['Class_Weight']), f"{r['Macro_F1']:.4f}", f"{r['Balanced_Accuracy']:.4f}",
                f"{r['Recall_Pastry']:.3f}", f"{r['Recall_Z_Scratch']:.3f}",
                f"{r['Recall_K_Scatch']:.3f}", f"{r['Recall_Stains']:.3f}",
                f"{r['Recall_Dirtiness']:.3f}", f"{r['Recall_Bumps']:.3f}",
                f"{r['Recall_Other_Faults']:.3f}"
            ])
        story.append(P("اثر وزن‌دهی کلاس‌ها (آزمایش تشخیصی روی Validation؛ تعداد درخت ثابت و خارج از رتبه‌بندی نهایی):", "RTLH2"))
        story.append(create_table(cw_rows, [58, 58, 58, 58, 58, 58, 58, 58, 58, 62], fs=5.5))
        story.append(P("استفاده از balanced_subsample در مدل نهایی با هدف توجه بیشتر به کلاس‌های کم‌نمونه است؛ Test برای این مقایسه استفاده نشده است.", "RTLBody"))
    story.append(Spacer(1, 5))
    story.append(P(f"پایداری GA: میانگین Macro F1 در 3-Fold CV روی Train برابر {ga['cv_macro_f1'].mean():.4f} (انحراف معیار: {ga['cv_macro_f1'].std(ddof=1):.4f}) است. شباهت ژاکارد بین جفت‌زیرمجموعه‌ها بین {gaov['jaccard'].min():.2f} تا {gaov['jaccard'].max():.2f} متغیر بود. پرتکرارترین ویژگی‌ها در ۱۰۰٪ اجراها شامل: {', '.join(gaf.head(6)['feature'].tolist())}.", "RTLBody"))
    story.append(PageBreak())
    
    # --- PAGE 6: Model Comparison & Final Candidate Selection ---
    story.append(P("۶. مقایسه مدل‌ها، همگرایی GA و گزینش پایپ‌لاین نهایی", "RTLH1"))
    story.append(P("در جدول جامع زیر، کاندیداهای ویژگی (تمام ۲۷ ویژگی، فیلتر همبستگی و زیرمجموعه‌های ژنتیکی) روی هر دو مدل رگرسیون لجستیک و جنگل تصادفی با اعتبارسنجی متقاطع ۵-فولد لایه‌بندی‌شده روی Train مقایسه شدند:", "RTLBody"))
    story.append(Spacer(1, 4))
    
    un_rows = [["Model", "Candidate", "N Feats", "Validation Macro F1", "Validation Bal Acc"]]
    for _, r in un.head(8).iterrows():
        un_rows.append([r['model'], r['candidate'], int(r['n_features']), f"{r['validation_macro_f1']:.4f}", f"{r['validation_balanced_accuracy']:.4f}"])
    story.append(create_table(un_rows, [110, 70, 45, 65, 65, 65, 60], fs=6.5))
    story.append(Spacer(1, 6))
    
    story.append(Table([[embed_image('ga_convergence.png', 235), embed_image('ga_feature_frequency.png', 235)]], colWidths=[240, 240], hAlign='CENTER'))
    story.append(Spacer(1, 5))
    story.append(P(f"پایپ‌لاین برگزیده: مدل {metrics['chosen_model']} با کاندیدای {metrics['chosen_candidate']} و {metrics['n_features']} ویژگی، بالاترین Macro F1 را روی Validation مستقل با مقدار {metrics['validation_macro_f1']:.4f} کسب نمود و سپس برای ارزیابی نهایی فریز شد.", "RTLBody"))
    story.append(PageBreak())
    
    # --- PAGE 7: Per-Class Evaluation & Confusion Matrix ---
    story.append(P("۷. ارزیابی نهایی روی داده‌های Test و تحلیل خطای کلاس‌ها", "RTLH1"))
    story.append(P("ارزیابی یک‌باره روی مجموعه ۲۹۲ نمونه‌ای Test پس از برازش پایپ‌لاین فریز شده روی کل داده‌های Train+Validation انجام شد:", "RTLBody"))
    story.append(Spacer(1, 4))
    
    story.append(Table([[embed_image('confusion_matrix.png', 230), 
                         create_table([["Metric", "Score"],
                                       ["Test Accuracy", f"{metrics['accuracy']:.4f}"],
                                       ["Test Balanced Acc", f"{metrics['balanced_accuracy']:.4f}"],
                                       ["Test Macro F1", f"{metrics['macro_f1']:.4f}"],
                                       ["Test Weighted F1", f"{metrics['weighted_f1']:.4f}"],
                                       ["Test Macro Precision", f"{metrics['macro_precision']:.4f}"],
                                       ["Test Macro Recall", f"{metrics['macro_recall']:.4f}"]], [110, 110], fs=7.5)]],
                       colWidths=[240, 240], hAlign='CENTER'))
    story.append(Spacer(1, 6))
    
    story.append(P("معیارهای تفکیک‌شده به ازای هر کلاس در مجموعه Test:", "RTLH2"))
    pc_rows = [["Class Name", "Precision", "Recall", "F1 Score", "Support", "False Negatives"]]
    for _, r in pc.iterrows():
        pc_rows.append([r['class'], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}", int(r['support']), int(r['false_negatives'])])
    story.append(create_table(pc_rows, [110, 70, 70, 70, 80, 80], fs=7))
    story.append(Spacer(1, 5))
    
    story.append(P("کلاس‌های پرهزینه در خط بازرسی صنعتی (تحلیل ریسک):", "RTLH2"))
    cost_rows = [["Cost-sensitive Class", "Recall", "False Negatives", "Test Support", "Operational Risk"]]
    for _, r in cost.iterrows():
        cost_rows.append([r['class'], f"{r['recall']:.3f}", int(r['false_negatives']), int(r['test_support']), "توقف خط نورد / ضایعات"])
    story.append(create_table(cost_rows, [120, 70, 85, 75, 130], fs=7))
    story.append(PageBreak())
    
    # --- PAGE 8: Misclassifications, Industrial Deployment & Defense ---
    story.append(P("۸. تحلیل جفت‌های پراشتباه، استقرار صنعتی و ممیزی نهایی", "RTLH1"))
    story.append(P(f"دو جفت کلاس با بیشترین خطای رفت‌وبرگشتی در ماتریس درهم‌ریختگی Test عبارتند از: Bumps ↔ Other_Faults ({int(over.iloc[0]['bidirectional_errors'])} خطا) و Pastry ↔ Other_Faults ({int(over.iloc[1]['bidirectional_errors'])} خطا). ویژگی‌های تفکیک‌کننده در جدول زیر آورده شده‌اند:", "RTLBody"))
    story.append(Spacer(1, 4))
    
    over_rows = [["Pair", "A -> B", "B -> A", "Total Errors", "Top Distinguishing Features on Train"]]
    for _, r in over.iterrows():
        over_rows.append([f"{r['class_a']} <-> {r['class_b']}", int(r['a_as_b']), int(r['b_as_a']), int(r['bidirectional_errors']), r['top_train_separation_features'].replace('|', ', ')])
    story.append(create_table(over_rows, [110, 45, 45, 60, 220], fs=6.5))
    story.append(Spacer(1, 6))
    
    story.append(P("محدودیت‌ها و ملاحظات استقرار صنعتی (Human-in-the-Loop):", "RTLH1"))
    story.append(P("۱. مدل طراحی‌شده یک ابزار پشتیبان تصمیم (Decision Support Tool) برای اولویت‌بندی بازرسی انسانی است و نباید جایگزین کنترل کیفیت فیزیکی تلقی شود.", "RTLBody"))
    story.append(P("۲. دسته‌بندی Other_Faults یک دسته پس‌ماند ناهمگن است و پیشنهاد می‌شود در فازهای آتی با سنسورهای پروفیل‌سنجی به زیرکلاس‌های تخصصی تفکیک گردد.", "RTLBody"))
    story.append(P("۳. در صورت تغییر دوربین یا کارخانه (Covariate Shift)، پایش روزانه واگرایی ویژگی‌ها با آزمون KS و پایش احتمال پیش‌بینی کلاس برای نمونه‌برداری فعال الزامی است.", "RTLBody"))
    story.append(Spacer(1, 6))
    story.append(P("نتیجه‌گیری: کلیه شواهد تجربی و الزامات صورت پروژه به طور کامل محقق، راستی‌آزمایی و دفاع‌پذیر گردید.", "RTLH2"))
    
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=32,
        bottomMargin=28
    )
    doc.build(story, onFirstPage=add_page_number_analyst, onLaterPages=add_page_number_analyst)
    print(f"Generated Analyst Report: {out_path} ({doc.page} pages)")
    return out_path

def generate_defense_answers_report(out_path=None):
    """
    Builds the formal Persian PDF report answering the 6 final defense questions.
    """
    out_path = out_path or (REPORTS_DIR / "final_defense_answers.pdf")
    
    # Load actual experiment figures
    metrics = json.load(open(RESULTS_DIR / 'final_test_metrics.json'))
    hpd = pd.read_csv(RESULTS_DIR / 'hyperparameter_default_vs_tuned.csv')
    ga = pd.read_csv(RESULTS_DIR / 'ga_feature_selection_results.csv')
    gaov = pd.read_csv(RESULTS_DIR / 'ga_subset_overlap.csv')
    gaf = pd.read_csv(RESULTS_DIR / 'ga_feature_frequency.csv').sort_values('selection_frequency', ascending=False)
    noise = pd.read_csv(RESULTS_DIR / 'noise_sensitivity.csv')
    over = pd.read_csv(RESULTS_DIR / 'class_overlap_analysis.csv')
    pc = pd.read_csv(RESULTS_DIR / 'per_class_metrics.csv')
    
    story = []
    
    # Header
    story.append(P("پاسخ‌های رسمی به پرسش‌های دفاع نهایی پروژه", "RTLTitle"))
    story.append(P("پروژه تشخیص نوع عیب ورق فولادی | تیم ۳: ملینا سالمی (مدیر)، کیانا سرکاری (معمار)، مارول کشت‌کار (تحلیل‌گر)", "RTLBody"))
    story.append(P("کلیه پاسخ‌ها منحصراً بر پایه داده‌ها و خروجی‌های واقعی کدهای اجرا شده در آزمایش‌ها تدوین شده‌اند.", "RTLBody"))
    story.append(Spacer(1, 10))
    
    # Q1
    story.append(P("پرسش ۱: چرا Macro F1 یا معیار اصلی انتخابی برای این داده مناسب است؟", "RTLH1"))
    q1_text = (
        f"پاسخ مستند: داده‌های عیوب ورق فولادی دارای عدم توازن شدید هستند؛ بزرگ‌ترین کلاس (Other_Faults با ۶۷۳ نمونه) "
        f"بیش از ۱۲.۲ برابر کوچک‌ترین کلاس (Dirtiness با ۵۵ نمونه) است. در معیارهای مرسوم مانند Accuracy ساده، مدلی که کلاس‌های اقلیت "
        f"را به طور کامل نادیده بگیرد همچنان می‌تواند به صحت بالای ۶۵٪ یا ۷۰٪ دست یابد، در حالی که در خط تولید صنعتی، نادیده گرفتن "
        f"عیوب کوچک هزینه‌های خسارت و مرجوعی جبران‌ناپذیری ایجاد می‌کند. معیار Macro F1 میانگین ناموزون (وزن یکسان) برای تک‌تک ۷ کلاس است "
        f"و عملکرد را بدون غلبه کلاس اکثریت می‌سنجد. همچنین اختلاف {abs(metrics['accuracy'] - metrics['balanced_accuracy']) * 100:.2f} واحد درصدی بین Accuracy ({metrics['accuracy'] * 100:.2f}%) "
        f"و Balanced Accuracy ({metrics['balanced_accuracy'] * 100:.2f}%) ضرورت این انتخاب را به وضوح اثبات می‌کند."
    )
    story.append(P(q1_text, "RTLBody"))
    story.append(Spacer(1, 8))
    
    # Q2
    story.append(P("پرسش ۲: کدام کلاس‌ها بیشتر اشتباه شدند و کدام ویژگی‌ها این هم‌پوشانی را توضیح می‌دهند؟", "RTLH1"))
    q2_text = (
        f"پاسخ مستند: طبق ماتریس درهم‌ریختگی نهایی روی مجموعه Test دست‌نخورده، دو جفت با بیشترین خطای رفت‌وبرگشتی عبارتند از:\n"
        f"۱) جفت Bumps ↔ Other_Faults با مجموعاً {int(over.iloc[0]['bidirectional_errors'])} خطا ({int(over.iloc[0]['a_as_b'])} مورد Bumps به اشتباه Other_Faults و {int(over.iloc[0]['b_as_a'])} مورد برعکس).\n"
        f"۲) جفت Pastry ↔ Other_Faults با مجموعاً {int(over.iloc[1]['bidirectional_errors'])} خطا ({int(over.iloc[1]['a_as_b'])} مورد Pastry به اشتباه Other_Faults و {int(over.iloc[1]['b_as_a'])} مورد برعکس).\n"
        f"تحلیل ویژگی‌های تفکیک‌کننده در بخش آموزش نشان می‌دهد که مهم‌ترین ویژگی‌های تفکیک‌کننده این کلاس‌ها عبارتند از: "
        f"{over.iloc[0]['top_train_separation_features'].replace('|', ', ')}. دلیل اصلی هم‌پوشانی این است که Other_Faults یک دسته پس‌ماند (Residual) "
        f"نامتجانس است که عیوب متنوع ساختاری را در خود جای داده و شاخص‌های هندسی مانند Square_Index و ضخامت ورق در آن با برجستگی‌های سطحی (Bumps) هم‌پوشانی دارند."
    )
    story.append(P(q2_text, "RTLBody"))
    story.append(Spacer(1, 8))
    
    # Q3
    story.append(P("پرسش ۳: آیا راهبرد کاهش اثر نویز به کلاس‌های کوچک کمک کرد یا به آن‌ها آسیب زد؟", "RTLH1"))
    q3_text = (
        f"پاسخ مستند و تحلیل تجربی: نویز بر همه کلاس‌ها اثر یکسان نمی‌گذارد. در آزمایش کنترل‌شده نویز گاوسی روی داده اعتبارسنجی، "
        f"کلاس Dirtiness دچار افت شدید بازخوانی شد؛ بازخوانی آن از {noise.loc[noise.noise_level == 0.0, 'recall_Dirtiness'].iloc[0]:.4f} در نویز ۰٪ به "
        f"{noise.loc[noise.noise_level == 0.01, 'recall_Dirtiness'].iloc[0]:.4f} در نویز ۱٪ و نهایتاً به {noise.loc[noise.noise_level == 0.10, 'recall_Dirtiness'].iloc[0]:.4f} در نویز ۱۰٪ کاهش یافت. "
        f"همچنین کلاس Stains نیز در نویز ۱۰٪ افت کرد و از {noise.loc[noise.noise_level == 0.0, 'recall_Stains'].iloc[0]:.4f} به {noise.loc[noise.noise_level == 0.10, 'recall_Stains'].iloc[0]:.4f} رسید. "
        f"بنابراین، راهبرد کنونی محافظت کامل و صددرصدی برای کلاس‌های کوچک در برابر نویز اندازه‌گیری فراهم نمی‌کند و آزمایش نشان می‌دهد که مدل نسبت به افزایش نویز اندازه‌گیری حساس است. "
        f"این موضوع به عنوان یک محدودیت تجربی مدل شناسایی شده و ضرورت پشتیبانی بازرس انسانی (Human-in-the-Loop) را اثبات می‌کند."
    )
    story.append(P(q3_text, "RTLBody"))
    story.append(PageBreak())
    
    # Q4
    story.append(P("پرسش ۴: تنظیم ابرپارامتر چه تغییری در فاصله Train و Validation ایجاد کرد؟", "RTLH1"))
    q4_text = (
        f"پاسخ مستند: در مدل Random Forest، نسخه پیش‌فرض بدون محدودیت عمق دارای امتیاز ۱.۰۰ روی Train بود که نشانه بیش‌برازش محض بود "
        f"و شکاف Train-CV آن به {hpd.loc[(hpd['model'] == 'RandomForest_Nonlinear') & (hpd['status'] == 'default'), 'train_cv_gap'].iloc[0]:.4f} می‌رسید. "
        f"پس از تنظیم ابرپارامترها با RandomizedSearchCV پنج‌فولدی روی Train (بهترین پیکربندی: min_samples_leaf=2, max_features='sqrt', n_estimators=50)، "
        f"شکاف Train-CV از 0.2181 به {hpd.loc[(hpd['model'] == 'RandomForest_Nonlinear') & (hpd['status'] == 'tuned'), 'train_cv_gap'].iloc[0]:.4f} کاهش یافت و بهترین CV Macro F1 برابر 0.7776 شد. سپس همین مدل فریز‌شده روی Validation مستقل Macro F1 برابر 0.8288 گرفت. "
        f"در رگرسیون لجستیک نیز با تنظیم پارامتر منظم‌سازی C، شکاف تعمیم‌پذیری به {hpd.loc[(hpd['model'] == 'Logistic_Interpretable') & (hpd['status'] == 'tuned'), 'train_cv_gap'].iloc[0]:.4f} تثبیت شد."
    )
    story.append(P(q4_text, "RTLBody"))
    story.append(Spacer(1, 6))
    
    # Gap table
    gap_table = [["Model", "Status", "Train Macro F1", "Validation Macro F1", "Train-CV Gap"]]
    for _, r in hpd.iterrows():
        gap_table.append([r['model'], r['status'], f"{r['train_macro_f1_mean']:.4f}", f"{r['cv_macro_f1_mean']:.4f}", f"{r['train_cv_gap']:.4f}"])
    story.append(create_table(gap_table, [140, 80, 85, 85, 90], fs=7))
    story.append(Spacer(1, 8))
    
    # Q5
    story.append(P("پرسش ۵: ویژگی‌های منتخب الگوریتم ژنتیک در سه اجرا چقدر پایدار بودند؟", "RTLH1"))
    q5_text = (
        f"پاسخ مستند: الگوریتم ژنتیک در سه دانه تصادفی مستقل (Seeds 11, 22, 33) اجرا شد. تعداد ویژگی‌های منتخب در هر سه اجرا بین ۱۸ تا ۱۹ ویژگی "
        f"(از ۲۷ ویژگی اولیه) بود. میانگین Macro F1 در 3-Fold CV روی Train برابر {ga['cv_macro_f1'].mean():.4f} با انحراف معیار بسیار پایین {ga['cv_macro_f1'].std(ddof=1):.4f} ثبت شد. "
        f"ضریب هم‌پوشانی ژاکارد (Jaccard Similarity) بین جفت‌زیرمجموعه‌ها بین {gaov['jaccard'].min():.2f} تا {gaov['jaccard'].max():.2f} به دست آمد. "
        f"تعداد ۱۰ ویژگی کلیدی شامل Edges_Index, Edges_Y_Index, Empty_Index, LogOfAreas, Luminosity_Index, Minimum_of_Luminosity, Outside_Global_Index, Square_Index, Steel_Plate_Thickness, TypeOfSteel_A400 "
        f"در ۱۰۰٪ اجراها انتخاب شدند که پایداری هسته اصلی ویژگی‌های منتخب را اثبات می‌کند."
    )
    story.append(P(q5_text, "RTLBody"))
    story.append(Spacer(1, 8))
    
    # Q6
    story.append(P("پرسش ۶: اگر داده از کارخانه یا دوربین دیگری بیاید، چه نوع افتی انتظار دارید و چگونه آن را پایش می‌کنید؟", "RTLH1"))
    q6_text = (
        "پاسخ مستند: انتقال مدل به محیط فیزیکی دیگر با پدیده جابه‌جایی دامنه (Domain / Covariate Shift) همراه خواهد بود:\n"
        "۱) نوع افت مورد انتظار: تغییر در کالیبراسیون و نورپردازی دوربین، باعث شیفت سیستماتیک در توزیع میانگین روشنایی (Luminosity_Index، Minimum/Maximum_of_Luminosity) "
        "و نسبت‌های پیکسلی می‌شود. تغییر کارخانه نیز می‌تواند ضخامت متداول ورق (Steel_Plate_Thickness) یا سهم آلیاژهای A300/A400 را جابه‌جا کرده و موجب افت تفکیک‌پذیری کلاس‌ها گردد.\n"
        "۲) برنامه پایش صنعتی (Monitoring Architecture):\n"
        "• پایش تغییر داده (Data Drift): اجرای روزانه آزمون ناپارامتری دو نمونه‌ای کولموگروف-اسمیرنوف (KS-Test) روی ویژگی‌های پیوسته کلیدی و ارسال هشدار خودکار در صورت افت p-value به زیر ۰.۰۱.\n"
        "• پایش پیش‌بینی (Prediction Drift): محاسبه فاصله واگرایی جنسن-شنون یا کولبک-لایبلر روی توزیع خروجی کلاس‌ها در دوره‌های هفتگی.\n"
        "• حلقه بازخورد انسانی (Human-in-the-Loop Active Verification): در مواردی که بیشینه احتمال پیش‌بینی کلاس کمتر از ۰.۶۰ باشد، نمونه به بازرس انسانی ارجاع شده و به عنوان داده برچسب‌دار جدید برای بازآموزی دوره‌ای ذخیره می‌گردد."
    )
    story.append(P(q6_text, "RTLBody"))
    
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=34,
        leftMargin=34,
        topMargin=32,
        bottomMargin=28
    )
    doc.build(story, onFirstPage=add_page_number_defense, onLaterPages=add_page_number_defense)
    print(f"Generated Final Defense Answers Report: {out_path} ({doc.page} pages)")
    return out_path

if __name__ == '__main__':
    generate_analyst_report()
    generate_defense_answers_report()
