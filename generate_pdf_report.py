import os, sys, json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#6b7280"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "Apache Spark Fraud Detection Prototype - Technical & Evaluation Report")
            self.setStrokeColor(colors.HexColor("#e5e7eb"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
        
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, footer_text)
        self.drawString(54, 36, "BE SEM-7 BIG DATA ANALYTICS ASSIGNMENT — TECHNICAL REPORT")
        self.setStrokeColor(colors.HexColor("#e5e7eb"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * 72 - 54, 48)
        
        self.restoreState()

def build_pdf(filename="Fraud_Detection_Project_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    c_primary = colors.HexColor("#1e1b4b")
    c_accent = colors.HexColor("#4338ca")
    c_dark = colors.HexColor("#111827")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=12
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_accent,
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=7
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=3
    )

    tbl_text = ParagraphStyle('TblText', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10.5, textColor=c_dark)
    tbl_hdr = ParagraphStyle('TblHdr', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10.5, textColor=colors.white)

    story = []

    # Title Banner
    story.append(Paragraph("Scalable Fraud Detection for Fintech Transactions with Apache Spark", title_style))
    story.append(Paragraph("Batch Pipeline Prototype & Rigorous Evaluation Report on Real PaySim Dataset (6.36M Transactions)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceAfter=12))

    # Metadata Card Table
    meta_data = [
      [Paragraph("<b>Dataset:</b> Real Kaggle PaySim (6,362,620 rows)", tbl_text), Paragraph("<b>Pipeline Engine:</b> Local-mode PySpark 4.2.0", tbl_text)],
      [Paragraph("<b>Feature Strategy:</b> Leak-Free <code>nobal</code> (12 features)", tbl_text), Paragraph("<b>Best Model:</b> Spark MLlib GBT (AUPRC = 0.7076)", tbl_text)],
      [Paragraph("<b>Evaluation Split:</b> Strict Temporal (Train 1-479 / Val 480-599 / Test 600-743)", tbl_text), Paragraph("<b>Detected Fraud Value:</b> 2.13B units (79.2% Value Recall)", tbl_text)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Section 1: Executive Summary & Scope
    story.append(Paragraph("1. Executive Summary & Project Scope", h1_style))
    story.append(Paragraph(
        "Digital financial services generate high-volume transaction streams where fraudulent activities are a rare minority (0.129% overall dataset prevalence). Evaluating models on random data splits often causes severe look-ahead data leakage, while uncalibrated models flood security operations teams with excessive false alarms.",
        body_style
    ))
    story.append(Paragraph(
        "This project implements a <b>batch pipeline prototype in Apache Spark</b> for mobile-money fraud detection on the public <b>PaySim dataset (6,362,620 transactions)</b>. The pipeline extracts leakage-free behavioral features with window functions under the strict <code>nobal</code> setting (excluding both pre- and post-transaction balance columns per PaySim guidelines), handles class imbalance via instance weighting (33.6x), enforces strict temporal train/validation/test splits, and evaluates operational trade-offs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Important Caveat:</b> Roughly half of the headline AUPRC gain over RAW (+0.1626 of +0.3589) stems from the <code>hour</code> feature. In PaySim, synthetic fraud is concentrated in specific hours of the day—a simulator artifact that will likely not transfer to real-world banking transaction streams.",
        body_style
    ))

    # Section 2: Pipeline Workflow & Methodology
    story.append(Paragraph("2. Pipeline Workflow & Methodology", h1_style))
    
    steps = [
        ("Step 1: Ingestion & Preprocessing", "Processed 6,362,620 raw transaction rows (470 MB) using a chunked CSV loader (<code>prepare_paysim.py</code>) to validate schema, drop leakage columns (<code>isFlaggedFraud</code>), and attach sequential transaction IDs."),
        ("Step 2: Leakage-Free Window Features", "Engineered 12 leak-safe features (<code>features.py</code>) under the <code>nobal</code> setting (excluding ALL balance fields). Sender velocity features (<code>orig_cnt_24h</code>, <code>orig_amt_24h</code>) use 24-step past frames (<code>rangeBetween(-24, -1)</code>). Destination in-degree features (<code>dest_prior_cnt</code>, <code>dest_prior_senders</code> via HyperLogLog) use unbounded prior frames."),
        ("Step 3: Strict Temporal Split Protocol", "Divided dataset chronologically: <b>Train</b> (steps 1-479; 6.04M rows, 0.088% fraud), <b>Validation</b> (steps 480-599; 219.6k rows, 0.570% fraud, used for threshold tuning), and <b>Test</b> (steps 600-743; 103.6k rows, 1.562% fraud, used for final evaluation)."),
        ("Step 4: Class-Weighted Ensemble Training", "Trained Spark MLlib Logistic Regression, Random Forest (40 trees), and Gradient Boosted Trees (40 iterations) using positive instance weights w_pos = sqrt(N_neg / N_pos) = 33.6. Class weighting shifts raw predicted probability outputs, necessitating validation-based threshold selection."),
        ("Step 5: Validation Threshold Selection", "Decision thresholds were selected on validation set to maximize F1 and frozen before test evaluation: <b>GBT threshold = 0.7230</b>, <b>RF threshold = 0.7384</b>, <b>LR threshold = 0.4065</b>."),
        ("Step 6: Statistical CI & Bootstrapping", "Computed 1,000 paired bootstrap resamples (<code>bootstrap_ci.py</code>) on test predictions to establish 95% confidence intervals and p-values."),
        ("Step 7: Multi-Dimensional Error & Alert Analysis", "Evaluated test recall across 4 transaction size quartiles (404 frauds each) and scored alert workload trade-offs across reviewer capacity tiers."),
        ("Step 8: Scalability Timing Study", "Benchmarked feature engineering and GBT training wall-clock times across 25%, 50%, and 100% temporal subsets (<code>scalability.py</code>).")
    ]

    for title, desc in steps:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))
    
    story.append(Spacer(1, 8))

    # Architecture Diagram Image
    if os.path.exists("results/paysim/figures/fig1_pipeline.png"):
        story.append(KeepTogether([
            Paragraph("Figure 1: End-to-End Spark Pipeline Architecture", h2_style),
            Image("results/paysim/figures/fig1_pipeline.png", width=440, height=170),
            Spacer(1, 8)
        ]))

    # Section 3: Experimental Results
    story.append(Paragraph("3. Detailed Experimental Results & Classifier Benchmarks", h1_style))
    story.append(Paragraph(
        "Table 1 reconciles test performance across all models at frozen validation thresholds on the held-out test split (103,591 transactions, 1,618 frauds). AUPRC is evaluated using Scikit-Learn's <code>average_precision_score</code>. Alerts per 10k transactions equal total test alerts (TP + FP) divided by 10.3591:",
        body_style
    ))

    # Reconciled Model Performance Table
    model_tbl_data = [
        [Paragraph("Method / Model", tbl_hdr), Paragraph("AUPRC [95% CI]", tbl_hdr), Paragraph("Prec.", tbl_hdr), Paragraph("Rec.", tbl_hdr), Paragraph("F1 Score", tbl_hdr), Paragraph("Value Rec.", tbl_hdr), Paragraph("Alerts/10k", tbl_hdr)],
        [Paragraph("Rule: Amount > 200k units*", tbl_text), Paragraph("—", tbl_text), Paragraph("0.058", tbl_text), Paragraph("0.670", tbl_text), Paragraph("0.106", tbl_text), Paragraph("<b>0.984</b>", tbl_text), Paragraph("1809.5", tbl_text)],
        [Paragraph("Rule: Account Drain*", tbl_text), Paragraph("—", tbl_text), Paragraph("0.068", tbl_text), Paragraph("<b>0.964</b>", tbl_text), Paragraph("0.126", tbl_text), Paragraph("0.819", tbl_text), Paragraph("2228.2", tbl_text)],
        [Paragraph("Logistic Regression (thr=0.407)", tbl_text), Paragraph("0.3973 [0.374, 0.423]", tbl_text), Paragraph("0.565", tbl_text), Paragraph("0.278", tbl_text), Paragraph("0.373", tbl_text), Paragraph("0.475", tbl_text), Paragraph("76.9", tbl_text)],
        [Paragraph("Random Forest (thr=0.738)", tbl_text), Paragraph("0.6936 [0.673, 0.714]", tbl_text), Paragraph("<b>0.909</b>", tbl_text), Paragraph("0.409", tbl_text), Paragraph("0.564", tbl_text), Paragraph("0.723", tbl_text), Paragraph("70.3", tbl_text)],
        [Paragraph("<b>GBT (thr=0.723)</b>", tbl_text), Paragraph("<b>0.7076 [0.688, 0.727]</b>", tbl_text), Paragraph("0.864", tbl_text), Paragraph("0.483", tbl_text), Paragraph("<b>0.619</b>", tbl_text), Paragraph("0.792", tbl_text), Paragraph("87.3", tbl_text)]
    ]

    model_table = Table(model_tbl_data, colWidths=[130, 110, 42, 42, 48, 60, 62])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor("#e0e7ff")),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<font size=7.5 color='#64748b'>* Note: Simple rules catch more raw fraud (recall 0.67–0.96) but flag 18%–22% of all traffic as alerts. The Drain rule inspects pre-transaction balance fields which GBT/RF/LR were denied under leak-free <code>nobal</code> guidelines.</font>", body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Model Comparison:</b> GBT and Random Forest are statistically distinguishable (+0.0140 AUPRC, p &lt; 0.001) but practically similar. GBT achieves higher count recall (0.483 vs 0.409) and overall value recall (0.792 vs 0.723) at the F1 threshold, while Random Forest achieves higher precision (0.909 vs 0.864) and higher value recall at larger alert capacity budgets (0.904 vs 0.839 at 117.1 alerts/10k, 0.929 vs 0.895 at 156.2 alerts/10k).",
        body_style
    ))

    # Alert Budget Capacity Table
    story.append(Paragraph("Alert Capacity Budget Trade-offs (Table 2)", h2_style))
    budget_tbl_data = [
        [Paragraph("Alerts/10k (Oracle Budget)", tbl_hdr), Paragraph("LR Count Rec.", tbl_hdr), Paragraph("LR Value Rec.", tbl_hdr), Paragraph("RF Count Rec.", tbl_hdr), Paragraph("RF Value Rec.", tbl_hdr), Paragraph("GBT Count Rec.", tbl_hdr), Paragraph("GBT Value Rec.", tbl_hdr)],
        [Paragraph("39.0 (404 alerts)", tbl_text), Paragraph("0.180", tbl_text), Paragraph("0.189", tbl_text), Paragraph("0.245", tbl_text), Paragraph("0.480", tbl_text), Paragraph("0.247", tbl_text), Paragraph("<b>0.513</b>", tbl_text)],
        [Paragraph("78.1 (809 alerts)", tbl_text), Paragraph("0.281", tbl_text), Paragraph("0.481", tbl_text), Paragraph("0.441", tbl_text), Paragraph("0.734", tbl_text), Paragraph("0.440", tbl_text), Paragraph("<b>0.745</b>", tbl_text)],
        [Paragraph("117.1 (1,213 alerts)", tbl_text), Paragraph("0.358", tbl_text), Paragraph("0.601", tbl_text), Paragraph("0.560", tbl_text), Paragraph("<b>0.904</b>", tbl_text), Paragraph("<b>0.567</b>", tbl_text), Paragraph("0.839", tbl_text)],
        [Paragraph("156.2 (1,618 alerts)", tbl_text), Paragraph("0.410", tbl_text), Paragraph("0.692", tbl_text), Paragraph("<b>0.650</b>", tbl_text), Paragraph("<b>0.929</b>", tbl_text), Paragraph("0.646", tbl_text), Paragraph("0.895", tbl_text)]
    ]
    budget_table = Table(budget_tbl_data, colWidths=[120, 64, 64, 64, 64, 64, 64])
    budget_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(budget_table)
    story.append(Spacer(1, 10))

    # Feature Ablation Section
    story.append(Paragraph("Feature Group Ablation Gains", h2_style))
    ablation_tbl_data = [
        [Paragraph("Feature Group", tbl_hdr), Paragraph("Features Included", tbl_hdr), Paragraph("AUPRC", tbl_hdr), Paragraph("Incremental Gain", tbl_hdr), Paragraph("Cumulative Gain", tbl_hdr)],
        [Paragraph("<code>RAW</code>", tbl_text), Paragraph("Amount, Type One-Hots (6)", tbl_text), Paragraph("0.3487", tbl_text), Paragraph("Baseline", tbl_text), Paragraph("Baseline", tbl_text)],
        [Paragraph("<code>RAW + TIME</code>", tbl_text), Paragraph("+ Hour of Day, Night Flag (8)", tbl_text), Paragraph("0.5113", tbl_text), Paragraph("+0.1626 over RAW", tbl_text), Paragraph("+0.1626", tbl_text)],
        [Paragraph("<b>ALL (RAW+TIME+BEH)</b>", tbl_text), Paragraph("+ 24h Velocity & Mule In-Degree (12)", tbl_text), Paragraph("<b>0.7076</b>", tbl_text), Paragraph("<b>+0.1963 over TIME</b>", tbl_text), Paragraph("<b>+0.3589 over RAW</b>", tbl_text)]
    ]
    ablation_table = Table(ablation_tbl_data, colWidths=[120, 154, 55, 115, 100])
    ablation_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_accent),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(ablation_table)
    story.append(Spacer(1, 10))

    # PR Curves & Feature Importance Figures
    if os.path.exists("results/paysim/figures/fig2_pr_curves.png") and os.path.exists("results/paysim/figures/fig3_importance.png"):
        story.append(KeepTogether([
            Paragraph("Figure 2: Precision-Recall Curves & GBT Feature Importances (All 12 Features)", h2_style),
            Table([
                [Image("results/paysim/figures/fig2_pr_curves.png", width=235, height=175),
                 Image("results/paysim/figures/fig3_importance.png", width=235, height=175)]
            ], colWidths=[245, 259]),
            Spacer(1, 8)
        ]))

    # Section 4: Scalability Study
    story.append(Paragraph("4. Scalability Timing Study (scalability.py)", h1_style))
    story.append(Paragraph(
        "Table 3 reports single-node execution times for feature engineering and 40-iteration GBT fitting across time-based data subsets (25%, 50%, and 100% of the dataset span):",
        body_style
    ))

    scale_tbl_data = [
        [Paragraph("Dataset Subset", tbl_hdr), Paragraph("Total Rows", tbl_hdr), Paragraph("Feature Engineering (s)", tbl_hdr), Paragraph("GBT Model Fit (s)", tbl_hdr), Paragraph("FE Throughput (rows/s)", tbl_hdr)],
        [Paragraph("25% Time Span", tbl_text), Paragraph("2,204,140", tbl_text), Paragraph("30.3 s", tbl_text), Paragraph("139.7 s", tbl_text), Paragraph("72,648", tbl_text)],
        [Paragraph("50% Time Span", tbl_text), Paragraph("5,256,373", tbl_text), Paragraph("42.8 s", tbl_text), Paragraph("524.2 s", tbl_text), Paragraph("122,703", tbl_text)],
        [Paragraph("<b>100% Full Dataset</b>", tbl_text), Paragraph("<b>6,362,620</b>", tbl_text), Paragraph("<b>113.0 s</b>", tbl_text), Paragraph("<b>712.1 s</b>", tbl_text), Paragraph("<b>56,287</b>", tbl_text)]
    ]
    scale_table = Table(scale_tbl_data, colWidths=[110, 85, 110, 95, 104])
    scale_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(scale_table)
    story.append(Spacer(1, 10))

    # Section 5: Demonstration Interface
    story.append(Paragraph("5. Client-Side Web Demonstration Interface", h1_style))
    story.append(Paragraph(
        "To illustrate model predictions interactively without requiring a live Spark cluster server, we built a client-side JavaScript demonstration interface (<code>index.html</code>) available at <code>http://localhost:5000</code>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Scoring Implementation:</b> The web UI evaluates transactions locally using a handwritten rule approximation derived from the trained GBT decision logic (weighting transaction amount, type, hour of day, and destination in-degree). It features an <b>Interactive Risk Predictor</b> and an <b>Alert Capacity Simulator</b> (39.0 to 156.2 alerts per 10k transactions).",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Section 6: Limitations & Critical Discussion
    story.append(Paragraph("6. Methodological Disclosures & Limitations", h1_style))
    
    limits = [
        ("Prevalence Shift Impact", "The test split has a 1.562% fraud prevalence due to Kaggle PaySim step distribution, compared to 0.088% in training and 0.129% in the full dataset. Assuming constant TPR (0.483) and FPR (0.00121), GBT precision would drop from 0.864 on test to ~0.341 at 0.129% full-dataset prevalence and ~0.261 at 0.088% training prevalence."),
        ("Simulator Artifact in Hour Feature", "Hour of day is GBT's top feature (impurity importance ~0.294) because PaySim concentrates synthetic fraud in specific hours of the day. This reflects simulator mechanics rather than real banking behavior, so time-of-day gains (+0.1626 AUPRC) may not transfer to real data."),
        ("Sender Velocity vs Destination In-Degree", "PaySim origin accounts are nearly all unique single-transaction IDs, so sender 24h velocity features (orig_cnt_24h, orig_amt_24h) carry near-zero importance (0.00). Destination-side in-degree features (dest_prior_cnt, dest_prior_senders) provide the primary behavioral lift."),
        ("Oracle Alert Budget Definitions", "The 4 alert budgets (39.0, 78.1, 117.1, 156.2 alerts/10k) correspond to 0.25, 0.50, 0.75, and 1.00 x the test fraud count (404, 809, 1213, 1618 alerts). This requires knowing the test fraud count in advance. At the 156.2 budget, precision equals recall (0.646)."),
        ("Workload vs Recall Trade-off", "Rules achieve higher recall (0.67-0.96) than GBT (0.48) but generate 17,661 to 21,522 false alarms (1809.5-2228.2 alerts/10k). GBT reduces alert volume by ~21x to 25x (87.3 alerts/10k) and false positives by ~144x to 175x (123 FP)."),
        ("Value Recall Breakdown", "GBT recovers 92.7% of CASH_OUT fraud value (1.23B of 1.33B units) versus 66.0% for TRANSFER fraud (903M of 1.37B units). Amount bands in Error Analysis correspond to fraud amount quartiles (~404 frauds each)."),
        ("Single-Node Execution", "Timings demonstrate local Spark execution on 6.36M rows but do not measure multi-node cluster speed-up.")
    ]

    for title, desc in limits:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
    story.append(Paragraph("Report generated automatically by Apache Spark Fraud Detection System Pipeline.", ParagraphStyle('FooterNote', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor("#64748b"))))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated corrected report PDF: {filename}")

if __name__ == "__main__":
    build_pdf()
