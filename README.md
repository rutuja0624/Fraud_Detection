# Scalable Fraud Detection for Fintech Transactions with Apache Spark

Big Data Analytics (BE Sem-7) - Assignment 10: Technical Project & GitHub Repository.

Batch Spark pipeline prototype for mobile-money fraud detection on the real **Kaggle PaySim dataset** (6.36 million transactions): leakage-free window features, class-weighted Logistic Regression / Random Forest / GBT, strict temporal evaluation, rule baselines, feature ablation, bootstrap confidence intervals, error analysis, alert-budget analysis, a single-node scalability timing study, and an interactive web demo dashboard.

---

## Benchmark Results (Real PaySim Dataset, 6,362,620 rows, Leak-Free `nobal` setting)

| Method / Model | AUPRC [95% CI] | Precision | Recall | F1 Score | Value Recall | Alerts/10k |
|---|---|---|---|---|---|---|
| Rule: amount > 200k units* | - | 0.058 | 0.670 | 0.106 | **0.984** | 1809.5 |
| Rule: drain on TRANSFER/CASH_OUT* | - | 0.068 | **0.964** | 0.126 | 0.819 | 2228.2 |
| Logistic Regression (thr=0.407) | 0.3973 [0.3735, 0.4233] | 0.565 | 0.278 | 0.373 | 0.475 | 76.9 |
| Random Forest (thr=0.738) | 0.6936 [0.6734, 0.7144] | **0.909** | 0.409 | 0.564 | 0.723 | 70.3 |
| **GBT (Gradient Boosted Trees, thr=0.723)** | **0.7076 [0.6881, 0.7272]** | 0.864 | 0.483 | **0.619** | 0.792 | 87.3 |

\* *Note: Simple rules achieve higher recall (0.67–0.96) than learned models (0.28–0.48) but generate 17,661 to 21,522 false alarms (1,809.5–2,228.2 alerts/10k). The Drain rule also inspects pre-transaction account balance fields which GBT/RF/LR were denied under leak-free `nobal` guidelines.*

**Model Comparison**: GBT and Random Forest are statistically distinguishable (+0.0140 AUPRC, $p < 0.001$) but practically similar. Random Forest achieves higher precision (90.9%) and higher value recall at restricted alert budgets, while GBT achieves higher count recall (0.483 vs 0.409).

---

## Quick Start & Setup Guide

### 1. Prerequisites & Environment Setup
- **Python**: 3.10+ (tested on Python 3.12)
- **Java**: Java 17+ or 21 (JDK 17 LTS / JDK 21 tested)
- **Apache Spark**: PySpark 4.2.0

Install python dependencies:
```bash
pip install -r requirements.txt
pip install reportlab
```

---

### 2. How to Run the Complete Pipeline (Step-by-Step)

#### Step A: Prepare the Kaggle PaySim Dataset
Ensure the raw Kaggle PaySim CSV file (`PS_20174392719_1491204439457_log.csv`, ~470 MB) is in the project root, then convert it using the streaming chunk loader:
```bash
python src/prepare_paysim.py PS_20174392719_1491204439457_log.csv --out data/paysim.csv
```
*Outputs `data/paysim.csv` (6,362,620 rows) and `data/paysim_summary.json`.*

#### Step B: Train & Evaluate Spark Models
Train Logistic Regression, Random Forest, and GBT with class weighting ($33.6\times$) under strict temporal split (Train 1–479, Val 480–599, Test 600–743):
```bash
python src/train_evaluate.py --data data/paysim.csv --out results/paysim --train-end 479 --val-end 599 --gbt-iter 40 --rf-trees 40 --feature-set nobal --mem 4g
```

#### Step C: Run Statistical & Error Analysis
Compute 95% bootstrap confidence intervals, error breakdowns by transaction size quartiles, and alert-budget review capacity trade-offs:
```bash
python src/bootstrap_ci.py results/paysim
python src/error_analysis.py results/paysim --bands auto
python src/alert_budget.py results/paysim --budgets auto
```

#### Step D: Generate Figures & Run Scalability Timing Study
Generate publication plots in `results/paysim/figures/` and execute single-node execution throughput benchmark across 25%, 50%, and 100% time spans:
```bash
python src/make_figures.py results/paysim
python src/scalability.py --data data/paysim.csv --out results/paysim/scalability.json --gbt-iter 40 --feature-set nobal --mem 4g
```

#### Step E: Generate Formal PDF Report
Compile the comprehensive 4-page PDF project report:
```bash
python generate_pdf_report.py
```
*Creates `Fraud_Detection_Project_Report.pdf` in the root directory.*

---

### 3. How to Launch & View the Interactive Web Demo Dashboard

Start a local web server from the project directory:
```bash
python -m http.server 5000
```
Open your browser and navigate to:
**`http://localhost:5000`** (or open `index.html` directly).

**Dashboard Features**:
- **Client-Side GBT Scoring Demo**: Test custom transactions (Type, Amount, Hour, Mule In-Degree) using a handwritten rule approximation derived from trained GBT decision logic.
- **KPI Cards**: Dataset scale (6.36M), best AUPRC (0.7076), simulated detected fraud value (2.13B units), and precision @ threshold (86.4%).
- **Model Comparison Table**: Reconciled benchmark matrix.
- **Alert Capacity Simulator**: Interactive slider (39.0 to 156.2 alerts per 10k transactions) updating recall and value recall dynamically.
- **Feature Ablation & Scalability Tables**.

---

## Feature Group Ablation Progression

| Feature Group | Features Included | AUPRC | Incremental Gain | Cumulative Gain |
|---|---|---|---|---|
| `RAW` | Amount, Type One-Hots (6) | 0.3487 | Baseline | Baseline |
| `RAW + TIME` | + Hour of Day, Night Flag (8) | 0.5113 | +0.1626 over RAW | +0.1626 |
| **`ALL (RAW+TIME+BEH)`** | + 24h Velocity & Mule In-Degree (12) | **0.7076** | **+0.1963 over TIME** | **+0.3589 over RAW** |

---

## Single-Node Scalability Benchmark (`scalability.py`)

| Dataset Subset | Total Rows | Feature Engineering Time | GBT Fitting Time | FE Throughput |
|---|---|---|---|---|
| **25% Time Span** | 2,204,140 | 30.3 s | 139.7 s | 72,648 rows/s |
| **50% Time Span** | 5,256,373 | 42.8 s | 524.2 s | 122,703 rows/s |
| **100% Full Dataset** | 6,362,620 | 113.0 s | 712.1 s | 56,287 rows/s |

---

## Methodological Considerations & Limitations

- **Prevalence Shift Impact**: Test set fraud prevalence is 1.562% (1,618 / 103,591) due to PaySim step distribution, compared to 0.088% in training and 0.129% in the full dataset. Assuming constant TPR (0.483) and FPR (0.00121), GBT precision would drop from 0.864 on test to ~0.341 at 0.129% full-dataset prevalence and ~0.261 at 0.088% training prevalence.
- **Hour Feature Simulator Artifact**: `hour` is GBT's top feature (impurity importance ~0.294) because PaySim concentrates synthetic fraud in specific hours. This reflects simulator mechanics rather than real banking behavior, so time-of-day gains (+0.1626 AUPRC) may not transfer to real data.
- **Sender Velocity vs Destination In-Degree**: PaySim origin accounts are nearly all unique single-transaction IDs, so sender 24h velocity features (`orig_cnt_24h`, `orig_amt_24h`) carry near-zero importance (0.00). Destination-side in-degree features (`dest_prior_cnt`, `dest_prior_senders`) provide the primary behavioral lift.
- **Oracle Alert Budget Definitions**: The 4 alert budgets (39.0, 78.1, 117.1, 156.2 alerts/10k) correspond to 0.25, 0.50, 0.75, and 1.00 x the test fraud count (404, 809, 1213, 1618 alerts). This requires knowing the test fraud count in advance. At the 156.2 budget, precision equals recall (0.646).
- **Drain Rule Leakage**: The Drain rule inspects pre-transaction account balance fields which GBT, RF, and LR were denied under leak-free `nobal` guidelines.
