#!/usr/bin/env python3
"""
Prepare the real PaySim dataset (Kaggle: "Synthetic Financial Datasets For Fraud Detection",
file PS_20174392719_1491204439457_log.csv, ~470 MB, 6.36 M rows) for this repository.

What it does (streaming, so it also works on small-memory machines):
  * accepts the .csv or the Kaggle .zip
  * validates the required columns and values
  * drops isFlaggedFraud (not used; it is a rule output, would be leakage-prone to use)
  * adds a sequential txnId
  * optionally keeps only the first --max-step time steps (time-based subset, NOT random rows,
    because random rows would break the account-history window features)
  * writes data/paysim.csv and data/paysim_summary.json (use for the paper's dataset table)

Usage:
    python src/prepare_paysim.py PS_20174392719_1491204439457_log.csv
    python src/prepare_paysim.py archive.zip --max-step 400 --out data/paysim_400.csv
"""
import argparse, json, os, sys
import pandas as pd

REQUIRED = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
TYPES = {"PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"}

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--out", default="data/paysim.csv")
ap.add_argument("--summary", default=None, help="default: <out without .csv>_summary.json")
ap.add_argument("--max-step", type=int, default=None, help="keep steps <= this value")
ap.add_argument("--chunksize", type=int, default=500_000)
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
rows = fraud = 0
last_step = -1
unsorted = False
by_type, nulls = {}, 0
steps_min, steps_max = 10**9, -1
first = True
txn = 0
for chunk in pd.read_csv(a.input, chunksize=a.chunksize):
    if first:
        miss = [c for c in REQUIRED if c not in chunk.columns]
        if miss:
            sys.exit(f"ERROR: missing columns {miss}. Found: {list(chunk.columns)}")
    if a.max_step is not None:
        chunk = chunk[chunk.step <= a.max_step]
        if chunk.empty:
            continue
    chunk = chunk[REQUIRED]
    nulls += int(chunk.isna().sum().sum())
    bad_t = set(chunk.type.unique()) - TYPES
    if bad_t:
        sys.exit(f"ERROR: unexpected transaction types {bad_t}")
    if not set(chunk.isFraud.unique()) <= {0, 1}:
        sys.exit("ERROR: isFraud must be 0/1")
    if chunk.step.iloc[0] < last_step or not chunk.step.is_monotonic_increasing:
        unsorted = True
    last_step = int(chunk.step.iloc[-1])
    steps_min, steps_max = min(steps_min, int(chunk.step.min())), max(steps_max, int(chunk.step.max()))
    chunk.insert(0, "txnId", range(txn, txn + len(chunk)))
    txn += len(chunk)
    chunk.to_csv(a.out, mode="w" if first else "a", header=first, index=False)
    first = False
    rows += len(chunk); fraud += int(chunk.isFraud.sum())
    for t, g in chunk.groupby("type"):
        d = by_type.setdefault(t, {"rows": 0, "fraud": 0})
        d["rows"] += len(g); d["fraud"] += int(g.isFraud.sum())
    print(f"  ... {rows:,} rows", end="\r", flush=True)

if rows == 0:
    sys.exit("ERROR: no rows written (check --max-step)")
summary = dict(source=os.path.basename(a.input), rows=rows, fraud=fraud, fraud_rate=fraud / rows,
               step_min=steps_min, step_max=steps_max, null_cells=nulls, by_type=by_type,
               sorted_by_step=not unsorted, max_step_filter=a.max_step)
sp = a.summary or os.path.splitext(a.out)[0] + "_summary.json"
json.dump(summary, open(sp, "w"), indent=2)
print(f"\nwrote {a.out}: {rows:,} rows, {fraud:,} frauds ({fraud/rows:.4%}), steps {steps_min}-{steps_max}")
for t, d in by_type.items():
    print(f"  {t:9s} rows={d['rows']:>9,} fraud={d['fraud']:>6,}")
if unsorted:
    print("WARNING: input is not sorted by step. Window features order by step inside Spark, so "
          "results stay correct, but txnId no longer follows time order.")
if nulls:
    print(f"WARNING: {nulls} null cells found")
print("summary ->", sp)
