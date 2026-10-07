#!/usr/bin/env python3
"""Recall / value recall when analysts can review only the top-k scored transactions per 10k.

Default budgets 10,20,30,50 alerts per 10k (suits ~0.3-0.4 % fraud, the synthetic data).
Use --budgets auto for rarer fraud (e.g. real PaySim, ~0.13 %): budgets become
25/50/75/100 % of the test period's frauds-per-10k.
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default="results")
ap.add_argument("--budgets", default="10,20,30,50", help="alerts per 10k, comma-separated, or 'auto'")
a = ap.parse_args()
R = a.results
P = pd.read_csv(f"{R}/predictions_test.csv.gz")
if a.budgets == "auto":
    f = P.y.mean() * 1e4
    budgets = sorted(set(round(f * x, 1) for x in (0.25, 0.5, 0.75, 1.0)))
else:
    budgets = [float(x) for x in a.budgets.split(",")]
out = {}
for budget in budgets:
    k = int(round(len(P) * budget / 1e4))
    row = {}
    for name, c in [("LR", "p_lr"), ("RF", "p_rf"), ("GBT", "p_gbt")]:
        top = P.nlargest(k, c)
        row[name] = dict(precision=float(top.y.mean()), recall=float(top.y.sum() / P.y.sum()),
                         value_recall=float(top[top.y == 1].amount.sum() / P[P.y == 1].amount.sum()))
    out[f"{budget:g}"] = dict(k=k, **row)
json.dump(out, open(f"{R}/alert_budget.json", "w"), indent=2)
for b, v in out.items():
    print(b, v["k"], {m: tuple(round(x, 3) for x in v[m].values()) for m in ("LR", "RF", "GBT")})
