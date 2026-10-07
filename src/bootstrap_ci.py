#!/usr/bin/env python3
"""Bootstrap 95% CIs for test AUPRC (and paired differences vs. GBT)."""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
R = sys.argv[1] if len(sys.argv) > 1 else "results"
P = pd.read_csv(f"{R}/predictions_test.csv.gz")
y = P.y.values; rng = np.random.default_rng(0); B = 1000
cols = {"GBT": "p_gbt", "RF": "p_rf", "LR": "p_lr"}
bs = {k: [] for k in cols}
n = len(P)
for _ in range(B):
    i = rng.integers(0, n, n)
    for k, c in cols.items():
        bs[k].append(average_precision_score(y[i], P[c].values[i]))
out = {}
for k in cols:
    a = np.array(bs[k]); out[k] = dict(mean=a.mean(), lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5))
for k in ("RF", "LR"):
    d = np.array(bs["GBT"]) - np.array(bs[k])
    out[f"GBT-{k}"] = dict(mean=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                           p_gbt_not_better=float((d <= 0).mean()))
json.dump(out, open(f"{R}/bootstrap.json", "w"), indent=2)
for k, v in out.items(): print(k, {a: round(b, 4) for a, b in v.items()})
