#!/usr/bin/env python3
"""Where does GBT miss fraud?  Recall by transaction type and by amount band (test period).

Amount bands: default = fixed edges used for the synthetic data (10k, 50k, 150k).
Use --bands auto for other datasets: edges become the quartiles of FRAUD amounts.
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="?", default="results")
ap.add_argument("--bands", default="10000,50000,150000", help="comma-separated edges, or 'auto'")
a = ap.parse_args()
R = a.results
P = pd.read_csv(f"{R}/predictions_test.csv.gz")
thr = json.load(open(f"{R}/metrics.json"))["models"]["GBT"]["threshold"]
P["pred"] = P.p_gbt >= thr
F = P[P.y == 1].copy()
out = {"threshold": thr, "by_type": {}, "by_amount": {}}
for t, g in F.groupby("type"):
    out["by_type"][t] = dict(n=len(g), recall=float(g.pred.mean()), value=float(g.amount.sum()),
                             value_recall=float(g[g.pred].amount.sum() / g.amount.sum()))
if a.bands == "auto":
    q = np.unique(np.quantile(F.amount, [0.25, 0.5, 0.75]).round(0))
    edges = [0, *q.tolist(), np.inf]
else:
    edges = [0, *[float(x) for x in a.bands.split(",")], np.inf]
fmt = lambda v: f"{v/1e3:g}k" if v < 1e6 else f"{v/1e6:g}M"
labels = [f"<{fmt(edges[1])}"] + [f"{fmt(edges[i])}-{fmt(edges[i+1])}" for i in range(1, len(edges) - 2)] \
         + [f">{fmt(edges[-2])}"]
F["band"] = pd.cut(F.amount, edges, labels=labels, right=False)
for b, g in F.groupby("band", observed=True):
    out["by_amount"][str(b)] = dict(n=len(g), recall=float(g.pred.mean()),
                                    value_recall=float(g[g.pred].amount.sum() / g.amount.sum()))
json.dump(out, open(f"{R}/error_analysis.json", "w"), indent=2)
print(json.dumps(out, indent=1))
