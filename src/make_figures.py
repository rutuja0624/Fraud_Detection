#!/usr/bin/env python3
"""Generate IEEE-width figures from results/metrics.json and results/predictions_test.csv.gz."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve, average_precision_score

R = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = os.path.join(R, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6,
                     "savefig.dpi": 300, "figure.dpi": 150})
M = json.load(open(os.path.join(R, "metrics.json")))

# ---------------------------------------------------------------- Fig 1: pipeline
fig, ax = plt.subplots(figsize=(3.45, 2.35)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
boxes = [(0.1, 5.1, "Transaction\nstream / CSV\n(PaySim schema)"),
         (3.6, 5.1, "Spark ingest\n(typed schema)"),
         (7.1, 5.1, "Feature\nengineering\n(window fns)"),
         (7.1, 2.6, "Temporal split\ntrain / val / test"),
         (3.6, 2.6, "Spark MLlib\nLR / RF / GBT\n(class weights)"),
         (0.1, 2.6, "Threshold on\nvalidation\n(max F1)"),
         (0.1, 0.1, "Alerts + metrics\nAUPRC, value\nrecall"),
         ]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.8, boxstyle="round,pad=0.05", fc="#f2f2f2", ec="k", lw=0.7))
    ax.text(x + 1.4, y + 0.9, t, ha="center", va="center", fontsize=6.3)
arr = dict(arrowstyle="->", lw=0.8, color="k")
for (x1, y1), (x2, y2) in [((2.95, 6.0), (3.55, 6.0)), ((6.45, 6.0), (7.05, 6.0)),
                           ((8.5, 5.05), (8.5, 4.45)), ((7.05, 3.5), (6.45, 3.5)),
                           ((3.55, 3.5), (2.95, 3.5)), ((1.5, 2.55), (1.5, 1.95))]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arr)
ax.text(8.5, 0.9, "Behavioural features:\nsender velocity,\ndest. in-degree", ha="center", va="center", fontsize=5.8, style="italic")
fig.savefig(os.path.join(FIG, "fig1_pipeline.png"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: PR curves
P = pd.read_csv(os.path.join(R, "predictions_test.csv.gz"))
fig, ax = plt.subplots(figsize=(3.45, 2.55))
for col, lab, ls in [("p_gbt", "GBT", "-"), ("p_rf", "Random Forest", "--"), ("p_lr", "Logistic Reg.", ":")]:
    pr, rc, _ = precision_recall_curve(P.y, P[col])
    ax.plot(rc, pr, ls, color="k", lw=1.1, label=f"{lab} (AP={average_precision_score(P.y, P[col]):.3f})")
rule = M["models"]["Rule: drain on TRANSFER/CASH_OUT"]
ax.plot(rule["recall"], rule["precision"], "s", color="k", ms=4, label="Drain rule")
rule2 = M["models"]["Rule: amount>200k (PaySim flag)"]
ax.plot(rule2["recall"], rule2["precision"], "^", color="gray", ms=4, label="Amount>200k rule")
ax.axhline(P.y.mean(), color="gray", lw=0.5, ls="-.")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3, lw=0.4); ax.legend(loc="upper right", fontsize=6, frameon=True)
fig.savefig(os.path.join(FIG, "fig2_pr_curves.png"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: importance
imp = pd.Series(M["importance"]["GBT"]).sort_values(ascending=True)  # plot all 12 nobal features
fig, ax = plt.subplots(figsize=(3.45, 2.7))
ax.barh(imp.index, imp.values, color="0.35", edgecolor="k", lw=0.5)
ax.set_xlabel("GBT feature importance (impurity-based)"); ax.grid(axis="x", alpha=0.3, lw=0.4)
fig.savefig(os.path.join(FIG, "fig3_importance.png"), bbox_inches="tight"); plt.close(fig)
print("figures written to", FIG)
