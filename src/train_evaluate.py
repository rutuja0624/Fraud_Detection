#!/usr/bin/env python3
"""
Train and evaluate fraud-detection models with Apache Spark MLlib.

Protocol
  * Temporal split (no shuffling): train  = steps [0, TRAIN_END]
                                   valid  = (TRAIN_END, VAL_END]   (threshold selection)
                                   test   = (VAL_END, 743]         (final report)
  * Class imbalance handled with instance weights  w_pos = sqrt(N_neg / N_pos)
  * Threshold is chosen on the validation split (max F1) and frozen for test.
  * Metrics: AUPRC (average precision), ROC-AUC, precision, recall, F1,
             value-weighted recall (share of fraudulent money caught), alerts / 10k txns.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve)

from features import get_spark, load, build_features, get_groups

MONEY = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
         "errorBalanceOrig", "errorBalanceDest", "orig_amt_24h"]


def with_slog(df, cols):
    """Signed log1p for heavy-tailed money columns (used for the linear model only)."""
    out = []
    for c in cols:
        if c in MONEY:
            df = df.withColumn(f"slog_{c}", F.signum(c) * F.log1p(F.abs(F.col(c))))
            out.append(f"slog_{c}")
        else:
            out.append(c)
    return df, out


def make_model(kind, args):
    if kind == "LR":
        return LogisticRegression(featuresCol="features", labelCol="isFraud", weightCol="w",
                                  maxIter=100, regParam=1e-4, standardization=False)
    if kind == "RF":
        return RandomForestClassifier(featuresCol="features", labelCol="isFraud", weightCol="w",
                                      numTrees=args.rf_trees, maxDepth=8, maxBins=64, seed=7)
    if kind == "GBT":
        return GBTClassifier(featuresCol="features", labelCol="isFraud", weightCol="w",
                             maxIter=args.gbt_iter, maxDepth=5, stepSize=0.1,
                             subsamplingRate=0.8, maxBins=64, seed=7)
    raise ValueError(kind)


def fit_predict(df_train, df_val, df_test, kind, cols, args):
    """Fit on train; return (val_scores, test_scores, fit_seconds, model, feature_cols)."""
    if kind == "LR":
        df_train, fcols = with_slog(df_train, cols)
        df_val, _ = with_slog(df_val, cols)
        df_test, _ = with_slog(df_test, cols)
        stages = [VectorAssembler(inputCols=fcols, outputCol="raw", handleInvalid="keep"),
                  StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
                  make_model(kind, args)]
    else:
        fcols = cols
        stages = [VectorAssembler(inputCols=fcols, outputCol="features", handleInvalid="keep"),
                  make_model(kind, args)]
    t0 = time.time()
    model = Pipeline(stages=stages).fit(df_train)
    fit_s = time.time() - t0

    def score(d):
        r = (model.transform(d)
             .select("txnId", "step", "amount", "type", "isFraud",
                     vector_to_array("probability")[1].alias("p")).collect())
        return pd.DataFrame(r, columns=["txnId", "step", "amount", "type", "y", "p"])
    return score(df_val), score(df_test), fit_s, model, fcols


def metrics_at(y, p, amount, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fraud_amt = amount[y == 1].sum()
    return dict(threshold=float(thr), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, fpr=fp / max(fp + tn, 1),
                value_recall=float(amount[(y == 1) & pred].sum() / max(fraud_amt, 1e-9)),
                alerts_per_10k=float(pred.mean() * 1e4))


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.nanargmax(f1))])


def evaluate(val, test):
    thr = best_threshold(val.y.values, val.p.values)
    m = metrics_at(test.y.values, test.p.values, test.amount.values, thr)
    m["auprc"] = float(average_precision_score(test.y, test.p))
    m["roc_auc"] = float(roc_auc_score(test.y, test.p))
    return m


def rule_metrics(test, name):
    t = test
    if name == "Rule: amount>200k (PaySim flag)":
        pred = t.type.isin(["TRANSFER", "CASH_OUT"]) & (t.amount > 200_000)
    else:  # Rule: drain
        pred = t.type.isin(["TRANSFER", "CASH_OUT"]) & (t.drain == 1)
    p = pred.astype(float).values
    m = metrics_at(t.y.values, p, t.amount.values, 0.5)
    m["auprc"] = float(average_precision_score(t.y, p))
    m["roc_auc"] = float(roc_auc_score(t.y, p))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--train-end", type=int, default=479)
    ap.add_argument("--val-end", type=int, default=599)
    ap.add_argument("--gbt-iter", type=int, default=40)
    ap.add_argument("--rf-trees", type=int, default=40)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--feature-set", default=os.environ.get("FEATURE_SET", "full"), choices=["full", "safe", "nobal"],
                    help="full = all columns (synthetic data); safe = no post-transaction balances; "
                         "nobal = no balance columns at all (recommended for real PaySim)")
    ap.add_argument("--mem", default=os.environ.get("SPARK_DRIVER_MEMORY", "2g"),
                    help="Spark driver memory, e.g. 2g (default) or 8g for the full PaySim")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spark = get_spark("fraud-train", mem=args.mem)
    spark.sparkContext.setLogLevel("ERROR")
    res = {"config": vars(args)}
    GROUPS = get_groups(args.feature_set)

    t0 = time.time()
    df = build_features(load(spark, args.data))
    df = df.cache()
    n = df.count()
    smin, smax = df.agg(F.min("step"), F.max("step")).first()
    res["data_rows"], res["step_range"] = n, [smin, smax]
    res["feature_engineering_seconds"] = time.time() - t0
    print(f"[data] {n:,} rows, features built in {res['feature_engineering_seconds']:.1f}s", flush=True)

    tr = df.filter(F.col("step") <= args.train_end)
    va = df.filter((F.col("step") > args.train_end) & (F.col("step") <= args.val_end))
    te = df.filter(F.col("step") > args.val_end)
    stats = {}
    for nm, d in [("train", tr), ("val", va), ("test", te)]:
        r = d.agg(F.count("*").alias("n"), F.sum("isFraud").alias("f"),
                  F.sum(F.when(F.col("isFraud") == 1, F.col("amount")).otherwise(0.0)).alias("fa")).first()
        if not r["n"] or not r["f"]:
            raise SystemExit(f"Split '{nm}' has no rows or no frauds. Data steps span "
                             f"{smin}-{smax}; adjust --train-end/--val-end (about 65/15/20 %).")
        stats[nm] = dict(rows=r["n"], fraud=int(r["f"]), fraud_rate=r["f"] / r["n"], fraud_amount=r["fa"])
    res["splits"] = stats
    print("[splits]", json.dumps(stats, indent=1), flush=True)

    npos = stats["train"]["fraud"]; nneg = stats["train"]["rows"] - npos
    wpos = float(np.sqrt(nneg / npos))
    res["positive_class_weight"] = wpos
    wcol = F.when(F.col("isFraud") == 1, F.lit(wpos)).otherwise(F.lit(1.0))
    tr, va, te = [d.withColumn("w", wcol) for d in (tr, va, te)]

    LAST = list(GROUPS)[-1]            # the full feature set of the chosen mode
    all_cols = GROUPS[LAST]
    res["models"], res["fit_seconds"], res["importance"] = {}, {}, {}
    preds = {}

    # ---------------------------------------------------- model comparison
    for kind in ["LR", "RF", "GBT"]:
        v, t, fit_s, model, fcols = fit_predict(tr, va, te, kind, all_cols, args)
        m = evaluate(v, t)
        res["models"][kind] = m
        res["fit_seconds"][kind] = fit_s
        preds[kind] = t
        if kind in ("RF", "GBT"):
            imp = model.stages[-1].featureImportances.toArray()
            res["importance"][kind] = dict(zip(fcols, map(float, imp)))
        print(f"[{kind}] fit={fit_s:.0f}s AUPRC={m['auprc']:.4f} ROC={m['roc_auc']:.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"valueR={m['value_recall']:.3f}", flush=True)

    # ------------------------------------------------------- rule baselines
    t_rows = te.select("txnId", "step", "amount", "type", "isFraud",
                         F.col("is_drain" if args.feature_set == "full" else "is_drain_pre").alias("is_drain")).collect()
    tdf = pd.DataFrame(t_rows, columns=["txnId", "step", "amount", "type", "y", "drain"])
    for rn in ["Rule: amount>200k (PaySim flag)", "Rule: drain on TRANSFER/CASH_OUT"]:
        m = rule_metrics(tdf, rn)
        res["models"][rn] = m
        print(f"[{rn}] P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"valueR={m['value_recall']:.3f}", flush=True)

    # ------------------------------------------------------------ ablation
    res["ablation"] = {}
    if not args.skip_ablation:
        for gname, cols in GROUPS.items():
            if gname == LAST:
                m = res["models"]["GBT"]
            else:
                v, t, _, _, _ = fit_predict(tr, va, te, "GBT", cols, args)
                m = evaluate(v, t)
            res["ablation"][gname] = dict(n_features=len(cols), **m)
            print(f"[ablation {gname}] AUPRC={m['auprc']:.4f} F1={m['f1']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f}", flush=True)

    # --------------------------------------------------------------- save
    out = pd.DataFrame({"txnId": preds["GBT"].txnId, "step": preds["GBT"].step,
                        "amount": preds["GBT"].amount, "type": preds["GBT"].type,
                        "y": preds["GBT"].y, "p_gbt": preds["GBT"].p.values,
                        "p_rf": preds["RF"].p.values, "p_lr": preds["LR"].p.values})
    # collect() order is not guaranteed identical across calls -> align on txnId
    for k, col in [("RF", "p_rf"), ("LR", "p_lr")]:
        out[col] = preds[k].set_index("txnId").loc[out.txnId, "p"].values
    out.to_csv(os.path.join(args.out, "predictions_test.csv.gz"), index=False)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("[done] total %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
