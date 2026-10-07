#!/usr/bin/env python3
"""Measure feature-engineering and GBT training time vs. data volume (single run per point).

Subsets are the first 25 / 50 / 100 % of the TIME span, so window features stay valid.
"""
import argparse, json, os, time
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from features import get_spark, load, build_features, get_groups

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/transactions.csv")
ap.add_argument("--out", default="results/scalability.json")
ap.add_argument("--gbt-iter", type=int, default=40)
ap.add_argument("--feature-set", default=os.environ.get("FEATURE_SET", "full"), choices=["full", "safe", "nobal"])
ap.add_argument("--mem", default=os.environ.get("SPARK_DRIVER_MEMORY", "2g"))
a = ap.parse_args()

spark = get_spark("fraud-scale", mem=a.mem)
spark.sparkContext.setLogLevel("ERROR")
raw = load(spark, a.data)
smin, smax = raw.agg(F.min("step"), F.max("step")).first()
span = smax - smin + 1
_g = get_groups(a.feature_set)
cols = _g[list(_g)[-1]]
out = []
for frac in (0.25, 0.5, 1.0):
    sub = raw.filter(F.col("step") < smin + int(span * frac))
    t0 = time.time()
    df = build_features(sub).cache()
    n = df.count()
    fe = time.time() - t0
    npos = df.filter(F.col("isFraud") == 1).count()
    wpos = float(np.sqrt((n - npos) / max(npos, 1)))   # same weighting rule as train_evaluate.py
    df = df.withColumn("w", F.when(F.col("isFraud") == 1, F.lit(wpos)).otherwise(F.lit(1.0)))
    t1 = time.time()
    Pipeline(stages=[
        VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep"),
        GBTClassifier(featuresCol="features", labelCol="isFraud", weightCol="w",
                      maxIter=a.gbt_iter, maxDepth=5, stepSize=0.1, subsamplingRate=0.8,
                      maxBins=64, seed=7)]).fit(df)
    fit = time.time() - t1
    out.append(dict(fraction=frac, rows=n, fe_seconds=fe, gbt_fit_seconds=fit,
                    rows_per_sec_fe=n / fe, positive_weight=wpos))
    print(out[-1], flush=True)
    df.unpersist()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
