"""
Spark feature engineering for transaction fraud detection.

All behavioural features are computed with window functions that look ONLY at
strictly earlier time steps (rangeBetween(..., -1)), so there is no look-ahead
leakage between a transaction and its own (or same-hour) neighbours.
"""
import os
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import (StructType, StructField, LongType, IntegerType,
                               StringType, DoubleType)

SCHEMA = StructType([
    StructField("txnId", LongType()), StructField("step", IntegerType()),
    StructField("type", StringType()), StructField("amount", DoubleType()),
    StructField("nameOrig", StringType()), StructField("oldbalanceOrg", DoubleType()),
    StructField("newbalanceOrig", DoubleType()), StructField("nameDest", StringType()),
    StructField("oldbalanceDest", DoubleType()), StructField("newbalanceDest", DoubleType()),
    StructField("isFraud", IntegerType()),
])

TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]

# Feature groups used for the ablation study (cumulative)
RAW = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"] \
      + [f"is_{t.lower()}" for t in TYPES]
BAL = ["errorBalanceOrig", "errorBalanceDest", "amount_ratio", "is_drain", "dest_empty_before"]
TIME = ["hour", "is_night"]
BEH = ["orig_cnt_24h", "orig_amt_24h", "dest_prior_cnt", "dest_prior_senders"]
GROUPS = {
    "RAW": RAW,
    "RAW+BAL": RAW + BAL,
    "RAW+BAL+TIME": RAW + BAL + TIME,
    "ALL (RAW+BAL+TIME+BEH)": RAW + BAL + TIME + BEH,
}


# ---- Leak-safe variant (recommended for REAL PaySim) ------------------------------------
# The PaySim dataset page states that fraudulent transactions are cancelled, so the
# balance columns must not be used for fraud detection: post-transaction balances
# (newbalanceOrig / newbalanceDest) reflect the cancellation, i.e. the label.  The safe
# set uses only information available BEFORE the decision: amount, type, balances BEFORE the
# transaction, time, and past behaviour.  (Even oldbalance* is flagged by the dataset
# owners, so treat the safe-mode results as an upper bound and say so in the paper.)
RAW_SAFE = ["amount", "oldbalanceOrg", "oldbalanceDest"] + [f"is_{t.lower()}" for t in TYPES]
BAL_SAFE = ["amount_ratio", "is_drain_pre", "dest_empty_before"]
GROUPS_SAFE = {
    "RAW": RAW_SAFE,
    "RAW+BAL": RAW_SAFE + BAL_SAFE,
    "RAW+BAL+TIME": RAW_SAFE + BAL_SAFE + TIME,
    "ALL (RAW+BAL+TIME+BEH)": RAW_SAFE + BAL_SAFE + TIME + BEH,
}


# ---- No-balance variant (strictest; follows the PaySim dataset note literally) --------------
# Uses NO balance column at all: amount, type, time of day and past behaviour only.
RAW_NOBAL = ["amount"] + [f"is_{t.lower()}" for t in TYPES]
GROUPS_NOBAL = {
    "RAW": RAW_NOBAL,
    "RAW+TIME": RAW_NOBAL + TIME,
    "ALL (RAW+TIME+BEH)": RAW_NOBAL + TIME + BEH,
}


def get_groups(mode="full"):
    if mode not in ("full", "safe", "nobal"):
        raise ValueError("feature set must be 'full', 'safe' or 'nobal'")
    return {"full": GROUPS, "safe": GROUPS_SAFE, "nobal": GROUPS_NOBAL}[mode]


def get_spark(app="fraud-bda", mem=None, cores="*"):
    mem = mem or os.environ.get("SPARK_DRIVER_MEMORY", "2g")
    return (SparkSession.builder.master(f"local[{cores}]").appName(app)
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.memory", mem)
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def load(spark, path):
    """Read a PaySim-style CSV, selecting and casting columns BY NAME.

    * extra columns (e.g. PaySim's isFlaggedFraud) are ignored
    * column order does not matter
    * if txnId is absent a unique id is generated
    """
    raw = spark.read.csv(path, header=True)
    cols = set(raw.columns)
    missing = [f.name for f in SCHEMA.fields if f.name != "txnId" and f.name not in cols]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}. "
                         f"Found: {sorted(cols)}. Run src/prepare_paysim.py first.")
    sel = []
    for f in SCHEMA.fields:
        if f.name in cols:
            sel.append(F.col(f.name).cast(f.dataType).alias(f.name))
        else:  # txnId
            sel.append(F.monotonically_increasing_id().alias("txnId"))
    return raw.select(*sel)


def build_features(df):
    """Return df with engineered columns (all groups)."""
    for t in TYPES:
        df = df.withColumn(f"is_{t.lower()}", (F.col("type") == t).cast("double"))

    # --- balance-consistency features (classic for mobile-money fraud) --------
    df = (df
          .withColumn("errorBalanceOrig", F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"))
          .withColumn("errorBalanceDest", F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"))
          .withColumn("amount_ratio", F.col("amount") / (F.col("oldbalanceOrg") + 1.0))
          .withColumn("is_drain", ((F.col("oldbalanceOrg") > 0) &
                                   (F.col("newbalanceOrig") == 0)).cast("double"))
          .withColumn("is_drain_pre", ((F.col("oldbalanceOrg") > 0) &
                                       (F.col("amount") >= F.col("oldbalanceOrg")) &
                                       (F.col("type").isin("TRANSFER", "CASH_OUT"))).cast("double"))
          .withColumn("dest_empty_before", (F.col("oldbalanceDest") == 0).cast("double")))

    # --- temporal features -----------------------------------------------------
    df = (df.withColumn("hour", (F.col("step") % 24).cast("double"))
            .withColumn("is_night", ((F.col("step") % 24) < 6).cast("double")))

    # --- behavioural / velocity features (strictly past) -----------------------
    w_o24 = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-24, -1)
    w_dst = (Window.partitionBy("nameDest").orderBy("step")
             .rangeBetween(Window.unboundedPreceding, -1))
    df = (df
          .withColumn("orig_cnt_24h", F.count(F.lit(1)).over(w_o24).cast("double"))
          .withColumn("orig_amt_24h", F.coalesce(F.sum("amount").over(w_o24), F.lit(0.0)))
          .withColumn("dest_prior_cnt", F.count(F.lit(1)).over(w_dst).cast("double"))
          .withColumn("dest_prior_senders", F.approx_count_distinct("nameOrig").over(w_dst).cast("double")))
    return df
