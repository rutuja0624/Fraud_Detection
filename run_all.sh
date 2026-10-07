#!/usr/bin/env bash
# Run the complete experiment suite on one dataset.
#
#   ./run_all.sh <data.csv> <results_dir> [--auto]
#
# Examples
#   ./run_all.sh data/transactions.csv results/synthetic          # synthetic data (paper v1)
#   ./run_all.sh data/paysim.csv results/paysim --auto            # real PaySim (rarer fraud)
#
# --auto  : alert budgets and amount bands are derived from the data (needed for PaySim,
#           whose fraud rate (~0.13 %) and amounts differ from the synthetic data)
#
# Optional environment variables: SPARK_DRIVER_MEMORY (default 2g; use 8g+ for full PaySim),
# FEATURE_SET (full|safe|nobal; use nobal for real PaySim), TRAIN_END (479), VAL_END (599), GBT_ITER (40), RF_TREES (40), SKIP_SCALABILITY=1
set -euo pipefail
DATA=${1:?usage: ./run_all.sh <data.csv> <results_dir> [--auto]}
OUT=${2:?usage: ./run_all.sh <data.csv> <results_dir> [--auto]}
AUTO=${3:-}
mkdir -p "$OUT"
TRAIN_END=${TRAIN_END:-479}; VAL_END=${VAL_END:-599}
GBT_ITER=${GBT_ITER:-40}; RF_TREES=${RF_TREES:-40}
FEATURE_SET=${FEATURE_SET:-full}   # full (synthetic) | safe (no post-txn balances) | nobal (no balances at all; REAL PaySim)
ABS=$(cd "$(dirname "$DATA")" && pwd)/$(basename "$DATA")
OUTABS=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
cd "$(dirname "$0")/src"

echo "== 1/6 train + evaluate"
python train_evaluate.py --data "$ABS" --out "$OUTABS" --train-end "$TRAIN_END" --val-end "$VAL_END" \
       --gbt-iter "$GBT_ITER" --rf-trees "$RF_TREES" --feature-set "$FEATURE_SET" 2>&1 | tee "$OUTABS/train_log.txt" | grep -E "^\[|Error|Traceback"
echo "== 2/6 bootstrap confidence intervals"
python bootstrap_ci.py "$OUTABS"
if [ "$AUTO" = "--auto" ]; then BUD="--budgets auto"; BANDS="--bands auto"; else BUD=""; BANDS=""; fi
echo "== 3/6 error analysis";  python error_analysis.py "$OUTABS" $BANDS > /dev/null
echo "== 4/6 alert budgets";   python alert_budget.py "$OUTABS" $BUD
echo "== 5/6 figures";         python make_figures.py "$OUTABS"
if [ "${SKIP_SCALABILITY:-0}" != "1" ]; then
  echo "== 6/6 scalability"
  python scalability.py --data "$ABS" --out "$OUTABS/scalability.json" --gbt-iter "$GBT_ITER" \
         --feature-set "$FEATURE_SET" 2>&1 | tee "$OUTABS/scalability_log.txt" | grep -E "^\{|Error|Traceback"
fi
echo "All done -> $OUTABS"
