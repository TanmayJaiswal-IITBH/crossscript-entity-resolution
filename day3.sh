set -e
export PYTHONHASHSEED=0
until grep -q "^saved" fitwide3.log; do sleep 20; done
echo "=== TRAIN (60 features) ==="
python -u src/train_matcher.py --data work/fitw --folds 4 --rounds 1400 --threads 8
echo "=== SCORE VAL ==="
python -u src/score_split.py --truth work/val_truth.tsv --out work/val_scored_w3.npz \
  --kn 60 --ka 60 --bn 8000 --ba 8000 --chunk 1500 --procs 7
echo "DAY3_MODEL_READY"
