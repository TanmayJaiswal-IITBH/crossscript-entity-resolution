set -e
export PYTHONHASHSEED=0
until [ -f work/test_s3_norm.parquet ]; do sleep 5; done
sleep 3
python -u src/split.py
python -u src/block2.py train
python -u src/block2.py test
python -u src/block3.py train
python -u src/block3.py test
echo "REBUILD COMPLETE"
