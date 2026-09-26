"""Assemble the final submission zip in the structure the challenge requires."""
import os
import sys
import shutil
import zipfile
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the pipeline as shipped; Day-1 experiments that were superseded
# (keys.py/blocking.py/features.py/run_pipeline.py) are kept because the
# methodology document refers to them as the blocking v1/v2 iterations.
SRC_FILES = ["common.py", "normalize.py", "split.py", "evaluate.py",
             "block2.py", "block3.py", "pairs.py", "features2.py", "engine.py",
             "build_training2.py", "train_matcher.py", "decide.py",
             "score_split.py", "run_pipeline2.py", "error_analysis.py",
             "measure_block2.py", "measure_block3.py", "diag_missed.py",
             "diag_tokens.py", "diag_rank.py", "make_package.py",
             # superseded Day-1 iterations, referenced in the write-up
             "keys.py", "blocking.py", "features.py", "run_pipeline.py",
             "measure_blocking.py", "sweep_threshold.py", "build_training.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="team")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    args = ap.parse_args()

    stage = os.path.join(ROOT, "work", "_package")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    code = os.path.join(stage, "code", "business_entity_resolution")
    os.makedirs(os.path.join(code, "src"))
    os.makedirs(os.path.join(stage, "output"))

    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = os.path.join(args.out_dir, f)
        if not os.path.exists(p):
            sys.exit("missing %s -- run the test pipeline first" % p)
        shutil.copy2(p, os.path.join(stage, "output", f))
    for f in SRC_FILES:
        p = os.path.join(ROOT, "src", f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(code, "src", f))
    shutil.copy2(os.path.join(ROOT, "requirements.txt"),
                 os.path.join(code, "requirements.txt"))
    shutil.copy2(os.path.join(ROOT, "PIPELINE_README.md"),
                 os.path.join(code, "README.md"))
    for f in ("Documentation_template.md", "RESULTS.md"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(stage, f))

    zpath = os.path.join(ROOT, "%s_submission.zip" % args.team)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _dirs, files in os.walk(stage):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, stage))
    print("wrote %s (%.1f MB)" % (zpath, os.path.getsize(zpath) / 1e6))


if __name__ == "__main__":
    main()
