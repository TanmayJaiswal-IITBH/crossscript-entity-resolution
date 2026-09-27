"""Assemble the final submission zip in the structure the challenge requires.

<team>_submission.zip
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/
      src/*.py, config/, work/<model files>, README.md,
      requirements.txt, requirements-embed.txt, embed_env.sh
  Documentation_template.md, RESULTS.md
"""
import argparse
import glob
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILES = ("matcher_lgb.txt", "matcher_calib.pkl", "best_rule.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="team")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    args = ap.parse_args()

    stage = os.path.join(ROOT, "work", "_package")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    code = os.path.join(stage, "code", "business_entity_resolution")
    for d in ("src", "config", "work"):
        os.makedirs(os.path.join(code, d))
    os.makedirs(os.path.join(stage, "output"))

    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = os.path.join(args.out_dir, f)
        if not os.path.exists(p):
            sys.exit("missing %s -- run the test pipeline first" % p)
        shutil.copy2(p, os.path.join(stage, "output", f))
    # every source file currently in src/ -- no hand-maintained list to go stale
    for p in glob.glob(os.path.join(ROOT, "src", "*.py")):
        shutil.copy2(p, os.path.join(code, "src", os.path.basename(p)))
    for p in glob.glob(os.path.join(ROOT, "config", "*")):
        shutil.copy2(p, os.path.join(code, "config", os.path.basename(p)))
    for f in MODEL_FILES:
        shutil.copy2(os.path.join(ROOT, "work", f), os.path.join(code, "work", f))
    for f in ("requirements.txt", "requirements-embed.txt", "embed_env.sh"):
        shutil.copy2(os.path.join(ROOT, f), os.path.join(code, f))
    shutil.copy2(os.path.join(ROOT, "PIPELINE_README.md"), os.path.join(code, "README.md"))
    for f in ("Documentation_template.md", "RESULTS.md"):
        shutil.copy2(os.path.join(ROOT, f), os.path.join(stage, f))

    zpath = os.path.join(ROOT, "%s_submission.zip" % args.team)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _dirs, files in os.walk(stage):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, stage))
    print("wrote %s (%.1f MB)" % (zpath, os.path.getsize(zpath) / 1e6))


if __name__ == "__main__":
    main()
