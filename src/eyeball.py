import pyarrow.csv as pv, pyarrow as pa, numpy as np, random, sys
OPTS = dict(parse_options=pv.ParseOptions(delimiter="\t", newlines_in_values=False),
            convert_options=pv.ConvertOptions(column_types={c:pa.string() for c in
              ["entity_id","business_name","business_address","country","source1_entity_id","matched_entity_ids"]},
              strings_can_be_null=True, null_values=[]))
def rd(p): return pv.read_csv(p, **OPTS)
random.seed(7)
gt=rd("dataset/train/train_ground_truth.tsv")
s1id=gt["source1_entity_id"].to_pylist(); mm=gt["matched_entity_ids"].to_pylist()
# pick 60 random non-singleton entities
idx=[i for i in random.sample(range(len(s1id)), 4000) if mm[i]][:60]
want1=set(s1id[i] for i in idx); want23=set()
for i in idx: want23.update(mm[i].split(","))
def index(p, want):
    t=rd(p); ids=t["entity_id"].to_pylist(); nm=t["business_name"].to_pylist(); ad=t["business_address"].to_pylist(); co=t["country"].to_pylist()
    return {e:(nm[j],ad[j],co[j]) for j,e in enumerate(ids) if e in want}
d={}
d.update(index("dataset/train/train_source1.tsv", want1))
d.update(index("dataset/train/train_source2.tsv", want23))
d.update(index("dataset/train/train_source3.tsv", want23))
for k,i in enumerate(idx[:25]):
    a=d.get(s1id[i]); print(f"\n--- [{k}] {s1id[i]} | {a[2]} ---")
    print(f"   S1 NAME: {a[0]!r}\n   S1 ADDR: {a[1]!r}")
    for mid in mm[i].split(","):
        b=d.get(mid)
        if b: print(f"   {mid[:2]} MATCH  name={b[0]!r}  addr={b[1]!r}  co={b[2]}")
