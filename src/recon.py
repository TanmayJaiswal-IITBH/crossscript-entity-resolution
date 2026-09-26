import pyarrow.csv as pv, pyarrow.compute as pc, pyarrow as pa, time, os, collections

OPTS = dict(parse_options=pv.ParseOptions(delimiter="\t", newlines_in_values=False),
            read_options=pv.ReadOptions(block_size=1<<26),
            convert_options=pv.ConvertOptions(column_types={"entity_id":pa.string(),"business_name":pa.string(),
                                                            "business_address":pa.string(),"country":pa.string(),
                                                            "source1_entity_id":pa.string(),"matched_entity_ids":pa.string()},
                                              strings_can_be_null=True, null_values=[""]))
def load(p):
    t0=time.time(); tb = pv.read_csv(p, **OPTS)
    print(f"  loaded {os.path.basename(p)}: {tb.num_rows:,} rows x {tb.num_columns} cols  ({time.time()-t0:.1f}s)")
    return tb

def profile(name, tb):
    print(f"\n### {name}  rows={tb.num_rows:,}")
    for c in tb.column_names:
        col = tb[c]
        nulls = col.null_count
        empt = 0
        print(f"  {c:20s} nulls={nulls:>9,} ({nulls/tb.num_rows*100:5.2f}%)")
    if "entity_id" in tb.column_names:
        pref = pc.utf8_slice_codeunits(tb["entity_id"], 0, 3)
        vc = pc.value_counts(pref)
        print("  id prefixes:", {v["values"]:v["counts"] for v in vc.to_pylist()})
        nuniq = pc.count_distinct(tb["entity_id"]).as_py()
        print(f"  unique entity_id = {nuniq:,} (dupes: {tb.num_rows-nuniq:,})")
    if "country" in tb.column_names:
        vc = pc.value_counts(tb["country"])
        d = sorted(({(v['values'] if v['values'] is not None else 'NULL'):v['counts']} for v in vc.to_pylist()), key=lambda x:-list(x.values())[0])
        print("  country:", {k:v for dd in d for k,v in dd.items()})

base="dataset"
for split in ["train","test"]:
    for s in ["source1","source2","source3"]:
        p=f"{base}/{split}/{split}_{s}.tsv"
        if os.path.exists(p):
            profile(f"{split}_{s}", load(p))
