import pyarrow.csv as pv, pyarrow as pa, pyarrow.compute as pc, numpy as np, collections, time
OPTS = dict(parse_options=pv.ParseOptions(delimiter="\t", newlines_in_values=False),
            convert_options=pv.ConvertOptions(column_types={"source1_entity_id":pa.string(),"matched_entity_ids":pa.string()},
                                              strings_can_be_null=True, null_values=[]))
t0=time.time()
gt = pv.read_csv("dataset/train/train_ground_truth.tsv", **OPTS)
print(f"gt rows {gt.num_rows:,}  ({time.time()-t0:.1f}s)")
s1 = gt["source1_entity_id"].to_numpy(zero_copy_only=False)
m  = gt["matched_entity_ids"]
print("gt null matched:", m.null_count)
m = m.fill_null("").to_numpy(zero_copy_only=False)

n_match=np.zeros(len(m),dtype=np.int32); n_s2=np.zeros(len(m),dtype=np.int32); n_s3=np.zeros(len(m),dtype=np.int32)
for i,s in enumerate(m):
    if not s: continue
    parts=s.split(",")
    n_match[i]=len(parts)
    a=b=0
    for p in parts:
        if p[1]=="2": a+=1
        else: b+=1
    n_s2[i]=a; n_s3[i]=b
N=len(m)
print(f"\nunique s1 in gt: {len(set(s1.tolist())):,}   (rows {N:,})")
print(f"\n--- match-count distribution (total matches per S1) ---")
cnt=collections.Counter(n_match.tolist())
cum=0
for k in sorted(cnt):
    if k>10: break
    cum+=cnt[k]
    print(f"  {k:>3} matches: {cnt[k]:>9,}  {cnt[k]/N*100:6.2f}%   cum {cum/N*100:6.2f}%")
big=sum(v for k,v in cnt.items() if k>10)
print(f"  >10      : {big:>9,}  {big/N*100:6.2f}%")
print(f"  mean={n_match.mean():.3f} max={n_match.max()} total_matches={n_match.sum():,}")
print(f"\nSINGLETON RATE (baseline macro-F0.5 floor for predict-nothing): {cnt[0]/N:.4f}")
print(f"\n--- bucket ---")
for name,mask in [("0",n_match==0),("1",n_match==1),("2",n_match==2),("3+",n_match>=3)]:
    print(f"  {name:>3}: {mask.sum():>9,}  {mask.sum()/N*100:6.2f}%")
print(f"\n--- source composition among non-singletons ---")
nz = n_match>0
both=((n_s2>0)&(n_s3>0)).sum(); only2=((n_s2>0)&(n_s3==0)).sum(); only3=((n_s2==0)&(n_s3>0)).sum()
print(f"  S2 only : {only2:>9,} ({only2/nz.sum()*100:5.2f}% of non-singletons)")
print(f"  S3 only : {only3:>9,} ({only3/nz.sum()*100:5.2f}%)")
print(f"  both    : {both:>9,} ({both/nz.sum()*100:5.2f}%)")
print(f"  total S2 matches {n_s2.sum():,}   total S3 matches {n_s3.sum():,}")
print(f"\n  n_s2 dist:", dict(sorted(collections.Counter(n_s2[nz].tolist()).items())[:8]))
print(f"  n_s3 dist:", dict(sorted(collections.Counter(n_s3[nz].tolist()).items())[:8]))
np.save("work/gt_nmatch.npy", n_match); np.save("work/gt_ns2.npy", n_s2); np.save("work/gt_ns3.npy", n_s3)
