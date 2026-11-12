"""Post hoc for test 8b (results/test8b_gated.md): test 8's full union adapter against test 8b's reference (own adapter if
kept, else base), and both adapters against the base. Run inside WSL from the repo root: python phase0/test8b_posthoc.py"""
import sys
sys.path.insert(0, "phase0")
import test7_families as T
import test8_union as U
import test8b_gated as G
for forgive in (False, True):
    rows = {f: G.rates(f, forgive) for f in T.FAMILIES}
    full_ref = U.stratified_boot([(ref, full) for _, ref, full, _ in rows.values()])
    gated_base = U.stratified_boot([(base, gated) for base, _, _, gated in rows.values()])
    full_base = U.stratified_boot([(base, full) for base, _, full, _ in rows.values()])
    print("forgive" if forgive else "as scored", "| full minus 8b reference:", U.ci(full_ref),
          "| gated minus base:", U.ci(gated_base), "| full minus base:", U.ci(full_base))
