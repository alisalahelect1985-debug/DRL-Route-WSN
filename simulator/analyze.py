"""Summarise results.csv and check seed separation."""
import csv, collections
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("results.csv")
# average the DRL-Route models per topology first (as in the manuscript)
g = df.groupby(["scen", "c", "channel", "topo", "protocol"], as_index=False)[["FND", "HND", "PDR"]].mean()
CH = ["static", "slow", "fast", "severe"]
PR = ["LEACH", "PEGASIS", "MTE-Tree", "EA-SPT", "EA-Greedy", "DRL-Route"]

print("=== mean FND / HND / PDR ===")
summ = g.groupby(["c", "scen", "protocol", "channel"])[["FND", "HND", "PDR"]].agg(["mean", "std"])
for c in [1.0, 0.0]:
    for s in ["S1", "S2"]:
        print(f"\n-- c={c} {s}")
        for p in PR:
            line = f"{p:10s}"
            for ch in CH:
                try:
                    m = summ.loc[(c, s, p, ch)]
                    line += f" | {m[('FND','mean')]:7.1f}±{m[('FND','std')]:5.1f} {m[('HND','mean')]:7.1f} {100*m[('PDR','mean')]:5.1f}"
                except KeyError:
                    line += " | -"
            print(line)

print("\n=== FND change vs static (paired, % mean over topologies) ===")
piv = g.pivot_table(index=["c", "scen", "protocol", "topo"], columns="channel", values="FND")
rows = []
for (c, s, p), sub in piv.groupby(level=[0, 1, 2]):
    out = {"c": c, "scen": s, "protocol": p}
    for ch in CH[1:]:
        out[ch] = 100 * np.mean(sub[ch] / sub["static"] - 1)
    rows.append(out)
chg = pd.DataFrame(rows)
print(chg.round(1).to_string(index=False))

print("\n=== DRL-Route vs baselines: per-topology FND change %, [min,max], Wilcoxon p ===")
for c in [1.0, 0.0]:
    for s in ["S1", "S2"]:
        for ch in CH:
            sub = g[(g.c == c) & (g.scen == s) & (g.channel == ch)].pivot(index="topo", columns="protocol", values="FND")
            for b in ["EA-Greedy", "MTE-Tree", "EA-SPT", "PEGASIS", "LEACH"]:
                d = 100 * (sub["DRL-Route"] / sub[b] - 1)
                try:
                    p = wilcoxon(sub["DRL-Route"], sub[b]).pvalue
                except ValueError:
                    p = float("nan")
                wins = int((sub["DRL-Route"] > sub[b]).sum())
                print(f"c={c} {s} {ch:7s} vs {b:9s}: {d.mean():+7.1f}% [{d.min():+.1f},{d.max():+.1f}] wins {wins}/{len(d)} p={p:.4f}")

print("\n=== seed separation check ===")
log = pd.read_csv("topo_log.csv", names=["phase", "run", "idx", "hash"])
tr = set(log[log.phase == "train"].hash); te = log[log.phase == "test"]
print("train topologies:", len(tr), " test runs logged:", len(te))
print("train ∩ test hashes:", len(tr & set(te.hash)))
te = te.assign(key=te.run.str.split("_").str[0] + "_" + te.idx.astype(str))
u = te.groupby("key").hash.nunique()
print("distinct hashes per (scenario, test topology):", u.value_counts().to_dict())
