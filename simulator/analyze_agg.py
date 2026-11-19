"""Summarise results_agg.csv: FND per protocol vs c, DRL-Route vs EA-Greedy, best gamma, Fig. 4."""
import json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results_agg.csv")
df["proto"] = np.where(df.protocol == "DRL-Route", "DRL-Route (γ = " + df.gamma.astype(str).str.replace(".0", "", regex=False) + ")", df.protocol)
df.loc[df.proto == "DRL-Route (γ = 0.9)", "proto"] = "DRL-Route (γ = 0.9)"
g = df.groupby(["scen", "c", "topo", "proto"], as_index=False)[["FND", "HND", "PDR"]].mean()
CS = sorted(df.c.unique())
PR = ["LEACH", "PEGASIS", "MTE-Tree", "EA-SPT", "EA-Greedy", "DRL-Route (γ = 0)", "DRL-Route (γ = 0.9)"]

mean = g.groupby(["scen", "proto", "c"]).FND.mean().unstack("c")
print("=== mean FND ==="); print(mean.round(0).to_string())

rows, table = [], []
for s in ["S1", "S2"]:
    for c in CS:
        sub = g[(g.scen == s) & (g.c == c)].pivot(index="topo", columns="proto", values="FND")
        m0, m9 = sub["DRL-Route (γ = 0)"].mean(), sub["DRL-Route (γ = 0.9)"].mean()
        best = "DRL-Route (γ = 0)" if m0 >= m9 else "DRL-Route (γ = 0.9)"
        out = dict(scen=s, c=c, best_gamma=best[-4:-1].strip("= "), FND_g0=m0, FND_g09=m9)
        for b in ["EA-Greedy", "EA-SPT", "MTE-Tree"]:
            d = 100 * (sub[best] / sub[b] - 1)
            try: p = wilcoxon(sub[best], sub[b]).pvalue
            except ValueError: p = float("nan")
            out[f"vs_{b}"] = d.mean(); out[f"wins_{b}"] = int((sub[best] > sub[b]).sum()); out[f"p_{b}"] = p
            out[f"min_{b}"] = d.min(); out[f"max_{b}"] = d.max()
        rows.append(out)
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== best DRL-Route vs baselines ==="); print(res.round(3).to_string(index=False))
res.to_csv("agg_summary.csv", index=False)
mean.to_csv("agg_mean_fnd.csv")

# -------- Fig. 4: FND vs aggregation coefficient
COL = {"MTE-Tree": "#2a78d6", "EA-SPT": "#eb6834", "EA-Greedy": "#1baf7a",
       "DRL-Route (γ = 0)": "#eda100", "DRL-Route (γ = 0.9)": "#e87ba4"}
MK = {"MTE-Tree": "s", "EA-SPT": "D", "EA-Greedy": "^", "DRL-Route (γ = 0)": "o", "DRL-Route (γ = 0.9)": "v"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=250)
for ax, s, title in zip(axes, ["S1", "S2"], ["(a) S1: sink outside the field", "(b) S2: sink at the centre"]):
    for p in COL:
        y = mean.loc[(s, p)].values
        ax.plot(CS, y, color=COL[p], lw=2, marker=MK[p], ms=7, mec="white", mew=1, label=p, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Aggregation coefficient c")
    ax.set_xticks(CS)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#8a8984"); ax.spines["bottom"].set_color("#8a8984")
    ax.set_ylim(0, None)
axes[0].set_ylabel("FND (rounds)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("fig4_aggregation.png", facecolor="white")
print("saved fig4_aggregation.png")
