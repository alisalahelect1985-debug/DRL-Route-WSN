"""Tables 3-6, statistics, ablation, scalability, distributed gateway and Figs. 1-3 from results_*.csv."""
import json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = {}
rng = np.random.default_rng(1)
BASE = ["LEACH", "PEGASIS", "MTE-Tree", "EA-SPT", "EA-Greedy"]

def load(g):
    df = pd.read_csv(f"results_{g}.csv")
    df["variant"] = df["variant"].fillna("")
    df["model"] = df["model"].fillna("")
    # the glob c1_g0_m* also matched the MAC-trained agents c1_g0_mac_m*: keep those only where they are meant
    df = df[~(df.model.str.contains("_mac_") & (df.variant != "random access, retrained"))].copy()
    df["name"] = df.protocol + np.where(df.variant != "", " (" + df.variant + ")", "")
    return df

def per_topo(df, cols=("FND", "HND", "LND", "PDR", "delivered", "J_per_reading")):
    cols = [c for c in cols if c in df.columns]
    return df.groupby(["scen", "c", "N", "topo", "name"], as_index=False)[cols].mean()

def boot_ci(x, B=10000):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

def compare(pt, scen, ref, others, metric="FND"):
    sub = pt[pt.scen == scen].pivot(index="topo", columns="name", values=metric)
    res = []
    for b in others:
        d = 100 * (sub[ref] / sub[b] - 1)
        try: p = wilcoxon(sub[ref], sub[b]).pvalue
        except ValueError: p = 1.0
        lo, hi = boot_ci(d)
        res.append(dict(scen=scen, metric=metric, ref=ref, vs=b, mean=d.mean(), lo=lo, hi=hi,
                        wins=int((sub[ref] > sub[b]).sum()), n=len(d), p=p))
    adj = holm([r["p"] for r in res])
    for r, a in zip(res, adj): r["p_holm"] = a
    return res

def fmt(m, s): return f"{m:.0f} ± {s:.0f}"

# ================= main (Table 3, Fig. 2) =================
main = load("main"); pt = per_topo(main)
order = BASE + ["DRL-Route (gamma=0)"]
t3 = []
for s in ["S1", "S2"]:
    for p in order:
        x = pt[(pt.scen == s) & (pt.name == p)]
        t3.append(dict(scen=s, protocol=p, FND=fmt(x.FND.mean(), x.FND.std()), HND=fmt(x.HND.mean(), x.HND.std()),
                       LND=fmt(x.LND.mean(), x.LND.std()), PDR=round(100 * x.PDR.mean(), 1),
                       readings=round(x.delivered.mean() / 1000, 1), J_per_reading_mJ=round(1000 * x.J_per_reading.mean(), 3),
                       FND_m=x.FND.mean(), HND_m=x.HND.mean(), LND_m=x.LND.mean()))
t3 = pd.DataFrame(t3); OUT["table3"] = t3.to_dict("records")
print("=== Table 3 ==="); print(t3[["scen", "protocol", "FND", "HND", "LND", "PDR", "readings", "J_per_reading_mJ"]].to_string(index=False))
cmp3 = []
for s in ["S1", "S2"]:
    for m in ["FND", "HND", "delivered"]:
        cmp3 += compare(pt, s, "DRL-Route (gamma=0)", BASE, m)
cmp3 = pd.DataFrame(cmp3); OUT["cmp3"] = cmp3.to_dict("records")
pd.set_option("display.width", 220)
print(cmp3.round(4).to_string(index=False))
# per-model FND (S1) and die-off span
dm = main[(main.protocol == "DRL-Route")]
OUT["models_S1"] = dm[dm.scen == "S1"].groupby("model").FND.mean().round(1).to_dict()
print("per-model S1 FND:", OUT["models_S1"])
for s in ["S1", "S2"]:
    x = pt[(pt.scen == s) & (pt.name == "DRL-Route (gamma=0)")]
    OUT[f"dieoff_{s}"] = float((x.LND - x.FND).mean())
print("die-off span (LND-FND) DRL:", OUT["dieoff_S1"], OUT["dieoff_S2"])
ex = [json.loads(l) for l in open("extra_main.jsonl")]
ag = np.array([e["agree"] for e in ex if "agree" in e]); de = np.array([e["direct_early"] for e in ex if "agree" in e]); dl = np.array([e["direct_late"] for e in ex if "agree" in e])
OUT["agree"] = float(ag[:, 0].sum() / ag[:, 1].sum()); OUT["direct_early"] = float(de[:, 0].sum() / de[:, 1].sum()); OUT["direct_late"] = float(dl[:, 0].sum() / max(dl[:, 1].sum(), 1))
print("agreement with EA-Greedy %.3f, direct early %.3f late %.3f" % (OUT["agree"], OUT["direct_early"], OUT["direct_late"]))

# ---- Fig. 2: alive nodes vs round
COL = {"MTE-Tree": "#2a78d6", "EA-SPT": "#eb6834", "EA-Greedy": "#1baf7a", "DRL-Route": "#eda100", "LEACH": "#008300", "PEGASIS": "#6250d6"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 11})
def style(ax):
    ax.grid(axis="y", color="#e5e4e0", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#8a8984"); ax.spines["bottom"].set_color("#8a8984")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=250, sharey=True)
deaths = {}
for e in ex:
    g, s, c, n, i, proto, var, model = e["key"]
    deaths.setdefault((s, proto), []).append(np.array(e["deaths"]))
T = np.arange(0, 1801)
for ax, s, title in zip(axes, ["S1", "S2"], ["(a) S1: sink outside the field", "(b) S2: sink at the centre"]):
    for proto in ["MTE-Tree", "EA-SPT", "EA-Greedy", "DRL-Route", "LEACH", "PEGASIS"]:
        arr = deaths[(s, proto)]
        curves = np.array([[100 - np.sum((d >= 0) & (d <= t)) for t in T[::5]] for d in arr]).mean(0)
        ax.plot(T[::5], curves, color=COL[proto], lw=2, label=proto)
    ax.set_title(title, fontsize=11); ax.set_xlabel("Round"); style(ax); ax.set_xlim(0, 1700); ax.set_ylim(0, 102); ax.set_xticks(range(0, 1701, 300))
axes[0].set_ylabel("Alive nodes")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig2_alive.png", facecolor="white"); plt.close(fig)

# ================= ablation =================
ab = pd.concat([main[(main.scen == "S1") & (main.topo < 10) & (main.protocol == "DRL-Route")], load("ablation")])
abt = ab.groupby(["topo", "name"], as_index=False)[["FND", "LND", "J_per_reading"]].mean().groupby("name")[["FND", "LND", "J_per_reading"]].mean()
OUT["ablation"] = abt.round(4).reset_index().to_dict("records")
print("=== ablation (S1, 10 topologies) ==="); print(abt.round(5))

# ================= scalability (Table 4) =================
sc = pd.concat([main[(main.scen == "S1")], load("scale")])
spt = per_topo(sc, ("FND", "HND"))
t4 = spt.groupby(["name", "N"]).FND.agg(["mean", "std"]).round(0)
OUT["table4"] = {f"{k[0]}|{k[1]}": [float(v["mean"]), float(v["std"])] for k, v in t4.iterrows()}
print("=== Table 4 FND ==="); print(t4.unstack("N"))
for n in [50, 100, 150, 200]:
    x = spt[spt.N == n].pivot(index="topo", columns="name", values="FND")
    d = 100 * (x["DRL-Route (gamma=0)"] / x["EA-Greedy"] - 1)
    OUT[f"scale_vs_greedy_{n}"] = [float(d.mean()), int((d > 0).sum()), len(d)]
    print(n, "DRL vs EA-Greedy %.1f%% wins %d/%d" % (d.mean(), (d > 0).sum(), len(d)))
h200 = spt[spt.N == 200].groupby("name").HND.mean(); OUT["hnd200"] = h200.round(0).to_dict(); print("HND N=200:", h200.round(0).to_dict())

# ================= no aggregation (Table 5, Fig. 3) =================
na = load("noagg"); npt = per_topo(na, ("FND", "HND"))
names5 = BASE + ["DRL-Route (gamma=0)", "DRL-Route (gamma=0.9)"]
t5 = []
for p in names5:
    r = dict(protocol=p)
    for s in ["S1", "S2"]:
        x = npt[(npt.scen == s) & (npt.name == p)]
        r[f"FND_{s}"] = fmt(x.FND.mean(), x.FND.std()); r[f"HND_{s}"] = fmt(x.HND.mean(), x.HND.std())
    t5.append(r)
t5 = pd.DataFrame(t5); OUT["table5"] = t5.to_dict("records"); print("=== Table 5 ==="); print(t5.to_string(index=False))
cmp5 = []
for s in ["S1", "S2"]:
    cmp5 += compare(npt, s, "DRL-Route (gamma=0.9)", BASE + ["DRL-Route (gamma=0)"], "FND")
cmp5 = pd.DataFrame(cmp5); OUT["cmp5"] = cmp5.to_dict("records"); print(cmp5.round(4).to_string(index=False))
# Fig. 3
COL5 = {"LEACH": "#008300", "PEGASIS": "#6250d6", "MTE-Tree": "#2a78d6", "EA-SPT": "#eb6834", "EA-Greedy": "#1baf7a",
        "DRL-Route (gamma=0)": "#eda100", "DRL-Route (gamma=0.9)": "#e87ba4"}
LAB = {"DRL-Route (gamma=0)": "DRL-Route\n(γ = 0)", "DRL-Route (gamma=0.9)": "DRL-Route\n(γ = 0.9)", "EA-SPT": "EA-SPT\n(oracle)"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=250)
for ax, s, title in zip(axes, ["S1", "S2"], ["(a) S1: sink outside the field", "(b) S2: sink at the centre"]):
    xs = np.arange(len(names5)); w = 0.38
    for k, p in enumerate(names5):
        x = npt[(npt.scen == s) & (npt.name == p)]
        for j, (met, alpha) in enumerate([("FND", 1.0), ("HND", 0.45)]):
            v = x[met].values; m = v.mean(); ci = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(k + (j - 0.5) * w, m, w * 0.92, color=COL5[p], alpha=alpha, yerr=ci, capsize=2,
                   error_kw=dict(lw=0.9, ecolor="#52514e"), edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([LAB.get(p, p) for p in names5], fontsize=8.5)
    ax.set_title(title, fontsize=11); style(ax)
axes[0].set_ylabel("Rounds")
from matplotlib.patches import Patch
fig.legend([Patch(color="#52514e"), Patch(color="#52514e", alpha=0.45)], ["FND (dark)", "HND (light)"], loc="lower center",
           ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("fig3_noagg.png", facecolor="white"); plt.close(fig)

# ================= MAC (Table 6) and distributed gateway =================
mac = load("mac"); mpt = per_topo(mac, ("FND", "HND"))
t6 = mpt.groupby(["name", "scen"])[["FND", "HND"]].mean().round(0).unstack("scen")
OUT["table6"] = {k: [float(v) for v in row.values] for k, row in t6.iterrows()}
print("=== Table 6 ==="); print(t6)
macc = []
for s in ["S1", "S2"]:
    sub = mpt[mpt.scen == s]
    for m in ["FND", "HND"]:
        x = sub.pivot(index="topo", columns="name", values=m)
        for a, b in [("DRL-Route (random access)", "PEGASIS (scheduled)"), ("DRL-Route (TDMA)", "PEGASIS (scheduled)"),
                     ("DRL-Route (random access, retrained)", "DRL-Route (random access)"),
                     ("DRL-Route (random access)", "EA-Greedy (random access)"), ("DRL-Route (TDMA)", "EA-Greedy (TDMA)"),
                     ("DRL-Route (random access)", "LEACH (scheduled)"), ("DRL-Route (TDMA)", "LEACH (scheduled)"),
                     ("DRL-Route (random access)", "DRL-Route (TDMA)")]:
            if a not in x or b not in x: continue
            d = 100 * (x[a] / x[b] - 1)
            try: p = wilcoxon(x[a], x[b]).pvalue
            except ValueError: p = 1.0
            macc.append(dict(scen=s, metric=m, a=a, b=b, mean=d.mean(), wins=int((x[a] > x[b]).sum()), p=p))
macc = pd.DataFrame(macc); OUT["mac_cmp"] = macc.to_dict("records"); print(macc.round(4).to_string(index=False))
# mac EA-Greedy names
dist = load("dist"); dpt = per_topo(dist, ("FND", "HND"))
for s in ["S1", "S2"]:
    base = pt[(pt.scen == s) & (pt.topo < 10) & (pt.name == "DRL-Route (gamma=0)")].set_index("topo").FND
    dd = dpt[(dpt.scen == s)].pivot(index="topo", columns="name", values="FND")
    drl, gr = dd["DRL-Route (distributed gateway)"], dd["EA-Greedy (distributed gateway)"] if "EA-Greedy (distributed gateway)" in dd else dd["EA-Greedy"]
    OUT[f"dist_{s}"] = dict(drl=float(drl.mean()), greedy=float(gr.mean()), drl_vs_sinkgw=float(100 * np.mean(drl / base - 1)),
                           greedy_vs_drl=float(100 * np.mean(gr / drl - 1)), p=float(wilcoxon(gr, drl).pvalue))
    print("dist", s, OUT[f"dist_{s}"])

# ================= Fig. 1: validation curves =================
def curves(pat):
    fs = sorted(glob.glob(f"models/{pat}_val.csv"))
    arr = np.array([pd.read_csv(f, header=None)[1].values for f in fs]); ep = pd.read_csv(fs[0], header=None)[0].values
    return ep, arr
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=250)
for ax, cfg, title in [(axes[0], [("c1_g0_m?", "γ = 0", "#eda100", "o"), ("c1_g05_m?", "γ = 0.5", "#6250d6", "s"), ("c1_g09_m?", "γ = 0.9", "#e87ba4", "v")], "(a) Perfect aggregation (c = 1)"),
                       (axes[1], [("c0_g0_m?", "γ = 0", "#eda100", "o"), ("c0_g09_m?", "γ = 0.9", "#e87ba4", "v")], "(b) No aggregation (c = 0)")]:
    for pat, lab, col, mk in cfg:
        ep, a = curves(pat)
        ax.fill_between(ep, a.min(0), a.max(0), color=col, alpha=0.18, lw=0)
        ax.plot(ep, a.mean(0), color=col, lw=2, marker=mk, ms=6, mec="white", mew=1, label=lab)
        OUT[f"val_{pat}"] = dict(first=float(a[:, 0].mean()), last=float(a[:, -1].mean()), peak=float(a.mean(0).max()))
    ax.set_title(title, fontsize=11); ax.set_xlabel("Training episode"); style(ax); ax.legend(frameon=False, fontsize=10)
axes[0].set_ylabel("Validation FND (rounds)")
fig.tight_layout(); fig.savefig("fig1_training.png", facecolor="white"); plt.close(fig)
print({k: v for k, v in OUT.items() if k.startswith("val_")})

# ================= seed separation =================
log = pd.read_csv("topo_log.csv", names=["phase", "run", "idx", "hash"])
tr = set(log[log.phase == "train"].hash); te = log[log.phase == "test"]
te = te.assign(key=te.run.str.replace(r"^(main|ablation|scale|noagg|mac|dist)_", "", regex=True) + "_" + te.idx.astype(str))
OUT["seed_check"] = dict(train=len(tr), test_runs=len(te), overlap=len(tr & set(te.hash)))
print("seed check:", OUT["seed_check"])
json.dump(OUT, open("paper_numbers.json", "w"), indent=1, default=float)
