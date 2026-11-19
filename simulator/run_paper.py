"""All experiments of Sections 3.1-3.5 (Tables 3-6, Figs. 2-3, ablation, scalability, distributed gateway).
Every protocol and every trained model runs on identical test topologies (positions, shadowing, energies,
sink); only the ARQ outcomes use run-specific random streams. Test runs with topology index < 10 and the
static channel coincide with the corresponding runs of run_experiments.py and run_aggregation.py.
usage: python run_paper.py [group ...]      groups: main ablation scale noagg mac dist
"""
import sys, csv, json, glob, time
import numpy as np
from multiprocessing import Pool
import params as P, sim, seeds
from dqn import Agent

SCEN = {"S1": 1, "S2": 2}
BASE = ["LEACH", "PEGASIS", "MTE-Tree", "EA-SPT", "EA-Greedy"]
M = lambda pat: sorted(f for f in glob.glob(f"models/{pat}") if "_mac_" not in f or "mac" in pat)

def topo(scen, idx, n):
    if n == 100:
        r = seeds.test_rng(idx); key_init = seeds.rng(seeds.TEST, idx, SCEN[scen])
    else:   # separate test seeds for the other network sizes
        r = seeds.rng(seeds.TEST, idx, 1000 + n); key_init = seeds.rng(seeds.TEST, idx, 1000 + n, SCEN[scen])
    pos, D, E = sim.make_topology(r, n, P.SINKS[scen])
    chan = sim.Channel(D, key_init, sigma=P.SIGMA, rho_t=None, rng_evo=seeds.chan_rng(idx, 0, SCEN[scen]))
    return pos, D, E, chan

def tasks_for(group):
    T = []
    def add(scen, c, idx, proto, variant, model, pid, n=100, mac=None, gw="sink", stop="hnd", diag=False):
        T.append(dict(group=group, scen=scen, c=c, idx=idx, n=n, proto=proto, variant=variant, model=model,
                      pid=pid, mac=mac, gw=gw, stop=stop, diag=diag))
    if group == "main":
        for s in SCEN:
            for i in range(30):
                for k, p in enumerate(BASE): add(s, 1.0, i, p, "", "", k, stop="lnd")
                for k, m in enumerate(M("c1_g0_m*.npz")): add(s, 1.0, i, "DRL-Route", "gamma=0", m, 5 + k, stop="lnd", diag=(s == "S1"))
    if group == "ablation":
        for i in range(10):
            for k, m in enumerate(M("c1_g05_m*.npz")): add("S1", 1.0, i, "DRL-Route", "gamma=0.5", m, 30 + k, stop="lnd")
            for k, m in enumerate(M("c1_g09_m*.npz")): add("S1", 1.0, i, "DRL-Route", "gamma=0.9", m, 20 + k, stop="lnd")
            for k, m in enumerate(M("c1_g0_a0_m*.npz")): add("S1", 1.0, i, "DRL-Route", "alpha=0", m, 40 + k, stop="lnd")
    if group == "scale":
        for n in (50, 150, 200):
            for i in range(10):
                for k, p in enumerate(BASE): add("S1", 1.0, i, p, "", "", k, n=n)
                for k, m in enumerate(M("c1_g0_m*.npz")): add("S1", 1.0, i, "DRL-Route", "gamma=0", m, 5 + k, n=n)
    if group == "noagg":
        for s in SCEN:
            for i in range(20):
                for k, p in enumerate(BASE): add(s, 0.0, i, p, "", "", k)
                for k, m in enumerate(M("c0_g09_m*.npz")): add(s, 0.0, i, "DRL-Route", "gamma=0.9", m, 5 + k)
                for k, m in enumerate(M("c0_g0_m*.npz")): add(s, 0.0, i, "DRL-Route", "gamma=0", m, 20 + k)
    if group == "mac":
        for s in SCEN:
            for i in range(10):
                add(s, 1.0, i, "PEGASIS", "scheduled", "", 101, mac="random")
                add(s, 1.0, i, "LEACH", "scheduled", "", 100, mac="random")
                add(s, 1.0, i, "EA-Greedy", "random access", "", 104, mac="random")
                add(s, 1.0, i, "EA-Greedy", "TDMA", "", 114, mac="tdma")
                for k, m in enumerate(M("c1_g0_m*.npz")):
                    add(s, 1.0, i, "DRL-Route", "random access", m, 105 + k, mac="random")
                    add(s, 1.0, i, "DRL-Route", "TDMA", m, 115 + k, mac="tdma")
                for k, m in enumerate(M("c1_g0_mac_m*.npz")):
                    add(s, 1.0, i, "DRL-Route", "random access, retrained", m, 125 + k, mac="random")
    if group == "dist":
        for s in SCEN:
            for i in range(10):
                add(s, 1.0, i, "EA-Greedy", "distributed gateway", "", 204, gw="dist")
                for k, m in enumerate(M("c1_g0_m*.npz")):
                    add(s, 1.0, i, "DRL-Route", "distributed gateway", m, 205 + k, gw="dist")
    return T

def one(t):
    pos, D, E, chan = topo(t["scen"], t["idx"], t["n"])
    h = sim.topo_hash(pos, D, E, chan.lns / P.SIGMA)
    diag = {} if t["diag"] else None
    net = sim.Net(pos, D, E, chan, t["c"], seeds.arq_rng(t["idx"], 0, SCEN[t["scen"]] + (10 * t["n"] if t["n"] != 100 else 0), t["pid"]),
                  mac=t["mac"], gw_mode=t["gw"], diag=diag)
    agent = Agent.load(t["model"]) if t["model"] else None
    res = sim.run(net, t["proto"], agent=agent, stop=t["stop"], max_rounds=8000)
    row = [t["group"], t["scen"], t["c"], t["n"], t["idx"], t["proto"], t["variant"], t["model"],
           res["FND"], res["HND"], res["LND"], round(res["PDR"], 5), res["delivered"], res["J_per_reading"]]
    extra = {}
    if t["group"] == "main":
        extra["deaths"] = net.death.tolist()
    if diag:
        a = np.array(diag["agree"]); d = np.array(diag["direct"])
        extra["agree"] = [int(a[:, 1].sum()), len(a)]
        extra["direct_early"] = [int(d[d[:, 0] < 300, 1].sum()), int((d[:, 0] < 300).sum())]
        extra["direct_late"] = [int(d[d[:, 0] > 600, 1].sum()), int((d[:, 0] > 600).sum())]
    return row, extra, ("test", f"{t['group']}_{t['scen']}_N{t['n']}", t["idx"], h)

if __name__ == "__main__":
    groups = sys.argv[1:] or ["main", "ablation", "scale", "noagg", "mac", "dist"]
    for grp in groups:
        T = tasks_for(grp)
        T.sort(key=lambda t: (-t["n"], t["c"] == 0.0))
        t0 = time.time()
        with Pool(2) as pool, open(f"results_{grp}.csv", "w", newline="") as f, \
                open(f"extra_{grp}.jsonl", "w") as fx, open("topo_log.csv", "a", newline="") as g:
            w = csv.writer(f)
            w.writerow(["group", "scen", "c", "N", "topo", "protocol", "variant", "model", "FND", "HND", "LND",
                        "PDR", "delivered", "J_per_reading"])
            hw = csv.writer(g)
            for k, (row, extra, hrow) in enumerate(pool.imap_unordered(one, T, chunksize=2)):
                w.writerow(row); hw.writerow(hrow)
                if extra:
                    fx.write(json.dumps({"key": row[:8], **extra}) + "\n")
                if (k + 1) % 50 == 0 or k + 1 == len(T):
                    f.flush(); print(f"{grp}: {k+1}/{len(T)} ({time.time()-t0:.0f}s)", flush=True)
