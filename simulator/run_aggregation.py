"""Intermediate aggregation levels (static channel). Same test topologies, initial
shadowing and ARQ streams as run_experiments.py, so the c = 0 and c = 1 rows
coincide with the static rows of Table 7."""
import csv, glob, itertools, time
from multiprocessing import Pool
import params as P, sim, seeds
from dqn import Agent

CS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCEN = {"S1": 1, "S2": 2}
BASELINES = ["LEACH", "PEGASIS", "MTE-Tree", "EA-SPT", "EA-Greedy"]

def tag(c):
    return {0.0: "0", 1.0: "1"}.get(c, str(c))

def models(c, g):
    return sorted(glob.glob(f"models/c{tag(c)}_g{g}_m*.npz"))

def one(task):
    scen, c, idx = task
    runs = [(p, None, None, pid) for pid, p in enumerate(BASELINES)]
    prim, sec = ("09", "0") if c == 0.0 else ("0", "09")      # primary gamma keeps ARQ ids 5.. (as Table 7)
    runs += [("DRL-Route", m, prim, 5 + k) for k, m in enumerate(models(c, prim))]
    runs += [("DRL-Route", m, sec, 20 + k) for k, m in enumerate(models(c, sec))]
    rows = []
    for proto, model, g, pid in runs:
        r = seeds.test_rng(idx)
        pos, D, E = sim.make_topology(r, 100, P.SINKS[scen])
        chan = sim.Channel(D, seeds.rng(seeds.TEST, idx, SCEN[scen]), sigma=P.SIGMA, rho_t=None,
                           rng_evo=seeds.chan_rng(idx, 0, SCEN[scen]))
        net = sim.Net(pos, D, E, chan, c, seeds.arq_rng(idx, 0, SCEN[scen], pid))
        agent = Agent.load(model) if model else None
        res = sim.run(net, proto, agent=agent)
        gamma = "" if g is None else ("0.9" if g == "09" else "0")
        rows.append([scen, c, idx, proto, gamma, model or "", res["FND"], res["HND"], round(res["PDR"], 4)])
    return rows

if __name__ == "__main__":
    tasks = list(itertools.product(["S1", "S2"], CS, range(10)))
    tasks.sort(key=lambda t: -t[1])          # heaviest (large c) first
    t0 = time.time()
    with Pool(2) as pool, open("results_agg.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scen", "c", "topo", "protocol", "gamma", "model", "FND", "HND", "PDR"])
        for k, rows in enumerate(pool.imap_unordered(one, tasks)):
            w.writerows(rows); f.flush()
            print(f"{k+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
