"""Static vs time-varying channel experiment (new Section on channel robustness).
Every protocol runs on identical test topologies, initial shadowing and channel
evolution; only the ARQ outcomes use protocol-specific random streams."""
import sys, csv, glob, itertools, time
import numpy as np
from multiprocessing import Pool
import params as P, sim, seeds
from dqn import Agent

CHANNELS = {0: ("static", P.SIGMA, None), 1: ("slow", P.SIGMA, 0.9),
            2: ("fast", P.SIGMA, 0.0), 3: ("severe", 0.6, 0.5)}
SCEN = {"S1": 1, "S2": 2}
BASELINES = ["LEACH", "PEGASIS", "MTE-Tree", "EA-SPT", "EA-Greedy"]
MODELS = {1.0: sorted(glob.glob("models/c1_g0_m*.npz")), 0.0: sorted(glob.glob("models/c0_g09_m*.npz"))}
N_TOPO = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def one(task):
    scen, c, ch_id, idx = task
    name, sigma, rho = CHANNELS[ch_id]
    rows, hashes = [], []
    runs = [(p, None) for p in BASELINES] + [("DRL-Route", m) for m in MODELS[c]]
    for pid, (proto, model) in enumerate(runs):
        r = seeds.test_rng(idx)
        pos, D, E = sim.make_topology(r, 100, P.SINKS[scen])
        chan = sim.Channel(D, seeds.rng(seeds.TEST, idx, SCEN[scen]), sigma=sigma, rho_t=rho,
                           rng_evo=seeds.chan_rng(idx, ch_id, SCEN[scen]))
        hashes.append(("test", f"{scen}_c{c}_{name}_{proto}_{model}", idx, sim.topo_hash(pos, D, E, chan.lns / sigma)))
        net = sim.Net(pos, D, E, chan, c, seeds.arq_rng(idx, ch_id, SCEN[scen], pid))
        agent = Agent.load(model) if model else None
        res = sim.run(net, proto, agent=agent)
        rows.append([scen, c, name, idx, proto, model or "", res["FND"], res["HND"], round(res["PDR"], 4)])
    return rows, hashes

if __name__ == "__main__":
    tasks = list(itertools.product(["S1", "S2"], [1.0, 0.0], CHANNELS.keys(), range(N_TOPO)))
    t0 = time.time()
    with Pool(2) as pool, open("results.csv", "w", newline="") as f, open("topo_log.csv", "a", newline="") as g:
        w = csv.writer(f); w.writerow(["scen", "c", "channel", "topo", "protocol", "model", "FND", "HND", "PDR"])
        hw = csv.writer(g)
        for k, (rows, hashes) in enumerate(pool.imap_unordered(one, tasks)):
            w.writerows(rows); hw.writerows(hashes); f.flush()
            print(f"{k+1}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)
