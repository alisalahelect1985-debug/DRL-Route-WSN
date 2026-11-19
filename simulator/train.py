"""Train a DRL-Route agent (Section 2.2 / 2.3 of the manuscript).
usage: python train.py <c> <gamma> <model_seed> <out.npz> [mac]
(set WSN_ALPHA=0 for the ablation agents without energy weighting)
"""
import sys, time, csv
import numpy as np
import params as P, sim, seeds
from dqn import Agent, Replay

def train(c, gamma, model_seed, out, log_topos="topo_log.csv", verbose=True, mac=None):
    rng_pol = seeds.policy_rng(model_seed)
    agent = Agent(seeds.rng(seeds.POLICY, model_seed, 1), gamma)
    buf = Replay()
    steps, val_hist = 0, []
    t0 = time.time()
    for ep in range(P.EPISODES):
        r = seeds.train_rng(model_seed, ep)
        n = int(r.integers(50, 201))
        sink = (float(r.uniform(0, P.FIELD)), float(r.uniform(0, 2 * P.FIELD)))
        e0 = r.uniform(0.025, P.E0, size=n)
        pos, D, E = sim.make_topology(r, n, sink, e0)
        ch = sim.Channel(D, r)
        with open(log_topos, "a", newline="") as f:
            csv.writer(f).writerow(["train", f"c{c}_g{gamma}_m{model_seed}", ep,
                                    sim.topo_hash(pos, D, E, ch.lns)])
        net = sim.Net(pos, D, E, ch, c, r, mac=mac)
        eps = P.EPS_START + (P.EPS_END - P.EPS_START) * min(1.0, ep / (0.8 * P.EPISODES))
        for _ in range(P.ROUNDS_PER_EPISODE):
            if not net.alive.any():
                break
            trans = []
            sim.tree_round(net, "DRL", agent=agent, eps=eps, rng_pol=rng_pol, collect=trans)
            for t in trans:
                buf.add(*t)
            if len(buf) >= 2000:
                for _ in range(max(1, len(trans) // 25)):
                    agent.train_step(buf); steps += 1
        if (ep + 1) % 20 == 0:
            fnd = validate(agent, c, mac)
            val_hist.append((ep + 1, fnd))
            if verbose:
                print(f"ep {ep+1} eps {eps:.2f} steps {steps} val FND {fnd} ({time.time()-t0:.0f}s)", flush=True)
    agent.save(out)
    return val_hist

def validate(agent, c, mac=None):
    r = seeds.val_rng()
    pos, D, E = sim.make_topology(r, 100, P.SINKS["S1"])
    ch = sim.Channel(D, r)
    net = sim.Net(pos, D, E, ch, c, seeds.rng(seeds.VAL, 1), mac=mac)
    res = sim.run(net, "DRL", agent=agent, stop="fnd")
    return res["FND"]

if __name__ == "__main__":
    c, gamma, ms, out = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    mac = sys.argv[5] if len(sys.argv) > 5 else None
    hist = train(c, gamma, ms, out, mac=mac)
    with open(out.replace(".npz", "_val.csv"), "w") as f:
        for e, v in hist: f.write(f"{e},{v}\n")
