"""Round-based WSN simulator: radio, ARQ, shadowing (static or time-varying),
aggregation, and the protocols of the manuscript (LEACH, PEGASIS, MTE-Tree,
EA-SPT, EA-Greedy, DRL-Route).

Index N in node arrays denotes the sink.
"""
import hashlib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import params as P

# ----------------------------------------------------------------- energy
def e_tx(b, d):
    d = np.asarray(d, dtype=float)
    return b * P.E_ELEC + np.where(d < P.D0, b * P.EPS_FS * d * d, b * P.EPS_MP * d ** 4)

def e_rx(b):
    return b * P.E_ELEC

def pkt_bits(m, c):
    """Eq. (3): size of a packet carrying m readings."""
    return P.K_BITS * (1.0 + (1.0 - c) * (m - 1))

def weight(E):
    """Eq. (5)."""
    E = np.maximum(np.asarray(E, dtype=float), 1e-12)
    return np.minimum((P.E0 / E) ** P.ALPHA, P.W_MAX)

def etx_expected(p):
    """Expected number of attempts with at most MAX_ATTEMPTS attempts."""
    p = np.asarray(p, dtype=float)
    return (1 - p ** P.MAX_ATTEMPTS) / np.maximum(1 - p, 1e-9)

# ----------------------------------------------------------------- channel
class Channel:
    """Log-normal shadowing on every link (sink included).

    rho_t = None  -> static shadowing (manuscript, Section 2.1)
    rho_t in [0,1) -> AR(1) time-varying log-shadowing per round with the same
                     marginal sigma; nodes refresh their PRR estimates only at
                     HELLO rounds, so estimates become stale between HELLOs.
    """
    def __init__(self, D, rng, sigma=P.SIGMA, rho_t=None, rng_evo=None):
        n = D.shape[0]
        self.D, self.sigma, self.rho_t = D, sigma, rho_t
        self.rng = rng_evo if rng_evo is not None else rng
        z = rng.standard_normal((n, n)); z = np.triu(z, 1); z = z + z.T
        self.lns = sigma * z
        self.base = P.P0 + P.P1 * (D / P.D_REF) ** 2
        self._update_p()
        self.prr_hat = 1.0 - self.p

    def _update_p(self):
        self.p = np.minimum(P.P_MAX, self.base * np.exp(self.lns))

    def step(self):
        if self.rho_t is None:
            return
        n = self.lns.shape[0]
        z = self.rng.standard_normal((n, n)); z = np.triu(z, 1); z = z + z.T
        self.lns = self.rho_t * self.lns + np.sqrt(1 - self.rho_t ** 2) * self.sigma * z
        self._update_p()

    def hello(self):
        self.prr_hat = 1.0 - self.p

# ----------------------------------------------------------------- topology
def make_topology(rng, n, sink, e0=None):
    pos = rng.uniform(0, P.FIELD, size=(n, 2))
    pts = np.vstack([pos, np.asarray(sink, dtype=float)[None, :]])
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    E = np.full(n, P.E0) if e0 is None else np.asarray(e0, dtype=float)
    return pos, D, E

def topo_hash(pos, D, E, lns):
    h = hashlib.sha256()
    for a in (pos, E, lns):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]

# ----------------------------------------------------------------- network state
class Net:
    def __init__(self, pos, D, E, chan, c, rng_arq, mac=None, gw_mode="sink", diag=None):
        """mac: None (no MAC costs, Sections 3.2-3.4), 'random' or 'tdma' (MAC-aware model of
        Section 2.1; LEACH and PEGASIS always use scheduled access). gw_mode: 'sink' (Eq. (4) every
        round) or 'dist' (hand-over to the best neighbour every 10 rounds)."""
        self.n = len(pos)
        self.mac, self.gw_mode, self.diag = mac, gw_mode, diag
        self.gw = None
        self.death = np.full(len(pos), -1)
        self.lnd = None
        self.pc = None
        self.pos, self.D, self.E = pos, D, E.copy()
        self.E_init = E.copy()
        self.Ehat = E.copy()
        self.alive = self.E > 0
        self.chan, self.c, self.rng = chan, c, rng_arq
        self.round = 0
        self.fnd = None; self.hnd = None
        self.generated = 0; self.delivered = 0
        self.spent = 0.0

    # energy ---------------------------------------------------------
    def spend(self, i, x):
        if i == self.n:                 # sink is mains-powered
            return True
        if not self.alive[i]:
            return False
        if self.E[i] < x:
            self.spent += self.E[i]; self.E[i] = 0.0; self.alive[i] = False
            self.death[i] = self.round
            return False
        self.E[i] -= x; self.spent += x
        return True

    def spend_many(self, idx, x):
        """Charge x joules to every node in idx (vectorised; used for overhearing)."""
        idx = idx[self.alive[idx]]
        if len(idx) == 0:
            return
        e = self.E[idx]
        dead = e < x
        self.spent += float(np.minimum(e, x).sum())
        self.E[idx] = np.where(dead, 0.0, e - x)
        if dead.any():
            di = idx[dead]; self.alive[di] = False; self.death[di] = self.round

    def mac_round_start(self, scheduled):
        """Sensing and idle listening for every alive node; collision probabilities for random access."""
        al = np.where(self.alive)[0]
        idle = P.IDLE_SCHED if (scheduled or self.mac == "tdma") else P.IDLE_RAND
        self.spend_many(al, P.E_SENSE + idle)
        if self.mac == "random" and not scheduled:
            near = (self.D[:, :self.n] <= P.R_INT) & self.alive[None, :]
            np.fill_diagonal(near[:self.n, :], False)
            cnt = near.sum(1)
            self.pc = 1.0 - (1.0 - 1.0 / P.SLOTS) ** cnt
            # a header can only be decoded inside the communication range R_c
            hear = (self.D[:self.n, :self.n] <= P.R_C) & self.alive[None, :]
            np.fill_diagonal(hear, False)
            self.near = hear
        else:
            self.pc = None

    def send(self, i, j, bits):
        """ARQ transmission i -> j. Returns delivered, attempts, Etx, Erx."""
        d = self.D[i, j]; etx = float(e_tx(bits, d)); erx = float(e_rx(bits))
        p = self.chan.p[i, j]
        if self.pc is not None:
            p = 1.0 - (1.0 - p) * (1.0 - self.pc[j])
            ov = np.where(self.near[i])[0] if i < self.n else np.array([], int)
            ov = ov[ov != j]
            e_hdr = float(e_rx(P.HDR_BITS))
        tx_tot = rx_tot = 0.0
        for a in range(1, P.MAX_ATTEMPTS + 1):
            if not self.spend(i, etx):
                return False, a, tx_tot, rx_tot
            tx_tot += etx
            if self.pc is not None:
                self.spend_many(ov, e_hdr)          # neighbours overhear the header
            rx_ok = True
            if j != self.n:
                if self.alive[j]:
                    rx_ok = self.spend(j, erx); rx_tot += erx
                else:
                    rx_ok = False
            if rx_ok and self.rng.random() >= p:
                return True, a, tx_tot, rx_tot
            if j != self.n and not self.alive[j]:
                return False, a, tx_tot, rx_tot
        return False, P.MAX_ATTEMPTS, tx_tot, rx_tot

    def broadcast(self, i, bits, rng_m):
        """Control broadcast within range rng_m; receivers pay reception."""
        if not self.spend(i, float(e_tx(bits, rng_m))):
            return
        rx = np.where(self.alive & (self.D[i, :self.n] <= rng_m))[0]
        er = float(e_rx(bits))
        for j in rx:
            if j != i:
                self.spend(j, er)

    def hello(self):
        for i in np.where(self.alive)[0]:
            self.broadcast(i, P.CTRL_BITS, P.R_C)
        self.Ehat = self.E.copy()
        self.chan.hello()

    def bookkeeping(self):
        dead = self.n - int(self.alive.sum())
        if self.fnd is None and dead >= 1:
            self.fnd = self.round
        if self.hnd is None and dead >= self.n / 2:
            self.hnd = self.round
        if self.lnd is None and dead >= self.n:
            self.lnd = self.round

    def prr_est(self, i, cj):
        """Estimated PRR of links i -> cj, including collisions under random access."""
        q = self.chan.prr_hat[i, cj]
        return q if self.pc is None else q * (1.0 - self.pc[cj])

# ----------------------------------------------------------------- tree-based protocols
def candidate_sets(net, g):
    """Up to K nearest alive neighbours within R_c closer to g, plus g."""
    n = net.n
    dg = net.D[:n, g]
    alive = net.alive
    cand = {}
    for i in np.where(alive)[0]:
        if i == g:
            continue
        di = net.D[i, :n]
        ok = alive & (di <= P.R_C) & (dg < dg[i])
        ok[g] = False; ok[i] = False
        idx = np.where(ok)[0]
        if len(idx) > P.K_CAND:
            idx = idx[np.argsort(di[idx])[:P.K_CAND]]
        cand[i] = np.append(idx, g).astype(int)
    return cand

def choose_gateway(net):
    n = net.n
    al = np.where(net.alive)[0]
    score = net.E[al] / e_tx(P.K_BITS, net.D[al, n])
    return int(al[np.argmax(score)])

def gateway_for_round(net):
    if net.gw_mode == "sink":
        return choose_gateway(net)
    n = net.n
    def best_near(h, include_self):
        nb = np.where(net.alive & (net.D[h, :n] <= P.R_C))[0]
        if not include_self:
            nb = nb[nb != h]
        if len(nb) == 0:
            return choose_gateway(net)
        return int(nb[np.argmax(net.E[nb] / e_tx(P.K_BITS, net.D[nb, n]))])
    if net.gw is None:
        net.gw = choose_gateway(net)
    elif not net.alive[net.gw]:
        net.gw = best_near(net.gw, include_self=False)
    elif (net.round - 1) % P.GW_HANDOVER == 0:
        net.gw = best_near(net.gw, include_self=False)
    return net.gw

def features(net, i, cj, g, b_i, m, children, ncand):
    n = net.n
    f = np.empty((len(cj), 12), dtype=np.float32)
    f[:, 0] = net.E[i] / P.E0
    f[:, 1] = net.Ehat[cj] / P.E0
    f[:, 2] = net.D[i, cj] / 100.0
    f[:, 3] = net.D[cj, g] / P.DIAG
    f[:, 4] = net.D[i, g] / P.DIAG
    f[:, 5] = net.prr_est(i, cj)
    f[:, 6] = children[cj] / 10.0
    f[:, 7] = (cj == g)
    f[:, 8] = np.log1p(e_tx(b_i, net.D[i, cj]) / P.E_REF)
    f[:, 9] = ncand / (P.K_CAND + 1)
    f[:, 10] = np.log1p(b_i / P.K_BITS)
    f[:, 11] = np.log1p(m[cj]) / 5.0
    return f

def greedy_cost(net, i, cj, b_i):
    """EA-Greedy: expected energy part of the one-step reward (Eq. (6))."""
    p = 1.0 - net.prr_est(i, cj)
    ex = etx_expected(p)
    return ex * (e_tx(b_i, net.D[i, cj]) * weight(net.E[i])
                 + e_rx(b_i) * weight(net.Ehat[cj])) \
        + (1 - p ** P.MAX_ATTEMPTS) * net.c * P.K_BITS * P.E_DA * weight(net.Ehat[cj])

def spt_parents(net, g, cand, energy_aware):
    """MTE-Tree / EA-SPT: shortest expected-energy path tree to the gateway
    over the candidate graph (k-bit link costs, as in Section 2.3)."""
    rows, cols, w = [], [], []
    k = P.K_BITS
    for i, cj in cand.items():
        p = 1.0 - net.prr_est(i, cj)
        ex = etx_expected(p)
        if energy_aware:
            cost = ex * (e_tx(k, net.D[i, cj]) * weight(net.E[i])
                         + (e_rx(k) + net.c * k * P.E_DA) * weight(net.E[cj]))
        else:
            cost = ex * (e_tx(k, net.D[i, cj]) + e_rx(k)) + net.c * k * P.E_DA
        rows.extend([i] * len(cj)); cols.extend(cj.tolist()); w.extend(np.maximum(cost, 1e-15).tolist())
    n = net.n
    G = csr_matrix((w, (cols, rows)), shape=(n, n))      # reversed edges: from parent to child
    dist, pred = dijkstra(G, directed=True, indices=g, return_predecessors=True)
    parents = {}
    for i in cand:
        parents[i] = int(pred[i]) if pred[i] >= 0 else g
    return parents

def tree_round(net, policy, agent=None, eps=0.0, rng_pol=None, collect=None):
    """One round of a tree-based protocol. policy in
    {'MTE-Tree','EA-SPT','EA-Greedy','DRL'}. Returns list of transitions if collect."""
    n, c = net.n, net.c
    net.round += 1
    net.chan.step()
    if net.round % P.HELLO_EVERY == 1:
        net.hello()
    al = np.where(net.alive)[0]
    if len(al) == 0:
        return
    net.generated += len(al)
    if net.mac:
        net.mac_round_start(scheduled=False)
        al = np.where(net.alive)[0]
        if len(al) == 0:
            net.bookkeeping(); return
    g = gateway_for_round(net)
    cand = candidate_sets(net, g)
    order = sorted(cand.keys(), key=lambda i: -net.D[i, g])
    m = np.zeros(n + 1); m[al] = 1
    children = np.zeros(n + 1)
    parent = {}
    dec = {}
    if policy in ("MTE-Tree", "EA-SPT"):
        parent = spt_parents(net, g, cand, policy == "EA-SPT")
        for i in order:
            m[parent[i]] += m[i]; children[parent[i]] += 1
    else:
        for i in order:
            cj = cand[i]
            b_i = pkt_bits(m[i], c)
            if policy == "EA-Greedy":
                k = int(np.argmin(greedy_cost(net, i, cj, b_i)))
                F = None
            else:
                F = features(net, i, cj, g, b_i, m, children, len(cj))
                if rng_pol is not None and rng_pol.random() < eps:
                    k = int(rng_pol.integers(len(cj)))
                else:
                    k = int(np.argmax(agent.q(F)))
                if net.diag is not None:
                    kg = int(np.argmin(greedy_cost(net, i, cj, b_i)))
                    net.diag.setdefault("agree", []).append((net.round, int(kg == k)))
            j = int(cj[k])
            if net.diag is not None:
                net.diag.setdefault("direct", []).append((net.round, int(j == g)))
            parent[i] = j
            dec[i] = (F, k, b_i, net.E[i], net.Ehat[j])
            m[j] += m[i]; children[j] += 1
    # transmission phase
    held = np.zeros(n + 1); held[al] = 1
    nrecv = np.zeros(n + 1)
    trans = []
    for i in order:
        j = parent[i]
        if not net.alive[i] or held[i] <= 0:
            if collect is not None and i in dec:
                trans.append((i, j, None))
            continue
        if not net.spend(i, c * (nrecv[i] + 1) * P.K_BITS * P.E_DA):
            continue
        bits = pkt_bits(held[i], c)
        ok, att, etx, erx = net.send(i, j, bits)
        if ok:
            held[j] += held[i]; nrecv[j] += 1
        if collect is not None and i in dec:
            trans.append((i, j, (ok, etx, erx)))
        held[i] = 0
    # gateway uploads to the sink
    if net.alive[g] and held[g] > 0:
        if net.spend(g, c * (nrecv[g] + 1) * P.K_BITS * P.E_DA):
            ok, *_ = net.send(g, n, pkt_bits(held[g], c))
            if ok:
                net.delivered += held[g]
    net.bookkeeping()
    if collect is not None:
        for (i, j, res) in trans:
            F, k, b_i, Ei, Ehj = dec[i]
            if res is None:
                continue
            ok, etx, erx = res
            eda = c * P.K_BITS * P.E_DA if ok else 0.0
            s_i = b_i / P.K_BITS
            r = -(etx * weight(Ei) + erx * weight(Ehj) + eda * weight(Ehj)) / (P.E_REF * s_i)
            if not ok:
                r -= P.RHO
            done = (not ok) or (j == g)
            nxt = dec[j][0] if (not done and j in dec) else None
            if not done and nxt is None:
                done = True
            collect.append((F[k], float(r), nxt, done))

# ----------------------------------------------------------------- LEACH
def leach_round(net, state):
    n, c = net.n, net.c
    net.round += 1
    net.chan.step()
    al = np.where(net.alive)[0]
    if len(al) == 0:
        return
    net.generated += len(al)
    if net.mac:
        net.mac_round_start(scheduled=True)
        al = np.where(net.alive)[0]
        if len(al) == 0:
            net.bookkeeping(); return
    p = P.LEACH_P; period = int(round(1 / p))
    r = net.round - 1
    if r % period == 0:
        state["G"] = np.ones(n, dtype=bool)
    T = p / (1 - p * (r % period))
    elig = net.alive & state["G"]
    ch = np.where(elig & (net.rng.random(n) < T))[0]
    state["G"][ch] = False
    held = np.zeros(n + 1); nrecv = np.zeros(n + 1)
    held[al] = 1
    if len(ch) == 0:
        for i in al:
            if net.spend(i, c * P.K_BITS * P.E_DA):
                ok, *_ = net.send(i, n, pkt_bits(1, c))
                if ok: net.delivered += 1
        net.bookkeeping(); return
    members = [i for i in al if i not in set(ch)]
    # advertisement: each CH broadcasts to the whole field
    for h in ch:
        net.broadcast(h, P.CTRL_BITS, P.DIAG)
    assign = {}
    for i in members:
        if not net.alive[i]: continue
        h = int(ch[np.argmin(net.D[i, ch])]); assign[i] = h
        if net.spend(i, float(e_tx(P.CTRL_BITS, net.D[i, h]))):       # join request
            net.spend(h, float(e_rx(P.CTRL_BITS)))
    for h in ch:                                                        # TDMA schedule
        mem = [i for i, hh in assign.items() if hh == h]
        if mem:
            net.broadcast(h, P.CTRL_BITS, float(net.D[h, mem].max()))
    for i, h in assign.items():                                         # data (TDMA, no collisions)
        if not net.alive[i]: continue
        if not net.spend(i, c * P.K_BITS * P.E_DA): continue
        ok, *_ = net.send(i, h, pkt_bits(1, c))
        held[i] = 0
        if ok: held[h] += 1; nrecv[h] += 1
    for h in ch:
        if net.alive[h] and held[h] > 0 and net.spend(h, c * (nrecv[h] + 1) * P.K_BITS * P.E_DA):
            ok, *_ = net.send(h, n, pkt_bits(held[h], c))
            if ok: net.delivered += held[h]
    net.bookkeeping()

# ----------------------------------------------------------------- PEGASIS
def build_chain(net):
    n = net.n
    al = list(np.where(net.alive)[0])
    if not al: return []
    start = max(al, key=lambda i: net.D[i, n])
    chain = [start]; rest = set(al); rest.remove(start)
    while rest:
        last = chain[-1]
        nxt = min(rest, key=lambda j: net.D[last, j])
        chain.append(nxt); rest.remove(nxt)
    return chain

def pegasis_round(net, state):
    n, c = net.n, net.c
    net.round += 1
    net.chan.step()
    al = np.where(net.alive)[0]
    if len(al) == 0:
        return
    net.generated += len(al)
    if net.mac:
        net.mac_round_start(scheduled=True)
        al = np.where(net.alive)[0]
        if len(al) == 0:
            net.bookkeeping(); return
    if state.get("nalive") != len(al):
        state["chain"] = build_chain(net); state["nalive"] = len(al)
    chain = state["chain"]
    L = len(chain)
    li = (net.round - 1) % L
    leader = chain[li]
    # token passing from the leader to both ends
    for seg in (range(li, 0, -1), range(li, L - 1)):
        for a in seg:
            b = a - 1 if seg.step < 0 else a + 1
            u, v = chain[a], chain[b]
            if net.spend(u, float(e_tx(P.CTRL_BITS, net.D[u, v]))):
                net.spend(v, float(e_rx(P.CTRL_BITS)))
    held = np.zeros(n + 1); nrecv = np.zeros(n + 1); held[chain] = 1
    for seg in (range(0, li), range(L - 1, li, -1)):
        for a in seg:
            b = a + 1 if a < li else a - 1
            u, v = chain[a], chain[b]
            if not net.alive[u] or held[u] <= 0:
                continue
            if not net.spend(u, c * (nrecv[u] + 1) * P.K_BITS * P.E_DA):
                continue
            ok, *_ = net.send(u, v, pkt_bits(held[u], c))
            if ok: held[v] += held[u]; nrecv[v] += 1
            held[u] = 0
    if net.alive[leader] and held[leader] > 0 and \
            net.spend(leader, c * (nrecv[leader] + 1) * P.K_BITS * P.E_DA):
        ok, *_ = net.send(leader, n, pkt_bits(held[leader], c))
        if ok: net.delivered += held[leader]
    net.bookkeeping()

# ----------------------------------------------------------------- run to HND
def run(net, protocol, agent=None, max_rounds=6000, stop="hnd"):
    state = {}
    while net.round < max_rounds:
        if protocol == "LEACH":
            leach_round(net, state)
        elif protocol == "PEGASIS":
            pegasis_round(net, state)
        else:
            tree_round(net, "DRL" if protocol.startswith("DRL") else protocol, agent=agent)
        if stop == "hnd" and net.hnd is not None:
            break
        if stop == "fnd" and net.fnd is not None:
            break
        if not net.alive.any():
            break
    return dict(FND=net.fnd, HND=net.hnd, LND=net.lnd, rounds=net.round,
                PDR=net.delivered / max(net.generated, 1), delivered=net.delivered,
                J_per_reading=net.spent / max(net.delivered, 1))
