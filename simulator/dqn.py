"""Double DQN with a 12-64-64-1 ReLU network, implemented in NumPy."""
import numpy as np
import params as P

class MLP:
    def __init__(self, rng, sizes=(12, P.HIDDEN, P.HIDDEN, 1)):
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))
            self.b.append(np.zeros(b, dtype=np.float32))

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    def forward(self, X, keep=False):
        acts = [X]; h = X
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if k < len(self.W) - 1 else z
            acts.append(h)
        return (h[:, 0], acts) if keep else h[:, 0]

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]; self.b = [b.copy() for b in other.b]

    def soft_update(self, other, tau):
        for k in range(len(self.W)):
            self.W[k] += tau * (other.W[k] - self.W[k]); self.b[k] += tau * (other.b[k] - self.b[k])

class Adam:
    def __init__(self, net, lr=P.LR):
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in net.W + net.b]
        self.v = [np.zeros_like(p) for p in net.W + net.b]

    def step(self, net, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = net.W + net.b
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

class Replay:
    def __init__(self, cap=P.BUFFER, maxc=P.K_CAND + 1):
        self.cap, self.maxc = cap, maxc
        self.s = np.zeros((cap, 12), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.ns = np.zeros((cap, maxc, 12), np.float32)
        self.nm = np.zeros((cap, maxc), bool)
        self.d = np.zeros(cap, bool)
        self.i = 0; self.full = False

    def add(self, s, r, nxt, done):
        k = self.i
        self.s[k] = s; self.r[k] = r; self.d[k] = done
        self.nm[k] = False
        if nxt is not None and not done:
            L = len(nxt); self.ns[k, :L] = nxt; self.nm[k, :L] = True
        self.i = (k + 1) % self.cap; self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, rng, n):
        idx = rng.integers(len(self), size=n)
        return self.s[idx], self.r[idx], self.ns[idx], self.nm[idx], self.d[idx]

class Agent:
    def __init__(self, rng, gamma):
        self.rng, self.gamma = rng, gamma
        self.online = MLP(rng); self.target = MLP(rng); self.target.copy_from(self.online)
        self.opt = Adam(self.online)

    def q(self, F):
        return self.online.forward(F)

    def train_step(self, buf, batch=P.BATCH):
        s, r, ns, nm, d = buf.sample(self.rng, batch)
        y = r.copy()
        if self.gamma > 0:
            live = (~d) & nm.any(1)
            if live.any():
                B, C, _ = ns[live].shape
                flat = ns[live].reshape(-1, 12)
                qo = self.online.forward(flat).reshape(B, C)
                qt = self.target.forward(flat).reshape(B, C)
                qo = np.where(nm[live], qo, -np.inf)
                a = qo.argmax(1)
                y[live] += self.gamma * qt[np.arange(B), a]
        q, acts = self.online.forward(s, keep=True)
        err = q - y
        g = np.clip(err, -1.0, 1.0) / batch            # Huber loss gradient
        grads_W, grads_b = [], []
        delta = g[:, None].astype(np.float32)
        for k in range(len(self.online.W) - 1, -1, -1):
            grads_W.insert(0, acts[k].T @ delta); grads_b.insert(0, delta.sum(0))
            if k > 0:
                delta = (delta @ self.online.W[k].T) * (acts[k] > 0)
        self.opt.step(self.online, grads_W + grads_b)
        self.target.soft_update(self.online, P.TAU)
        return float(np.mean(np.minimum(err ** 2, np.abs(err))))

    def save(self, path):
        np.savez(path, *self.online.W, *self.online.b, gamma=self.gamma)

    @staticmethod
    def load(path, rng=None):
        z = np.load(path)
        a = Agent(rng or np.random.default_rng(0), float(z["gamma"]))
        a.online.W = [z["arr_%d" % k] for k in range(3)]
        a.online.b = [z["arr_%d" % k] for k in range(3, 6)]
        a.target.copy_from(a.online)
        return a
