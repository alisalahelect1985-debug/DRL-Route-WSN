# wsnsim: independent re-implementation of the DRL-Route simulator

This package re-implements the round-based simulator described in Sections 2.1–2.3 and Table 1 of the manuscript
"When Does Deep Reinforcement Learning Pay Off? Energy-Sustainable Routing for Wireless Sensor Networks under
Different Data-Aggregation Regimes". It was written during the revision and used for Section 3.6 (robustness to
time-varying channels) and Section 3.7 (intermediate aggregation levels). It is **not** the simulator that produced Tables 3–6.

Requirements: Python 3.10+, NumPy, SciPy, pandas. No GPU or deep-learning framework is needed; the 12-64-64-1
Double DQN (5,057 parameters) is implemented in NumPy (`dqn.py`).

## Files

| File | Content |
| --- | --- |
| `params.py` | All parameters of Table 1 |
| `sim.py` | Radio model (Eq. 1), link failures with ARQ (Eq. 2), aggregation (Eq. 3), gateway (Eq. 4), reward (Eqs. 5–7), static and AR(1) time-varying shadowing, LEACH, PEGASIS, MTE-Tree, EA-SPT, EA-Greedy, DRL-Route |
| `dqn.py` | Double DQN, Adam, replay buffer (Eq. 8) |
| `seeds.py` | Disjoint seed spaces for training, validation, test, channel and ARQ streams |
| `train.py` | Training (200 episodes x 40 rounds, N = 50–200, random sink and initial energies) |
| `run_experiments.py` | Static vs time-varying channel experiment (paired test topologies) |
| `analyze.py` | Tables, paired Wilcoxon tests and the seed-separation check |
| `run_aggregation.py` / `analyze_agg.py` | Intermediate aggregation (c = 0, 0.25, 0.5, 0.75, 1), Table 8 and Fig. 4 |
| `results_agg.csv`, `agg_summary.csv`, `fig4_aggregation.png` | Raw results, summary and figure for Section 3.7 |
| `models/` | 32 trained agents (gamma = 0 and 0.9 for every c; 5 models for c = 1, gamma = 0, 3 otherwise) and validation curves |
| `results.csv` | Raw per-run results (1,440 runs) |
| `topo_log.csv` | SHA-256 fingerprint of every training and test topology |
| `logs/analysis.txt` | Output of `analyze.py` |

## Reproduce

```bash
for s in 1 2 3 4 5; do python train.py 1 0 $s models/c1_g0_m$s.npz; done
for s in 1 2 3;     do python train.py 0 0.9 $s models/c0_g09_m$s.npz; done
python run_experiments.py 10      # about 40 min on 2 CPU cores
python analyze.py
./train_agg.sh A & ./train_agg.sh B   # models for intermediate c
python run_aggregation.py         # about 25 min on 2 CPU cores
python analyze_agg.py
```

## Choices not fixed by the manuscript

- Log-normal shadowing s_ij = exp(sigma z), symmetric, median 1; the sink link has its own factor.
- PRR estimates are exact under a static channel and refreshed only at HELLO rounds (every 10 rounds) under a
  time-varying channel.
- Expected attempts ETX = (1 - p^4)/(1 - p) (at most four attempts) in MTE-Tree, EA-SPT and EA-Greedy.
- The receiver pays reception for every attempt; the aggregation cost c k E_DA is charged only if the packet arrives.
- LEACH: cluster-head advertisements are broadcast over the field diagonal, members send a 200-bit join request,
  heads broadcast a TDMA schedule to their farthest member; no collisions.
- PEGASIS: greedy chain rebuilt when a node dies, leader rotates every round, 200-bit token passed along the chain.
- Training: epsilon decays linearly from 1.0 to 0.05 over the first 80% of episodes; one gradient step per 25 new
  transitions after 2,000 transitions; Huber loss.
- Experiments stop when half of the nodes have died (FND, HND and PDR up to HND are reported).

## Seed separation

Training topologies use seed key 10, validation 20 and test 30 (see `seeds.py`), so they cannot coincide.
`analyze.py` confirms it from the fingerprints: 0 shared fingerprints between the 1,000 training topologies and the
test topologies, and exactly one fingerprint per (scenario, test topology) across all protocols, models and channels.
