# DRL-Route simulator

Round-based wireless sensor network simulator, trained agents and raw results for the paper
"When Does Deep Reinforcement Learning Pay Off? Energy-Sustainable Routing for Wireless Sensor Networks under
Different Data-Aggregation Regimes". Every number, table and figure in the paper comes from this code.

Requirements: Python 3.10+, NumPy, SciPy, pandas, matplotlib. No GPU or deep-learning framework is needed; the
12-64-64-1 Double DQN (5,057 parameters) is implemented in NumPy (`dqn.py`).

## Files

| File | Content |
| --- | --- |
| `params.py` | All parameters of Table 1 |
| `sim.py` | Radio model (Eq. 1), link failures with ARQ (Eq. 2), aggregation (Eq. 3), gateway (Eq. 4), reward (Eqs. 5–7), static and time-varying shadowing, MAC-aware model, distributed gateway, LEACH, PEGASIS, MTE-Tree, EA-SPT, EA-Greedy, DRL-Route |
| `dqn.py` | Double DQN, Adam, replay buffer (Eq. 8) |
| `seeds.py` | Disjoint seed spaces for training, validation, test, channel and ARQ streams |
| `train.py`, `train_agg.sh`, `train_extra.sh` | Training of all 41 agents |
| `run_paper.py` / `analyze_paper.py` | Sections 3.1–3.5: Tables 3–6, Figs. 1–3, ablation, scalability, distributed gateway |
| `run_experiments.py` / `analyze.py` | Section 3.6: time-varying channels (Table 7) |
| `run_aggregation.py` / `analyze_agg.py` | Section 3.7: intermediate aggregation (Table 8, Fig. 4) |
| `models/` | The 41 trained agents and their validation curves (Fig. 1) |
| `results_*.csv`, `results.csv` | Raw per-run results |
| `extra_main.jsonl` | Per-node death rounds (Fig. 2) and decision diagnostics |
| `paper_numbers.json` | Every statistic quoted in Sections 3.1–3.5 |
| `topo_log.csv` | SHA-256 fingerprint of every training and test topology |
| `logs/` | Training logs and analysis outputs |

## Agents (`models/`)

| File pattern | Aggregation c | Discount γ | Other | Models |
| --- | --- | --- | --- | --- |
| `c1_g0_m*` | 1 | 0 | main model | 5 |
| `c1_g05_m*`, `c1_g09_m*` | 1 | 0.5, 0.9 | Fig. 1, ablation | 3 + 3 |
| `c1_g0_a0_m*` | 1 | 0 | α = 0 (no energy weighting), ablation | 3 |
| `c1_g0_mac_m*` | 1 | 0 | trained under the MAC-aware model | 3 |
| `c0_g0_m*`, `c0_g09_m*` | 0 | 0, 0.9 | Table 5 | 3 + 3 |
| `c0.25_*`, `c0.5_*`, `c0.75_*` | 0.25–0.75 | 0, 0.9 | Section 3.7 | 18 |

## Reproduce

The stored results reproduce every table and figure in seconds:

```bash
python analyze_paper.py   # Tables 3-6, Figs. 1-3, all statistics of Sections 3.1-3.5
python analyze.py         # Table 7 and the seed-separation check
python analyze_agg.py     # Table 8 and Fig. 4
```

To rerun everything from scratch (about 6 hours on 2 CPU cores):

```bash
for s in 1 2 3 4 5; do python train.py 1 0 $s models/c1_g0_m$s.npz; done
for s in 1 2 3;     do python train.py 0 0.9 $s models/c0_g09_m$s.npz; done
./train_agg.sh A & ./train_agg.sh B; wait
./train_extra.sh A & ./train_extra.sh B; wait
python run_paper.py
python run_experiments.py 10
python run_aggregation.py
```

## Implementation details

- Log-normal shadowing s_ij = exp(σz), symmetric, median 1; the sink link has its own factor.
- PRR estimates are exact under a static channel and refreshed only at HELLO rounds (every 10 rounds) under a
  time-varying channel; under random access they include the collision probability.
- Expected attempts ETX = (1 − p^4)/(1 − p) (at most four attempts) in MTE-Tree, EA-SPT and EA-Greedy.
- The receiver pays reception for every attempt; the aggregation cost c k E_DA is charged only if the packet arrives.
- MAC-aware model: sensing 10 µJ and idle listening 10 µJ (scheduled) or 40 µJ (random access) per node and round;
  under random access every attempt collides with probability 1 − (1 − 1/512)^n, n = alive nodes within 60 m of the
  receiver, and the alive nodes within R_c = 30 m of the sender overhear a 200-bit header. LEACH, PEGASIS and the
  TDMA variant have no collisions and no overhearing.
- Distributed gateway: the gateway hands its role to the neighbour (within R_c) with the best Eq. (4) score every
  10 rounds, or at once if it dies.
- LEACH: cluster-head advertisements are broadcast over the field diagonal, members send a 200-bit join request,
  heads broadcast a TDMA schedule to their farthest member.
- PEGASIS: greedy chain rebuilt when a node dies, leader rotates every round, 200-bit token passed along the chain.
- Training: ε decays linearly from 1.0 to 0.05 over the first 80% of episodes; one gradient step per 25 new
  transitions after 2,000 transitions; Huber loss; α = 0 agents are trained with `WSN_ALPHA=0`.

## Seed separation

Training topologies use seed key 10, validation 20 and test 30 (see `seeds.py`). `analyze_paper.py` confirms from the
fingerprints that none of the 1,000 training topologies appears among the 3,780 test runs, and every protocol and
model sees the same instance of each test topology.
