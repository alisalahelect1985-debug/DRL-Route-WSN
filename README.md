# DRL-Route: energy-sustainable routing for wireless sensor networks

Code, trained models and raw results for the paper:

> A. S. M. Khafaga, "When Does Deep Reinforcement Learning Pay Off? Energy-Sustainable Routing for Wireless Sensor
> Networks under Different Data-Aggregation Regimes," *Al Furat Journal of Innovations in Engineering Applications
> (AJIEA)*, 2026 (under review).

## Contents

| Folder | What it contains | Paper sections |
| --- | --- | --- |
| `original_simulator/` | Simulator, 26 trained DRL-Route models and raw results | Tables 3–6, Figs. 1–3 |
| `reimplementation/` | Independent re-implementation (NumPy/SciPy), 32 trained models, raw results | Sections 3.6–3.7, Tables 7–8, Fig. 4 |

Each folder has its own README with instructions to reproduce its results.

## Requirements

Python 3.10 or later with NumPy, SciPy and pandas (matplotlib for Fig. 4). No GPU is needed.

```bash
pip install numpy scipy pandas matplotlib
```

## Quick start (re-implementation)

```bash
cd reimplementation
python analyze.py        # Table 7 and the seed-separation check, from the stored results
python analyze_agg.py    # Table 8 and Fig. 4, from the stored results
```

## License

MIT (see `LICENSE`). If you use this code, please cite the paper (see `CITATION.cff`).
