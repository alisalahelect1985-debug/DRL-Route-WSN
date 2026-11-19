# DRL-Route: energy-sustainable routing for wireless sensor networks

Simulator, trained models and raw results for the paper:

> A. S. M. Khafaga, "When Does Deep Reinforcement Learning Pay Off? Energy-Sustainable Routing for Wireless Sensor
> Networks under Different Data-Aggregation Regimes," *Al Furat Journal of Innovations in Engineering Applications
> (AJIEA)*, 2026 (under review).

Everything is in `simulator/`: the code, the 41 trained DRL-Route agents, the raw per-run results and the scripts that
produce every table and figure of the paper. See `simulator/README.md` for details.

## Requirements

Python 3.10 or later with NumPy, SciPy, pandas and matplotlib. No GPU is needed.

```bash
pip install numpy scipy pandas matplotlib
```

## Quick start

```bash
cd simulator
python analyze_paper.py   # Tables 3-6 and Figs. 1-3
python analyze.py         # Table 7
python analyze_agg.py     # Table 8 and Fig. 4
```

## License

MIT (see `LICENSE`). If you use this code, please cite the paper (see `CITATION.cff`).
