#!/bin/bash
# extra agents: gamma = 0.5 (Fig. 1, ablation), alpha = 0 (ablation), trained under the MAC model (Table 6)
cd /home/claude/work/wsnsim
if [ "$1" = A ]; then
  for s in 1 2 3; do [ -f models/c1_g05_m$s.npz ] || python3 train.py 1 0.5 $s models/c1_g05_m$s.npz > logs/c1_g05_m$s.log 2>&1; done
  for s in 1 2; do [ -f models/c1_g0_mac_m$s.npz ] || python3 train.py 1 0 $s models/c1_g0_mac_m$s.npz random > logs/c1_g0_mac_m$s.log 2>&1; done
else
  for s in 1 2 3; do [ -f models/c1_g0_a0_m$s.npz ] || WSN_ALPHA=0 python3 train.py 1 0 $s models/c1_g0_a0_m$s.npz > logs/c1_g0_a0_m$s.log 2>&1; done
  for s in 3; do [ -f models/c1_g0_mac_m$s.npz ] || python3 train.py 1 0 $s models/c1_g0_mac_m$s.npz random > logs/c1_g0_mac_m$s.log 2>&1; done
fi
