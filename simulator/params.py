"""Simulation parameters, taken from Table 1 of the manuscript."""
import math

FIELD = 100.0                 # m, square field side L
DIAG = FIELD * math.sqrt(2)   # normalisation for distance features
SINKS = {"S1": (50.0, 175.0), "S2": (50.0, 50.0)}

E0 = 0.5                      # J, initial energy
E_ELEC = 50e-9                # J/bit
E_DA = 5e-9                   # J/bit/signal
EPS_FS = 10e-12               # J/bit/m^2
EPS_MP = 0.0013e-12           # J/bit/m^4
D0 = math.sqrt(EPS_FS / EPS_MP)   # ~87.7 m

K_BITS = 4000                 # data packet
CTRL_BITS = 200               # control / HELLO / header packet

# link model, Eq. (2)
P0, P1, D_REF, P_MAX = 0.01, 0.15, 100.0, 0.6
SIGMA = 0.3                   # log-normal shadowing
MAX_ATTEMPTS = 4              # one transmission + up to three retries

R_C = 30.0                    # m, candidate radius
K_CAND = 10                   # nearest candidates (+ gateway)
HELLO_EVERY = 10              # rounds

LEACH_P = 0.05

# DRL-Route
import os
ALPHA = float(os.environ.get("WSN_ALPHA", "2.0"))   # alpha = 0 only for the ablation agents
W_MAX, RHO = 50.0, 5.0
HIDDEN = 64
LR = 5e-4
BATCH = 128
BUFFER = 200_000
TAU = 0.01
EPISODES = 200
ROUNDS_PER_EPISODE = 40
EPS_START, EPS_END = 1.0, 0.05

# reference energy E_ref = E_Tx(k, 50 m) + E_Rx(k)
def e_tx_scalar(b, d):
    return b * E_ELEC + (b * EPS_FS * d * d if d < D0 else b * EPS_MP * d ** 4)
E_REF = e_tx_scalar(K_BITS, 50.0) + K_BITS * E_ELEC

# MAC-aware model (Table 1)
E_SENSE = 10e-6               # J per round
IDLE_SCHED, IDLE_RAND = 10e-6, 40e-6   # J per round, scheduled / random access
HDR_BITS = 200                # overheard header
SLOTS = 512                   # contention slots per round
R_INT = 60.0                  # m, interference range
GW_HANDOVER = 10              # rounds, distributed gateway variant
