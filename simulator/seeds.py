"""Seed management. Training, validation and test topologies come from
disjoint key spaces, so no test topology can appear in training."""
import numpy as np

BASE = 20261115
TRAIN, VAL, TEST, CHAN, ARQ, POLICY = 10, 20, 30, 40, 50, 60

def rng(*key):
    return np.random.default_rng([BASE, *key])

def train_rng(model_seed, episode):     return rng(TRAIN, model_seed, episode)
def val_rng():                          return rng(VAL, 0)
def test_rng(idx):                      return rng(TEST, idx)
def chan_rng(idx, chan_id, scen_id):    return rng(CHAN, idx, chan_id, scen_id)
def arq_rng(idx, chan_id, scen_id, proto_id):
    return rng(ARQ, idx, chan_id, scen_id, proto_id)
def policy_rng(model_seed):             return rng(POLICY, model_seed)
