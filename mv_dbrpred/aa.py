"""
aa.py – Amino-acid vocabulary, encoding, and residue-level physicochemical features.
Path: /home/liusicen/methods/DBR_pred/Multi-view_DBRpred/mv_dbrpred/aa.py
"""

from __future__ import annotations

import torch
import numpy as np

# 20 standard amino acids + X(unknown) + PAD
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
UNK_IDX = len(AA_LIST)       # 20
PAD_IDX = len(AA_LIST) + 1   # 21
VOCAB_SIZE = len(AA_LIST) + 2  # 22

def encode_sequence(seq: str) -> list[int]:
    """Encode amino acid sequence to integer indices."""
    return [AA_TO_IDX.get(aa, UNK_IDX) for aa in seq.upper()]

# ---------------------------------------------------------------------------
# Physicochemical properties (7 descriptors per residue)
# Charge, Hydrophobicity (Kyte-Doolittle), Polarity (Grantham),
# Flexibility (Vihinen), Volume (Zamyatnin), Molecular weight, pI
# ---------------------------------------------------------------------------

_PHYSCHEM_TABLE = {
    # AA: [charge, hydrophobicity, polarity, flexibility, volume, mol_weight, pI]
    'A': [ 0.0,  1.8,  8.1, 0.360,  88.6,  89.09, 6.01],
    'C': [ 0.0,  2.5,  5.5, 0.350, 108.5, 121.16, 5.07],
    'D': [-1.0, -3.5, 13.0, 0.510, 111.1, 133.10, 2.77],
    'E': [-1.0, -3.5, 12.3, 0.500, 138.4, 147.13, 3.22],
    'F': [ 0.0,  2.8,  5.2, 0.310, 189.9, 165.19, 5.48],
    'G': [ 0.0, -0.4,  9.0, 0.540,  60.1,  75.03, 5.97],
    'H': [ 0.5, -3.2, 10.4, 0.320, 153.2, 155.16, 7.59],
    'I': [ 0.0,  4.5,  5.2, 0.460, 166.7, 131.17, 6.02],
    'K': [ 1.0, -3.9, 11.3, 0.470, 168.6, 146.19, 9.74],
    'L': [ 0.0,  3.8,  4.9, 0.370, 166.7, 131.17, 5.98],
    'M': [ 0.0,  1.9,  5.7, 0.300, 162.9, 149.21, 5.74],
    'N': [ 0.0, -3.5, 11.6, 0.460, 114.1, 132.12, 5.41],
    'P': [ 0.0, -1.6,  8.0, 0.510, 112.7, 115.13, 6.30],
    'Q': [ 0.0, -3.5, 10.5, 0.490, 143.8, 146.15, 5.65],
    'R': [ 1.0, -4.5, 10.5, 0.530, 173.4, 174.20, 10.76],
    'S': [ 0.0, -0.8,  9.2, 0.510,  89.0, 105.09, 5.68],
    'T': [ 0.0, -0.7,  8.6, 0.440, 116.1, 119.12, 5.60],
    'V': [ 0.0,  4.2,  5.9, 0.390, 140.0, 117.15, 5.97],
    'W': [ 0.0, -0.9,  5.4, 0.310, 227.8, 204.23, 5.89],
    'Y': [ 0.0, -1.3,  6.2, 0.420, 193.6, 181.19, 5.66],
}

PHYSCHEM_DIM = 7

# Pre-compute normalised table as numpy
_all_vals = np.array([_PHYSCHEM_TABLE[aa] for aa in AA_LIST], dtype=np.float32)  # (20, 7)
_mean = _all_vals.mean(axis=0, keepdims=True)
_std = _all_vals.std(axis=0, keepdims=True) + 1e-8
_normed = (_all_vals - _mean) / _std

# Add unknown and pad rows (zeros)
_normed = np.concatenate([_normed, np.zeros((2, PHYSCHEM_DIM), dtype=np.float32)], axis=0)  # (22, 7)

def physchem_features(seq_indices: torch.Tensor) -> torch.Tensor:
    """
    Given integer-encoded sequence tensor of shape (L,), return (L, PHYSCHEM_DIM).
    """
    table = torch.from_numpy(_normed).to(seq_indices.device)
    return table[seq_indices.clamp(0, VOCAB_SIZE - 1)]

def physchem_features_batch(seq_indices: torch.Tensor) -> torch.Tensor:
    """
    Given integer-encoded batch tensor of shape (B, L), return (B, L, PHYSCHEM_DIM).
    """
    table = torch.from_numpy(_normed).to(seq_indices.device)
    return table[seq_indices.clamp(0, VOCAB_SIZE - 1)]