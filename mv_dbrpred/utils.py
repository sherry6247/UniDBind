"""
utils.py – Shared utilities.
Path: /home/liusicen/methods/DBR_pred/Multi-view_DBRpred/mv_dbrpred/utils.py
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

@dataclass
class Batch:
    """Collated mini-batch."""
    seq_idx: torch.Tensor           # (B, L) integer-encoded sequence
    pssm: torch.Tensor              # (B, L, 20)
    phys: torch.Tensor              # (B, L, PHYSCHEM_DIM)
    esm2: torch.Tensor              # (B, L, esm2_dim)
    mask: torch.Tensor              # (B, L) bool – True for valid positions
    y_res: torch.Tensor             # (B, L) residue-level labels
    y_prot: torch.Tensor            # (B,) protein-level labels
    lengths: torch.Tensor           # (B,)
    # Privileged (training only; may be None)
    ss: Optional[torch.Tensor] = None        # (B, L, ss_dim) secondary structure one-hot
    sasa: Optional[torch.Tensor] = None      # (B, L, 1) solvent accessibility
    disorder: Optional[torch.Tensor] = None  # (B, L, 1) disorder annotation
    subset: Optional[list[str]] = None       # "structure" or "disorder" per sample
    pids: Optional[list[str]] = None         # protein IDs

    def to(self, device: torch.device) -> "Batch":
        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if t is not None else None

        return Batch(
            seq_idx=self.seq_idx.to(device),
            pssm=self.pssm.to(device),
            phys=self.phys.to(device),
            esm2=self.esm2.to(device),
            mask=self.mask.to(device),
            y_res=self.y_res.to(device),
            y_prot=self.y_prot.to(device),
            lengths=self.lengths.to(device),
            ss=_mv(self.ss),
            sasa=_mv(self.sasa),
            disorder=_mv(self.disorder),
            subset=self.subset,
            pids=self.pids,
        )

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Any, path: Path | str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def resolve_esm2_root(base: Path | str) -> Path:
    """Find the ESM-2 embeddings root directory."""
    base = Path(base)
    for candidate in [base / "esm2_embeddings", base / "esm2", base]:
        if candidate.is_dir():
            return candidate
    return base

def resolve_esm2_dim(esm2_root: Path | str, default: int = 1280) -> int:
    """Infer ESM-2 embedding dimension from a sample .npy file."""
    esm2_root = Path(esm2_root)
    for npy in esm2_root.rglob("*.npy"):
        try:
            arr = np.load(npy)
            return arr.shape[-1]
        except Exception:
            continue
    return default

def infer_use_bilstm_refine(ckpt: dict) -> bool:
    """Check if checkpoint was trained with BiLSTM refinement."""
    state = ckpt.get("model_state_dict", ckpt)
    return any("bilstm" in k.lower() for k in state.keys())