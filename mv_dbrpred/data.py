"""
data.py – HybridDBRpred2024 dataset loader.
Loads ESM-2 embeddings, PSSM, physicochemical features,
and optional privileged structure/disorder annotations.
Path: /home/liusicen/methods/DBR_pred/Multi-view_DBRpred/mv_dbrpred/data.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from mv_dbrpred.aa import (
    PHYSCHEM_DIM,
    PAD_IDX,
    encode_sequence,
    physchem_features,
)
from mv_dbrpred.utils import Batch

# ─── Secondary structure encoding ──────────────────────────────────────────
SS_TYPES = ["H", "E", "C"]  # Helix, Sheet, Coil
SS_DIM = len(SS_TYPES)

def _encode_ss(ss_string: str) -> torch.Tensor:
    """One-hot encode secondary structure string -> (L, SS_DIM)."""
    mapping = {s: i for i, s in enumerate(SS_TYPES)}
    indices = [mapping.get(c, 2) for c in ss_string]
    return torch.nn.functional.one_hot(
        torch.tensor(indices, dtype=torch.long), num_classes=SS_DIM
    ).float()

# ─── PSSM loading ──────────────────────────────────────────────────────────

def _load_pssm(path: Path, expected_len: int) -> torch.Tensor:
    """Load PSSM from .npy or .pssm/.txt files -> (L, 20)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif suffix in (".pssm", ".txt"):
        rows = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 22:
                    try:
                        int(parts[0])  # position number
                        vals = [float(x) for x in parts[2:22]]
                        rows.append(vals)
                    except ValueError:
                        continue
        if not rows:
            return torch.zeros(expected_len, 20)
        arr = np.array(rows, dtype=np.float32)
    else:
        return torch.zeros(expected_len, 20)

    # Normalise PSSM (sigmoid-like scaling)
    arr = 1.0 / (1.0 + np.exp(-arr))
    t = torch.from_numpy(arr)
    if t.shape[0] != expected_len:
        # Truncate or pad
        if t.shape[0] > expected_len:
            t = t[:expected_len]
        else:
            t = torch.nn.functional.pad(t, (0, 0, 0, expected_len - t.shape[0]))
    return t

def _find_pssm(pssm_root: Path, pid: str, split: str) -> Optional[Path]:
    """Search for PSSM file in multiple candidate locations."""
    candidates = [
        pssm_root / f"{pid}.npy",
        pssm_root / f"{pid}.txt",
        pssm_root / f"{pid}.pssm",
        pssm_root / split / "pssm" / f"{pid}.pssm",
        pssm_root / "pssm" / f"{pid}.pssm",
        pssm_root / "Disorder_annotatred proteins" / f"disorder_annotated_{split}" / "pssm" / f"{pid}.pssm",
        pssm_root / "Structure_annotated proteins" / f"sturcture_annotated_{split}" / "pssm" / f"{pid}.pssm",
        pssm_root / "Disorder_annotatred proteins" / f"disorder_annotated_{split}" / f"{pid}.pssm",
        pssm_root / "Structure_annotated proteins" / f"sturcture_annotated_{split}" / f"{pid}.pssm",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

# ─── ESM-2 loading ─────────────────────────────────────────────────────────

def _find_esm2(esm2_root: Path, pid: str, split: str, subset: str) -> Optional[Path]:
    """Search for ESM-2 embedding .npy in multiple candidate locations."""
    candidates = [
        esm2_root / subset / split / f"{pid}.npy",
        esm2_root / split / f"{pid}.npy",
        esm2_root / f"{pid}.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

# ─── SS / SASA loading ────────────────────────────────────────────────────

def _find_ss(ss_root: Optional[Path], pid: str, split: str) -> Optional[str]:
    """Load secondary structure annotation.

    Supported formats:
      - Text: {pid}.ss or {pid}.txt (string of 'H'/'E'/'C' per residue)
      - NPY:  {pid}.npy (int array 0/1/2), where 0->H, 1->E, 2->C
    Returns a secondary-structure string of length L.
    """
    if ss_root is None:
        return None
    candidates = [
        ss_root / f"{pid}.ss",
        ss_root / split / f"{pid}.ss",
        ss_root / f"{pid}.txt",
        ss_root / f"{pid}.npy",
        ss_root / split / f"{pid}.npy",
    ]
    for c in candidates:
        if c.exists():
            if c.suffix == ".npy":
                arr = np.load(c).astype(np.int64).flatten()
                # Map 0/1/2 -> H/E/C (fallback to C)
                m = {0: "H", 1: "E", 2: "C"}
                return "".join(m.get(int(v), "C") for v in arr.tolist())
            with open(c) as f:
                lines = [l.strip() for l in f if l.strip()]
                return lines[-1] if lines else None
    return None

def _find_sasa(sasa_root: Optional[Path], pid: str, split: str, seq_len: int) -> Optional[torch.Tensor]:
    """Load SASA values -> (L, 1)."""
    if sasa_root is None:
        return None
    candidates = [
        sasa_root / f"{pid}.npy",
        sasa_root / split / f"{pid}.npy",
        sasa_root / f"{pid}.txt",
    ]
    for c in candidates:
        if c.exists():
            if c.suffix == ".npy":
                arr = np.load(c).astype(np.float32).flatten()
            else:
                arr = np.loadtxt(c, dtype=np.float32).flatten()
            t = torch.from_numpy(arr).unsqueeze(-1)
            if t.shape[0] != seq_len:
                if t.shape[0] > seq_len:
                    t = t[:seq_len]
                else:
                    t = torch.nn.functional.pad(t, (0, 0, 0, seq_len - t.shape[0]))
            return t
    return None

# ─── Label parsing ─────────────────────────────────────────────────────────

def _parse_labels(label_line: str, subset: str):
    """
    Parse residue-level labels.
    Returns: (y_res (L,), y_prot scalar, disorder (L,1) or None)
    Label encoding follows hybridDBRpred2024 readme:
      Structure track: 0=non-binding, 1=DNA-binding, 2=non-DNA ligand binding (negative)
      Disorder track:  0=non-disordered (negative), 1=disordered DNA-binding (positive),
                       2/3/4 are disordered but non-DNA-binding (negative)
    """
    y_digits = torch.tensor([int(c) for c in label_line.strip()], dtype=torch.float32)
    # Only label "1" is positive DNA-binding for both tracks.
    y_res = (y_digits == 1).float()
    y_prot = (y_res.sum() > 0).float()

    disorder = None
    if subset == "disorder":
        # Disorder annotation for privileged branch: non-zero means disordered residue.
        disorder = (y_digits != 0).float().unsqueeze(-1)
    
    return y_res, y_prot, disorder

# ─── Main Dataset ──────────────────────────────────────────────────────────

class HybridDBRpred2024(Dataset):
    """
    Loads hybrid DBRpred examples.
    - PSSM features: from pssm_root
    - ESM-2 embeddings: from esm2_root
    - Physicochemical: computed in-memory from sequence
    - Privileged structure: SS labels (ss_root), SASA (sasa_root)
    - Privileged disorder: encoded from subset annotation
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        pssm_root: Optional[str | Path] = None,
        esm2_root: Optional[str | Path] = None,
        ss_root: Optional[str | Path] = None,
        sasa_root: Optional[str | Path] = None,
        max_len: int = 1500,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.pssm_root = Path(pssm_root) if pssm_root else self.data_root
        self.esm2_root = Path(esm2_root) if esm2_root else None
        self.ss_root = Path(ss_root) if ss_root else None
        self.sasa_root = Path(sasa_root) if sasa_root else None
        self.max_len = max_len

        self.examples = []
        self._load_examples()

    def _load_examples(self):
        """Load all examples from the data directory."""
        # Map internal split names to hybridDBRpred2024 file suffixes
        # train -> train, val -> validation, test -> test
        if self.split == "val":
            ds_split = "validation"
        else:
            ds_split = self.split

        for subset in ["structure", "disorder"]:
            subset_dir = self.data_root / subset / ds_split
            if not subset_dir.exists():
                # Kurgan hybridDBRpred2024 layout: separate files for structure / disorder
                #   Structure_annotated proteins/sturcture_annotated_{split}.txt
                #   Disorder_annotatred proteins/disorder_annotated_{split}.txt
                if subset == "structure":
                    alt = (
                        self.data_root
                        / "Structure_annotated proteins"
                        / f"sturcture_annotated_{ds_split}.txt"
                    )
                else:
                    alt = (
                        self.data_root
                        / "Disorder_annotatred proteins"
                        / f"disorder_annotated_{ds_split}.txt"
                    )
                if alt.exists():
                    # Directly parse this FASTA-like file
                    self.examples.extend(self._parse_fasta(alt, subset))
                    continue
                # If neither conventional nor Kurgan-style paths exist, skip this subset
                continue

            # Look for FASTA-like files in directory layout (if present)
            for fasta_file in sorted(subset_dir.glob("*.fasta")) + sorted(
                subset_dir.glob("*.txt")
            ):
                examples = self._parse_fasta(fasta_file, subset)
                self.examples.extend(examples)

        # If no fasta files found, try loading from a single combined file
        if not self.examples:
            combined = self.data_root / f"{self.split}.txt"
            if combined.exists():
                self.examples = self._parse_combined(combined)

    def _parse_fasta(self, path: Path, subset: str) -> list[dict]:
        """Parse a FASTA-like file with sequence and labels."""
        examples = []
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]

        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                pid = lines[i][1:].split()[0]
                if i + 2 < len(lines):
                    seq = lines[i + 1]
                    labels = lines[i + 2]
                    if len(seq) <= self.max_len and len(seq) == len(labels):
                        examples.append({
                            "pid": pid,
                            "seq": seq,
                            "labels": labels,
                            "subset": subset,
                        })
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        return examples

    def _parse_combined(self, path: Path) -> list[dict]:
        """Parse combined format file."""
        examples = []
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                parts = lines[i][1:].split()
                pid = parts[0]
                subset = parts[1] if len(parts) > 1 else "structure"
                if i + 2 < len(lines):
                    seq = lines[i + 1]
                    labels = lines[i + 2]
                    if len(seq) <= self.max_len and len(seq) == len(labels):
                        examples.append({
                            "pid": pid,
                            "seq": seq,
                            "labels": labels,
                            "subset": subset,
                        })
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        pid = ex["pid"]
        seq = ex["seq"]
        labels = ex["labels"]
        subset = ex["subset"]
        L = len(seq)

        # 1. Sequence encoding
        seq_idx = torch.tensor(encode_sequence(seq), dtype=torch.long)

        # 2. Physicochemical features
        phys = physchem_features(seq_idx)  # (L, PHYSCHEM_DIM)

        # 3. PSSM
        pssm_path = _find_pssm(self.pssm_root, pid, self.split)
        if pssm_path is not None:
            pssm = _load_pssm(pssm_path, L)
        else:
            pssm = torch.zeros(L, 20)

        # 4. ESM-2 embeddings
        esm2 = None
        if self.esm2_root is not None:
            esm2_path = _find_esm2(self.esm2_root, pid, self.split, subset)
            if esm2_path is not None:
                arr = np.load(esm2_path).astype(np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                esm2 = torch.from_numpy(arr)
                if esm2.shape[0] != L:
                    if esm2.shape[0] > L:
                        esm2 = esm2[:L]
                    else:
                        esm2 = torch.nn.functional.pad(esm2, (0, 0, 0, L - esm2.shape[0]))

        # 5. Labels
        y_res, y_prot, disorder = _parse_labels(labels, subset)

        # 6. Privileged structure annotations
        ss = None
        ss_str = _find_ss(self.ss_root, pid, self.split)
        if ss_str is not None and len(ss_str) == L:
            ss = _encode_ss(ss_str)

        sasa = _find_sasa(self.sasa_root, pid, self.split, L)

        # # For structure-annotated proteins, disorder = 0
        # if subset == "structure" and disorder is None:
        #     disorder = torch.zeros(L, 1)

        return {
            "pid": pid,
            "seq_idx": seq_idx,
            "phys": phys,
            "pssm": pssm,
            "esm2": esm2,
            "y_res": y_res,
            "y_prot": y_prot,
            "ss": ss,
            "sasa": sasa,
            "disorder": disorder,
            "subset": subset,
            "length": L,
        }

def collate_batch(samples: list[dict]) -> Batch:
    """Collate list of samples into a padded Batch."""
    B = len(samples)
    lengths = torch.tensor([s["length"] for s in samples], dtype=torch.long)
    max_len = int(lengths.max().item())

    # Determine ESM-2 dim
    esm2_dim = None
    for s in samples:
        if s["esm2"] is not None:
            esm2_dim = s["esm2"].shape[-1]
            break

    seq_idx = torch.full((B, max_len), PAD_IDX, dtype=torch.long)
    pssm = torch.zeros(B, max_len, 20)
    phys = torch.zeros(B, max_len, PHYSCHEM_DIM)
    esm2 = torch.zeros(B, max_len, esm2_dim if esm2_dim else 1280)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    y_res = torch.zeros(B, max_len)
    y_prot = torch.zeros(B)

    # Check if any sample has privileged features
    has_ss = any(s["ss"] is not None for s in samples)
    has_sasa = any(s["sasa"] is not None for s in samples)
    has_disorder = any(s["disorder"] is not None for s in samples)

    ss_dim = SS_DIM
    ss_batch = torch.zeros(B, max_len, ss_dim) if has_ss else None
    sasa_batch = torch.zeros(B, max_len, 1) if has_sasa else None
    disorder_batch = torch.zeros(B, max_len, 1) if has_disorder else None

    subsets = []
    pids = []

    for i, s in enumerate(samples):
        L = s["length"]
        seq_idx[i, :L] = s["seq_idx"]
        pssm[i, :L] = s["pssm"]
        phys[i, :L] = s["phys"]
        if s["esm2"] is not None:
            esm2[i, :L] = s["esm2"]
        mask[i, :L] = True
        y_res[i, :L] = s["y_res"]
        y_prot[i] = s["y_prot"]

        if has_ss and s["ss"] is not None:
            ss_batch[i, :L] = s["ss"]
        if has_sasa and s["sasa"] is not None:
            sasa_batch[i, :L] = s["sasa"]
        if has_disorder and s["disorder"] is not None:
            disorder_batch[i, :L] = s["disorder"]

        subsets.append(s["subset"])
        pids.append(s["pid"])

    return Batch(
        seq_idx=seq_idx,
        pssm=pssm,
        phys=phys,
        esm2=esm2,
        mask=mask,
        y_res=y_res,
        y_prot=y_prot,
        lengths=lengths,
        ss=ss_batch,
        sasa=sasa_batch,
        disorder=disorder_batch,
        subset=subsets,
        pids=pids,
    )