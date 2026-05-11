"""
metrics.py – Evaluation metrics for binary classification at residue and protein level.
Path: /home/liusicen/methods/DBR_pred/Multi-view_DBRpred/mv_dbrpred/metrics.py
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute AUROC, returning None if undefined."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute AUPRC (average precision), returning None if undefined."""
    # if len(np.unique(y_true)) < 2:
    #     return None
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return None
    

def _bin_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """
    Compute binary classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    auroc = _safe_auroc(y_true, y_prob)
    auprc = _safe_auprc(y_true, y_prob)
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Sensitivity (recall / TPR) and specificity (TNR)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    sn = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    sp = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "acc": acc,
        "bacc": bacc,
        "mcc": mcc,
        "precision": precision,
        "f1": f1,
        "sn": sn,
        "sp": sp,
    }

class Accumulator:
    """Accumulate predictions across batches for global metric computation."""

    def __init__(self):
        self.prot_true: list[float] = []
        self.prot_prob: list[float] = []
        self.res_true: list[np.ndarray] = []
        self.res_prob: list[np.ndarray] = []
        self.subsets: list[str] = []

    def add_protein(self, y_true: np.ndarray, y_prob: np.ndarray, subsets: Optional[list[str]] = None):
        """Add protein-level predictions for a batch."""
        self.prot_true.extend(y_true.tolist())
        self.prot_prob.extend(y_prob.tolist())
        if subsets:
            self.subsets.extend(subsets)

    def add_residue(self, y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray):
        """Add residue-level predictions for a batch (masked)."""
        B = y_true.shape[0]
        for i in range(B):
            valid = mask[i].astype(bool)
            self.res_true.append(y_true[i][valid])
            self.res_prob.append(y_prob[i][valid])

    def protein_metrics(self, threshold: float = 0.5) -> dict[str, Any]:
        if not self.prot_true:
            return {}
        return _bin_metrics(
            np.array(self.prot_true),
            np.array(self.prot_prob),
            threshold,
        )

    def residue_metrics(self, threshold: float = 0.5) -> dict[str, Any]:
        if not self.res_true:
            return {}
        all_true = np.concatenate(self.res_true)
        all_prob = np.concatenate(self.res_prob)
        return _bin_metrics(all_true, all_prob, threshold)

    def stratified_metrics(self, threshold: float = 0.5) -> dict[str, dict[str, Any]]:
        """Compute metrics stratified by subset (structure vs disorder)."""
        results = {}
        if not self.subsets:
            return results

        for subset in ["structure", "disorder"]:
            indices = [i for i, s in enumerate(self.subsets) if s == subset]
            if not indices:
                continue

            # Protein-level
            pt = np.array([self.prot_true[i] for i in indices])
            pp = np.array([self.prot_prob[i] for i in indices])
            results[f"{subset}_protein"] = _bin_metrics(pt, pp, threshold)

            # Residue-level
            if len(self.res_true) > max(indices):
                rt = np.concatenate([self.res_true[i] for i in indices if i < len(self.res_true)])
                rp = np.concatenate([self.res_prob[i] for i in indices if i < len(self.res_prob)])
                results[f"{subset}_residue"] = _bin_metrics(rt, rp, threshold)

        return results