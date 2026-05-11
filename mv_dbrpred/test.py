"""
test.py – Inference script for MultiViewDBRpred.
Operates in sequence-only mode: ESM-2 + PSSM + physicochemical features.
No structure or disorder annotations required.
Path: /home/liusicen/methods/DBR_pred/Multi-view_DBRpred/mv_dbrpred/test.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mv_dbrpred.aa import encode_sequence, physchem_features
from mv_dbrpred.model import MultiViewDBRpred
from mv_dbrpred.utils import Batch, infer_use_bilstm_refine, resolve_esm2_dim

def predict_single(
    model: MultiViewDBRpred,
    sequence: str,
    esm2_embedding: Optional[np.ndarray] = None,
    pssm: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, np.ndarray]:
    """
    Predict DNA-binding for a single protein sequence.
    
    Args:
        model: Trained MultiViewDBRpred model
        sequence: Amino acid sequence string
        esm2_embedding: Pre-computed ESM-2 embedding (L, esm2_dim). If None, zeros.
        pssm: PSSM matrix (L, 20). If None, zeros.
        device: Compute device
    
    Returns:
        dict with 'res_prob' (L,), 'prot_prob' (scalar), 'alpha' (K,)
    """
    model.eval()
    L = len(sequence)

    seq_idx = torch.tensor(encode_sequence(sequence), dtype=torch.long).unsqueeze(0)
    phys = physchem_features(seq_idx.squeeze(0)).unsqueeze(0)

    if pssm is not None:
        pssm_t = torch.from_numpy(pssm.astype(np.float32)).unsqueeze(0)
    else:
        pssm_t = torch.zeros(1, L, 20)

    if esm2_embedding is not None:
        esm2_t = torch.from_numpy(esm2_embedding.astype(np.float32)).unsqueeze(0)
    else:
        esm2_t = torch.zeros(1, L, 1280)

    mask = torch.ones(1, L, dtype=torch.bool)
    lengths = torch.tensor([L], dtype=torch.long)

    batch = Batch(
        seq_idx=seq_idx,
        pssm=pssm_t,
        phys=phys,
        esm2=esm2_t,
        mask=mask,
        y_res=torch.zeros(1, L),
        y_prot=torch.zeros(1),
        lengths=lengths,
    ).to(device)

    with torch.no_grad():
        pred = model.predict(batch)

    return {
        "res_prob": pred["res_prob"][0, :L].cpu().numpy(),
        "prot_prob": float(pred["prot_prob"][0].cpu()),
        "alpha": pred["alpha"][0].cpu().numpy(),
    }

def main():
    parser = argparse.ArgumentParser(description="Predict DNA-binding (single sequence)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sequence", type=str, default=None, help="Amino acid sequence")
    parser.add_argument("--fasta", type=str, default=None, help="FASTA file")
    parser.add_argument("--esm2_npy", type=str, default=None, help="ESM-2 embedding .npy")
    parser.add_argument("--pssm_file", type=str, default=None, help="PSSM file")
    parser.add_argument("--output", type=str, default=None, help="Output .npz file")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    use_bilstm = infer_use_bilstm_refine(ckpt)

    esm2_dim = ckpt_args.get("esm2_dim", 1280)
    if args.esm2_npy:
        esm2_dim = np.load(args.esm2_npy).shape[-1]

    model = MultiViewDBRpred(
        esm2_dim=esm2_dim,
        hidden_dim=ckpt_args.get("hidden_dim", 256),
        topk=ckpt_args.get("topk", 0),
        use_bilstm_refine=use_bilstm,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Read sequence
    if args.sequence:
        seq = args.sequence.strip()
    elif args.fasta:
        with open(args.fasta) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith(">")]
            seq = "".join(lines)
    else:
        raise ValueError("Provide --sequence or --fasta")

    # Load optional features
    esm2_emb = np.load(args.esm2_npy) if args.esm2_npy else None
    pssm = None
    if args.pssm_file:
        if args.pssm_file.endswith(".npy"):
            pssm = np.load(args.pssm_file)
        else:
            # TODO: parse PSSM text format
            pass

    # Predict
    result = predict_single(model, seq, esm2_emb, pssm, device)

    # Output
    prot_label = "DNA-binding" if result["prot_prob"] >= args.threshold else "Non-binding"
    print(f"Protein prediction: {prot_label} (prob={result['prot_prob']:.4f})")
    print(f"Expert routing weights: {result['alpha']}")

    n_binding = (result["res_prob"] >= args.threshold).sum()
    print(f"Binding residues: {n_binding}/{len(seq)} (threshold={args.threshold})")

    # Detailed residue output
    for i, (aa, p) in enumerate(zip(seq, result["res_prob"])):
        flag = "*" if p >= args.threshold else " "
        print(f"  {i+1:4d} {aa} {p:.4f} {flag}")

    if args.output:
        np.savez(
            args.output,
            sequence=np.array(list(seq)),
            res_prob=result["res_prob"],
            prot_prob=result["prot_prob"],
            alpha=result["alpha"],
        )
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()