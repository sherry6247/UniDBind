"""
train.py – Training script for MultiViewDBRpred.
Path: /home/liusicen/methods/DBR_pred/Multi-view_DBRpred/mv_dbrpred/train.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mv_dbrpred.data import HybridDBRpred2024, collate_batch
from mv_dbrpred.metrics import Accumulator
from mv_dbrpred.model import  MultiViewMoEModel
from mv_dbrpred.utils import ensure_dir, resolve_esm2_root, resolve_esm2_dim, save_json, set_seed

def _evaluate(model: MultiViewMoEModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Evaluate model on a dataset."""
    model.eval()
    acc = Accumulator()
    loss_sums: dict[str, float] = {}
    n_steps = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = raw_batch.to(device)
            out, losses = model(batch)

            for k, v in losses.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu().item())
            n_steps += 1

            prot_prob = torch.sigmoid(out["prot_logits"]).detach().cpu().numpy()
            prot_true = batch.y_prot.detach().cpu().numpy()
            acc.add_protein(prot_true, prot_prob, batch.subset)

            res_prob = torch.sigmoid(out["res_logits"].squeeze(-1)).detach().cpu().numpy()
            res_true = batch.y_res.detach().cpu().numpy()
            mask_np = batch.mask.detach().cpu().numpy()
            acc.add_residue(res_true, res_prob, mask_np)

    avg_losses = {k: v / max(n_steps, 1) for k, v in loss_sums.items()}
    prot_m = acc.protein_metrics()
    res_m = acc.residue_metrics()
    strat = acc.stratified_metrics()

    return {
        "losses": avg_losses,
        "protein": prot_m,
        "residue": res_m,
        "stratified": strat,
    }

def main():
    parser = argparse.ArgumentParser(description="Train MultiViewDBRpred")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir of hybridDBRpred2024 dataset")
    parser.add_argument("--esm2_root", type=str, default=None,
                        help="Root dir of ESM-2 embeddings")
    parser.add_argument("--pssm_root", type=str, default=None,
                        help="Root dir of PSSM features")
    parser.add_argument("--ss_root", type=str, default=None,
                        help="Root dir of secondary structure annotations")
    parser.add_argument("--sasa_root", type=str, default=None,
                        help="Root dir of SASA annotations")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=0,
                        help="Top-k experts for routing (0=all)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--bilstm_refine", default=True,
                        help="Use BiLSTM post-MoE refinement")
    parser.add_argument("--struct_pos_boost", type=float, default=0.0)
    parser.add_argument("--res_pos_weight", type=float, default=1.0,
                        help="Global positive-class weight for residue BCEWithLogits")
    parser.add_argument("--prot_pos_weight", type=float, default=1.0,
                        help="Global positive-class weight for protein BCEWithLogits")
    parser.add_argument("--max_len", type=int, default=5500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pssm_dim", type=int, default=20)
    parser.add_argument("--phys_dim", type=int, default=7)
    parser.add_argument("--ss_dim", type=int, default=3)
    parser.add_argument("--sasa_dim", type=int, default=1)
    parser.add_argument("--disorder_dim", type=int, default=1)
    parser.add_argument("--distill_temperature", type=float, default=4.0)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--route_loss_weight", type=float, default=0.1,
                        help="Weight for expert-agnostic routing regularization")
    parser.add_argument("--route_entropy_target", type=float, default=0.6,
                        help="Target normalized routing entropy ratio in [0,1]")

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = ensure_dir(Path(args.out_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Data ──────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    esm2_root = resolve_esm2_root(args.esm2_root or data_root)
    pssm_root = Path(args.pssm_root) if args.pssm_root else data_root

    common_kwargs = dict(
        data_root=data_root,
        pssm_root=pssm_root,
        esm2_root=esm2_root,
        ss_root=args.ss_root,
        sasa_root=args.sasa_root,
        max_len=args.max_len,
    )

    train_ds = HybridDBRpred2024(split="train", **common_kwargs)
    val_ds = HybridDBRpred2024(split="val", **common_kwargs)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=args.num_workers,
        pin_memory=True,
    )

    # ─── Model ─────────────────────────────────────────────────────────
    esm2_dim = resolve_esm2_dim(esm2_root)
    print(f"ESM-2 dim: {esm2_dim}")

    model = MultiViewMoEModel(
        esm_dim=esm2_dim,
        pssm_dim=args.pssm_dim,
        phys_dim=args.phys_dim,
        ss_dim=args.ss_dim,
        sasa_dim=args.sasa_dim,
        disorder_dim=args.disorder_dim,
        hidden_dim=args.hidden_dim,
        distill_temperature=args.distill_temperature,
        num_experts=args.num_experts,
        dropout=args.dropout,
        route_loss_weight=args.route_loss_weight,
        route_entropy_target=args.route_entropy_target,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = -math.inf
    best_path = out_dir / "best.pt"
    patience_counter = 0

    # ─── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: dict[str, float] = {}
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for raw_batch in pbar:
            batch = raw_batch.to(device)
            opt.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out, losses = model(batch)
                
                total_loss = losses['loss']

            scaler.scale(total_loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v.detach().cpu().item())
            n_steps += 1

            pbar.set_postfix({"loss": f"{float(total_loss):.4f}"})

        scheduler.step()

        avg_train = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

        # ─── Validation ────────────────────────────────────────────────
        val_result = _evaluate(model, val_loader, device)
        val_losses = val_result["losses"]

        # Primary score: average of protein AUROC and residue AUROC
        prot_auroc = val_result["protein"].get("auroc")
        res_auroc = val_result["residue"].get("auroc")
        score_parts = [x for x in [prot_auroc, res_auroc] if x is not None]
        val_score = sum(score_parts) / len(score_parts) if score_parts else 0.0

        print(
            f"  Epoch {epoch}: train_loss={avg_train.get('loss', 0):.4f} | "
            f"val_loss={val_losses.get('loss', 0):.4f} | "
            f"prot_auroc={prot_auroc or 'N/A'} | res_auroc={res_auroc or 'N/A'} | "
            f"score={val_score:.4f}"
        )

        # ─── Checkpointing ────────────────────────────────────────────
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_score": val_score,
                "val_result": val_result,
                "args": vars(args),
            }, best_path)
            print(f"  → Saved best model (score={val_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }, out_dir / "last.pt")

    # ─── Save training summary ─────────────────────────────────────────
    summary = {
        "best_val_score": best_score,
        "args": vars(args),
    }
    save_json(summary, out_dir / "train_summary.json")
    print(f"\nTraining finished. Best val score: {best_score:.4f}")
    print(f"Best model saved to: {best_path}")

if __name__ == "__main__":
    main()