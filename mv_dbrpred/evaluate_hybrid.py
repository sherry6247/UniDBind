"""
evaluate_hybrid.py – Evaluate MultiViewMoEModel on hybridDBRpred2024 test split,
stratified by structure vs disorder track.
Path: /home/liusicen/methods/DBR_pred/Multi-view_DBRpred/mv_dbrpred/evaluate_hybrid.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mv_dbrpred.data import HybridDBRpred2024, collate_batch
from mv_dbrpred.metrics import _bin_metrics, Accumulator
from mv_dbrpred.model import MultiViewMoEModel
from mv_dbrpred.utils import ensure_dir, infer_use_bilstm_refine, resolve_esm2_dim, resolve_esm2_root, save_json

def _fmt4(x: Any) -> str:
    if x is None:
        return "N/A"
    return f"{float(x):.4f}"

def _metrics_row(name: str, level: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {level} | {_fmt4(m.get('auroc'))} | {_fmt4(m.get('auprc'))} | {_fmt4(m.get('bacc'))} | "
        f"{_fmt4(m.get('mcc'))} | {_fmt4(m.get('acc'))} | {_fmt4(m.get('precision'))} | "
        f"{_fmt4(m.get('f1'))} | {_fmt4(m.get('sn'))} | {_fmt4(m.get('sp'))} |"
    )

def _collect_predictions(
    model: MultiViewMoEModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    acc = Accumulator()
    pids: list[str] = []
    subsets: list[str] = []
    res_prob_per_sample: list[np.ndarray] = []
    res_true_per_sample: list[np.ndarray] = []

    with torch.no_grad():
        for raw_batch in tqdm(loader, desc="Evaluating"):
            batch = raw_batch.to(device)
            pred, losses = model(batch)

            prot_prob = torch.sigmoid(pred["prot_logits"]).cpu().numpy()
            # prot_prob = pred["prot_logits"].cpu().numpy()
            prot_true = batch.y_prot.cpu().numpy()
            acc.add_protein(prot_true, prot_prob, batch.subset)

            res_prob = torch.sigmoid(pred["res_logits"]).cpu().numpy()
            # res_prob = pred["res_logits"].cpu().numpy()
            res_true = batch.y_res.cpu().numpy()
            mask_np = batch.mask.cpu().numpy()
            acc.add_residue(res_true, res_prob, mask_np)

            # Keep per-sample predictions with pid/subset for exporting.
            B = res_true.shape[0]
            batch_pids = batch.pids if batch.pids else [""] * B
            batch_subsets = batch.subset if batch.subset else ["structure"] * B
            for i in range(B):
                pid = str(batch_pids[i])
                subset = str(batch_subsets[i])
                valid = mask_np[i].astype(bool)
                pids.append(pid)
                subsets.append(subset)
                res_true_per_sample.append(res_true[i][valid].astype(np.int64, copy=False))
                res_prob_per_sample.append(res_prob[i][valid].astype(np.float32, copy=False))

    results = {
        "prot_true": np.array(acc.prot_true, dtype=np.int64),
        "prot_prob": np.array(acc.prot_prob, dtype=np.float32),
        "res_true": np.concatenate(acc.res_true) if acc.res_true else np.array([], dtype=np.int64),
        "res_prob": np.concatenate(acc.res_prob) if acc.res_prob else np.array([], dtype=np.float32),
        # For stratified metrics (uses Accumulator.subsets ordering = sample ordering)
        "subsets": list(acc.subsets),
        # For exporting per sample by pid/subset
        "pids": pids,
        "subsets_per_sample": subsets,
        "res_true_per_sample": res_true_per_sample,
        "res_prob_per_sample": res_prob_per_sample,
    }
    return results


def _save_predictions_by_subset(
    out_dir: Path,
    split: str,
    col: dict[str, Any],
) -> dict[str, Any]:
    """
    Save per-protein residue outputs and protein outputs, separated by
    `structure/` and `disorder/` folders (HybridDBRpred2024 naming: files keyed by protein id).

    Layout:
      out_dir/predictions/{subset}/{split}/{pid}_residue_prob.npy   (residue probs, 1D float32)
      out_dir/predictions/{subset}/{split}/{pid}_residue_true.npy   (residue labels, 1D int64)
      out_dir/predictions/{subset}/{split}/protein_probs.csv         (pid,true,prob)
      out_dir/predictions/{subset}/{split}/residue_probs.csv         (pid,res_index,true,prob)
    """
    base = ensure_dir(out_dir / "predictions")
    pids: list[str] = col.get("pids", [])
    subsets: list[str] = col.get("subsets_per_sample", [])
    prot_true = col.get("prot_true")
    prot_prob = col.get("prot_prob")
    res_prob_per_sample: list[np.ndarray] = col.get("res_prob_per_sample", [])
    res_true_per_sample: list[np.ndarray] = col.get("res_true_per_sample", [])

    if prot_true is None or prot_prob is None:
        return {"written": False, "reason": "Missing prot_true/prot_prob in collected results."}

    if not (
        len(pids)
        == len(subsets)
        == len(res_prob_per_sample)
        == len(res_true_per_sample)
        == int(len(prot_true))
        == int(len(prot_prob))
    ):
        return {
            "written": False,
            "reason": (
                "Mismatched lengths among collected arrays: "
                f"pids={len(pids)}, subsets={len(subsets)}, res_prob_per_sample={len(res_prob_per_sample)}, "
                f"res_true_per_sample={len(res_true_per_sample)}, prot_true={len(prot_true)}, prot_prob={len(prot_prob)}"
            ),
        }

    written: dict[str, Any] = {"structure": {"n": 0}, "disorder": {"n": 0}}
    for subset in ["structure", "disorder"]:
        sub_dir = ensure_dir(base / subset / split)
        protein_rows: list[str] = ["pid,true,prob\n"]
        residue_rows: list[str] = ["pid,res_index,true,prob\n"]
        for i, (pid, s) in enumerate(zip(pids, subsets)):
            if s != subset:
                continue
            safe_pid = pid.strip()
            if not safe_pid:
                safe_pid = f"sample_{i:06d}"

            # Residue-level outputs (evaluated/masked positions) for this protein
            cur_prob = res_prob_per_sample[i].astype(np.float32, copy=False)
            cur_true = res_true_per_sample[i].astype(np.int64, copy=False)
            np.save(sub_dir / f"{safe_pid}_residue_prob.npy", cur_prob)
            np.save(sub_dir / f"{safe_pid}_residue_true.npy", cur_true)
            for j in range(cur_prob.shape[0]):
                residue_rows.append(f"{safe_pid},{j},{int(cur_true[j])},{float(cur_prob[j])}\n")

            protein_rows.append(f"{safe_pid},{int(prot_true[i])},{float(prot_prob[i])}\n")
            written[subset]["n"] += 1

        (sub_dir / "protein_probs.csv").write_text("".join(protein_rows), encoding="utf-8")
        (sub_dir / "residue_probs.csv").write_text("".join(residue_rows), encoding="utf-8")
        written[subset]["dir"] = str(sub_dir.resolve())
        written[subset]["protein_csv"] = str((sub_dir / "protein_probs.csv").resolve())
        written[subset]["residue_csv"] = str((sub_dir / "residue_probs.csv").resolve())

    return {"written": True, "paths": written}


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "mcc") -> float:
    if y_true.size == 0:
        return 0.5
    best_t, best_s = 0.5, -1e18
    for t in np.linspace(0.01, 0.99, 99):
        m = _bin_metrics(y_true, y_prob, threshold=float(t))
        s = m.get(metric)
        if s is None:
            continue
        if float(s) > best_s:
            best_s = float(s)
            best_t = float(t)
    return best_t


def _evaluate_from_collected(col: dict[str, Any], thr_prot: float, thr_res: float) -> dict[str, Any]:
    prot_true = col["prot_true"]
    prot_prob = col["prot_prob"]
    res_true = col["res_true"]
    res_prob = col["res_prob"]
    subsets = col["subsets"]
    res_true_per_sample = col["res_true_per_sample"]
    res_prob_per_sample = col["res_prob_per_sample"]

    strat: dict[str, Any] = {}
    if subsets:
        for subset in ["structure", "disorder"]:
            ids = [i for i, s in enumerate(subsets) if s == subset]
            if not ids:
                continue
            pt = np.array([prot_true[i] for i in ids], dtype=np.int64)
            pp = np.array([prot_prob[i] for i in ids], dtype=np.float32)
            strat[f"{subset}_protein"] = _bin_metrics(pt, pp, threshold=thr_prot)
            rt = np.concatenate([res_true_per_sample[i] for i in ids if i < len(res_true_per_sample)])
            rp = np.concatenate([res_prob_per_sample[i] for i in ids if i < len(res_prob_per_sample)])
            strat[f"{subset}_residue"] = _bin_metrics(rt, rp, threshold=thr_res)

    return {
        "thresholds": {"protein": float(thr_prot), "residue": float(thr_res)},
        "overall_protein": _bin_metrics(prot_true, prot_prob, threshold=thr_prot),
        "overall_residue": _bin_metrics(res_true, res_prob, threshold=thr_res),
        "stratified": strat,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate MultiViewMoEModel")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--esm2_root", type=str, default=None)
    parser.add_argument("--pssm_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument("--max_len", type=int, default=5500)
    parser.add_argument("--thr_prot", type=float, default=0.5, help="Protein decision threshold")
    parser.add_argument("--thr_res", type=float, default=0.5, help="Residue decision threshold")
    parser.add_argument("--auto_thr_on_val", action="store_true",
                        help="Tune thresholds on validation split by MCC, then evaluate on target split")
    parser.add_argument("--pssm_dim", type=int, default=20)
    parser.add_argument("--phys_dim", type=int, default=7)
    parser.add_argument("--ss_dim", type=int, default=3)
    parser.add_argument("--sasa_dim", type=int, default=1)
    parser.add_argument("--disorder_dim", type=int, default=1)
    parser.add_argument("--distill_temperature", type=float, default=4.0)
    parser.add_argument("--num_experts", type=int, default=4)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    data_root = Path(args.data_root)
    esm2_root = resolve_esm2_root(args.esm2_root or ckpt_args.get("esm2_root") or data_root)
    pssm_root = Path(args.pssm_root) if args.pssm_root else data_root
    esm2_dim = resolve_esm2_dim(esm2_root)

    # Build dataset
    test_ds = HybridDBRpred2024(
        data_root=data_root,
        split=args.split,
        pssm_root=pssm_root,
        esm2_root=esm2_root,
        max_len=args.max_len,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=2,
    )

    print(f"Test samples: {len(test_ds)}")

    # Build model
    use_bilstm = infer_use_bilstm_refine(ckpt)
    model = MultiViewMoEModel(
        esm_dim=esm2_dim,
        pssm_dim=args.pssm_dim,
        phys_dim=args.phys_dim,
        ss_dim=args.ss_dim,
        sasa_dim=args.sasa_dim,
        disorder_dim=args.disorder_dim,
        distill_temperature=args.distill_temperature,
        num_experts=args.num_experts,
        hidden_dim=ckpt_args.get("hidden_dim", 256),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("Model loaded successfully.")

    thr_prot = float(args.thr_prot)
    thr_res = float(args.thr_res)

    if args.auto_thr_on_val:
        val_ds = HybridDBRpred2024(
            data_root=data_root,
            split="val",
            pssm_root=pssm_root,
            esm2_root=esm2_root,
            max_len=args.max_len,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_batch, num_workers=2,
        )
        val_col = _collect_predictions(model, val_loader, device)
        thr_prot = _best_threshold(val_col["prot_true"], val_col["prot_prob"], metric="mcc")
        thr_res = _best_threshold(val_col["res_true"], val_col["res_prob"], metric="mcc")
        print(f"Auto thresholds from val (MCC): prot={thr_prot:.3f}, res={thr_res:.3f}")

    # Evaluate
    test_col = _collect_predictions(model, test_loader, device)
    results = _evaluate_from_collected(test_col, thr_prot=thr_prot, thr_res=thr_res)

    # Print results
    header = "| Subset | Level | AUROC | AUPRC | BACC | MCC | ACC | Precision | F1 | Sn | Sp |"
    sep = "|" + "---|" * 11
    print("\n" + header)
    print(sep)
    print(_metrics_row("Overall", "Protein", results["overall_protein"]))
    print(_metrics_row("Overall", "Residue", results["overall_residue"]))
    for key, m in results["stratified"].items():
        parts = key.rsplit("_", 1)
        print(_metrics_row(parts[0], parts[1], m))

    # Save
    out_dir = ensure_dir(Path(args.out_dir))
    save_json(results, out_dir / f"metrics_{args.split}.json")
    print(f"\nResults saved to {out_dir / f'metrics_{args.split}.json'}")

    export_info = _save_predictions_by_subset(out_dir, split=args.split, col=test_col)
    save_json(export_info, out_dir / f"predictions_{args.split}_export.json")
    if export_info.get("written"):
        paths = export_info.get("paths", {})
        print("\nSaved per-protein probabilities under:")
        for subset in ["structure", "disorder"]:
            if subset in paths:
                print(f"  {subset}: {paths[subset].get('dir')}")
    else:
        print(f"\nPrediction export skipped: {export_info.get('reason')}")

if __name__ == "__main__":
    main()