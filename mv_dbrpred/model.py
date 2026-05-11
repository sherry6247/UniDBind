"""model.py – Mechanism-aware Multi-View MoE for DNA-binding prediction."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# 0.  Batch dataclass (copied from your definition for reference)
# ──────────────────────────────────────────────────────────────
@dataclass
class Batch:
    """Collated mini-batch."""
    seq_idx: torch.Tensor
    pssm: torch.Tensor
    phys: torch.Tensor
    esm2: torch.Tensor
    mask: torch.Tensor
    y_res: torch.Tensor
    y_prot: torch.Tensor
    lengths: torch.Tensor
    ss: Optional[torch.Tensor] = None
    sasa: Optional[torch.Tensor] = None
    disorder: Optional[torch.Tensor] = None
    subset: Optional[list[str]] = None
    pids: Optional[list[str]] = None

# ──────────────────────────────────────────────────────────────
# 1.  Building blocks
# ──────────────────────────────────────────────────────────────
class ViewProjector(nn.Module):
    """Project one input view into the shared hidden space."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GatedFusion(nn.Module):
    """Gated fusion of three projected views → shared representation H."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        h_esm: torch.Tensor,
        h_pssm: torch.Tensor,
        h_phys: torch.Tensor,
    ) -> torch.Tensor:
        cat = torch.cat([h_esm, h_pssm, h_phys], dim=-1)   # (B, L, 3*d)
        g = self.gate(cat)                                    # (B, L, 3)
        stacked = torch.stack([h_esm, h_pssm, h_phys], dim=-1)  # (B, L, d, 3)
        fused = (stacked * g.unsqueeze(-2)).sum(dim=-1)       # (B, L, d)
        return fused

class Expert(nn.Module):
    """A single feed-forward expert."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MoELayer(nn.Module):
    """Mechanism-aware Mixture-of-Experts with 4 experts."""
    def __init__(self, hidden_dim: int, num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(hidden_dim, dropout) for _ in range(num_experts)]
        )
        # Routing network: sequence-conditioned (no privileged info)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, H: torch.Tensor):
        """
        Args:
            H: (B, L, d)
        Returns:
            Z: (B, L, d)  – mixed expert output
            alpha: (B, L, num_experts)  – routing weights
        """
        alpha = F.softmax(self.router(H), dim=-1)             # (B, L, K)
        expert_outs = torch.stack(
            [expert(H) for expert in self.experts], dim=-1
        )                                                      # (B, L, d, K)
        Z = (expert_outs * alpha.unsqueeze(-2)).sum(dim=-1)    # (B, L, d)
        return Z, alpha

class PrivilegedBranch(nn.Module):
    """Auxiliary branch that encodes privileged annotations (training only)."""
    def __init__(self, aux_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Auxiliary residue-level head (for logit distillation)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, aux_dim) concatenated privileged features
        Returns:
            h: (B, L, d) representation
            logits: (B, L, 1) auxiliary logits
        """
        h = self.net(x)
        logits = self.head(h)
        return h, logits

class MaskedAttentionPooling(nn.Module):
    """Attention-weighted pooling over sequence positions (masked)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, Z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: (B, L, d)
            mask: (B, L) bool
        Returns:
            pooled: (B, d)
        """
        scores = self.attn(Z).squeeze(-1)                     # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)     # (B, L, 1)
        pooled = (Z * weights).sum(dim=1)                     # (B, d)
        return pooled

# ──────────────────────────────────────────────────────────────
# sequence context encoder
# ──────────────────────────────────────────────────────────────
class SequenceContextEncoder(nn.Module):
    """Transformer encoder for contextual sequence modeling before prediction."""
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d)
            mask: (B, L) bool – True for valid positions
        Returns:
            (B, L, d)
        """
        # nn.TransformerEncoder expects key_padding_mask where True = IGNORED
        key_padding_mask = ~mask  # (B, L)
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(out)

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss with dynamic positive weight.
    """
    def __init__(self, gamma_pos=2.0, gamma_neg=4.0, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip  # asymmetric clipping for negatives

    def forward(self, logits, targets, mask):
        """
        logits:  (B, L)
        targets: (B, L)
        mask:    (B, L)
        """
        probs = torch.sigmoid(logits)
        
        # Asymmetric focusing
        pos_part = targets * (1 - probs) ** self.gamma_pos * F.logsigmoid(logits)
        
        # Clipping for negatives: shift probability to reduce easy-negative contribution
        neg_probs = (probs + self.clip).clamp(max=1.0)
        neg_part = (1 - targets) * neg_probs ** self.gamma_neg * F.logsigmoid(-logits)
        
        loss = -(pos_part + neg_part)
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        return loss

class DiceLoss(nn.Module):
    """Soft dice loss for binary segmentation."""
    def __init__(self, smooth=1.0, square_denominator=True):
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self, logits, targets, mask):
        probs = torch.sigmoid(logits) * mask.float()
        targets = targets.float() * mask.float()
        
        intersection = (probs * targets).sum()
        if self.square_denominator:
            denominator = (probs ** 2).sum() + (targets ** 2).sum()
        else:
            denominator = probs.sum() + targets.sum()
        
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice

class LocalContextEncoder(nn.Module):
    """Multi-scale 1D CNN for local binding-site pattern capture."""
    def __init__(self, hidden_dim, kernel_sizes=(3, 5, 7, 11), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // len(kernel_sizes),
                          kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim // len(kernel_sizes)),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, Z, mask):
        """
        Z: (B, L, d), mask: (B, L)
        """
        x = Z.transpose(1, 2)  # (B, d, L)
        x = x * mask.unsqueeze(1).float()
        
        conv_outs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)  # (B, d, L)
        x = x.transpose(1, 2)            # (B, L, d)
        return self.proj(x)

class EnhancedResidueHead(nn.Module):
    """Multi-layer residue prediction head with skip connections."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim // 4, 1)
        
        # Skip projection
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim // 2)

    def forward(self, Z):
        h1 = self.layer1(Z)          # (B, L, d)
        h2 = self.layer2(h1)         # (B, L, d//2)
        h2 = h2 + self.skip_proj(Z)  # skip connection
        h3 = self.layer3(h2)         # (B, L, d//4)
        return self.out(h3)          # (B, L, 1)

class EnhancedPrivilegedBranch(nn.Module):
    """Stronger auxiliary branch with cross-attention to sequence."""
    def __init__(self, aux_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Cross-attention: sequence queries, structure keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, aux_feats, H_seq, mask):
        """
        aux_feats: (B, L, aux_dim)
        H_seq:     (B, L, d)  — the fused sequence representation
        mask:      (B, L)
        """
        h_aux = self.proj(aux_feats)
        key_padding_mask = ~mask
        
        # Cross-attention: sequence attends to structure
        h_cross, _ = self.cross_attn(
            query=H_seq,
            key=h_aux,
            value=h_aux,
            key_padding_mask=key_padding_mask,
        )
        h = H_seq + h_cross
        h = h + self.ffn(h)
        logits = self.head(h)
        return h, logits

class ContrastiveDistillLoss(nn.Module):
    """Contrastive loss between sequence and privileged representations at binding sites."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, H_seq, H_priv, y_res, mask):
        """
        H_seq, H_priv: (B, L, d)
        y_res: (B, L) binary
        mask: (B, L)
        """
        # Gather positive residue representations
        pos_mask = (y_res == 1) & mask  # (B, L)
        
        if pos_mask.sum() < 2:
            return torch.tensor(0.0, device=H_seq.device)

        # Flatten
        h_s = F.normalize(H_seq[pos_mask], dim=-1)   # (N_pos, d)
        h_p = F.normalize(H_priv[pos_mask], dim=-1)   # (N_pos, d)

        # InfoNCE: each seq repr should be close to its corresponding priv repr
        sim = torch.mm(h_s, h_p.t()) / self.temperature  # (N_pos, N_pos)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return loss

def ohem_loss(logits, targets, mask, top_k_ratio=0.3):
    """Online Hard Example Mining for residue-level loss."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction='none'
    )
    bce = bce * mask.float()
    
    # 所有正样本保留
    pos_mask = (targets == 1) & mask
    pos_loss = (bce * pos_mask.float()).sum()
    
    # 负样本只取最难的 top_k
    neg_mask = (targets == 0) & mask
    neg_losses = bce[neg_mask]
    k = max(int(neg_losses.numel() * top_k_ratio), pos_mask.sum().item())
    hard_neg_loss, _ = neg_losses.topk(min(k, neg_losses.numel()))
    
    total = pos_loss + hard_neg_loss.sum()
    count = pos_mask.sum() + hard_neg_loss.numel()
    return total / count.clamp(min=1.0)
    

# ──────────────────────────────────────────────────────────────
# 2.  Main Model
# ──────────────────────────────────────────────────────────────
class MultiViewMoEModel(nn.Module):
    """
    Sequence-centered multi-view MoE model for unified DNA-binding
    protein & residue prediction (structured + disordered proteins).
    """

    def __init__(
        self,
        esm_dim: int = 2560,
        pssm_dim: int = 20,
        phys_dim: int = 10,
        hidden_dim: int = 512,
        num_experts: int = 4,
        ss_dim: int = 8,
        sasa_dim: int = 1,
        disorder_dim: int = 1,
        dropout: float = 0.1,
        distill_temperature: float = 4.0,
        route_loss_weight: float = 0.1,
        route_entropy_target: float = 0.6,
        num_layers_ctx: int = 2,
        num_heads_ctx: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.distill_temperature = distill_temperature
        self.route_loss_weight = route_loss_weight
        self.route_entropy_target = route_entropy_target

        # ---- View projectors ----
        self.proj_esm = ViewProjector(esm_dim, hidden_dim, dropout)
        self.proj_pssm = ViewProjector(pssm_dim, hidden_dim, dropout)
        self.proj_phys = ViewProjector(phys_dim, hidden_dim, dropout)

        # ---- Multi-view fusion ----
        self.fusion = GatedFusion(hidden_dim)

        # ---- MoE layer ----
        self.moe = MoELayer(hidden_dim, num_experts, dropout)

        # ---- Contextual encoder before residue prediction ----
        self.context_encoder = SequenceContextEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers_ctx,       # 新增超参数，建议默认 2
            num_heads=num_heads_ctx,         # 新增超参数，建议默认 8
            ff_dim=hidden_dim * 4,
            dropout=dropout,
        )
        self.local_encoder = LocalContextEncoder(hidden_dim, kernel_sizes=(3, 5, 7, 11), dropout=dropout)

        # ---- Prediction heads ----
        self.res_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pool = MaskedAttentionPooling(hidden_dim)
        self.prot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ---- Privileged branches (training only) ----
        struct_input_dim = ss_dim + sasa_dim   # concat of SS + SASA
        self.struct_branch = PrivilegedBranch(struct_input_dim, hidden_dim, dropout)

        dis_input_dim = disorder_dim           # disorder annotation
        self.dis_branch = PrivilegedBranch(dis_input_dim, hidden_dim, dropout)

        self.asym_focal = AsymmetricFocalLoss()
        self.dice_loss = DiceLoss()
        self.contrastive_distill = ContrastiveDistillLoss()

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------
    @staticmethod
    def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MSE loss only over valid positions."""
        diff = (a - b) ** 2                        # (B, L, d)
        diff = diff.mean(dim=-1)                   # (B, L)
        return (diff * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    def _kl_distill(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence distillation on residue-level logits (with temperature)."""
        T = self.distill_temperature
        # Convert to probabilities with temperature
        s = torch.sigmoid(student_logits.squeeze(-1) / T)   # (B, L)
        t = torch.sigmoid(teacher_logits.squeeze(-1) / T)   # (B, L)
        # Binary KL: t*log(t/s) + (1-t)*log((1-t)/(1-s))
        eps = 1e-7
        kl = t * torch.log((t + eps) / (s + eps)) + \
             (1 - t) * torch.log((1 - t + eps) / (1 - s + eps))
        kl = kl * mask.float()
        return (kl.sum() / mask.float().sum().clamp(min=1.0)) * (T ** 2)

    def _routing_regularization(self, alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Expert-agnostic routing regularization.

        1) Balance global expert usage to avoid routing collapse.
        2) Control token-level routing entropy (neither too flat nor too sharp).
        """
        eps = 1e-8
        valid = mask.float().unsqueeze(-1)  # (B, L, 1)
        n_valid = valid.sum().clamp(min=1.0)

        # Global load balance across experts.
        mean_prob = (alpha * valid).sum(dim=(0, 1)) / n_valid  # (K,)
        uniform = torch.full_like(mean_prob, 1.0 / self.num_experts)
        loss_balance = F.kl_div((mean_prob + eps).log(), uniform, reduction="batchmean")

        # Token-level entropy ratio in [0,1] (normalized by log(K)).
        token_entropy = -(alpha * torch.log(alpha + eps)).sum(dim=-1)  # (B, L)
        token_entropy = (token_entropy * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        max_entropy = math.log(self.num_experts + eps)
        entropy_ratio = token_entropy / max_entropy
        loss_entropy = (entropy_ratio - self.route_entropy_target).abs()

        return loss_balance + 0.1 * loss_entropy
    
    # 在 forward 中计算动态 pos_weight
    def _compute_pos_weight(self, y_res, mask):
        """Per-batch dynamic positive weight."""
        valid = mask.float()
        n_pos = (y_res * valid).sum().clamp(min=1.0)
        n_neg = ((1 - y_res) * valid).sum().clamp(min=1.0)
        # 限制最大权重避免梯度爆炸
        weight = (n_neg / n_pos).clamp(max=50.0)
        return weight

    # ----------------------------------------------------------
    # forward
    # ----------------------------------------------------------
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Batch dataclass.

        Returns:
            dict with keys:
                'res_logits'   : (B, L)     residue-level logits
                'prot_logits'  : (B,)       protein-level logits
                'alpha'        : (B, L, K)  routing weights
            During training, additionally:
                'loss_distill_rep'   : scalar
                'loss_distill_logit' : scalar
                'loss_route'         : scalar
                'loss_hier'          : scalar
                'loss_res'           : scalar
                'loss_prot'          : scalar
                'loss'               : scalar  (total)
        """
        mask = batch.mask                            # (B, L)

        # ====== 1. View projection ======
        h_esm  = self.proj_esm(batch.esm2)          # (B, L, d)
        h_pssm = self.proj_pssm(batch.pssm)         # (B, L, d)
        h_phys = self.proj_phys(batch.phys)          # (B, L, d)

        # ====== 2. Multi-view fusion → H ======
        H = self.fusion(h_esm, h_pssm, h_phys)      # (B, L, d)

        # ====== 3. MoE routing & mixing → Z ======
        Z, alpha = self.moe(H)                       # Z: (B,L,d), alpha: (B,L,K)
        # ====== 3.5 Local context + Transformer ======
        Z_local = self.local_encoder(Z, mask)
        Z = Z + Z_local
        Z_ctx = self.context_encoder(Z, mask)
        Z_ctx = Z + Z_ctx

        # ====== 4. Prediction heads ======
        res_logits = self.res_head(Z_ctx).squeeze(-1)    # (B, L)
        pooled = self.pool(Z_ctx, mask)                   # (B, d)
        prot_logits = self.prot_head(pooled).squeeze(-1)  # (B,)

        outputs: Dict[str, torch.Tensor] = {
            "res_logits": res_logits,
            "prot_logits": prot_logits,
            "alpha": alpha,
        }
        losses: Dict[str, torch.Tensor] = {
            "loss": torch.tensor(0.0, device=mask.device),
            "loss_res": torch.tensor(0.0, device=mask.device),
            "loss_prot": torch.tensor(0.0, device=mask.device),
            "loss_distill_rep": torch.tensor(0.0, device=mask.device),
            "loss_distill_logit": torch.tensor(0.0, device=mask.device),
            "loss_route": torch.tensor(0.0, device=mask.device),
            "loss_hier": torch.tensor(0.0, device=mask.device),
        }

        # ====== 5. Training losses ======
        # if self.training:
        # --- 5a. Residue-level loss (focal-style BCE) ---
        # Apply mask
        # Residue loss: Asymmetric Focal + Dice + OHEM
        loss_focal = self.asym_focal(res_logits, batch.y_res, mask)
        loss_dice = self.dice_loss(res_logits, batch.y_res, mask)
        loss_ohem = ohem_loss(res_logits, batch.y_res, mask, top_k_ratio=0.3)
        loss_res = loss_focal + 0.5 * loss_dice + 0.3 * loss_ohem

        # Protein loss
        loss_prot = F.binary_cross_entropy_with_logits(
            prot_logits, batch.y_prot.float()
        )

        # Hierarchical consistency
        res_probs = torch.sigmoid(res_logits)
        max_res_prob, _ = (res_probs * mask.float()).max(dim=1)
        prot_prob = torch.sigmoid(prot_logits)
        loss_hier = (prot_prob - max_res_prob).abs().mean()


        if self.training:
            # --- 5d. Privileged distillation (if annotations available) ---
            # Privileged distillation (per-sample masked)
            loss_distill_rep = torch.tensor(0.0, device=mask.device)
            loss_distill_logit = torch.tensor(0.0, device=mask.device)
            loss_contrast = torch.tensor(0.0, device=mask.device)
            n_priv = 0

            # Structure branch
            if batch.ss is not None and batch.sasa is not None and batch.subset is not None:
                struct_feats = torch.cat([batch.ss, batch.sasa], dim=-1)
                # H_struct, struct_logits = self.struct_branch(struct_feats)

                # 只对 subset=="structure" 的样本计算
                struct_mask = torch.tensor(
                    [s == "structure" for s in batch.subset], device=mask.device
                ).unsqueeze(1) & mask  # (B, L)

                if struct_mask.any():
                    H_struct, struct_logits = self.struct_branch(struct_feats)
                    loss_distill_rep += self._masked_mse(H, H_struct, struct_mask)
                    loss_distill_logit += self._kl_distill(
                        res_logits.unsqueeze(-1), struct_logits, struct_mask
                    )
                    loss_contrast += self.contrastive_distill(
                        H, H_struct, batch.y_res, struct_mask
                    )
                    n_priv += 1

            # Disorder branch
            if batch.disorder is not None and batch.subset is not None:
                H_dis, dis_logits = self.dis_branch(batch.disorder)

                dis_mask = torch.tensor(
                    [s == "disorder" for s in batch.subset], device=mask.device
                ).unsqueeze(1) & mask  # (B, L)

                if dis_mask.any():
                    # H_dis, dis_logits = self.dis_branch(batch.disorder, H, mask)
                    H_dis, dis_logits = self.dis_branch(batch.disorder)
                    loss_distill_rep += self._masked_mse(H, H_dis, dis_mask)
                    loss_distill_logit += self._kl_distill(
                        res_logits.unsqueeze(-1), dis_logits, dis_mask
                    )
                    loss_contrast += self.contrastive_distill(
                        H, H_dis, batch.y_res, dis_mask
                    )

                    n_priv += 1

            if n_priv > 0:
                loss_distill_rep /= n_priv
                loss_distill_logit /= n_priv
                loss_contrast /= n_priv

            loss_distill = loss_distill_rep + loss_distill_logit + 0.1 * loss_contrast

        # --- 5e. Expert-agnostic routing regularization ---
        loss_route = self._routing_regularization(alpha, mask)

        # --- Total loss ---
        if self.training:
            # loss = loss_res + loss_prot + loss_distill + loss_route + loss_hier
            # Total (with tunable weights)
            loss = (1.0 * loss_res +
                    0.5 * loss_prot +
                    0.3 * loss_distill +
                    self.route_loss_weight * loss_route +
                    0.2 * loss_hier)
            losses.update({
                "loss": loss,
                "loss_res": loss_res,
                "loss_prot": loss_prot,
                "loss_distill_rep": loss_distill_rep,
                "loss_distill_logit": loss_distill_logit,
                "loss_route": loss_route,
                "loss_hier": loss_hier,
            })
        else:
            # loss = loss_res + loss_prot + loss_route + loss_hier
            loss = loss_res + loss_prot + self.route_loss_weight * loss_route + loss_hier
            losses.update({
                "loss": loss,
                "loss_res": loss_res,
                "loss_prot": loss_prot,
                "loss_route": loss_route,
                "loss_hier": loss_hier,
            })

        return outputs, losses