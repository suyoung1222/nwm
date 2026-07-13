# OCNWM Stage 1: Slot Transition Predictor.
#
# Predicts next-frame object slots from a past slot sequence, conditioned on action and
# timeshift, and emits a per-slot ego-explainability gate g_s in [0,1]. Dynamic slots are
# modeled with a per-slot Gaussian mixture (GMM: pi, mu, diagonal Sigma) trained with a
# Winner-Takes-All (WTA / MTP-style) loss; static (background) slots collapse to a single
# effective mode (num_modes=1 => pure point prediction).
#
# Compositional transition (fixed for OCNWM v2):
#
#     z_ego^s          = f_ego(z^s_t, a)                          # shared, ALL slots, ungated
#     mu_k^s           = z_ego^s + (1 - g_s) * Delta_mu_k^s       # exo residual, gated per slot
#     z_hat^s_{t+1}    = sum_k pi_k^s * mu_k^s                    # mixture mean
#
#   * f_ego: an action-conditioned ego-transform SHARED across slots and applied
#     UNCONDITIONALLY to every slot (no gate). Residual form f_ego(z,a) = z + h_theta(z, psi_a),
#     with h_theta's last layer zero-init => f_ego == identity at start, ego opens gradually.
#   * Delta_mu_k: the exogenous residual on top of ego, per GMM mode. Gated per slot by
#     (1 - g_s); the same (1 - g_s) scales the Stage-2 exo cross-attention value.
#   * g_s = sigmoid(MLP(slot)) per slot (scalar) = EGO-EXPLAINABILITY (how well the slot is
#     explained by ego motion). Polarity: static/background -> 1 (exo suppressed),
#     independent/dynamic agent -> 0 (exo fully expressed). Action enters ONLY through f_ego
#     (role separation): the exo head sees only slot history + timeshift, never the action.
#   * Predicted slots flow into Stage-2 CDiT WITHOUT detach, so g_s receives gradient from
#     BOTH L_slot (here, via the (1-g_s) gate on Delta_mu) and L_diff (through the CDiT exo
#     value scaling). The L_slot *target* is always detached.
#   * v1 is per-slot independent. Optional slot<->slot self-attention lives ONLY in this
#     module (never in CDiT); it is OFF by default (use_slot_self_attn=False).
#
# Shapes:  past_slots (B, T_ctx, K, D)  action (B, A)  rel_t (B,)  ->
#   mu (B,K,M,D)  logvar (B,K,M,D)  pi_logits (B,K,M)  g_s (B,K)
#   z_ego (B,K,D)  expected_slot (B,K,D)

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TimestepEmbedder, ActionEmbedder  # reuse NWM's embedding style


def _mlp(sizes, act=nn.SiLU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class EgoTransform(nn.Module):
    """Shared, action-conditioned ego-transform applied UNCONDITIONALLY to every slot.

    f_ego(z, psi_a) = z + h_theta([z, psi_a]).  Weights are shared across slots (the slot
    axis is folded into the batch of the MLP). The last layer of h_theta is zero-init'd by
    the parent's _init_weights, so f_ego starts as the identity and the ego component opens
    up gradually during training.
    """

    def __init__(self, slot_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp([slot_dim + cond_dim, hidden_dim, slot_dim])

    def forward(self, z: torch.Tensor, psi_a: torch.Tensor) -> torch.Tensor:
        # z (B,K,D), psi_a (B,cond) -> (B,K,D)
        B, K, D = z.shape
        a = psi_a[:, None, :].expand(-1, K, -1)          # broadcast action over slots
        delta = self.net(torch.cat([z, a], dim=-1))      # (B,K,D)
        return z + delta


class SlotTransitionPredictor(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        num_slots: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        num_modes: int = 4,
        use_slot_self_attn: bool = False,
        n_self_attn_layers: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_modes = num_modes
        self.use_slot_self_attn = use_slot_self_attn

        # --- per-slot temporal encoder over the context frames (shared across slots) ---
        self.in_proj = nn.Linear(slot_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # --- optional slot<->slot self-attention (this module only; off by default) ---
        if use_slot_self_attn:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=4 * hidden_dim,
                batch_first=True, activation="gelu",
            )
            self.slot_self_attn = nn.TransformerEncoder(enc_layer, num_layers=n_self_attn_layers)
        else:
            self.slot_self_attn = None

        # --- per-slot ego-explainability gate g_s = sigmoid(MLP(slot)) ---
        self.gate_mlp = _mlp([slot_dim, hidden_dim, 1])

        # --- conditioning embedders (action + timeshift), NWM-style ---
        self.action_embed = ActionEmbedder(cond_dim)     # psi_a  (feeds f_ego ONLY)
        self.time_embed = TimestepEmbedder(cond_dim)      # psi_k  (timeshift, feeds exo head)

        # --- shared ego-transform f_ego (unconditional, all slots) ---
        self.f_ego = EgoTransform(slot_dim, cond_dim, hidden_dim)

        # --- exo GMM head: conditioned on slot history + timeshift ONLY (no action) ---
        head_in = hidden_dim + cond_dim
        self.head = _mlp([head_in, hidden_dim, hidden_dim], last_act=True)
        self.mu_head = nn.Linear(hidden_dim, num_modes * slot_dim)     # exo residual deltas
        self.logvar_head = nn.Linear(hidden_dim, num_modes * slot_dim)
        self.pi_head = nn.Linear(hidden_dim, num_modes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Start near identity: ego == identity (h_theta last layer zero), exo residual ~ 0,
        # mixture near-uniform, unit variance. At init z_hat == last_slot for every slot.
        nn.init.zeros_(self.f_ego.net[-1].weight); nn.init.zeros_(self.f_ego.net[-1].bias)
        nn.init.zeros_(self.mu_head.weight); nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.pi_head.weight); nn.init.zeros_(self.pi_head.bias)
        nn.init.zeros_(self.logvar_head.weight); nn.init.zeros_(self.logvar_head.bias)

    def forward(self, past_slots: torch.Tensor, action: torch.Tensor,
                rel_t: torch.Tensor,
                g_s_override: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, K, D = past_slots.shape
        assert K == self.num_slots and D == self.slot_dim

        # temporal encoding per slot: (B,T,K,D) -> (B*K, T, D) -> GRU -> (B,K,H)
        seq = past_slots.permute(0, 2, 1, 3).reshape(B * K, T, D)
        seq = self.in_proj(seq)
        _, h_last = self.temporal(seq)            # (1, B*K, H)
        h = h_last[0].reshape(B, K, -1)           # (B, K, H)

        if self.slot_self_attn is not None:
            h = self.slot_self_attn(h)            # (B, K, H)

        last_slot = past_slots[:, -1]             # (B, K, D)

        # per-slot ego-explainability gate from the (last) slot state
        g_s = torch.sigmoid(self.gate_mlp(last_slot)).squeeze(-1)  # (B, K)
        if g_s_override is not None:
            g_s = g_s_override                     # clamp/ablation hook (used by tests, A1)

        # --- ego branch: shared, unconditional, applied to ALL slots ---
        psi_a = self.action_embed(action)         # (B, cond)
        z_ego = self.f_ego(last_slot, psi_a)      # (B, K, D)

        # --- exo branch: GMM residual deltas, conditioned on history + timeshift only ---
        psi_k = self.time_embed(rel_t[..., None]) # (B, cond)
        cond_k = psi_k[:, None, :].expand(-1, K, -1)          # (B, K, cond)
        feat = self.head(torch.cat([h, cond_k], dim=-1))      # (B, K, H)
        delta = self.mu_head(feat).reshape(B, K, self.num_modes, D)   # exo residual deltas
        logvar = self.logvar_head(feat).reshape(B, K, self.num_modes, D)
        pi_logits = self.pi_head(feat)                        # (B, K, M)

        # --- compose: mu_k = z_ego + (1 - g_s) * Delta_mu_k ---
        gate = (1.0 - g_s)[:, :, None, None]                  # (B, K, 1, 1)
        mu = z_ego[:, :, None, :] + gate * delta              # (B, K, M, D)

        pi = torch.softmax(pi_logits, dim=-1)
        expected_slot = (pi[..., None] * mu).sum(dim=2)       # (B, K, D)

        return {
            "mu": mu, "logvar": logvar, "pi_logits": pi_logits,
            "g_s": g_s, "z_ego": z_ego, "expected_slot": expected_slot,
        }


def expected_slot(pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Mixture-mean predicted slot (B,K,D) for feeding Stage-2 CDiT (no detach)."""
    return pred["expected_slot"]


class SlotGate(nn.Module):
    """Standalone per-slot ego-explainability gate for the A1 (slots-as-context) ablation,
    where there is NO Stage-1 predictor and NO L_slot. Same form as the predictor's gate
    (g_s = sigmoid(MLP(slot))) but its own module; it is trained by L_diff only, through the
    Stage-2 exo value scaling (1 - g_s)."""

    def __init__(self, slot_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gate_mlp = _mlp([slot_dim, hidden_dim, 1])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_mlp(slots)).squeeze(-1)   # (B,K,D) -> (B,K)


def _slot_stats(g_s: torch.Tensor, ent: torch.Tensor,
                reg=None, ce=None) -> Dict[str, float]:
    """Scalar diagnostics for logging (no grad). ent is per-(B,K) pi entropy in nats."""
    with torch.no_grad():
        ent_mean = ent.mean().item()
        stats = {
            "g_s_mean": g_s.mean().item(),
            "g_s_per_slot": [v.item() for v in g_s.mean(dim=0)],   # (K,) averaged over batch
            "pi_entropy": ent_mean,
            "k_eff": math.exp(ent_mean),                            # effective #modes
        }
    if reg is not None:
        stats["reg"] = float(reg)
    if ce is not None:
        stats["ce"] = float(ce)
    return stats


def compute_slot_conditioning(
    slot_mode: str,
    slot_model: nn.Module,
    past_slots: torch.Tensor,      # (Bf, ctx, K, D)  flattened over goals, frozen (no grad)
    target_slots: torch.Tensor,    # (Bf, K, D)       next-frame targets, frozen (no grad)
    action: torch.Tensor,          # (Bf, A)
    rel_t: torch.Tensor,           # (Bf,)
    *,
    eps_wta: float = 0.0,
    ce_weight: float = 1.0,
    regression: str = "mse",
) -> Dict[str, object]:
    """Single source of truth for the CDiT exo conditioning, shared by train.py and tests.

    Returns dict with:
      slots  : (Bf,K,D) exo key/value slots -- for 'full' the predicted mixture mean WITHOUT
               detach (so L_diff reaches Stage-1); for 'context' the frozen current slots.
      g_s    : (Bf,K) per-slot ego-explainability gate (grad-carrying).
      l_slot : scalar L_slot tensor (0 for 'context').
      stats  : scalar diagnostics for logging.
    """
    if slot_mode == "full":
        pred = slot_model(past_slots, action, rel_t)
        ld = slot_transition_loss(pred, target_slots, eps_wta=eps_wta,
                                  ce_weight=ce_weight, regression=regression)
        pi = torch.softmax(pred["pi_logits"], dim=-1)              # (Bf,K,M)
        ent = -(pi * (pi + 1e-9).log()).sum(dim=-1)                # (Bf,K) nats
        return {
            "slots": pred["expected_slot"],                        # NO detach
            "g_s": pred["g_s"],
            "l_slot": ld["loss"],
            "stats": _slot_stats(pred["g_s"], ent, reg=ld["reg"].detach(), ce=ld["ce"].detach()),
        }
    elif slot_mode == "context":
        slots = past_slots[:, -1]                                  # frozen current-frame slots
        g_s = slot_model(slots)                                    # SlotGate -> (Bf,K)
        l_slot = action.new_zeros(())                             # exact 0, no grad path
        return {
            "slots": slots,
            "g_s": g_s,
            "l_slot": l_slot,
            "stats": _slot_stats(g_s, torch.zeros_like(g_s)),      # single effective mode
        }
    else:
        raise ValueError(f"compute_slot_conditioning: unknown slot_mode={slot_mode!r}")


def slot_transition_loss(
    pred: Dict[str, torch.Tensor],
    target: torch.Tensor,        # (B, K, D) next-frame target slots
    eps_wta: float = 0.0,        # 0 = pure WTA; >0 spreads reg mass to avoid dead modes
    ce_weight: float = 1.0,
    regression: str = "mse",     # "mse" or "nll" (diagonal Gaussian, uses logvar)
) -> Dict[str, torch.Tensor]:
    """WTA regression on the target-closest mode + cross-entropy on pi. Target is detached.

    Both the WTA winner distance and the pi cross-entropy are computed in the *composed*
    space (pred["mu"] already equals z_ego + (1-g_s)*Delta_mu), so g_s receives gradient
    through the winning mode.
    """
    mu = pred["mu"]                          # (B,K,M,D) composed
    logvar = pred["logvar"]
    pi_logits = pred["pi_logits"]            # (B,K,M)
    B, K, M, D = mu.shape

    tgt = target.detach()[:, :, None, :]     # (B,K,1,D)
    sqerr = ((mu - tgt) ** 2).mean(dim=-1)   # (B,K,M)

    if regression == "mse":
        per_mode = sqerr
    elif regression == "nll":
        per_mode = 0.5 * (logvar + (mu - tgt) ** 2 / logvar.exp()).mean(dim=-1)  # (B,K,M)
    else:
        raise ValueError(f"unknown regression={regression!r}")

    m_star = per_mode.argmin(dim=-1)         # (B,K) closest mode selects the WTA winner
    if regression == "mse":
        m_star = sqerr.argmin(dim=-1)

    # eps-WTA weights: (1-eps) on winner, eps spread across all modes
    w = torch.full_like(per_mode, eps_wta / M)
    w.scatter_(-1, m_star[..., None], 1.0 - eps_wta + eps_wta / M)
    reg = (w * per_mode).sum(dim=-1).mean()

    ce = F.cross_entropy(pi_logits.reshape(B * K, M), m_star.reshape(B * K))
    loss = reg + ce_weight * ce
    return {"loss": loss, "reg": reg, "ce": ce, "m_star": m_star}
