# Integration smoke test for the OCNWM train step (A0 / A1 / A2) on synthetic latents.
#
# Runnable without pytest / GPU / dataset:  python test_train_integration.py
#
# It exercises the SAME code path train.py uses: build the CDiT (use_slots per mode) + the
# Stage-1 slot_model, call compute_slot_conditioning() to get (slots, g_s, L_slot), inject
# into diffusion.training_losses(), and form L_total = L_diff + lambda * L_slot. This checks
# the integration math (shapes, mode switching, gradient flow, overfit) minus the DDP/VAE/
# dataloader plumbing, which needs a real cluster run (Stage D).

import torch

import models
from diffusion import create_diffusion
from slot_transition import (
    SlotTransitionPredictor, SlotGate, compute_slot_conditioning,
)

SEED = 0
CFG = dict(input_size=8, context_size=2, patch_size=2, in_channels=4,
           hidden_size=48, depth=2, num_heads=4)
SLOT_DIM, K, MODES = 16, 5, 4
BF = 4                      # B * goals (already flattened)
LAMBDA = 1.0


def _build(slot_mode):
    torch.manual_seed(SEED)
    use_slots = slot_mode != "off"
    model = models.CDiT(slot_dim=SLOT_DIM, use_slots=use_slots, **CFG)
    if slot_mode == "full":
        sm = SlotTransitionPredictor(slot_dim=SLOT_DIM, num_slots=K, action_dim=3,
                                     hidden_dim=32, cond_dim=48, num_modes=MODES)
    elif slot_mode == "context":
        sm = SlotGate(slot_dim=SLOT_DIM, hidden_dim=32)
    else:
        sm = None
    return model, sm


def _batch():
    torch.manual_seed(SEED + 5)
    hw = CFG["input_size"]
    x_start = torch.randn(BF, CFG["in_channels"], hw, hw)
    x_cond = torch.randn(BF, CFG["context_size"], CFG["in_channels"], hw, hw)
    y = torch.randn(BF, 3)
    rel_t = torch.rand(BF)
    past = torch.randn(BF, CFG["context_size"], K, SLOT_DIM)
    target = torch.randn(BF, K, SLOT_DIM)
    noise = torch.randn_like(x_start)
    t = torch.randint(0, 1000, (BF,))
    return x_start, x_cond, y, rel_t, past, target, noise, t


def _step(model, sm, slot_mode, diffusion, b):
    """Mirror of the train.py inner step. Returns (l_total, l_diff, l_slot, stats)."""
    x_start, x_cond, y, rel_t, past, target, noise, t = b
    model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
    l_slot = x_start.new_zeros(())
    stats = None
    if slot_mode != "off":
        cond = compute_slot_conditioning(slot_mode, sm, past, target, y, rel_t,
                                         eps_wta=0.0, regression="mse")
        model_kwargs["slots"] = cond["slots"]
        model_kwargs["g_s"] = cond["g_s"]
        l_slot = cond["l_slot"]
        stats = cond["stats"]
    ld = diffusion.training_losses(model, x_start, t, model_kwargs, noise=noise)
    l_diff = ld["loss"].mean()
    return l_diff + LAMBDA * l_slot, l_diff, l_slot, stats


# --------------------------------------------------------------------------------------
def test_all_modes_run_finite():
    diffusion = create_diffusion(timestep_respacing="")
    b = _batch()
    for mode in ("off", "context", "full"):
        model, sm = _build(mode)
        total, l_diff, l_slot, stats = _step(model, sm, mode, diffusion, b)
        assert torch.isfinite(total), f"{mode}: L_total not finite"
        assert torch.isfinite(l_diff), f"{mode}: L_diff not finite"
        if mode == "off":
            assert sm is None and not hasattr(model, "slot_proj"), "A0 must have no slot machinery"
            assert float(l_slot) == 0.0
        else:
            assert stats is not None and "g_s_mean" in stats
        if mode == "full":
            assert "k_eff" in stats and stats["k_eff"] >= 1.0 - 1e-6


def test_grad_flow_full():
    # Perturb mu_head (L_slot -> g_s path) and the CDiT exo/readout (L_diff -> g_s path) so
    # both gradient routes to the gate are live, then assert the Stage-1 gate/ego/proj learn.
    diffusion = create_diffusion(timestep_respacing="")
    model, sm = _build("full")
    torch.manual_seed(SEED + 2)
    with torch.no_grad():
        sm.mu_head.weight.normal_(0, 0.5); sm.f_ego.net[-1].weight.normal_(0, 0.5)
        model.final_layer.linear.weight.normal_(0, 0.3)
        for blk in model.blocks:
            blk.adaLN_exo[-1].weight.normal_(0, 0.3)
    total, _, l_slot, _ = _step(model, sm, "full", diffusion, _batch())
    model.zero_grad(); sm.zero_grad()
    total.backward()
    assert l_slot.item() > 0
    for name, p in [("gate", sm.gate_mlp[-1].weight), ("f_ego", sm.f_ego.net[-1].weight),
                    ("mu_head", sm.mu_head.weight), ("slot_proj", model.slot_proj.weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad reached {name}"


def test_grad_flow_context():
    # A1: SlotGate has no L_slot; it must still learn from L_diff via the exo (1-g_s) scaling.
    diffusion = create_diffusion(timestep_respacing="")
    model, sm = _build("context")
    torch.manual_seed(SEED + 2)
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(0, 0.3)
        for blk in model.blocks:
            blk.adaLN_exo[-1].weight.normal_(0, 0.3)
    total, _, l_slot, _ = _step(model, sm, "context", diffusion, _batch())
    model.zero_grad(); sm.zero_grad()
    total.backward()
    assert float(l_slot) == 0.0
    g = sm.gate_mlp[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "SlotGate got no gradient from L_diff"


def test_overfit_decreases_full():
    # Fixed batch/noise/t: L_total must fall substantially under joint optimization.
    diffusion = create_diffusion(timestep_respacing="")
    model, sm = _build("full")
    b = _batch()
    opt = torch.optim.AdamW(list(model.parameters()) + list(sm.parameters()), lr=2e-3)
    losses = []
    for _ in range(60):
        total, _, _, _ = _step(model, sm, "full", diffusion, b)
        opt.zero_grad(); total.backward(); opt.step()
        losses.append(total.item())
    assert all(l == l for l in losses), "NaN encountered during overfit"
    assert losses[-1] < 0.7 * losses[0], f"L_total did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback; traceback.print_exc()
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_main())
