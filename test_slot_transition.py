# Unit tests for the OCNWM Stage-1 SlotTransitionPredictor (compositional f_ego + exo).
#
# Runnable without pytest:  python test_slot_transition.py
# (Also importable by pytest: every check is a `test_*` function with asserts.)
#
# The predictor is init'd near-identity (f_ego == identity via zero-init h_theta, exo
# residual Delta == 0 via zero-init mu_head). Several role/gradient properties are therefore
# only observable once the relevant weights are OFF zero, so those tests deliberately perturb
# f_ego / mu_head to random values first -- they verify the WIRING, not the initial values.

import torch

from slot_transition import SlotTransitionPredictor, slot_transition_loss

SEED = 0
B, T, K, D, A = 2, 4, 7, 64, 3


def _model(num_modes=4, **kw):
    torch.manual_seed(SEED)
    return SlotTransitionPredictor(slot_dim=D, num_slots=K, action_dim=A,
                                   hidden_dim=32, cond_dim=48, num_modes=num_modes, **kw)


def _inputs():
    torch.manual_seed(SEED + 1)
    past = torch.randn(B, T, K, D)
    action = torch.randn(B, A)
    rel_t = torch.rand(B)
    return past, action, rel_t


def _perturb_(layer, std=0.5):
    """Push a Linear (or Sequential's last Linear) off zero-init so wiring becomes observable."""
    lin = layer[-1] if hasattr(layer, "__getitem__") else layer
    with torch.no_grad():
        lin.weight.normal_(0.0, std)
        if lin.bias is not None:
            lin.bias.normal_(0.0, std)


# --------------------------------------------------------------------------------------
def test_shapes():
    m = _model(num_modes=4)
    past, action, rel_t = _inputs()
    out = m(past, action, rel_t)
    assert out["mu"].shape == (B, K, 4, D)
    assert out["logvar"].shape == (B, K, 4, D)
    assert out["pi_logits"].shape == (B, K, 4)
    assert out["g_s"].shape == (B, K)
    assert out["z_ego"].shape == (B, K, D)
    assert out["expected_slot"].shape == (B, K, D)
    # gate is a valid probability
    assert torch.all((out["g_s"] >= 0) & (out["g_s"] <= 1))


def test_identity_at_init():
    # zero-init => ego == identity, Delta == 0  => predict "no change" (z_hat == last_slot).
    m = _model()
    past, action, rel_t = _inputs()
    out = m(past, action, rel_t)
    last = past[:, -1]
    assert torch.allclose(out["z_ego"], last, atol=1e-6), "f_ego not identity at init"
    assert torch.allclose(out["expected_slot"], last, atol=1e-6), "prediction != last_slot at init"


def test_grad_reaches_gs():
    # With Delta != 0, dL/dg_s = dL/dmu * (-Delta) != 0, so gate_mlp must receive gradient.
    m = _model()
    _perturb_(m.mu_head)                       # Delta != 0
    past, action, rel_t = _inputs()
    out = m(past, action, rel_t)               # note: no override -> gate_mlp is in the graph
    target = torch.randn(B, K, D) * 3.0        # far target -> clear gradient
    loss = slot_transition_loss(out, target)["loss"]
    m.zero_grad(); loss.backward()
    g = m.gate_mlp[-1].weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no gradient reached g_s gate"


def test_fego_all_slots_and_exo_gating():
    # ego is ungated (independent of g_s) and applied to every slot; exo scales by (1-g_s).
    m = _model()
    _perturb_(m.f_ego.net)                     # ego != identity
    _perturb_(m.mu_head)                       # Delta != 0
    past, action, rel_t = _inputs()

    g0 = torch.zeros(B, K)
    g1 = torch.full((B, K), 0.25)
    o0 = m(past, action, rel_t, g_s_override=g0)
    o1 = m(past, action, rel_t, g_s_override=g1)

    # ego branch does not depend on the gate
    assert torch.allclose(o0["z_ego"], o1["z_ego"], atol=1e-6), "ego depends on g_s (should be ungated)"
    # ego actually transformed the slot (not a no-op) for EVERY slot
    last = past[:, -1]
    per_slot = (o0["z_ego"] - last).norm(dim=-1)          # (B,K)
    assert torch.all(per_slot > 1e-4), "f_ego not applied to all slots"
    # exo contribution = mu - ego ; must scale exactly by (1-g_s): 1.0 vs 0.75
    exo0 = o0["mu"] - o0["z_ego"][:, :, None, :]
    exo1 = o1["mu"] - o1["z_ego"][:, :, None, :]
    assert torch.allclose(exo1, 0.75 * exo0, atol=1e-6), "exo residual not gated by (1-g_s)"


def test_M1_point_prediction():
    # num_modes=1 must reduce to an exact point prediction: expected == the single mode,
    # and the pi cross-entropy is exactly 0 (softmax over one logit).
    m = _model(num_modes=1)
    _perturb_(m.mu_head); _perturb_(m.f_ego.net)
    past, action, rel_t = _inputs()
    out = m(past, action, rel_t)
    assert out["mu"].shape == (B, K, 1, D)
    assert torch.allclose(out["expected_slot"], out["mu"][:, :, 0, :], atol=1e-6)
    target = torch.randn(B, K, D)
    ld = slot_transition_loss(out, target)
    assert abs(ld["ce"].item()) < 1e-6, f"CE should be 0 for M=1, got {ld['ce'].item()}"


def test_role_action_changes_ego():
    # Action enters only via f_ego; changing it must move z_ego for EVERY slot.
    m = _model()
    _perturb_(m.f_ego.net)                     # wire action -> ego
    past, action, rel_t = _inputs()
    a2 = action + 1.0
    z1 = m(past, action, rel_t)["z_ego"]
    z2 = m(past, a2, rel_t)["z_ego"]
    per_slot = (z1 - z2).norm(dim=-1)          # (B,K)
    assert torch.all(per_slot > 1e-4), "changing action did not change z_ego for all slots"


def test_role_gs1_equals_ego():
    # Clamping g_s=1 suppresses exo entirely => prediction collapses to z_ego.
    m = _model()
    _perturb_(m.mu_head); _perturb_(m.f_ego.net)   # Delta != 0 so this is a real test of the gate
    past, action, rel_t = _inputs()
    out = m(past, action, rel_t, g_s_override=torch.ones(B, K))
    assert torch.allclose(out["expected_slot"], out["z_ego"], atol=1e-6), "g_s=1 did not collapse to z_ego"
    # sanity: every mode equals ego under g_s=1
    assert torch.allclose(out["mu"], out["z_ego"][:, :, None, :].expand_as(out["mu"]), atol=1e-6)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_main())
