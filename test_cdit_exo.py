# Unit tests for the OCNWM extended CDiT (exo cross-attention branch) in models.py.
#
# Runnable without pytest:  python test_cdit_exo.py
#
# The load-bearing guarantee: with slots=None the extended CDiT is BIT-IDENTICAL to the
# untouched baseline in nwm_original/nwm/models.py (A0). We verify this directly by loading
# the baseline's weights into the extended model (strict=False copies every shared param;
# the exo modules stay at their zero-init) and comparing forward outputs exactly.

import importlib.util
import os

import torch

import models as ext  # the extended CDiT (cwd = ocnwm/nwm)

_ORIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "nwm_original", "nwm", "models.py")
)


def _load_orig():
    spec = importlib.util.spec_from_file_location("orig_models", _ORIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# small config so the test runs fast on CPU
CFG = dict(input_size=8, context_size=2, patch_size=2, in_channels=4,
           hidden_size=48, depth=3, num_heads=4)
SLOT_DIM, K = 16, 5
N = 2
SEED = 0


def _inputs():
    torch.manual_seed(SEED + 7)
    p = CFG["patch_size"]; hw = CFG["input_size"]
    x = torch.randn(N, CFG["in_channels"], hw, hw)
    x_cond = torch.randn(N, CFG["context_size"], CFG["in_channels"], hw, hw)
    t = torch.randint(0, 1000, (N,))
    y = torch.randn(N, 3)
    rel_t = torch.rand(N)
    slots = torch.randn(N, K, SLOT_DIM)
    g_s = torch.rand(N, K)
    return x, t, y, x_cond, rel_t, slots, g_s


def _build_pair():
    orig_mod = _load_orig()
    torch.manual_seed(SEED)
    orig = orig_mod.CDiT(**CFG)
    torch.manual_seed(SEED)
    model = ext.CDiT(slot_dim=SLOT_DIM, **CFG)
    # Copy every shared parameter from baseline; exo modules keep their (zero) init.
    missing, unexpected = model.load_state_dict(orig.state_dict(), strict=False)
    assert unexpected == [], f"baseline had keys absent from extended model: {unexpected}"
    # every 'missing' key must belong to the new exo/slot modules only
    for k in missing:
        assert ("exo" in k) or k.startswith("slot_proj"), f"unexpected missing shared key: {k}"
    orig.eval(); model.eval()
    return orig, model


# --------------------------------------------------------------------------------------
def test_bit_identical_slots_none():
    orig, model = _build_pair()
    x, t, y, x_cond, rel_t, _, _ = _inputs()
    with torch.no_grad():
        o = orig(x, t, y, x_cond, rel_t)
        e = model(x, t, y, x_cond, rel_t, slots=None)
    assert o.shape == e.shape
    assert torch.equal(o, e), f"slots=None not bit-identical: max|Δ|={(o-e).abs().max().item():.3e}"


def test_use_slots_false_is_structurally_baseline():
    # A0: use_slots=False must build NO exo params -> state_dict keys identical to baseline,
    # loadable with strict=True, and forward bit-identical. (DDP sees no unused parameters.)
    orig_mod = _load_orig()
    torch.manual_seed(SEED)
    orig = orig_mod.CDiT(**CFG)
    torch.manual_seed(SEED)
    a0 = ext.CDiT(slot_dim=SLOT_DIM, use_slots=False, **CFG)
    assert set(a0.state_dict().keys()) == set(orig.state_dict().keys()), "A0 state_dict differs from baseline"
    a0.load_state_dict(orig.state_dict(), strict=True)  # must load with NO missing/unexpected
    orig.eval(); a0.eval()
    x, t, y, x_cond, rel_t, _, _ = _inputs()
    with torch.no_grad():
        o = orig(x, t, y, x_cond, rel_t)
        e = a0(x, t, y, x_cond, rel_t)
    assert torch.equal(o, e), "use_slots=False forward not bit-identical to baseline"


def test_zero_init_slots_have_no_effect():
    # Fresh extended model (adaLN_exo zero-init): supplying slots must not change the output.
    _, model = _build_pair()
    x, t, y, x_cond, rel_t, slots, g_s = _inputs()
    with torch.no_grad():
        none = model(x, t, y, x_cond, rel_t, slots=None)
        withs = model(x, t, y, x_cond, rel_t, slots=slots, g_s=g_s)
    assert torch.allclose(none, withs, atol=1e-6), \
        f"zero-init exo changed output: max|Δ|={(none-withs).abs().max().item():.3e}"


def _activate_exo_(model, std=0.3):
    # A fresh DiT is adaLN-zero: block gates AND final_layer.linear are zero, so the output
    # is identically 0 and no internal change is observable. Push adaLN_exo off zero (so the
    # exo branch is live) AND the readout off zero (so internal changes reach the output).
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for b in model.blocks:
            b.adaLN_exo[-1].weight.normal_(0.0, std)
            b.adaLN_exo[-1].bias.normal_(0.0, std)
        model.final_layer.linear.weight.normal_(0.0, std)
        model.final_layer.linear.bias.normal_(0.0, std)


def test_exo_active_after_perturb():
    # Once adaLN_exo is off zero, slots must actually influence the output.
    _, model = _build_pair()
    _activate_exo_(model)
    x, t, y, x_cond, rel_t, slots, g_s = _inputs()
    with torch.no_grad():
        none = model(x, t, y, x_cond, rel_t, slots=None)
        withs = model(x, t, y, x_cond, rel_t, slots=slots, g_s=g_s)
    assert not torch.allclose(none, withs, atol=1e-5), "exo branch had no effect after activation"


def test_gs_gating_polarity():
    # g_s=1 => value scaled by (1-g_s)=0 and add_bias_kv=False => exo fully suppressed
    # (output == slots-off). g_s=0 => exo fully expressed (output != slots-off).
    _, model = _build_pair()
    _activate_exo_(model)
    x, t, y, x_cond, rel_t, slots, _ = _inputs()
    ones = torch.ones(N, K)
    zeros = torch.zeros(N, K)
    with torch.no_grad():
        none = model(x, t, y, x_cond, rel_t, slots=None)
        g1 = model(x, t, y, x_cond, rel_t, slots=slots, g_s=ones)
        g0 = model(x, t, y, x_cond, rel_t, slots=slots, g_s=zeros)
    assert torch.allclose(g1, none, atol=1e-6), \
        f"g_s=1 did not suppress exo: max|Δ|={(g1-none).abs().max().item():.3e}"
    assert not torch.allclose(g0, none, atol=1e-5), "g_s=0 should express exo but output unchanged"


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
