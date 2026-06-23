"""
Additive wandb + tensorboard logging for NWM fine-tuning.

This module is SELF-CONTAINED and never affects training math. Only rank 0
should create a real TrainLogger; pass enabled=False on other ranks for a no-op.

Graceful fallback (nothing here ever raises into the training loop):
  - tensorboard missing            -> wandb only
  - wandb missing / not logged in  -> tensorboard only
  - neither available              -> silent no-op
  - offline cluster               -> set WANDB_MODE=offline (wandb honors it);
                                      sync later with `wandb sync <dir>`.

Configuration via env vars (no secrets in code):
  WANDB_PROJECT   (default "nwm-finetune")
  WANDB_ENTITY    (optional team/user; default = your default entity)
  WANDB_RUN_ID    (optional; set to resume the same wandb run after a requeue)
  WANDB_MODE      (online|offline|disabled — handled natively by wandb)
  WANDB_DISABLED  (1/true -> skip wandb entirely, tensorboard only)
  WANDB_API_KEY   (auth; keep it in your shell env, never in the repo)
"""

import os
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def gpu_utilization():
    """Best-effort GPU utilization %. Needs pynvml; returns None if unavailable."""
    try:
        return float(torch.cuda.utilization())
    except Exception:
        return None


def gpu_mem_gb():
    try:
        return torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    except Exception:
        return None


class TrainLogger:
    """Thin wrapper that fans every log call out to tensorboard and/or wandb."""

    def __init__(self, run_name, log_dir, config=None, enabled=True):
        self.enabled = enabled
        self.tb = None
        self.wandb = None
        if not enabled:
            return

        # ---- tensorboard (per-run log_dir) ----
        if _HAS_TB:
            try:
                os.makedirs(log_dir, exist_ok=True)
                self.tb = SummaryWriter(log_dir=log_dir)
                print(f"[logging] tensorboard -> {log_dir}")
            except Exception as e:
                print(f"[logging] tensorboard init failed: {e}")
        else:
            print("[logging] tensorboard not installed; `pip install tensorboard` to enable it")

        # ---- wandb (offline-friendly, graceful) ----
        disabled = os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
        if _HAS_WANDB and not disabled:
            try:
                self.wandb = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "nwm-finetune"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    name=run_name,
                    dir=log_dir,
                    config=config or {},
                    resume="allow",
                    id=os.environ.get("WANDB_RUN_ID") or None,
                )
                mode = os.environ.get("WANDB_MODE", "online")
                print(f"[logging] wandb initialized (mode={mode})")
            except Exception as e:
                print(f"[logging] wandb init failed ({e}); continuing without wandb")
                self.wandb = None
        elif not _HAS_WANDB:
            print("[logging] wandb not installed; tensorboard only")
        elif disabled:
            print("[logging] WANDB_DISABLED set; tensorboard only")

    def log_scalars(self, metrics, step):
        if not self.enabled:
            return
        metrics = {k: v for k, v in metrics.items() if v is not None}
        if self.tb:
            for k, v in metrics.items():
                try:
                    self.tb.add_scalar(k, v, step)
                except Exception:
                    pass
        if self.wandb:
            try:
                self.wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_images(self, tag, images, step, captions=None):
        """images: list of HxWx3 float arrays in [0,1]."""
        if not self.enabled or not images:
            return
        if self.tb:
            for i, im in enumerate(images):
                try:
                    self.tb.add_image(f"{tag}/{i}", np.transpose(im, (2, 0, 1)), step)
                except Exception:
                    pass
        if self.wandb:
            try:
                imgs = [wandb.Image(im, caption=(captions[i] if captions else None))
                        for i, im in enumerate(images)]
                self.wandb.log({tag: imgs}, step=step)
            except Exception:
                pass

    def close(self):
        if self.tb:
            try:
                self.tb.flush(); self.tb.close()
            except Exception:
                pass
        if self.wandb:
            try:
                self.wandb.finish()
            except Exception:
                pass


@torch.no_grad()
def make_rollout_images(ema_model, vae, sample_diffusion, batch, latent_size,
                        device, num_cond, max_items=4):
    """Sample the first goal frame for a small val batch and return
    [context | prediction | ground-truth] strips (list of HxWx3 float arrays).

    Reuses the EXACT inference path used by train.evaluate (model_forward_wrapper)
    so the visualization matches real rollouts. EMA model is restored to its
    prior train/eval state afterwards. Imports are local to avoid import cycles.
    """
    from isolated_nwm_infer import model_forward_wrapper
    from misc import unnormalize

    x_img, y_act, rel_t = batch
    x_img = x_img.to(device)
    y_act = y_act.to(device)
    rel_t = rel_t.to(device)

    obs = x_img[:, :num_cond]            # (B, num_cond, 3, H, W)
    y0 = y_act[:, 0:1]                    # first goal action  (B,1,3)
    rel0 = rel_t[:, 0:1].flatten(0, 1)    # matching rel_t      (B,)

    was_training = ema_model.training
    ema_model.eval()
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        pred = model_forward_wrapper(
            (ema_model, sample_diffusion, vae), obs, y0,
            num_timesteps=None, latent_size=latent_size, device=device,
            num_cond=num_cond, num_goals=1, rel_t=rel0,
        )  # (B,3,H,W) in [-1,1]
    if was_training:
        ema_model.train()

    gt = x_img[:, num_cond]               # first goal frame GT
    images, captions = [], []
    n = min(x_img.shape[0], max_items)
    for i in range(n):
        ctx = unnormalize(obs[i, -1].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        pr = unnormalize(pred[i].float().cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        g = unnormalize(gt[i].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        images.append(np.concatenate([ctx, pr, g], axis=1))  # side-by-side strip
        captions.append("context | prediction | ground-truth")
    return images, captions
