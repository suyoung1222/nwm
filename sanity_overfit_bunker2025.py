"""
Single-GPU sanity / overfit check for fine-tuning NWM on bunker2025, BEFORE the
real fine-tuning run. It does NOT modify train.py — it reuses the same model,
diffusion, dataset and inference wrapper.

What it does:
  1. loads CDiT-XL from the existing 5-dataset checkpoint (resume),
  2. grabs ONE small bunker2025 batch and overfits it for a few hundred steps
     with the exact training objective (VAE-encode -> diffusion.training_losses),
  3. asserts loss goes down and stays finite (no NaN / shape errors),
  4. predicts the goal frame for that batch (diffusion sampling) and saves a
     context | prediction | ground-truth strip so you can eyeball plausibility.

VAE stays frozen (used under no_grad), matching train.py.

Usage (single GPU):
  python sanity_overfit_bunker2025.py --steps 200 --batch_size 2
"""

import argparse
import os

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL

from models import CDiT_models
from diffusion import create_diffusion
from datasets import TrainingDataset
from misc import transform, unnormalize
from isolated_nwm_infer import model_forward_wrapper

BUNKER_DATA = "/home/suyoung/mydata/NWM/preprocessed/datasets/bunker2025"
BUNKER_TRAIN_SPLIT = "/home/suyoung/mydata/NWM/preprocessed/data_splits/bunker2025/train"
BASE_CKPT = "logs/nwm_cdit_xl/checkpoints/cdit_xl_ego4d_200000.pth.tar"


def load_config():
    with open("config/eval_config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("config/nwm_cdit_xl.yaml") as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def strip(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


@torch.no_grad()
def eval_fixed_loss(model, diffusion, x_start, kwargs, device, n_t=24):
    """Deterministic loss over a fixed grid of timesteps + fixed noise.
    Removes the per-step random-t variance so the overfit signal is readable."""
    was_training = model.training
    model.eval()
    g = torch.Generator(device=device).manual_seed(1234)
    noise = torch.randn(x_start.shape, generator=g, device=device, dtype=x_start.dtype)
    ts = torch.linspace(0, diffusion.num_timesteps - 1, n_t).long().to(device)
    total = 0.0
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for tv in ts:
            t = tv.repeat(x_start.shape[0])
            terms = diffusion.training_losses(model, x_start, t, kwargs, noise=noise)
            # use the denoising MSE term; the learned-sigma vb term is noisy at high t
            key = "mse" if "mse" in terms else "loss"
            total += terms[key].mean().item()
    if was_training:
        model.train()
    return total / len(ts)


def main(args):
    assert torch.cuda.is_available()
    device = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    out_dir = "logs/sanity_bunker2025"
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config()
    image_size = cfg["image_size"]
    num_cond = cfg["context_size"]
    latent_size = image_size // 8

    # ---- models ---------------------------------------------------------------
    print("Loading VAE (frozen) ...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    print(f"Building {cfg['model']} and loading base checkpoint ...")
    model = CDiT_models[cfg["model"]](context_size=num_cond, input_size=latent_size,
                                      in_channels=4).to(device)
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    # this base checkpoint is ema-only (no "model"/"opt"); load from ema
    weights = ckpt["model"] if "model" in ckpt else ckpt["ema"]
    res = model.load_state_dict(strip(weights), strict=True)
    print(f"  load_state_dict (from {'model' if 'model' in ckpt else 'ema'}):", res)
    if "train_steps" in ckpt:
        print(f"  resumed from train_steps={ckpt['train_steps']}")

    diffusion = create_diffusion(timestep_respacing="")        # 1000 steps, for training loss
    sample_diffusion = create_diffusion(timestep_respacing="250")  # fewer steps for viz sampling

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # ---- one fixed bunker2025 batch ------------------------------------------
    ds = TrainingDataset(
        data_folder=BUNKER_DATA, data_split_folder=BUNKER_TRAIN_SPLIT,
        dataset_name="bunker2025", image_size=image_size,
        min_dist_cat=cfg["distance"]["min_dist_cat"], max_dist_cat=cfg["distance"]["max_dist_cat"],
        len_traj_pred=cfg["len_traj_pred"], context_size=num_cond, normalize=cfg["normalize"],
        goals_per_obs=args.goals, transform=transform, predefined_index=None, traj_stride=1,
    )
    print(f"bunker2025 train samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    x_img, y_act, rel_t = next(iter(loader))
    x_img, y_act, rel_t = x_img.to(device), y_act.to(device), rel_t.to(device)
    print(f"batch shapes: obs={tuple(x_img.shape)} act={tuple(y_act.shape)} rel_t={tuple(rel_t.shape)}")

    # pre-encode latents once (VAE frozen) — mirrors train.py
    with torch.no_grad():
        B, T = x_img.shape[:2]
        flat = x_img.flatten(0, 1)
        lat = vae.encode(flat).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
    num_goals = T - num_cond
    x_start = lat[:, num_cond:].flatten(0, 1)
    x_cond = lat[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, lat.shape[2],
                                                   lat.shape[3], lat.shape[4]).flatten(0, 1)
    y_flat = y_act.flatten(0, 1)
    rel_t_flat = rel_t.flatten(0, 1)
    print(f"x_start={tuple(x_start.shape)} x_cond={tuple(x_cond.shape)} (B*num_goals={B*num_goals})")

    eval_kwargs = dict(y=y_flat, x_cond=x_cond, rel_t=rel_t_flat)
    pre_loss = eval_fixed_loss(model, diffusion, x_start, eval_kwargs, device)
    print(f"\nfixed-grid loss BEFORE overfit: {pre_loss:.4f}")

    # ---- overfit loop ---------------------------------------------------------
    model.train()
    losses = []
    kw = dict(y=y_flat, x_cond=x_cond, rel_t=rel_t_flat)
    print(f"Overfitting one batch for {args.steps} steps "
          f"(lr={args.lr}, grad-accum over {args.accum} timesteps/step) ...")
    for step in range(args.steps):
        opt.zero_grad()
        step_loss = 0.0
        # average the gradient over several random timesteps to cut the t-variance
        for _ in range(args.accum):
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)
                loss = diffusion.training_losses(model, x_start, t, kw)["loss"].mean()
            (loss / args.accum).backward()
            step_loss += loss.item() / args.accum
        opt.step()
        losses.append(step_loss)
        if not np.isfinite(step_loss):
            raise RuntimeError(f"NaN/Inf loss at step {step}!")
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss(avg over {args.accum} t)={step_loss:.4f}")

    post_loss = eval_fixed_loss(model, diffusion, x_start, eval_kwargs, device)
    print(f"\nfixed-grid loss AFTER overfit:  {post_loss:.4f}  "
          f"(BEFORE {pre_loss:.4f}, drop {pre_loss - post_loss:+.4f} = "
          f"{100*(pre_loss-post_loss)/pre_loss:+.1f}%)")
    plt.figure(figsize=(6, 4))
    plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss"); plt.title("bunker2025 overfit")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=130)
    plt.close()

    # ---- prediction / rollout viz on the same batch ---------------------------
    print("\nSampling goal-frame predictions for visualization ...")
    model.eval()
    obs = x_img[:, :num_cond]                 # (B, num_cond, 3, H, W)
    y0 = y_act[:, 0:1]                         # first goal action  (B,1,3)
    rel0 = rel_t[:, 0:1].flatten(0, 1)         # matching rel_t      (B,)
    with torch.no_grad():
        pred = model_forward_wrapper(
            (model, sample_diffusion, vae), obs, y0,
            num_timesteps=None, latent_size=latent_size, device=device,
            num_cond=num_cond, num_goals=1, rel_t=rel0,
        )  # (B,3,H,W) in [-1,1]
    gt_goal = x_img[:, num_cond]               # first goal frame GT

    n = min(B, 4)
    fig, ax = plt.subplots(n, 3, figsize=(7, 2.4 * n))
    if n == 1:
        ax = ax[None, :]
    for i in range(n):
        ctx = unnormalize(obs[i, -1].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        pr = unnormalize(pred[i].float().cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        gt = unnormalize(gt_goal[i].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        ax[i, 0].imshow(ctx); ax[i, 0].set_title("context (last)", fontsize=9)
        ax[i, 1].imshow(pr);  ax[i, 1].set_title("prediction", fontsize=9)
        ax[i, 2].imshow(gt);  ax[i, 2].set_title("GT goal", fontsize=9)
        for j in range(3):
            ax[i, j].axis("off")
    plt.tight_layout()
    viz = os.path.join(out_dir, "prediction.png")
    plt.savefig(viz, dpi=130)
    plt.close()
    print(f"saved -> {viz}")
    print(f"saved -> {os.path.join(out_dir, 'loss_curve.png')}")

    ok = post_loss < pre_loss
    print(f"\n[{'OK' if ok else 'WARN'}] fixed-grid loss {'decreased' if ok else 'did NOT decrease'}; "
          f"no NaN over {args.steps} steps.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--goals", type=int, default=1,
                   help="goals_per_obs; effective model batch = batch_size*goals (keep small on 24GB)")
    p.add_argument("--accum", type=int, default=8,
                   help="timesteps to average per optimizer step (smooths diffusion t-variance)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()
    main(args)
