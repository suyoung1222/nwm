"""
Core NWM online planner.

Maintains a rolling context buffer (4 frames) and runs CEM planning
given a goal image. Call push_frame() each time a new observation arrives,
then call plan() to get the best next action.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL
import lpips

from diffusion import create_diffusion
from models import CDiT_models
from misc import unnormalize_data, calculate_delta_yaw, transform


# From data_config.yaml — global normalization bounds for (dx, dy) in waypoint units
_ACTION_STATS = {
    'min': torch.tensor([-2.5, -4.0]),
    'max': torch.tensor([ 5.0,  4.0]),
}


class NWMPlanner:
    """
    Wraps the NWM model for online goal-conditioned navigation.

    Typical usage:
        planner = NWMPlanner(ckp_path="logs/nwm_cdit_xl/checkpoints/0100000.pth.tar")
        planner.set_goal(goal_image_np)
        # each timestep:
        planner.push_frame(current_image_np)
        step_dx, step_dy, total_dyaw, debug_img = planner.plan()
    """

    def __init__(
        self,
        ckp_path: str,
        model_name: str = 'CDiT-XL/2',
        context_size: int = 4,
        device: str = 'cuda',
        # CEM hyperparameters — tune for speed vs quality
        num_samples: int = 30,       # paper uses 120; reduce for speed
        topk: int = 5,
        opt_steps: int = 1,          # 1 iteration is enough for short horizon
        num_repeat_eval: int = 1,    # >1 reduces stochasticity at cost of time
        rollout_stride: int = 4,     # aggregate 4 steps → 1 NWM call; 1s per call at 4 FPS
        len_traj_pred: int = 8,      # plan 8 steps = 8s at rollout_stride=4, 4 FPS
        diffusion_steps: int = 50,   # 250 in paper; 50 is practical without distillation
        # CEM initial distribution (in normalized action space, dataset-specific)
        mu_init: list = [-0.1, 0.0, 0.0],       # (norm_dx, norm_dy, yaw_scale)
        sigma_init: list = [0.05, 0.1, 0.1],
        # Robot physical parameters
        metric_waypoint_spacing: float = 0.05,   # meters per waypoint unit (from data_config.yaml)
    ):
        self.device = torch.device(device)
        self.context_size = context_size
        self.num_samples = num_samples
        self.topk = topk
        self.opt_steps = opt_steps
        self.num_repeat_eval = num_repeat_eval
        self.rollout_stride = rollout_stride
        self.len_traj_pred = len_traj_pred
        self.latent_size = 224 // 8  # 28
        self.action_dim = 3          # (norm_dx, norm_dy, yaw_offset_scale)
        self.mu_init = torch.tensor(mu_init, dtype=torch.float32)
        self.sigma_init = torch.tensor(sigma_init, dtype=torch.float32)
        self.metric_waypoint_spacing = metric_waypoint_spacing

        self._context_buf: list[torch.Tensor] = []  # list of (1, C, H, W) tensors
        self._goal_tensor: torch.Tensor | None = None

        print(f"Loading NWM checkpoint: {ckp_path}")
        model = CDiT_models[model_name](
            context_size=context_size,
            input_size=self.latent_size,
            in_channels=4,
        )
        ckp = torch.load(ckp_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckp['ema'], strict=True)
        model.eval().to(self.device)
        self.model = torch.compile(model)

        self.diffusion = create_diffusion(str(diffusion_steps))
        self.vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').to(self.device)
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        print("NWM ready.")

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def set_goal(self, goal_img: np.ndarray):
        """Register the goal image. Must be called before plan()."""
        self._goal_tensor = self._preprocess(goal_img).to(self.device)

    def push_frame(self, img: np.ndarray):
        """Add the latest observation frame to the rolling context buffer."""
        t = self._preprocess(img)
        self._context_buf.append(t)
        if len(self._context_buf) > self.context_size:
            self._context_buf.pop(0)

    @torch.no_grad()
    def plan(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Run one CEM planning step.

        Returns
        -------
        step_actions_metric : np.ndarray, shape (len_traj_pred, 2)
            Per-step (dx, dy) displacement in **meters** in robot frame.
            The trajectory is straight-line, so all rows are identical.
        total_dyaw : float
            Total yaw change over the trajectory, in radians.
        debug_img : np.ndarray, shape (224, 224, 3), uint8
            NWM's prediction of the final frame — useful for debugging.
        """
        if self._goal_tensor is None:
            raise RuntimeError("Call set_goal() before plan().")
        obs = self._get_context()          # (1, ctx, C, H, W)
        goal = self._goal_tensor           # (1, C, H, W)
        mu    = self.mu_init.clone().to(self.device)
        sigma = self.sigma_init.clone().to(self.device)

        for _ in range(self.opt_steps):
            samples = torch.randn(self.num_samples, self.action_dim, device=self.device) * sigma + mu
            deltas_3d = self._samples_to_deltas(samples)                      # (N, T, 3)
            obs_exp  = obs.expand(self.num_samples, -1, -1, -1, -1)
            goal_exp = goal.expand(self.num_samples, -1, -1, -1)

            loss = self._score_candidates(obs_exp, deltas_3d, goal_exp)       # (N,)

            topk_idx = torch.argsort(loss)[:self.topk]
            mu    = samples[topk_idx].mean(0)
            sigma = samples[topk_idx].std(0).clamp(min=1e-4)

        # Final rollout with the best mean action
        best_deltas_3d = self._samples_to_deltas(mu.unsqueeze(0))             # (1, T, 3)
        preds = self._autoregressive_rollout(obs, best_deltas_3d)             # (1, T, C, H, W)
        debug_img = self._tensor_to_uint8(preds[0, -1])

        # Convert to metric
        norm_dxdy = mu[:2].unsqueeze(0).unsqueeze(0).repeat(1, self.len_traj_pred, 1)  # (1, T, 2)
        step_metric = (
            unnormalize_data(norm_dxdy, _ACTION_STATS)[0].cpu().numpy()
            * self.metric_waypoint_spacing
        )   # (T, 2) meters per step
        total_dyaw = float(best_deltas_3d[0, :, -1].sum().cpu())

        return step_metric, total_dyaw, debug_img

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 → (1, C, H, W) float32 in [-1, 1]."""
        return transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0)

    def _get_context(self) -> torch.Tensor:
        """Return (1, context_size, C, H, W), padding with first frame if needed."""
        buf = self._context_buf
        if not buf:
            raise RuntimeError("Context buffer is empty. Push at least one frame first.")
        while len(buf) < self.context_size:
            buf = [buf[0]] + buf
        return torch.cat(buf[-self.context_size:], dim=0).unsqueeze(0).to(self.device)

    def _samples_to_deltas(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Convert CEM samples (N, 3) → trajectory deltas (N, T, 3).
        The CEM parameterises a straight-line endpoint: all T steps share
        the same (dx_norm, dy_norm). A yaw offset is added at the final step.
        """
        N = samples.shape[0]
        norm_dxdy = samples[:, :2]                                         # (N, 2)
        deltas_xy = norm_dxdy.unsqueeze(1).repeat(1, self.len_traj_pred, 1)  # (N, T, 2)

        unnorm = unnormalize_data(deltas_xy, _ACTION_STATS)                  # waypoint units
        delta_yaw = calculate_delta_yaw(unnorm)                              # (N, T, 1)
        deltas_3d = torch.cat([deltas_xy, delta_yaw.to(deltas_xy.device)], dim=-1)  # (N, T, 3)
        deltas_3d[:, -1, -1] = deltas_3d[:, -1, -1] + samples[:, 2] * torch.pi

        return deltas_3d

    def _autoregressive_rollout(self, obs: torch.Tensor, deltas_3d: torch.Tensor) -> torch.Tensor:
        """
        obs      : (B, ctx, C, H, W)
        deltas_3d: (B, len_traj_pred, 3) normalised
        Returns  : (B, n_calls, C, H, W)
        """
        # Time-skip: sum rollout_stride adjacent steps → fewer NWM calls
        deltas_agg = deltas_3d.unflatten(1, (-1, self.rollout_stride)).sum(2)  # (B, n_calls, 3)
        curr_obs = obs.clone()
        preds = []
        for i in range(deltas_agg.shape[1]):
            curr_delta = deltas_agg[:, i:i+1]                # (B, 1, 3)
            pred = self._nwm_step(curr_obs, curr_delta)       # (B, C, H, W)
            preds.append(pred.unsqueeze(1))
            curr_obs = torch.cat([curr_obs, pred.unsqueeze(1)], dim=1)[:, 1:]
        return torch.cat(preds, dim=1)

    @torch.no_grad()
    def _nwm_step(self, obs: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        obs  : (B, ctx, C, H, W)
        delta: (B, 1, 3)
        → (B, C, H, W) predicted next frame in [-1, 1]
        """
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = obs.shape[:2]
            rel_t = (torch.ones(B, device=self.device) * (self.rollout_stride / 128.0))

            # VAE encode all context frames
            flat = obs.flatten(0, 1)
            latents = self.vae.encode(flat).latent_dist.sample().mul_(0.18215)
            latents = latents.unflatten(0, (B, T))              # (B, ctx, 4, h, w)

            x_cond = latents[:, :self.context_size]             # (B, ctx, 4, h, w)
            z = torch.randn(B, 4, self.latent_size, self.latent_size, device=self.device)
            y = delta.flatten(0, 1)                             # (B, 3)

            model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
            sample = self.diffusion.p_sample_loop(
                self.model.forward, z.shape, z,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=self.device,
            )
            decoded = self.vae.decode(sample / 0.18215).sample
            return torch.clip(decoded, -1.0, 1.0)

    def _score_candidates(
        self,
        obs_exp: torch.Tensor,
        deltas_3d: torch.Tensor,
        goal_exp: torch.Tensor,
        chunk_size: int = 5,
    ) -> torch.Tensor:
        """Score N candidate trajectories in chunks to save GPU memory. Returns (N,) LPIPS losses."""
        N = obs_exp.shape[0]
        losses = []
        for start in range(0, N, chunk_size):
            o = obs_exp[start:start + chunk_size]
            d = deltas_3d[start:start + chunk_size]
            g = goal_exp[start:start + chunk_size]
            if self.num_repeat_eval > 1:
                runs = []
                for _ in range(self.num_repeat_eval):
                    preds = self._autoregressive_rollout(o, d)
                    runs.append(self.loss_fn(preds[:, -1], g).flatten())
                losses.append(torch.stack(runs).mean(0))
            else:
                preds = self._autoregressive_rollout(o, d)
                losses.append(self.loss_fn(preds[:, -1], g).flatten())
            torch.cuda.empty_cache()
        return torch.cat(losses)

    @staticmethod
    def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
        """(C, H, W) in [-1, 1] → (H, W, 3) uint8."""
        return (
            ((t.float().cpu().permute(1, 2, 0) + 1.0) / 2.0 * 255)
            .clamp(0, 255).byte().numpy()
        )
