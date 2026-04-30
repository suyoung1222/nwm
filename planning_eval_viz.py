# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import yaml
import os
import numpy as np
import lpips
import matplotlib
matplotlib.use('TkAgg')   # interactive backend; fall back to 'Qt5Agg' if TkAgg missing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from diffusers.models import AutoencoderKL

from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from diffusion import create_diffusion
from datasets import TrajectoryEvalDataset
from isolated_nwm_infer import model_forward_wrapper
from misc import calculate_delta_yaw, get_action_torch, save_planning_pred, transform, unnormalize_data, unnormalize
import distributed as dist
from isolated_nwm_eval import save_metric_to_disk
from models import CDiT_models


with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])


def tensor_to_numpy_img(t):
    """Convert a CHW [-1,1] tensor to HWC [0,1] numpy array for imshow."""
    img = unnormalize(t.detach().cpu().float())
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


class LiveViz:
    """Interactive matplotlib window showing real vs imagined observations."""

    def __init__(self, context_size):
        self.context_size = context_size
        plt.ion()
        # Layout: row 0 = real context frames + goal, row 1 = imagined rollout frames
        n_cols = max(context_size + 1, 2)  # +1 for goal
        self.fig = plt.figure(figsize=(n_cols * 3, 7))
        self.fig.suptitle("Real Observations  vs  Imagined Rollout", fontsize=13)
        gs = gridspec.GridSpec(2, n_cols, figure=self.fig,
                               hspace=0.35, wspace=0.05)

        self.ax_real = []
        for c in range(context_size):
            ax = self.fig.add_subplot(gs[0, c])
            ax.axis('off')
            self.ax_real.append(ax)

        self.ax_goal = self.fig.add_subplot(gs[0, context_size])
        self.ax_goal.axis('off')
        self.ax_goal.set_title("Goal", fontsize=10, color='green')

        self.ax_imag = []
        for c in range(n_cols):
            ax = self.fig.add_subplot(gs[1, c])
            ax.axis('off')
            self.ax_imag.append(ax)

        self.im_real = [ax.imshow(np.zeros((224, 224, 3))) for ax in self.ax_real]
        self.im_goal = self.ax_goal.imshow(np.zeros((224, 224, 3)))
        self.im_imag = [ax.imshow(np.zeros((224, 224, 3))) for ax in self.ax_imag]

        self.fig.text(0.01, 0.72, "Real\nContext", va='center', ha='left', fontsize=10,
                      color='steelblue', fontweight='bold')
        self.fig.text(0.01, 0.28, "Imagined\nRollout", va='center', ha='left', fontsize=10,
                      color='darkorange', fontweight='bold')

        plt.tight_layout()
        plt.pause(0.001)

    def update(self, obs_images, goal_image, imagined_frames, sample_idx, cem_iter, lpips_loss):
        """
        obs_images:      (context_size, C, H, W) tensor in [-1, 1]
        goal_image:      (C, H, W) tensor
        imagined_frames: list of (C, H, W) tensors — best-action rollout
        """
        # --- real context row ---
        for c in range(self.context_size):
            img = tensor_to_numpy_img(obs_images[c])
            self.im_real[c].set_data(img)
            title = f"t-{self.context_size - 1 - c}" if c < self.context_size - 1 else "t (now)"
            self.ax_real[c].set_title(title, fontsize=9, color='steelblue')

        # --- goal ---
        self.im_goal.set_data(tensor_to_numpy_img(goal_image))

        # --- imagined rollout row ---
        for c, ax in enumerate(self.ax_imag):
            if c < len(imagined_frames):
                img = tensor_to_numpy_img(imagined_frames[c])
                self.im_imag[c].set_data(img)
                ax.set_title(f"step {c + 1}", fontsize=9, color='darkorange')
                ax.set_visible(True)
            else:
                ax.set_visible(False)

        loss_str = f"{lpips_loss:.4f}" if lpips_loss is not None else "—"
        self.fig.suptitle(
            f"Sample {sample_idx} | CEM iter {cem_iter} | Best LPIPS: {loss_str}",
            fontsize=12
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def get_dataset_eval(config, dataset_name):
    data_cfg = config["eval_datasets"][dataset_name]
    predefined_index = f"data_splits/{dataset_name}/test/navigation_eval.pkl"

    dataset = TrajectoryEvalDataset(
        data_folder=data_cfg["data_folder"],
        data_split_folder=data_cfg["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["trajectory_eval_distance"]["min_dist_cat"],
        max_dist_cat=config["trajectory_eval_distance"]["max_dist_cat"],
        len_traj_pred=config["trajectory_eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["trajectory_eval_context_size"],
        normalize=config["normalize"],
        transform=transform,
        predefined_index=predefined_index,
        traj_names="rollout_traj_names.txt",
    )
    return dataset


class WM_Planning_Evaluator_Viz:
    def __init__(self, args):
        self.args = args
        self.exp = args.exp
        _, _, device, _ = dist.init_distributed()
        self.device = torch.device(device)

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        self.exp_eval = self.exp
        self.get_eval_name()

        with open("config/eval_config.yaml", "r") as f:
            default_config = yaml.safe_load(f)
        self.config = default_config
        with open(self.exp_eval, "r") as f:
            user_config = yaml.safe_load(f)
        self.config.update(user_config)

        latent_size = self.config['image_size'] // 8
        self.latent_size = latent_size
        self.num_cond = self.config['eval_context_size']

        if self.args.save_preds:
            exp_name = os.path.basename(self.args.exp).split('.')[0]
            self.args.save_output_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(self.args.save_output_dir, exist_ok=True)

        self.dataset_names = self.args.datasets.split(',')
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = get_dataset_eval(self.config, dataset_name)

            if len(dataset_val) % num_tasks != 0:
                print('Warning: dataset size not divisible by num processes.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            curr_data_loader = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=1,           # must be 1 for live visualisation
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            self.datasets[dataset_name] = curr_data_loader

        print("Loading model …")
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond,
            input_size=latent_size,
        )
        ckp = torch.load(
            f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.ckp}.pth.tar',
            map_location='cpu', weights_only=False,
        )
        model.load_state_dict(ckp["ema"], strict=True)
        model.eval()
        model.to(self.device)
        self.model = torch.compile(model)
        self.diffusion = create_diffusion(str(args.diffusion_steps))
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.device], find_unused_parameters=False)
        self.model_without_ddp = self.model.module

        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mode = 'cem'
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 3

        # Optional custom goal image: shape (1, 1, C, H, W) on CPU, broadcast over batch
        if args.goal_image:
            from PIL import Image as PILImage
            raw = PILImage.open(args.goal_image).convert("RGB")
            self.custom_goal = transform(raw).unsqueeze(0).unsqueeze(0)  # (1,1,C,H,W)
            print(f"Using custom goal image: {args.goal_image}")
        else:
            self.custom_goal = None

        self.viz = LiveViz(context_size=self.config['trajectory_eval_context_size'])

    def init_mu_sigma(self, obs_0):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim)
        mu[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        sigma = torch.ones([n_evals, self.action_dim])
        sigma[:, ] = torch.tensor(data_hyperparams[self.args.datasets]['var_scale'])
        return mu, sigma

    def autoregressive_rollout(self, obs_image, deltas, rollout_stride):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)

        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i+1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(
                all_models, curr_obs, curr_delta, self.args.rollout_stride,
                self.latent_size, num_cond=self.num_cond, device=self.device,
            )
            x_pred_pixels = x_pred_pixels.unsqueeze(1)
            curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)
            curr_obs = curr_obs[:, 1:]
            preds.append(x_pred_pixels)

        return torch.cat(preds, 1)

    def _cem_one_horizon(self, obs_image, goal_image, len_traj_pred, sample_idx, horizon_idx, cem_label_prefix):
        """
        Run CEM for a single planning horizon.
        Returns:
          best_deltas  : (1, len_traj_pred, 3)  — optimal action sequence
          best_preds   : (len_traj_pred, C, H, W) — imagined frames (batch elem 0)
          best_loss    : float
        """
        mu, sigma = self.init_mu_sigma(obs_image)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        best_preds_frames = None
        best_loss_val = None

        for cem_iter in range(self.opt_steps):
            sample = torch.randn(self.num_samples, self.action_dim).to(self.device) * sigma[0] + mu[0]
            single_delta = sample[:, :2]
            deltas = single_delta.unsqueeze(1).repeat(1, len_traj_pred, 1)
            unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
            delta_yaw = calculate_delta_yaw(unnorm_deltas)
            deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
            deltas[:, -1, -1] += sample[:, -1] * np.pi

            cur_obs = obs_image[0].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)
            cur_goal = goal_image[0].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1).squeeze(1)

            preds = self.autoregressive_rollout(cur_obs, deltas, self.args.rollout_stride)
            loss = self.loss_fn(preds[:, -1].to(self.device), cur_goal.to(self.device)).flatten(0)

            sorted_idx = torch.argsort(loss)
            topk_idx = sorted_idx[:self.topk]
            mu[0] = deltas[topk_idx][:, -1].mean(dim=0)
            sigma[0] = deltas[topk_idx][:, -1].std(dim=0)

            best_idx = topk_idx[0].item()
            best_preds_frames = [preds[best_idx, s] for s in range(preds.shape[1])]
            best_loss_val = loss[topk_idx[0]].item()

            self.viz.update(
                obs_images=obs_image[0],
                goal_image=goal_image[0, 0],
                imagined_frames=best_preds_frames,
                sample_idx=sample_idx,
                cem_iter=f"{cem_label_prefix} CEM {cem_iter + 1}",
                lpips_loss=best_loss_val,
            )

        # Final rollout with converged mu
        deltas = mu[:, :2].unsqueeze(1).repeat(1, len_traj_pred, 1)
        unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
        delta_yaw = calculate_delta_yaw(unnorm_deltas)
        deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
        deltas[:, -1, -1] += mu[:, -1] * np.pi

        preds = self.autoregressive_rollout(obs_image, deltas, self.args.rollout_stride)
        final_loss = self.loss_fn(
            preds[:, -1].to(self.device), goal_image.squeeze(1).to(self.device)
        ).flatten(0)

        final_frames = [preds[0, s] for s in range(preds.shape[1])]
        self.viz.update(
            obs_images=obs_image[0],
            goal_image=goal_image[0, 0],
            imagined_frames=final_frames,
            sample_idx=sample_idx,
            cem_iter=f"{cem_label_prefix} FINAL",
            lpips_loss=final_loss[0].item(),
        )

        return deltas, final_frames, final_loss[0].item()

    def generate_actions(self, sample_idx, obs_image, goal_image, gt_actions, len_traj_pred):
        """
        Receding horizon planning: repeatedly plan `len_traj_pred` steps,
        slide the context window forward using imagined frames, replan toward
        the same goal until `max_steps` total steps are accumulated or the
        goal LPIPS threshold is reached.
        """
        current_obs = obs_image.clone()   # (1, context, C, H, W) — starts with real frames
        all_pred_actions = []             # cumulative waypoints across horizons
        cumulative_offset = torch.zeros(1, 2)  # tracks absolute position across horizons

        max_steps = self.args.max_steps
        steps_done = 0
        horizon_idx = 0

        while steps_done < max_steps:
            steps_this_horizon = min(len_traj_pred, max_steps - steps_done)
            cem_label = f"H{horizon_idx + 1} ({steps_done}→{steps_done + steps_this_horizon})"

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                best_deltas, imagined_frames, best_loss = self._cem_one_horizon(
                    current_obs, goal_image, steps_this_horizon,
                    sample_idx, horizon_idx, cem_label,
                )

            # Convert deltas → absolute waypoints for this horizon
            horizon_actions = get_action_torch(best_deltas[:, :, :2], ACTION_STATS_TORCH)  # (1, steps, 2)
            horizon_actions = horizon_actions + cumulative_offset.to(horizon_actions.device)
            all_pred_actions.append(horizon_actions)

            # Update cumulative position offset (last waypoint of this horizon)
            cumulative_offset = horizon_actions[0, -1:].cpu()

            # Slide context window: last `num_cond` imagined frames become new context
            new_frames = torch.stack(imagined_frames, dim=0)           # (steps, C, H, W)
            new_frames = new_frames.unsqueeze(0)                       # (1, steps, C, H, W)
            current_obs = torch.cat([current_obs, new_frames], dim=1)  # append imagined
            current_obs = current_obs[:, -self.num_cond:]              # keep last 4 frames

            steps_done += steps_this_horizon
            horizon_idx += 1

            print(f"  Horizon {horizon_idx}: steps {steps_done}/{max_steps}, LPIPS={best_loss:.4f}")

            # Early stopping: imagined final frame already looks like goal
            if best_loss < self.args.goal_threshold:
                print(f"  Goal reached at horizon {horizon_idx} (LPIPS {best_loss:.4f} < {self.args.goal_threshold})")
                break

        pred_actions = torch.cat(all_pred_actions, dim=1)  # (1, total_steps, 2)
        pred_yaw = torch.zeros(1)                          # simplified: full yaw tracking omitted
        return pred_actions, pred_yaw

    def get_eval_name(self):
        self.eval_name = f'CEM_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'

    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :2] = actions
        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4))
        orientations_quat_wxyz[:, -1] = 1
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        return PoseTrajectory3D(positions_xyz=positions_xyz,
                                orientations_quat_wxyz=orientations_quat_wxyz,
                                timestamps=timestamps)

    def eval_metrics(self, traj_ref, traj_pred):
        traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)

        result = main_ape.ape(traj_ref, traj_pred, est_name='traj',
                              pose_relation=PoseRelation.translation_part,
                              align=False, correct_scale=False)
        ate = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
                              pose_relation=PoseRelation.rotation_angle_deg,
                              align=False, correct_scale=False,
                              delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_rot = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
                              pose_relation=PoseRelation.translation_part,
                              align=False, correct_scale=False,
                              delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_trans = result.stats['rmse']

        return ate, rpe_trans, rpe_rot

    @torch.no_grad()
    def evaluate(self):
        for dataset_name in self.dataset_names:
            metric_logger = dist.MetricLogger(delimiter="  ")
            header = 'Test:'

            if self.args.save_preds:
                os.makedirs(self.args.save_output_dir, exist_ok=True)

            curr_data_loader = self.datasets[dataset_name]
            for (idxs, obs_image, goal_image, gt_actions, goal_pos) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond:]
                sample_idx = int(idxs.flatten()[0].item())

                # Replace goal with custom image if provided (broadcast to current batch size)
                if self.custom_goal is not None:
                    goal_image = self.custom_goal.expand(obs_image.shape[0], -1, -1, -1, -1)

                pred_actions, pred_yaw = self.generate_actions(
                    sample_idx,
                    obs_image, goal_image, gt_actions,
                    self.config["trajectory_eval_len_traj_pred"],
                )

                # Trim both trajectories to the shorter length for fair metric comparison
                n_steps = min(pred_actions.shape[1], gt_actions.shape[1])
                pred_traj = self.actions_to_traj(pred_actions[0, :n_steps, :2])
                gt_traj = self.actions_to_traj(gt_actions[0, :n_steps, :2])
                ate, rpe_trans, _ = self.eval_metrics(gt_traj, pred_traj)

                pred_final_pos = pred_actions[0, -1, :2].cpu()
                pred_final_yaw = pred_yaw[0].cpu()
                goal_final_pos = goal_pos[0, 0, :2]
                goal_final_yaw = goal_pos[0, 0, -1]
                pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                yaw_diff = pred_final_yaw - goal_final_yaw
                yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()

                metric_logger.meters[f'{dataset_name}_ate'].update(ate, n=1)
                metric_logger.meters[f'{dataset_name}_rpe_trans'].update(rpe_trans, n=1)
                metric_logger.meters[f'{dataset_name}_pos_diff_norm'].update(pos_diff_norm, n=1)
                metric_logger.meters[f'{dataset_name}_yaw_diff_norm'].update(yaw_diff_norm, n=1)

            if self.args.save_preds:
                output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.eval_name}.json')
                save_metric_to_disk(metric_logger, output_fn)

        metric_logger.synchronize_between_processes()
        self.viz.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--ckp", type=str, default='0100000')
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument('--save_preds', action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)

    # CEM args
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--rollout_stride", type=int, default=2)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--opt_steps", type=int, default=5)
    parser.add_argument("--num_repeat_eval", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=100,
                        help="DDIM steps (250=default quality, 50-100=faster)")
    parser.add_argument("--goal_image", type=str, default=None,
                        help="Path to a custom goal image (.jpg/.png). Overrides dataset goals for every sample.")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Total steps toward goal across all horizons. 8=one-shot, 32/64=receding horizon.")
    parser.add_argument("--goal_threshold", type=float, default=0.3,
                        help="LPIPS value below which planning stops early (goal considered reached).")

    args = parser.parse_args()
    evaluator = WM_Planning_Evaluator_Viz(args)
    evaluator.evaluate()
