# OCNWM Stage-1 support: dataloader extension that attaches frozen-VideoSAUR slots.
#
# Additive & opt-in. `datasets.TrainingDataset` is left completely untouched (so the
# NWM baseline is bit-identical when slots are off). `SlotTrainingDataset` subclasses it
# and returns two extra tensors per sample:
#   past_slots   : (context_size,  K, slot_dim)  slots of the context frames  -> Stage-1 input
#   target_slots : (goals_per_obs, K, slot_dim)  slots of the goal/next frames -> L_slot target
# Both come from the frozen precomputed cache (no gradient); the L_slot target is a
# constant w.r.t. the trainable model. The predicted-slot "no-detach" path lives in the
# Stage-1 predictor / train loop (Steps 2-4), not here.
#
# The base __getitem__ is re-implemented (not super()-called) so the *same* random
# goal_offset drives both the loaded images and the looked-up slots, keeping them aligned.

import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from PIL import Image

from datasets import TrainingDataset
from misc import get_data_path, normalize_data


class SlotCache:
    """Reads per-trajectory slot .npz files (written by cache_slots.py), with a small
    in-process LRU so repeated frame lookups within a trajectory don't re-read disk."""

    def __init__(self, cache_dir: str, max_cached_trajs: int = 32):
        self.cache_dir = cache_dir
        self.max_cached = max_cached_trajs
        self._mem: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def _load(self, traj: str) -> np.ndarray:
        if traj in self._mem:
            self._mem.move_to_end(traj)
            return self._mem[traj]
        path = os.path.join(self.cache_dir, f"{traj}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"slot cache missing for trajectory {traj!r}: {path}")
        with np.load(path) as z:
            slots = z["slots"]  # (L, K, slot_dim) float16
        self._mem[traj] = slots
        self._mem.move_to_end(traj)
        if len(self._mem) > self.max_cached:
            self._mem.popitem(last=False)
        return slots

    def get_frames(self, traj: str, frame_indices) -> torch.Tensor:
        """Return slots for the given frame indices -> (len(indices), K, slot_dim) float32."""
        slots = self._load(traj)
        idx = np.asarray(frame_indices, dtype=np.int64)
        if idx.min() < 0 or idx.max() >= slots.shape[0]:
            raise IndexError(
                f"frame index out of range for {traj}: got {idx.tolist()}, L={slots.shape[0]}"
            )
        return torch.from_numpy(slots[idx].astype(np.float32))


class SlotTrainingDataset(TrainingDataset):
    """TrainingDataset that additionally returns (past_slots, target_slots) from the cache."""

    def __init__(self, *args, cache_root: str, cache_dataset_name: Optional[str] = None,
                 max_cached_trajs: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        name = cache_dataset_name if cache_dataset_name is not None else self.dataset_name
        self.slot_cache = SlotCache(os.path.join(cache_root, name), max_cached_trajs)

    def __getitem__(self, i: int):
        # Mirror of TrainingDataset.__getitem__, with aligned slot lookup added.
        f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
        goal_time = (curr_time + goal_offset).astype("int")
        rel_time = (goal_offset).astype("float") / 128.0

        context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
        context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]
        obs_image = torch.stack(
            [self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context]
        )

        curr_traj_data = self._get_trajectory(f_curr)
        _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

        # Aligned slot lookup (same context_times / goal_time as the images above).
        past_slots = self.slot_cache.get_frames(f_curr, context_times)       # (context_size, K, D)
        target_slots = self.slot_cache.get_frames(f_curr, goal_time.tolist())  # (goals_per_obs, K, D)

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(rel_time, dtype=torch.float32),
            past_slots,     # frozen (no grad): Stage-1 input
            target_slots,   # frozen (no grad): L_slot target
        )
