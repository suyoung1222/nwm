# OCNWM Stage-1 support: frozen VideoSAUR slot extractor (INFERENCE ONLY).
#
# This module is an ADDITIVE extension for the object-centric NWM (OCNWM). It never
# trains or modifies VideoSAUR / the VAE / the CDiT world model. It wraps a frozen,
# pretrained VideoSAUR (YTVIS DINO ViT-B/16 by default) and extracts per-frame object
# slots from a sequence of frames, using the native *video* path so slot identities are
# temporally consistent across frames (ScanOverTime carries state frame-to-frame).
#
# Design decisions (fixed for OCNWM):
#   * VideoSAUR is frozen (eval, requires_grad=False). No gradients ever flow into it.
#   * We drive encoder -> initializer -> processor directly and SKIP the decoder/losses
#     (we only need slots + slot-attention masks), which is faster and lighter.
#   * Long trajectories: the encoder is run in chunks (memory bound) but the recurrent
#     processor (ScanOverTime) is run ONCE over the full feature sequence, so temporal
#     slot identity is globally consistent along the whole trajectory and the
#     first-frame corrector (n_iters=3) fires only at true t=0.
#   * Preprocessing defaults to NWM's 4:3 CenterCropAR + Resize(224) so slots describe
#     the SAME field of view the world model's VAE sees, with ImageNet normalization
#     (what DINO expects). Both crop and normalization are configurable for ablation.
#
# Returned slots have shape (T, K, slot_dim); masks (T, K, num_patches).

import os
import sys
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF

# --- make the vendored videosaur package importable ---------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOSAUR_ROOT = os.path.join(_THIS_DIR, "thirdparty", "videosaur")
if _VIDEOSAUR_ROOT not in sys.path:
    sys.path.insert(0, _VIDEOSAUR_ROOT)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MOVI_MEAN = [0.5, 0.5, 0.5]
MOVI_STD = [0.5, 0.5, 0.5]

# NWM crops every image to a 4:3 aspect ratio before resizing (see misc.CenterCropAR).
_IMAGE_ASPECT_RATIO = 4.0 / 3.0

# Default frozen model (native video, DINO ViT-B/16 @224, K=7, slot_dim=64).
DEFAULT_CONFIG = os.path.join(_VIDEOSAUR_ROOT, "configs", "videosaur", "ytvis.yml")
DEFAULT_CHECKPOINT = os.path.join(_VIDEOSAUR_ROOT, "checkpoints", "videosaur-ytvis.ckpt")


class _CenterCropAR:
    """Replicates nwm/misc.py::CenterCropAR (kept local to avoid importing misc, which
    pulls matplotlib and reads config/ at import time)."""

    def __init__(self, ar: float = _IMAGE_ASPECT_RATIO):
        self.ar = ar

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w > h:
            return TF.center_crop(img, (h, int(h * self.ar)))
        return TF.center_crop(img, (int(w / self.ar), w))


def build_slot_transform(
    input_size: int = 224,
    crop_mode: str = "nwm_ar",
    normalization: str = "imagenet",
) -> T.Compose:
    """Preprocessing applied per frame before VideoSAUR.

    crop_mode: 'nwm_ar' (4:3 CenterCropAR, matches the world-model view) or 'none'
    normalization: 'imagenet' (DINO default) or 'movi'
    """
    ops: List[object] = []
    if crop_mode == "nwm_ar":
        ops.append(_CenterCropAR())
    elif crop_mode == "none":
        pass
    else:
        raise ValueError(f"unknown crop_mode={crop_mode!r}")
    ops.append(T.Resize((input_size, input_size)))
    ops.append(T.ToTensor())
    if normalization == "imagenet":
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif normalization == "movi":
        ops.append(T.Normalize(MOVI_MEAN, MOVI_STD))
    else:
        raise ValueError(f"unknown normalization={normalization!r}")
    return T.Compose(ops)


class SlotExtractor:
    """Frozen VideoSAUR wrapper. Extracts temporally-consistent object slots.

    Attributes:
        num_slots (int): K, number of slots per frame.
        slot_dim (int): slot feature dimension (needed to size the CDiT exo projection).
        num_patches (int): number of encoder patch tokens (for reshaping masks to a grid).
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        videosaur_config: str = DEFAULT_CONFIG,
        device: Union[str, torch.device] = "cuda",
        n_slots: Optional[int] = None,
        input_size: int = 224,
        crop_mode: str = "nwm_ar",
        normalization: str = "imagenet",
        encoder_chunk: int = 64,
        seed: int = 0,
        backbone_pretrained: bool = False,
    ):
        from videosaur import configuration, models  # imported lazily (heavy deps)

        self.device = torch.device(device)
        self.seed = int(seed)
        self.encoder_chunk = int(encoder_chunk)
        self.input_size = int(input_size)
        self.crop_mode = crop_mode
        self.normalization = normalization
        self.config_path = videosaur_config
        self.checkpoint_path = checkpoint

        cfg = configuration.load_config(videosaur_config)
        # The full VideoSAUR checkpoint already contains the (frozen) backbone weights,
        # so we don't need timm to fetch pretrained backbone weights at build time.
        if not backbone_pretrained:
            try:
                cfg.model.encoder.backbone.pretrained = False
            except Exception:
                pass
        assert cfg.model.get("input_type", "image") == "video", (
            "SlotExtractor requires a *video* VideoSAUR config (native temporal slots); "
            f"got input_type={cfg.model.get('input_type', 'image')!r} in {videosaur_config}"
        )

        model = models.build(cfg.model, cfg.optimizer)
        state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) or len(unexpected):
            # Not fatal, but surface it: a healthy load is missing=0 / unexpected=0.
            print(
                f"[SlotExtractor] load_state_dict: missing={len(missing)} "
                f"unexpected={len(unexpected)} (first missing={missing[:3]})"
            )

        if n_slots is not None:
            # Safe for RandomInit (samples noise per forward using n_slots). Would break
            # a FixedLearnedInit (learned slot params have a fixed count).
            init_name = type(model.initializer).__name__
            if init_name != "RandomInit":
                raise ValueError(
                    f"n_slots override requested but initializer is {init_name}, not RandomInit."
                )
            model.initializer.n_slots = int(n_slots)

        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        self.num_slots = int(model.initializer.n_slots)
        self.slot_dim = int(cfg.model.initializer.dim)
        self.num_patches: Optional[int] = None  # filled in on first extraction

        self.transform = build_slot_transform(input_size, crop_mode, normalization)

    # -- preprocessing ---------------------------------------------------------------
    def preprocess_paths(self, paths: Sequence[str]) -> torch.Tensor:
        pil = [Image.open(p).convert("RGB") for p in paths]
        return self.preprocess_pils(pil)

    def preprocess_pils(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(im) for im in images])  # (T, 3, H, W)

    # -- extraction ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, frames: torch.Tensor, return_masks: bool = False) -> dict:
        """frames: (T, 3, H, W) preprocessed tensor. Returns dict with:
        slots (T, K, slot_dim) float32 on CPU, and optionally masks (T, K, num_patches)."""
        assert frames.ndim == 4, f"expected (T,3,H,W), got {tuple(frames.shape)}"
        video = frames.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        T_len = video.shape[1]

        # Encoder in chunks (bounds memory for long trajectories); features are cheap
        # to hold for the whole sequence.
        feats = []
        for s in range(0, T_len, self.encoder_chunk):
            chunk = video[:, s : s + self.encoder_chunk]
            feats.append(self.model.encoder(chunk)["features"])
        features = torch.cat(feats, dim=1)  # (1, T, P, slot_dim)
        self.num_patches = int(features.shape[2])

        # Deterministic initial slots, then one recurrent pass over the whole sequence.
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        slots_initial = self.model.initializer(batch_size=1)  # (1, K, slot_dim)
        processor_output = self.model.processor(slots_initial, features)

        slots = processor_output["state"][0].float().cpu()  # (T, K, slot_dim)
        out = {"slots": slots}
        if return_masks:
            out["masks"] = processor_output["corrector"]["masks"][0].float().cpu()  # (T,K,P)
        return out

    @torch.no_grad()
    def extract_paths(self, paths: Sequence[str], return_masks: bool = False) -> dict:
        return self.extract(self.preprocess_paths(paths), return_masks=return_masks)
