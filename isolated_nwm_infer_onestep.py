# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from distributed import init_distributed
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import os

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import misc
from models import CDiT_models
from datasets import EvalDataset
from PIL import Image
import time

def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = misc.unnormalize(img)

    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

    image.save(output_file)


def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/{eval_type}.pkl"
    else:
        predefined_index = None

    dataset = EvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
                max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
                len_traj_pred=config["eval_len_traj_pred"],
                traj_stride=config["traj_stride"],
                context_size=config["eval_context_size"],
                normalize=config["normalize"],
                transform=misc.transform,
                goals_per_obs=4,
                predefined_index=predefined_index,
                traj_names='traj_names.txt'
            )

    return dataset

@torch.no_grad()
def model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps, latent_size, device, num_cond, num_goals=1, rel_t=None, progress=False):
    model, diffusion, vae = all_models
    x = curr_obs.to(device)  # 현재관측
    y = curr_delta.to(device)  # 액션

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B) * (1. / 128.)).to(device)
            rel_t *= num_timesteps

        x = x.flatten(0, 1)
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
        z = torch.randn(B * num_goals, 4, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
        samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
        )  # NWM 출력
        samples = vae.decode(samples / 0.18215).sample

        return torch.clip(samples, -1., 1.)


@torch.no_grad
def main(args):
    _, _, device, _ = init_distributed()
    print(args)
    device = torch.device(device)
    exp_eval = args.exp

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, 'gt')
    else:
        exp_name = os.path.basename(exp_eval).split('.')[0]
        args.save_output_dir = os.path.join(args.output_dir, exp_name)

    if args.ckp != '0100000':
        args.save_output_dir = args.save_output_dir + "_%s" % (args.ckp)

    os.makedirs(args.save_output_dir, exist_ok=True)

    with open("config/eval_config.yaml", "r") as f:  # data_folder, test, goals_per_obs 불러옴.
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    latent_size = config['image_size'] // 8
    args.latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    print("loading")
    model_lst = (None, None, None)
    if not args.gt:
        model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4)  # 모델 불러오기
        ckp = torch.load(f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar', map_location='cpu', weights_only=False)
        print(model.load_state_dict(ckp["ema"], strict=True))
        model.eval()
        model.to(device)
        model = torch.compile(model)
        diffusion = create_diffusion(str(250))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
        model_lst = (model, diffusion, vae)

    # Load dataset and grab one sample by index
    dataset_name = args.datasets
    dataset_val = get_dataset_eval(config, dataset_name, args.eval_type, predefined_index=True)
    print(f"Dataset size: {len(dataset_val)}")

    if args.sample_idx < 0 or args.sample_idx >= len(dataset_val):
        raise ValueError(f"sample_idx {args.sample_idx} out of range [0, {len(dataset_val)})")

    idxs, obs_image, gt_image, delta = dataset_val[args.sample_idx]
    # add batch dim (B=1)
    idxs = idxs.unsqueeze(0)
    obs_image = obs_image.unsqueeze(0)
    gt_image = gt_image.unsqueeze(0)
    delta = delta.unsqueeze(0)

    dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
    os.makedirs(dataset_save_output_dir, exist_ok=True)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        obs_image = obs_image[:, -num_cond:].to(device)
        gt_image = gt_image.to(device)

        sec = args.sec
        timestep = sec * args.input_fps  # 하나의 sec와 timestep = sec*fps는 대응됨.
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)

        if args.gt:
            x_pred_pixels = gt_image[:, timestep - 1].clone().to(device)
        else:
            start_time = time.time()
            x_pred_pixels = model_forward_wrapper(model_lst, obs_image, curr_delta, timestep, args.latent_size, num_cond=num_cond, num_goals=1, device=device)
            end_time = time.time()
            print(f"Time taken for model_forward_wrapper: {end_time - start_time:.4f} seconds")

        sample_folder = os.path.join(dataset_save_output_dir, f'id_{args.sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{sec}.png')
        save_image(image_file, x_pred_pixels[0], True)
        print(f"Saved prediction to {image_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='cdit_xl_ego4d_200000')
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--datasets", type=str, default=None, help="dataset name (single)")
    parser.add_argument("--eval_type", type=str, default='time', help="type of evaluation has to be either 'time' or 'rollout' (selects predefined index pkl)")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    parser.add_argument("--sample_idx", type=int, default=0, help="index of the sample in the dataset to predict")
    parser.add_argument("--sec", type=int, default=1, help="timestep in seconds (model timesteps = sec * input_fps)")
    args = parser.parse_args()

    main(args)
