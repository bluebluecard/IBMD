"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch
import torchvision.utils as vutils
import torch.distributed as dist

from ddbm import dist_util, logger
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.karras_diffusion import karras_sample

from datasets import load_data

from pathlib import Path
from torch import nn
from ddbm.nn import append_dims
import ipdb

@torch.no_grad()
def main():
    args = create_argparser().parse_args()
    args.use_fp16 = False

    workdir = os.path.join("workdir_imagenet_teacher", os.path.basename(args.model_path)[:-3])

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.replace("_adapted", "").split("_")
    step = int(split[-1].split(".")[0])
    if args.sampler == "dbim":
        sample_dir = Path(workdir) / f"sample_{step}/split={args.split}/dbim_eta={int(args.eta)}/steps={args.steps}"
    elif args.sampler == "dbim_high_order":
        sample_dir = Path(workdir) / f"sample_{step}/split={args.split}/dbim_order={args.order}/steps={args.steps}"
    else:
        sample_dir = Path(workdir) / f"sample_{step}/split={args.split}/{args.sampler}/steps={args.steps}"
    dist_util.setup_dist()
    if dist.get_rank() == 0:

        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(sample_dir))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    if args.add_noise:
        out_ch, in_ch, kernel_size, _ = model.input_blocks[0][0].weight.shape
        final_conv = nn.Conv2d(args.noise_channels + in_ch, out_ch, kernel_size, padding=1).to(dist_util.dev())
        model.input_blocks[0][0] = final_conv
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(dist_util.dev())

    if args.use_fp16:
        model = model.half()
    model.eval()

    logger.log("sampling...")

    all_images = []
    all_labels = []

    all_dataloaders = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        include_test=(args.split == "test"),
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.split == "train":
        dataloader = all_dataloaders[1]
    elif args.split == "test":
        dataloader = all_dataloaders[2]
    else:
        raise NotImplementedError
    args.num_samples = len(dataloader.dataset)
    num = 0
    m = 0
    for i, data in enumerate(dataloader):
        x0_image = data[0]
        x0 = x0_image.to(dist_util.dev())

        y0_image = data[1].to(dist_util.dev())
        y0 = y0_image
        if args.draw:
            path = []

        model_kwargs = {"xT": torch.cat([y0, torch.randn_like(y0)[:, :args.noise_channels]], dim=1)} if args.add_noise else {"xT": y0}

        if "inpaint" in args.dataset:
            _, mask, label = data[2]
            mask = mask.to(dist_util.dev())
            label = label.to(dist_util.dev())
            model_kwargs["y"] = label
        else:
            mask = None
        nfe = 0
        ones = torch.ones((y0.shape[0],), device=dist_util.dev())
        timesteps = torch.linspace(diffusion.t_max, diffusion.t_min,  steps=args.num_steps + 1, device = dist_util.dev())[:-1]
        x = y0.clone()
        if args.draw:
            path.append(x)
        for j in range(len(timesteps) - 1):
            if j == 0:
                s = timesteps[j]
                t = timesteps[j + 1]
                _, x0_hat, _ = diffusion.denoise(model, x, s * ones, **model_kwargs)
                nfe += 1
                if mask is not None:
                    x0_hat = x0_hat * mask + y0 * (1 - mask)
                noise = torch.randn_like(x0_hat)
                x = diffusion.bridge_sample(x0_hat, y0, t * ones, noise)
                if args.draw:
                    path.append(x)
            else:
                s = timesteps[j]
                t = timesteps[j + 1]

                _, x0_hat, _ = diffusion.denoise(model, x, s * ones, **model_kwargs)
                nfe += 1
                if mask is not None:
                    x0_hat = x0_hat * mask + y0 * (1 - mask)

                a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
                a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

                _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(s * ones)]
                alpha_t, _, rho_t, _ = [
                    append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(t * ones)
                ]

                omega_st = (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
                tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
                coeff_xs = tmp_var
                coeff_x0_hat = b_t - tmp_var * b_s
                coeff_xT = a_t - tmp_var * a_s

                noise = torch.randn_like(x0_hat)

                x = coeff_x0_hat * x0_hat + coeff_xT * y0 + coeff_xs * x + omega_st * noise
                if args.draw:
                    path.append(x)
            _, x0_hat, _ = diffusion.denoise(model, x, timesteps[-1] * ones, **model_kwargs)
            nfe += 1
            sample = x0_hat.clamp(-1, 1)
            if args.draw:
                path.append(sample)
            if dist.get_rank() == 0:
                print("nfe:", nfe)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_samples = torch.cat(gathered_samples)
        if "inpaint" in args.dataset:
            gathered_labels = [torch.zeros_like(label) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, label)
            gathered_labels = torch.cat(gathered_labels)
        num += gathered_samples.shape[0]

        num_display = min(32, sample.shape[0])
        if i == 0 and dist.get_rank() == 0:
            vutils.save_image(
                sample.permute(0, 3, 1, 2)[:num_display].float() / 255,
                f"{sample_dir}/sample_{i}.png",
                nrow=int(np.sqrt(num_display)),
            )
            if x0 is not None:
                vutils.save_image(
                    x0_image[:num_display] / 2 + 0.5,
                    f"{sample_dir}/x_{i}.png",
                    nrow=int(np.sqrt(num_display)),
                )
            vutils.save_image(
                y0_image[:num_display] / 2 + 0.5,
                f"{sample_dir}/y_{i}.png",
                nrow=int(np.sqrt(num_display)),
            )
            
                    
        all_images.append(gathered_samples.detach().cpu().numpy())
        if "inpaint" in args.dataset:
            all_labels.append(gathered_labels.detach().cpu().numpy())

        if dist.get_rank() == 0:
            logger.log(f"sampled {num} images")

    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if "inpaint" in args.dataset:
        labels = np.concatenate(all_labels, axis=0)
        labels = labels[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        if "inpaint" in args.dataset:
            shape_str = "x".join([str(x) for x in labels.shape])
            out_path = os.path.join(sample_dir, f"labels_{shape_str}_nfe{nfe}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, labels)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",  ## only used in bridge
        dataset="edges2handbags",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split="train",
        churn_step_ratio=0.0,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=42,
        num_workers=8,
        eta=1.0,
        order=1,
        add_noise=False,
        noise_channels=1,
        num_steps=0,
        draw=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
