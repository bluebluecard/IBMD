# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch

from guided_diffusion.script_util import create_model

from . import util
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

from ipdb import set_trace as debug

class Image256Net(torch.nn.Module):
    def __init__(self, log, noise_levels, use_fp16=False, cond=False, student_noise_input=False, 
                    pretrained_adm=True, ckpt_dir="data/"):
        super(Image256Net, self).__init__()

        # initialize model
        ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_COND_PKL if cond else I2SB_IMG256_UNCOND_PKL)
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        self.diffusion_model = create_model(**kwargs)
        log.info(f"[Net] Initialized network from {ckpt_pkl=}! Size={util.count_parameters(self.diffusion_model)}!")

        # load (modified) adm ckpt
        if pretrained_adm:
            ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT if cond else I2SB_IMG256_UNCOND_CKPT)
            out = torch.load(ckpt_pt, map_location="cpu")
            self.diffusion_model.load_state_dict(out)
            log.info(f"[Net] Loaded pretrained adm {ckpt_pt=}!")

        self.diffusion_model.eval()
        self.cond = cond
        self.student_noise_input = student_noise_input
        self.noise_levels = noise_levels
        
        # self.register_buffer("noise_levels", noise_levels)

    def add_noise_channels(self):
        out_ch, in_ch, kernel_size, _ = self.diffusion_model.input_blocks[0][0].weight.shape
        noise_conv = torch.nn.Conv2d(3, out_ch, kernel_size, padding=1)
        # Initialize weights to zero
        torch.nn.init.zeros_(noise_conv.weight)
        final_conv = torch.nn.Conv2d(3 + in_ch, out_ch, kernel_size, padding=1)
        final_conv.weight.data = torch.cat([self.diffusion_model.input_blocks[0][0].weight.data, noise_conv.weight.data], dim=1)
        # Copy bias from original conv layer
        final_conv.bias.data = self.diffusion_model.input_blocks[0][0].bias.data
        self.diffusion_model.input_blocks[0][0] = final_conv
        self.student_noise_input = True

    def forward(self, x, steps, cond=None):
        # t = self.noise_levels[steps].detach()
        t = self.noise_levels[steps].detach().to(x.device)
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        x = torch.cat([x, torch.randn_like(x[:, :3, :, :])], dim=1) if self.student_noise_input else x
        return self.diffusion_model(x, t)

class DebugIdentityNet(torch.nn.Module):
    def __init__(self, log, noise_levels=None, use_fp16=False, cond=False, student_noise_input=False):
        super().__init__()
        self.log = log
        self.noise_levels = noise_levels
        self.use_fp16 = use_fp16
        self.cond = cond
        self.student_noise_input = student_noise_input
        
        # Calculate input channels based on conditional and noise inputs
        base_channels = 3  # RGB
        extra_channels = 0
        if cond:
            extra_channels += 3  # Conditional input
        if student_noise_input:
            extra_channels += 3  # Noise input
        
        in_channels = base_channels + extra_channels
        out_channels = 3  # Output is always RGB
        
        # Create the input convolution layer
        self.input_blocks = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                )
            ])
        ])
        
        # Initialize weights to handle identity mapping
        # Set the weights for the RGB channels to 1.0 (identity)
        with torch.no_grad():
            weights = torch.zeros(out_channels, in_channels, 1, 1)
            # Set the first 3 channels (RGB) to identity
            weights[:, :3, 0, 0] = torch.eye(3)
            self.input_blocks[0][0].weight.copy_(weights)
        
        log.info(f"[Debug] Created identity network with {in_channels} input channels (cond={cond}, noise={student_noise_input})")

    def forward(self, x, step, cond=None):
        # Concatenate inputs based on configuration
        if self.cond and cond is not None:
            x = torch.cat([x, cond], dim=1)
        if self.student_noise_input:
            noise = torch.randn_like(x[:, :3])  # Generate noise for RGB channels
            x = torch.cat([x, noise], dim=1)
        
        # Process through the convolution layer
        x = self.input_blocks[0][0](x)
        return x
