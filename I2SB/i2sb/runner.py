# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net, DebugIdentityNet
from .diffusion import Diffusion

from ipdb import set_trace as debug


def resolve_cond_x1(opt, log):
    cond_x1 = getattr(opt, "cond_x1", False)
    if cond_x1 or not getattr(opt, "load", None):
        return cond_x1

    ckpt_path = Path(opt.load)
    opt_pkl_path = ckpt_path.parent / "options.pkl"
    if opt_pkl_path.exists():
        try:
            with open(opt_pkl_path, "rb") as f:
                ckpt_opt = pickle.load(f)
            if getattr(ckpt_opt, "cond_x1", False):
                log.info(f"[Net] Inferred cond_x1=True from {opt_pkl_path}!")
                return True
        except Exception as exc:
            log.warning(f"[Net] Failed to inspect {opt_pkl_path}: {exc}")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_state = checkpoint.get(getattr(opt, "model_name_in_ckpt", "net"))
        if isinstance(model_state, dict):
            input_weight = model_state.get("input_blocks.0.0.weight")
            if input_weight is not None and hasattr(input_weight, "shape") and len(input_weight.shape) >= 2:
                in_channels = int(input_weight.shape[1])
                if in_channels > 3:
                    log.info(f"[Net] Inferred cond_x1=True from {ckpt_path} input channels={in_channels}!")
                    return True
    except Exception as exc:
        log.warning(f"[Net] Failed to inspect checkpoint architecture from {ckpt_path}: {exc}")

    return False


def should_load_pretrained_adm(opt):
    return not bool(getattr(opt, "load", None))

def load_ema_with_compat(ema, model, ema_state, log, ckpt_path, role="Ema"):
    if ema_state is None:
        log.warning(f"[{role}] No EMA state provided for {ckpt_path}; using model weights.")
        return ema

    try:
        ema.load_state_dict(ema_state)
        log.info(f"[{role}] Loaded ema ckpt: {ckpt_path}!")
        return ema
    except (KeyError, ValueError, TypeError) as exc:
        shadow = ema_state.get("shadow") if isinstance(ema_state, dict) else None
        if not isinstance(shadow, dict):
            raise

        decay = float(ema_state.get("decay", getattr(ema, "decay", 0.999)))
        loaded = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow_param = shadow.get(name)
                if shadow_param is None:
                    continue
                if param.shape != shadow_param.shape:
                    log.warning(
                        f"[{role}] Skip EMA tensor with mismatched shape for {name}: "
                        f"{tuple(shadow_param.shape)} != {tuple(param.shape)}"
                    )
                    continue
                param.copy_(shadow_param.detach().to(param.device))
                loaded += 1

        compat_ema = ExponentialMovingAverage(model.parameters(), decay=decay)
        log.warning(
            f"[{role}] Converted codec-style EMA from {ckpt_path} after {type(exc).__name__}: "
            f"loaded {loaded} shadow tensors into model weights."
        )
        return compat_ema


def nchw_to_bcwh(x):
    return x.permute(0, 1, 3, 2).contiguous()


def bcwh_to_nchw(x):
    return x.permute(0, 1, 3, 2).contiguous()


def pad_width_to_target(x, target_w, mode="reflect"):
    if target_w is None:
        return x, 0, 0
    width = x.shape[-1]
    if width > target_w:
        raise ValueError(f"Input width W={width} > target_w={target_w}")
    if width == target_w:
        return x, 0, 0
    total = target_w - width
    pad_left = total // 2
    pad_right = total - pad_left
    return F.pad(x, (pad_left, pad_right, 0, 0), mode=mode), pad_left, pad_right


def crop_width(x, pad_left, pad_right):
    if pad_left == 0 and pad_right == 0:
        return x
    return x[..., pad_left : x.shape[-1] - pad_right]

def build_optimizer_sched(opt, net, log, load_student=False, load_bridge_model=False):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load and not load_student and not load_bridge_model:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    RESULT_DIR = Path("results")

    if hasattr(opt, "distillation_checkpoint") and opt.distillation_checkpoint and load_student:
        distillation_load = RESULT_DIR/ opt.distillation_checkpoint / opt.distillation_checkpoint_name
        checkpoint = torch.load(distillation_load, map_location="cpu")
        if "optimizer_student" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer_student"])
            log.info(f"[Opt] Loaded optimizer ckpt {distillation_load}!")
        else:
            log.warning(f"[Opt] Ckpt {distillation_load} has no optimizer!")
            
        if sched is not None and "sched_student" in checkpoint.keys() and checkpoint["sched_student"] is not None:
            sched.load_state_dict(checkpoint["sched_student"])
            log.info(f"[Opt] Loaded lr sched ckpt {distillation_load}!")
        else:
            log.warning(f"[Opt] Ckpt {distillation_load} has no lr sched!")

    if hasattr(opt, "distillation_checkpoint") and opt.distillation_checkpoint and load_bridge_model:
        distillation_load = RESULT_DIR / opt.distillation_checkpoint / opt.distillation_checkpoint_name
        checkpoint = torch.load(distillation_load, map_location="cpu")
        if "optimizer_bridge" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer_bridge"])
            log.info(f"[Opt] Loaded optimizer ckpt {distillation_load}!")
        else:
            log.warning(f"[Opt] Ckpt {distillation_load} has no optimizer!")
        if sched is not None and "sched_bridge" in checkpoint.keys() and checkpoint["sched_bridge"] is not None:
            sched.load_state_dict(checkpoint["sched_bridge"])
            log.info(f"[Opt] Loaded lr sched ckpt {distillation_load}!")
        else:
            log.warning(f"[Opt] Ckpt {distillation_load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        print(f"opt.beta_max = {opt.beta_max}")
        # Original I2SB schedule:
        # betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        # betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        #
        # Use a constant per-step beta so the discrete schedule sums to opt.beta_max.
        # This better matches a bridge-matching style process with total noise budget eps=beta_max.
        betas = np.full(opt.interval, opt.beta_max / opt.interval, dtype=np.float64)
        self.diffusion = Diffusion(betas, opt.device)
        print("loaded diffusion!")
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        opt.cond_x1 = resolve_cond_x1(opt, log)
        pretrained_adm = should_load_pretrained_adm(opt)

        is_debug_mode = getattr(opt, "debug_mode", False)
        NetworkClass = DebugIdentityNet if is_debug_mode else Image256Net
        print(f"opt.use_fp16 = {opt.use_fp16}")
        self.net = NetworkClass(
            log,
            noise_levels=noise_levels,
            use_fp16=opt.use_fp16,
            cond=opt.cond_x1,
            pretrained_adm=pretrained_adm,
        )

        print(f"opt.model_add_noise_input = {opt.model_add_noise_input}")
        if opt.model_add_noise_input:
            self.net.add_noise_channels()
        
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load and not is_debug_mode:  # Only load checkpoint if not in debug mode
            print(f"loading {opt.load}")
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint[opt.model_name_in_ckpt])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema = load_ema_with_compat(
                self.ema,
                self.net,
                checkpoint.get(opt.model_ema_name_in_ckpt),
                log,
                opt.load,
                role="Ema",
            )
        elif is_debug_mode:
            log.info("[Debug Mode] Using identity network - no checkpoint loaded")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log
        self.tensor_layout = getattr(opt, "tensor_layout", "nchw").lower()
        self.pad_to_width = getattr(opt, "pad_to_width", None)
        self.pad_mode = getattr(opt, "pad_mode", "reflect")
        self.last_pad_left = 0
        self.last_pad_right = 0

    def _to_model_layout(self, x):
        if self.tensor_layout == "bcwh":
            return nchw_to_bcwh(x)
        return x

    def _from_model_layout(self, x):
        if self.tensor_layout == "bcwh":
            return bcwh_to_nchw(x)
        return x

    def _prepare_input_tensor(self, x):
        x, pad_left, pad_right = pad_width_to_target(x, self.pad_to_width, mode=self.pad_mode)
        x = self._to_model_layout(x)
        return x, pad_left, pad_right

    def _restore_output_tensor(self, x, pad_left=None, pad_right=None):
        if pad_left is None:
            pad_left = self.last_pad_left
        if pad_right is None:
            pad_right = self.last_pad_right
        x = self._from_model_layout(x)
        return crop_width(x, pad_left, pad_right)

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        # debug()
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if util.is_paired_dataset_mode(opt):
            clean_img, corrupt_img = next(loader)
            mask = None
            y = None
        elif opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()

        y = None if y is None else y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if util.is_paired_dataset_mode(opt):
            x0, pad_left, pad_right = self._prepare_input_tensor(x0)
            x1, x1_pad_left, x1_pad_right = self._prepare_input_tensor(x1)
            if (pad_left, pad_right) != (x1_pad_left, x1_pad_right):
                raise RuntimeError("Clean and corrupt padding offsets differ; expected identical shapes.")
            self.last_pad_left = pad_left
            self.last_pad_right = pad_right
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch, opt.num_workers)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch, opt.num_workers)

        if not util.is_paired_dataset_mode(opt):
            self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000).to(opt.device)
            self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it == 500 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # debug()

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y = all_cat_cpu(opt, log, y) if y is not None else None
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        if util.is_paired_dataset_mode(opt):
            img_clean = self._restore_output_tensor(img_clean)
            img_corrupt = self._restore_output_tensor(img_corrupt)
            xs = self._restore_output_tensor(xs.reshape(-1, *xs.shape[2:])).reshape(xs.shape[0], xs.shape[1], *img_clean.shape[1:])
            pred_x0s = self._restore_output_tensor(pred_x0s.reshape(-1, *pred_x0s.shape[2:])).reshape(pred_x0s.shape[0], pred_x0s.shape[1], *img_clean.shape[1:])

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        if y is not None:
            assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        def log_accuracy(tag, img):
            pred = self.resnet(img.to(opt.device)) # input range [-1,1]
            accu = self.accuracy(pred, y.to(opt.device))
            self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        if y is not None:
            log.info("Logging accuracies ...")
            log_accuracy("accuracy/clean",   img_clean)
            log_accuracy("accuracy/corrupt", img_corrupt)
            log_accuracy("accuracy/recon",   img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
