# import os
# import numpy as np
# import pickle

from pathlib import Path

import torch
import torch.nn.functional as F
# from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

# import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net, DebugIdentityNet
# from .diffusion import Diffusion

from .runner import Runner, all_cat_cpu, build_optimizer_sched
from torch.optim import AdamW, lr_scheduler

from ipdb import set_trace as debug

# def build_new_optimizer_sched(opt, net, log):
#     optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
#     optimizer = AdamW(net.parameters(), **optim_dict)
#     log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

#     if opt.lr_gamma < 1.0:
#         sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
#         sched = lr_scheduler.StepLR(optimizer, **sched_dict)
#         log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
#     else:
#         sched = None

#     return optimizer, sched


class RunnerDistill(Runner):
    def __init__(self, opt, log, save_opt=True):
        super().__init__(opt, log, save_opt=save_opt)

        # # Save opt.
        # if save_opt:
        #     opt_pkl_path = opt.ckpt_path / "options.pkl"
        #     with open(opt_pkl_path, "wb") as f:
        #         pickle.dump(opt, f)
        #     log.info("Saved options pickle to {}!".format(opt_pkl_path))

        # betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        # betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        # self.diffusion = Diffusion(betas, opt.device)
        # log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")
        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

        NetworkClass = DebugIdentityNet if hasattr(opt, "debug_mode") else Image256Net
        
        self.bridge_model = NetworkClass(log, noise_levels=noise_levels,
                                          use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.bridge_model_ema = ExponentialMovingAverage(self.bridge_model.parameters(), decay=opt.ema)

        self.student = NetworkClass(log, noise_levels=noise_levels,
                                     use_fp16=opt.use_fp16, cond=opt.cond_x1,
                                     student_noise_input=opt.student_noise_input)
        
        self.student_ema = ExponentialMovingAverage(self.student.parameters(), decay=opt.ema)
        
        RESULT_DIR = Path("results")

        if not hasattr(opt, "debug_mode"):  # Only load checkpoints if not in debug mode
            if hasattr(opt, "distillation_checkpoint") and opt.distillation_checkpoint:
                distillation_load = RESULT_DIR / opt.distillation_checkpoint / opt.distillation_checkpoint_name
                checkpoint = torch.load(distillation_load, map_location="cpu")
                self.bridge_model.load_state_dict(checkpoint['bridge_model'])
                log.info(f"[Bridge Model] Loaded network ckpt: {distillation_load}!")
                self.bridge_model_ema.load_state_dict(checkpoint["bridge_model_ema"])
                log.info(f"[Bridge Model Ema] Loaded ema ckpt: {distillation_load}!")

                # checkpoint = torch.load(distillation_load, map_location="cpu")
                self.student.load_state_dict(checkpoint['student'])
                log.info(f"[Student] Loaded network ckpt: {distillation_load}!")
                self.student_ema.load_state_dict(checkpoint["student_ema"])
                log.info(f"[Student Ema] Loaded ema ckpt: {distillation_load}!")
            else:
                checkpoint = torch.load(opt.load, map_location="cpu")
                self.bridge_model.load_state_dict(checkpoint['net'])
                log.info(f"[Bridge Model] Loaded network ckpt: {opt.load}!")
                self.bridge_model_ema.load_state_dict(checkpoint["ema"])
                log.info(f"[Bridge Model Ema] Loaded ema ckpt: {opt.load}!")

                # checkpoint = torch.load(opt.load, map_location="cpu")
                self.student.load_state_dict(checkpoint['net'])
                log.info(f"[Student] Loaded network ckpt: {opt.load}!")
                self.student_ema.load_state_dict(checkpoint["ema"])
                log.info(f"[Student Ema] Loaded ema ckpt: {opt.load}!")

                if opt.init_student_from_ema:
                    self.student_ema.copy_to(self.student.parameters())
        else:
            log.info("[Debug Mode] Using identity networks - no checkpoints loaded")

        if opt.student_noise_input:
            self.student.add_noise_channels()
            self.student_ema_new = ExponentialMovingAverage(self.student.parameters(), decay=opt.ema)
            self.student_ema_new.num_updates = self.student_ema.num_updates
            self.student_ema = self.student_ema_new

        self.student.to(opt.device)
        self.student_ema.to(opt.device)

        self.bridge_model.to(opt.device)
        self.bridge_model_ema.to(opt.device)

    def compute_label(self, step, x0, xt, detach=True):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd

        if detach:
            label = label.detach()

        return label

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        # net = DataParallel(self.net, device_ids=[i for i in range(opt.n_gpu_per_node)])
        ema = self.ema

        bridge_model = DDP(self.bridge_model, device_ids=[opt.device])
        # bridge_model = DataParallel(self.bridge_model, device_ids=[i for i in range(opt.n_gpu_per_node)])
        bridge_model_ema = self.bridge_model_ema

        student = DDP(self.student, device_ids=[opt.device])
        # student = DataParallel(self.student, device_ids=[i for i in range(opt.n_gpu_per_node)])
        student_ema = self.student_ema

        if not hasattr(opt, "distillation_checkpoint"):
            optimizer_bridge, sched_bridge = build_optimizer_sched(opt, bridge_model, log)
            optimizer_student, sched_student = build_optimizer_sched(opt, student, log)
        else:
            optimizer_bridge, sched_bridge = build_optimizer_sched(opt, bridge_model, log, load_bridge_model=True)
            optimizer_student, sched_student = build_optimizer_sched(opt, student, log, load_student=True)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000).to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        bridge_model.train()
        student.train()
        net.eval()

        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        n_bridge_loop = opt.n_bridge_loop

        for it in range(opt.num_itr):
            bridge_model.train()
            student.eval()

            for _ in range(n_bridge_loop):
                optimizer_bridge.zero_grad()
                for __ in range(n_inner_loop):
                    # ===== sample boundary pair =====
                    x0_original, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)
                    with torch.no_grad():
                        # with student_ema.average_parameters():
                        student.eval()

                        if opt.multistep_student_full_sampling:
                            x1 = x1.to(opt.device)
                            xs, pred_x0s = self.ddpm_sampling(
                                opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=False, 
                                nfe=opt.multistep_student_num_fixed_steps, train_mode=True
                            )
                            x0 = xs[:, 0, ...]
                            net_out = pred_x0s[:, 0, ...]
                        else:
                            step_student = torch.full((x1.shape[0],), opt.interval-1, device=opt.device, dtype=torch.long)
                            net_out = student(x1, step_student, cond=cond)
                            x0 = self.compute_pred_x0(step_student, x1, net_out, clip_denoise=False)

                            if mask is not None:
                                x0 = (1. - mask) * x0_original + mask * x0

                        if opt.bridge_use_student_intermediate_steps and opt.multistep_student:
                            net_out = net_out.detach()
                            x0 = x0.detach()

                            if opt.multistep_student_use_fixed_steps:
                                steps = util.space_indices(opt.interval, opt.multistep_student_num_fixed_steps+1)
                                steps = torch.tensor(steps, device=opt.device)
                                step_indices = torch.randint(1, opt.multistep_student_num_fixed_steps+1, (x0.shape[0],), device=opt.device)
                                intermediate_step_student = steps[step_indices].long()
                            else:
                                intermediate_step_student = torch.randint(1, opt.interval, (x0.shape[0],), device=opt.device)
                            xt = self.diffusion.q_sample(intermediate_step_student, x0, x1, ot_ode=opt.ot_ode)
                            xt[intermediate_step_student == opt.interval-1] = x1[intermediate_step_student == opt.interval-1]

                            net_out = student(xt, intermediate_step_student, cond=cond)
                            x0 = self.compute_pred_x0(intermediate_step_student, xt, net_out, clip_denoise=False)

                            if mask is not None:
                                x0 = (1. - mask) * x0_original + mask * x0

                    # ===== compute loss =====
                    if opt.bridge_use_student_intermediate_steps and opt.multistep_student:
                        # Use the same intermediate steps and xt as the student
                        step = (torch.rand(x0.shape[0], device=opt.device) * intermediate_step_student).long()
                        xt = self.diffusion.p_posterior(step, intermediate_step_student, xt, x0, ot_ode=opt.ot_ode)
                    else:
                        step = torch.randint(0, opt.interval-1, (x0.shape[0],), device=opt.device)
                        xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                    
                    label = self.compute_label(step, x0, xt)

                    pred = bridge_model(xt, step, cond=cond)
                    assert xt.shape == label.shape == pred.shape

                    pred_x0  = self.compute_pred_x0(step, xt, pred, clip_denoise=False)

                    x0_loss = x0

                    if mask is not None:
                        pred = mask * pred
                        label = mask * label
                        pred_x0 = mask * pred_x0

                        x0_loss = mask*x0_loss
                        
                    if opt.x0_prediction_loss:
                        loss = F.mse_loss(pred_x0, x0_loss, reduction='none')
                    else:
                        loss = F.mse_loss(pred, label, reduction='none')

                    if opt.normalize_loss_by_loss:
                        loss = (loss/(loss.detach()+1e-8))
                    elif opt.normalize_bridge_loss_by_t_power_ten:
                        loss = loss * (10**(1 - step[:, None, None, None]/1000))

                    loss = loss.mean()
                    loss.backward()

                optimizer_bridge.step()
                bridge_model_ema.update()
                if sched_bridge is not None: sched_bridge.step()

            if it % 1 == 0:
                self.writer.add_scalar(it, 'bridge_loss', loss.detach())

            optimizer_student.zero_grad()

            bridge_model.eval()
            student.train()

            if it*n_bridge_loop < opt.bridge_pretrain_iters:
                continue

            for __ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0_original, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)
                
                if opt.multistep_student_full_sampling:
                    x1 = x1.to(opt.device)
                    xs, pred_x0s = self.ddpm_sampling(
                        opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=False, 
                        nfe=opt.multistep_student_num_fixed_steps, train_mode=True
                    )
                    x0 = xs[:, 0, ...]
                    net_out = pred_x0s[:, 0, ...]
                else:
                    step_student = torch.full((x1.shape[0],), opt.interval-1, device=opt.device, dtype=torch.long)
                    net_out = student(x1, step_student, cond=cond)
                    x0 = self.compute_pred_x0(step_student, x1, net_out, clip_denoise=False)

                    if mask is not None:
                        x0 = (1. - mask) * x0_original + mask * x0

                if opt.multistep_student:
                    net_out = net_out.detach()
                    x0 = x0.detach()

                    if opt.multistep_student_use_fixed_steps:
                        steps = util.space_indices(opt.interval, opt.multistep_student_num_fixed_steps+1)
                        steps = torch.tensor(steps, device=opt.device)
                        step_indices = torch.randint(1, opt.multistep_student_num_fixed_steps+1, (x0.shape[0],), device=opt.device)
                        intermediate_step_student = steps[step_indices].long()
                    else:
                        intermediate_step_student = torch.randint(1, opt.interval, (x0.shape[0],), device=opt.device)
                    xt = self.diffusion.q_sample(intermediate_step_student, x0, x1, ot_ode=opt.ot_ode)
                    xt[intermediate_step_student == opt.interval-1] = x1[intermediate_step_student == opt.interval-1]

                    net_out = student(xt, intermediate_step_student, cond=cond)
                    x0 = self.compute_pred_x0(intermediate_step_student, xt, net_out, clip_denoise=False)

                    if mask is not None:
                        x0 = (1. - mask) * x0_original + mask * x0
                    

                # ===== compute loss =====
                if opt.multistep_student:
                    # Generate random values in [0,1) and multiply by intermediate_step_student
                    rand = torch.rand(x0.shape[0], device=opt.device)
                    step = (rand * intermediate_step_student).long()
                    xt = self.diffusion.p_posterior(step, intermediate_step_student, xt, x0, ot_ode=opt.ot_ode)
                else:
                    step = torch.randint(0, opt.interval-1, (x0.shape[0],), device=opt.device)
                    xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)

                with ema.average_parameters():
                    net.eval()
                    pred = net(xt, step, cond=cond)
                    pred_x0  = self.compute_pred_x0(step, xt, pred, clip_denoise=False)

                # with bridge_model_ema.average_parameters():
                bridge_model.eval()
                pred_fake = bridge_model(xt, step, cond=cond)
                pred_x0_fake = self.compute_pred_x0(step, xt, pred_fake, clip_denoise=False)
                
                assert xt.shape == label.shape == pred.shape == pred_fake.shape

                x0_loss = x0
                if mask is not None:
                    pred = mask * pred
                    pred_fake = mask * pred_fake
                    label = mask * label

                    pred_x0 = mask * pred_x0
                    pred_x0_fake = mask * pred_x0_fake
                    x0_loss = mask * x0_loss

                if opt.x0_prediction_loss:
                    true_loss = F.mse_loss(pred_x0, x0_loss, reduction='none')
                    fake_loss = F.mse_loss(pred_x0_fake, x0_loss, reduction='none')
                else:
                    true_loss = F.mse_loss(pred, label, reduction='none')
                    fake_loss = F.mse_loss(pred_fake, label, reduction='none')

                if opt.normalize_loss_by_loss:
                    true_loss = (true_loss/(true_loss.detach()+1e-8))
                    fake_loss = (fake_loss/(fake_loss.detach()+1e-8))

                if opt.normalize_generator_loss_by_t_power_ten:
                    true_loss = true_loss * (10**(opt.normalize_generator_loss_by_t_power_ten_coeff*(1 - step[:, None, None, None]/1000)))
                    fake_loss = fake_loss * (10**(opt.normalize_generator_loss_by_t_power_ten_coeff*(1 - step[:, None, None, None]/1000)))
                    
                if opt.normalize_generator_loss_by_teacher_l1_loss:
                    true_l1 = torch.mean(torch.abs(pred_x0 - x0_loss), dim=(1,2,3)).detach()
                    true_loss = true_loss / (true_l1[:, None, None, None] + 1e-8)
                    fake_loss = fake_loss / (true_l1[:, None, None, None] + 1e-8)
                loss = (true_loss - fake_loss).mean()

                fake_bridge_matching_convergence_condition = torch.mean(pred_x0_fake**2) - torch.mean(pred_x0_fake*x0_loss)
                teacher_bridge_matching_convergence_condition = torch.mean(pred_x0**2) - torch.mean(pred_x0*x0_loss)

                loss.backward()

            optimizer_student.step()
            student_ema.update()
            if sched_student is not None: sched_student.step()

            if it % 2 == 0:
                self.writer.add_image(it, "x_0_data/pred_x0", tu.make_grid((pred_x0+1)/2, nrow=10)) # [1,1] -> [0,1]
                self.writer.add_image(it, "x_0_data/pred_x0_fake", tu.make_grid((pred_x0_fake+1)/2, nrow=10)) # [1,1] -> [0,1]
                self.writer.add_image(it, "x_0_data/x0", tu.make_grid((x0+1)/2, nrow=10)) # [1,1] -> [0,1]

                self.writer.add_image(it, "input/xt", tu.make_grid((xt+1)/2, nrow=10)) # [1,1] -> [0,1]
                self.writer.add_image(it, "input/x1", tu.make_grid((x1+1)/2, nrow=10)) # [1,1] -> [0,1]
                self.writer.add_image(it, "input/x0_original", tu.make_grid((x0_original+1)/2, nrow=10)) # [1,1] -> [0,1]

                self.writer.add_image(it, "label_data/pred", tu.make_grid((pred+1)/2, nrow=10)) # [1,1] -> [0,1]
                self.writer.add_image(it, "label_data/pred_fake", tu.make_grid((pred_fake+1)/2, nrow=10)) # [1,1] -> [0,1]
                self.writer.add_image(it, "label_data/label", tu.make_grid((label+1)/2, nrow=10)) # [1,1] -> [0,1]

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer_student.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 1 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())
                self.writer.add_scalar(it, 'fake_bridge_matching_convergence_condition', fake_bridge_matching_convergence_condition.detach())
                self.writer.add_scalar(it, 'teacher_bridge_matching_convergence_condition', teacher_bridge_matching_convergence_condition.detach())
            
            if it % 100 == 0:
                if opt.global_rank == 0:
                    save_dict = {
                        "student": self.student.state_dict(),
                        "student_ema": student_ema.state_dict(),
                    }
                                        
                    latest_save_dict = {
                        "student": self.student.state_dict(),
                        "student_ema": student_ema.state_dict(),
                        "bridge_model": self.bridge_model.state_dict(),
                        "bridge_model_ema": bridge_model_ema.state_dict(),
                        "optimizer_student": optimizer_student.state_dict(),
                        "sched_student": sched_student.state_dict() if sched_student is not None else sched_student,
                        "optimizer_bridge": optimizer_bridge.state_dict(),
                        "sched_bridge": sched_bridge.state_dict() if sched_bridge is not None else sched_bridge,
                    }
                    
                    # Find unique filename for checkpoint
                    base_filename = f"latest_{it}"
                    counter = 0
                    while (opt.ckpt_path / f"{base_filename}_{counter}.pt").exists():
                        counter += 1
                    checkpoint_path = opt.ckpt_path / f"{base_filename}_{counter}.pt"
                    
                    # Save checkpoint with unique name and latest.pt
                    torch.save(save_dict, checkpoint_path)
                    torch.save(latest_save_dict, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % 10 == 0: # 0, 0.5k, 3k, 6k 9k
                student.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                student.train()

        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=1, log_count=10, verbose=True, train_mode=False):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        # nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)

        if train_mode:
            log_steps = [0]
        else:
            log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
            assert log_steps[0] == 0
            self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.student_ema.average_parameters():
            self.student.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.student(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose, train_mode=train_mode
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
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0, nfe=opt.eval_nfe
        )

        # xs = xs.contiguous()
        # pred_x0s = pred_x0s.contiguous()

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
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

        log.info("Logging accuracies ...")
        log_accuracy("accuracy/clean",   img_clean)
        log_accuracy("accuracy/corrupt", img_corrupt)
        log_accuracy("accuracy/recon",   img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
