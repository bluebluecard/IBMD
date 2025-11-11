from .train_util import TrainLoop, update_ema, get_blob_logdir
import copy
from . import dist_util, logger
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import functools
import glob 
import os
import blobfile as bf
import wandb
import torchvision.utils as tu
from torch import nn
import numpy as np
from .nn import append_dims
import ipdb

class TrainLoopDistill(TrainLoop):
    def __init__(
        self, 
        *, 
        model, 
        diffusion, 
        train_data, 
        test_data, 
        batch_size, 
        microbatch, 
        lr, 
        ema_rate, 
        log_interval, 
        test_interval, 
        save_interval, 
        save_interval_for_preemption, 
        resume_checkpoint, 
        workdir, 
        use_fp16=False, 
        fp16_scale_growth=0.001, 
        schedule_sampler=None, 
        weight_decay=0, 
        lr_anneal_steps=0, 
        total_training_steps=10000000, 
        augment_pipe=None, 
        n_bridge_loop=1,
        resume_student_checkpoint="",
        add_noise=False,
        noise_channels=None,
        new_lr=0.,
        num_steps=1,
        **sample_kwargs
    ):
        assert resume_checkpoint is not None # checkpoint for the teacher 
        assert num_steps >= 1
        super().__init__(
            model=model, 
            diffusion=diffusion, 
            train_data=train_data, 
            test_data=test_data, 
            batch_size=batch_size, 
            microbatch=microbatch, 
            lr=lr, 
            ema_rate=ema_rate, 
            log_interval=log_interval, 
            test_interval=test_interval, 
            save_interval=save_interval, 
            save_interval_for_preemption=save_interval_for_preemption, 
            resume_checkpoint=resume_checkpoint, 
            workdir=workdir, 
            use_fp16=use_fp16, 
            fp16_scale_growth=fp16_scale_growth, 
            schedule_sampler=schedule_sampler, 
            weight_decay=weight_decay, 
            lr_anneal_steps=lr_anneal_steps, 
            total_training_steps=total_training_steps, 
            augment_pipe=augment_pipe,
            resume_train_flag=False,
            **sample_kwargs
        )
        self.add_noise = add_noise
        self.noise_channels = noise_channels if self.add_noise else None
        self.student_model = self._load_and_sync_parameters_models("student")
        self.bridge_model = self._load_and_sync_parameters_models("bridge")
        self.ema_model = self._load_and_sync_parameters_models("ema")
        self.new_lr = new_lr
        self.ddp_model.eval()
    
        self.resume_student_checkpoint = resume_student_checkpoint
        self._load_models_checkpoints()
        self.opt_student = RAdam(
            self.student_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        self.opt_bridge = RAdam(
            self.bridge_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        if self.step:
            self._load_optimizers("student")
            self._load_optimizers("bridge")
            self.ema_params_student = [self._load_ema_parameters(rate, "student") for rate in self.ema_rate]
            self.ema_params_bridge = [self._load_ema_parameters(rate, "bridge") for rate in self.ema_rate]
        else:
            self.ema_params_student = [
                    copy.deepcopy(list(self.student_model.parameters()))
                    for _ in range(len(self.ema_rate))
                ]
            
            self.ema_params_bridge = [
                    copy.deepcopy(list(self.bridge_model.parameters()))
                    for _ in range(len(self.ema_rate))
                ]
        
        if th.cuda.is_available():
            self.use_ddp = True
            local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_student_model = DDP(
                self.student_model,
                device_ids=[local_rank],
                output_device=local_rank
            )
            
            self.ddp_bridge_model = DDP(
                self.bridge_model,
                device_ids=[local_rank],
                output_device=local_rank
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_student_model = self.student_model
            self.ddp_bridge_model = self.bridge_model
        
        self.n_bridge_loop = n_bridge_loop
        self.timesteps = th.linspace(self.diffusion.t_max, self.diffusion.t_min,  steps=num_steps + 1, device = dist_util.dev())[:-1]
        self.num_steps = num_steps
        
    def _load_ema_parameters(self, rate, model_name):
        model = self.bridge_model if model_name == "bridge" else self.student_model
        ema_params = copy.deepcopy(list(model.parameters()))

        ema_checkpoint = find_ema_checkpoint(self.resume_student_checkpoint, self.step, rate, model_name)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA for {model_name} model from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = [state_dict[name] for name, _ in model.named_parameters()]

            dist.barrier()
        return ema_params
    
    def _load_optimizers(self, model_name):
        opt_checkpoint = bf.join(bf.dirname(self.resume_student_checkpoint), f"opt_{model_name}_{self.step:06}.pt")
        if bf.exists(opt_checkpoint):
            if dist.get_rank() == 0:
                logger.log(f"loading optimizer state for {model_name} model from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            opt = self.opt_bridge if model_name == "bridge" else self.opt_student
            opt.load_state_dict(state_dict)
            if self.new_lr and model_name == "student":
                if dist.get_rank() == 0:
                    logger.log(f"change optimizer LR for {model_name} model to: {self.new_lr}")
                for param_group in opt.param_groups:
                    param_group["lr"] = self.new_lr
            dist.barrier()
        
    def _load_models_checkpoints(self):
        if self.resume_student_checkpoint:
            parts_of_checkpoint_name = self.resume_student_checkpoint.split("_")
            self.step = int((parts_of_checkpoint_name[-1]).split(".")[-2])
            
            if dist.get_rank() == 0:
                logger.log(f"train will start from {self.step} iteration")
                logger.log(f"load student checkpoint: {self.resume_student_checkpoint}")
                
            self.student_model.load_state_dict(th.load(self.resume_student_checkpoint, map_location="cpu"))
            self.student_model.to(dist_util.dev())
            
            parts_of_checkpoint_name[-2] = "bridge"
            bridge_checkpoint = "_".join(parts_of_checkpoint_name)
            if dist.get_rank() == 0:
                logger.log(f"load bridge checkpoint: {bridge_checkpoint}")
            self.bridge_model.load_state_dict(th.load(bridge_checkpoint, map_location="cpu"))
            self.bridge_model.to(dist_util.dev())
            dist.barrier()
            
    
    def _load_and_sync_parameters_models(self, model_name):
        if dist.get_rank() == 0:
            logger.log(f"loading {model_name} model")
            
        model = copy.deepcopy(self.model)
        if self.add_noise and model_name != "bridge":
            if dist.get_rank() == 0:
                logger.log(f"expand input kernel for noise in {model_name} model")
            out_ch, in_ch, kernel_size, _ = model.input_blocks[0][0].weight.shape
            noise_conv = nn.Conv2d(self.noise_channels, out_ch, kernel_size, padding=1).to(dist_util.dev())
            nn.init.zeros_(noise_conv.weight)
            final_conv = nn.Conv2d(self.noise_channels + in_ch, out_ch, kernel_size, padding=1).to(dist_util.dev())
            final_conv.weight.data = th.cat([model.input_blocks[0][0].weight.data, noise_conv.weight.data], dim=1)
            final_conv.bias.data = model.input_blocks[0][0].bias.data
            model.input_blocks[0][0] = final_conv
        dist.barrier()
        return model
    
    def prepare_loader(self, data):
        while True:
            yield from data
            
    def run_loop(self):
        loader = self.prepare_loader(self.data)
        val_loader = self.prepare_loader(self.test_data)
        while True:
            self.ddp_student_model.eval()
            self.ddp_bridge_model.train()
            for _ in range(self.n_bridge_loop):
                batch, cond, _ = next(loader)
                
                if "inpaint" in self.workdir:
                    _, mask, label = _
                else:
                    mask = None
                
                if self.augment is not None:
                    batch, _ = self.augment(batch)
                if isinstance(cond, th.Tensor) and batch.ndim == cond.ndim:
                    xT = cond
                    
                    cond = {'xT': xT}
                else:
                    cond['xT'] = cond['xT']
                
                if mask is not None:
                    cond["mask"] = mask
                    cond["y"] = label
                
                loss, xt, pred_x0, pred_bridge, true_x0, cond = self.run_step(batch, cond)
            if dist.get_rank() == 0:
                wandb.log({"bridge_loss": loss.item()}, self.step)
                self.log_image("train_bridge/pred_x0", pred_x0[:10])
                self.log_image("train_bridge/true_x0", true_x0[:10])
                self.log_image("train_bridge/cond", cond[:10])
                self.log_image("train_bridge/xt", xt[:10])
                self.log_image("train_bridge/pred_bridge", pred_bridge[:10])
                grad_norm_bridge, param_norm_bridge = self._compute_norms("bridge")
                wandb.log({"bridge_grad_norm": grad_norm_bridge}, self.step)
                wandb.log({"bridge_param_norm": param_norm_bridge}, self.step)
            self.ddp_student_model.train()
            self.ddp_bridge_model.eval()

            batch, cond, _ = next(loader)
            
            if "inpaint" in self.workdir:
                _, mask, label = _
            else:
                mask = None
            
            if self.augment is not None:
                batch, _ = self.augment(batch)
            if isinstance(cond, th.Tensor) and batch.ndim == cond.ndim:
                xT = cond
                
                cond = {'xT': xT}
            else:
                cond['xT'] = cond['xT']
                
            if mask is not None:
                cond["mask"] = mask
                cond["y"] = label
                
            loss, xt, pred_teacher, pred_bridge = self.run_step_student(batch, cond)      
                
            if self.step % self.save_interval == 0:
                self.ddp_student_model.eval()
                self.save_student()
                if dist.get_rank() == 0:
                    self.run_test_step(val_loader)
                self.ddp_student_model.train()
                dist.barrier()
            
            self.step += 1
            
    @th.no_grad()
    def predict_x0_no_grad(self, cond, model):
        xT = cond['xT'] 
        
        cond_arr, result_arr = [], []
        for i in range(0, xT.shape[0], self.microbatch):
            micro = xT[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            ones = th.ones((micro.shape[0],), device=dist_util.dev())
            if self.add_noise:
                bs, channels, h, w = micro.shape
                noise_input = th.randn(bs, self.noise_channels, h, w, dtype=micro.dtype, device=micro.device)
                micro_cond['xT'] = th.cat([micro_cond['xT'], noise_input], dim = 1)
            mask = micro_cond.pop('mask', None)
            x = micro.clone()
            for i in range(len(self.timesteps) - 1):
                if i == 0:
                    s = self.timesteps[i]
                    t = self.timesteps[i + 1]
                    _, x0_hat, _ = self.diffusion.denoise(model, x, s * ones, **micro_cond)
                    if mask is not None:
                        x0_hat = x0_hat * mask + micro * (1 - mask)
                    noise = th.randn_like(x0_hat)
                    x = self.diffusion.bridge_sample(x0_hat, micro, t * ones, noise)
                else:
                    s = self.timesteps[i]
                    t = self.timesteps[i + 1]

                    _, x0_hat, _ = self.diffusion.denoise(model, x, s * ones, **micro_cond)
                    if mask is not None:
                        x0_hat = x0_hat * mask + micro * (1 - mask)

                    a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_abc(s * ones)]
                    a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_abc(t * ones)]

                    _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_alpha_rho(s * ones)]
                    alpha_t, _, rho_t, _ = [
                        append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_alpha_rho(t * ones)
                    ]

                    omega_st = (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
                    tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
                    coeff_xs = tmp_var
                    coeff_x0_hat = b_t - tmp_var * b_s
                    coeff_xT = a_t - tmp_var * a_s

                    noise = th.randn_like(x0_hat)

                    x = coeff_x0_hat * x0_hat + coeff_xT * micro + coeff_xs * x + omega_st * noise
            _, x0_hat, _ = self.diffusion.denoise(model, x, self.timesteps[-1] * ones, **micro_cond)
            if mask is not None:
                x0_hat = x0_hat*mask + micro*(1-mask)
            cond_arr.append(micro), result_arr.append(x0_hat)
        th_cond, th_result = th.cat(cond_arr, 0), th.cat(result_arr, 0)
        return th_cond, th_result
    
    def _load_ema_weights(self, params):
        state_dict = self.model.state_dict()
        for i, (name, _) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = params[i]
        self.ema_model.load_state_dict(state_dict)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    @th.no_grad()
    def run_test_step(self, val_loader):
        batch, cond, _ = next(val_loader)
        
        if "inpaint" in self.workdir:
            _, mask, label = _
        else:
            mask = None
                
        if self.augment is not None:
            batch, _ = self.augment(batch)
        if isinstance(cond, th.Tensor) and batch.ndim == cond.ndim:
            xT = cond
            
            cond = {'xT': xT}
        else:
            cond['xT'] = cond['xT']
        
        if mask is not None:
            cond["mask"] = mask
            cond["y"] = label
        
        new_cond, pred_batch = self.predict_x0_no_grad(cond, self.ddp_student_model)
        self.log_image("val/pred_x0", pred_batch[:10])
        self.log_image("val/true_x0", batch[:10])
        self.log_image("val/cond", new_cond[:10])
        
        for rate, params in zip(self.ema_rate, self.ema_params_student):
            self._load_ema_weights(params)
            
            new_cond, pred_batch = self.predict_x0_no_grad(cond, self.ema_model)
            self.log_image(f"val_ema_{rate}/pred_x0", pred_batch[:10])
            self.log_image(f"val_ema_{rate}/true_x0", batch[:10])
            self.log_image(f"val_ema_{rate}/cond", new_cond[:10])
        
    def log_image(self, key, image):
        image = tu.make_grid((image+1)/2, nrow=10).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", th.uint8).numpy()
        wandb.log({key: wandb.Image(image)}, step=self.step)
    def save_student(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):06d}')[0]+'*'
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)
                    

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params, model_name):
            model = self.bridge_model if model_name == "bridge" else self.student_model
            state_dict = model.state_dict()
            for i, (name, _) in enumerate(model.named_parameters()):
                assert name in state_dict
                state_dict[name] = params[i]
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_{model_name}_{(self.step):06d}.pt"
                else:
                    filename = f"ema_{model_name}_{rate}_{(self.step):06d}.pt"
                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params_bridge):
            save_checkpoint(rate, params, "bridge")
        
        for rate, params in zip(self.ema_rate, self.ema_params_student):
            save_checkpoint(rate, params, "student")

        if dist.get_rank() == 0:
            filename_bridge, filename_student  = f"opt_bridge_{(self.step):06d}.pt", f"opt_student_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)
                
            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename_bridge),
                "wb",
            ) as f:
                th.save(self.opt_bridge.state_dict(), f)
            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename_student),
                "wb",
            ) as f:
                th.save(self.opt_student.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, list(self.bridge_model.parameters()), "bridge")
        save_checkpoint(0, list(self.student_model.parameters()), "student")
        dist.barrier()
    
    def _compute_norms(self, model_name):
        grad_norm = 0.0
        param_norm = 0.0
        model = self.bridge_model if model_name == "bridge" else self.student_model
        for p in model.parameters():
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm), np.sqrt(param_norm)
    
    def run_step(self, batch, cond):
        loss, xt, pred_x0, pred_bridge, true_x0, cond = self.forward_backward(batch, cond)
        self.scaler.step(self.opt_bridge)
        self.scaler.update()
        self._update_ema_bridge()
        self._anneal_lr_bridge()
        return loss, xt, pred_x0, pred_bridge, true_x0, cond
    
    def run_step_student(self, batch, cond):
        loss, xt, pred_teacher, pred_bridge = self.forward_backward_student(batch, cond)
        if dist.get_rank() == 0:
            grad_norm_student, param_norm_student = self._compute_norms("student")
            wandb.log({"student_grad_norm": grad_norm_student}, self.step)
            wandb.log({"student_param_norm": param_norm_student}, self.step)
            
        self.scaler.step(self.opt_student)
        self.scaler.update()
        self._update_ema_student()
        self._anneal_lr_student()
        return loss, xt, pred_teacher, pred_bridge
    
    def _update_ema_bridge(self):
        for rate, params in zip(self.ema_rate, self.ema_params_bridge):
            update_ema(params, self.bridge_model.parameters(), rate=rate)
            
    def _update_ema_student(self):
        for rate, params in zip(self.ema_rate, self.ema_params_student):
            update_ema(params, self.student_model.parameters(), rate=rate)
    
    def _anneal_lr_student(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt_student.param_groups:
            param_group["lr"] = lr
            
    def _anneal_lr_bridge(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt_bridge.param_groups:
            param_group["lr"] = lr
    
    @th.no_grad()
    def sample_backward(self, micro_xT, student_micro_cond, mask):
        batch_size = micro_xT.shape[0]
        ones = th.ones((batch_size,), device=dist_util.dev())
        
        selected_step = th.randint(low=0, high=self.num_steps, size=(1,), dtype=th.long, device=dist_util.dev())
        dist.broadcast(selected_step, 0)
        micro_xT = micro_xT.to(th.float64)
        x = micro_xT.clone()
        
        for i in range(len(self.timesteps[:selected_step])):
            if i == 0:
                s = self.timesteps[i]
                t = self.timesteps[i + 1]
                _, x0_hat, _ = self.diffusion.denoise(self.ddp_student_model, x, s * ones, **student_micro_cond)
                if mask is not None:
                    x0_hat = x0_hat * mask + micro_xT * (1 - mask)
                noise = th.randn_like(x0_hat)
                x = self.diffusion.bridge_sample(x0_hat, micro_xT, t * ones, noise)
            else:
                s = self.timesteps[i]
                t = self.timesteps[i + 1]

                _, x0_hat, _ = self.diffusion.denoise(self.ddp_student_model, x, s * ones, **student_micro_cond)
                if mask is not None:
                    x0_hat = x0_hat * mask + micro_xT * (1 - mask)

                a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_abc(s * ones)]
                a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_abc(t * ones)]

                _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_alpha_rho(s * ones)]
                alpha_t, _, rho_t, _ = [
                    append_dims(item, x0_hat.ndim) for item in self.diffusion.noise_schedule.get_alpha_rho(t * ones)
                ]

                omega_st = (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
                tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
                coeff_xs = tmp_var
                coeff_x0_hat = b_t - tmp_var * b_s
                coeff_xT = a_t - tmp_var * a_s

                noise = th.randn_like(x0_hat)

                x = coeff_x0_hat * x0_hat + coeff_xT * micro_xT + coeff_xs * x + omega_st * noise
        return_timesteps = self.timesteps[selected_step] * th.ones(batch_size, device=dist_util.dev())
        pure_xT_mask = (return_timesteps == self.diffusion.t_max)
        x[pure_xT_mask] = micro_xT[pure_xT_mask]
        return x, return_timesteps
        
    def forward_backward(self, batch, cond, train=True):
        if train:
            self.opt_bridge.zero_grad()
        assert batch.shape[0] % self.microbatch == 0
        num_microbatches = batch.shape[0] // self.microbatch
        for i in range(0, batch.shape[0], self.microbatch):
            with th.autocast(device_type="cuda", dtype=th.float16, enabled=self.use_fp16):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i : i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                micro_xT = micro_cond['xT']
                student_micro_cond = micro_cond.copy()
                if self.add_noise:
                    bs, channels, h, w = micro_xT.shape
                    noise_input = th.randn(bs, self.noise_channels, h, w, dtype=micro_xT.dtype, device=micro_xT.device)
                    student_micro_cond['xT'] = th.cat([student_micro_cond['xT'], noise_input], dim = 1)
                mask = student_micro_cond.pop('mask', None)
                x, timesteps = self.sample_backward(micro_xT, student_micro_cond, mask)
                with th.no_grad():
                    _, pred_micro, _ = self.diffusion.denoise(self.ddp_student_model, x, timesteps, **student_micro_cond)
                if mask is not None:
                    pred_micro = pred_micro*mask + micro_xT*(1-mask)
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                        self.diffusion.training_bridge_losses,
                        self.ddp_bridge_model,
                        pred_micro.detach(),
                        t,
                        model_kwargs=micro_cond,
                    )

                if last_batch or not self.use_ddp:
                    losses, xt, denoised = compute_losses()
                else:
                    with self.ddp_bridge_model.no_sync():
                        losses, xt, denoised = compute_losses() 

                loss = (losses["loss"] * weights).mean() / num_microbatches
            if train:
                self.scaler.scale(loss).backward()
    
        return loss, xt, pred_micro, denoised, micro, micro_xT

    def forward_backward_student(self, batch, cond, train=True):
        if train:
            self.opt_student.zero_grad()
        assert batch.shape[0] % self.microbatch == 0
        num_microbatches = batch.shape[0] // self.microbatch
        for i in range(0, batch.shape[0], self.microbatch):
            with th.autocast(device_type="cuda", dtype=th.float16, enabled=self.use_fp16):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i : i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                micro_xT = micro_cond['xT']
                student_micro_cond = micro_cond.copy()
                if self.add_noise:
                    bs, channels, h, w = micro_xT.shape
                    noise_input = th.randn(bs, self.noise_channels, h, w, dtype=micro_xT.dtype, device=micro_xT.device)
                    student_micro_cond['xT'] = th.cat([student_micro_cond['xT'], noise_input], dim = 1)
                mask = student_micro_cond.pop('mask', None)
                x, timesteps = self.sample_backward(micro_xT, student_micro_cond, mask)
                _, pred_micro, _ = self.diffusion.denoise(self.ddp_student_model, x, timesteps, **student_micro_cond)
                if mask is not None:
                    pred_micro = pred_micro*mask + micro_xT*(1-mask)
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                        self.diffusion.training_student_losses,
                        self.ddp_model,
                        self.ddp_bridge_model,
                        pred_micro,
                        t,
                        model_kwargs=micro_cond,
                    )

                if last_batch or not self.use_ddp:
                    losses, xt, pred_teacher, pred_bridge = compute_losses()
                else:
                    with self.ddp_student_model.no_sync():
                        losses, xt, pred_teacher, pred_bridge = compute_losses()

                loss = (losses["loss"] * weights * 10**(1-t)).mean() / num_microbatches 
            if train:
                self.scaler.scale(loss).backward()
        if dist.get_rank() == 0:
            wandb.log({"student_loss": loss.item()}, self.step)
            self.log_image("train_student/pred_x0", pred_micro[:10])
            self.log_image("train_student/true_x0", micro[:10])
            self.log_image("train_student/cond", micro_xT[:10])
            self.log_image("train_student/xt", xt[:10])
            self.log_image("train_student/pred_teacher", pred_teacher[:10])
            self.log_image("train_student/pred_bridge", pred_bridge[:10])  
        return loss, xt, pred_teacher, pred_bridge
    

def find_ema_checkpoint(main_checkpoint, step, rate, model_name):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split("/")[-1].startswith("freq"):
        prefix = "freq_"
    else:
        prefix = ""
    filename = f"{prefix}ema_{model_name}_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None