# Inverse Bridge Matching Distillation

Official PyTorch implementation of **Inverse Bridge Matching Distillation**, accepted to ICML 2025.

**Paper:** [arXiv](https://arxiv.org/abs/2502.01362) | [ICML 2025 Proceedings](https://proceedings.mlr.press/v267/gushchin25b.html) | [ICML 2025 Poster](https://icml.cc/virtual/2025/poster/45134)

This repository provides the implementation of our distillation method for Diffusion Bridge Models, which can be easily integrated into existing training pipelines. We provide implementations for both [I²SB](https://github.com/NVlabs/I2SB) and [DDBM](https://github.com/alexzhou907/DDBM) setups.

## Overview

Inverse Bridge Matching Distillation (IBMD) is a novel distillation technique based on the inverse bridge matching formulation that accelerates Diffusion Bridge Models (DBMs) for image-to-image translation tasks. Unlike previously developed DBM distillation techniques, our method can:

- Distill both conditional and unconditional types of DBMs
- Distill models into a one-step generator
- Use only corrupted images for training
- Achieve inference acceleration from 4x to 100x
- Provide better generation quality than the teacher model in some setups

We evaluate our approach on a wide set of tasks including super-resolution, JPEG restoration, sketch-to-image, and inpainting.

## Repository Structure

- `I2SB/`: Implementation for I²SB (Image-to-Image Schrödinger Bridge) setups
- `DiffusionBridge/`: Implementation for DDBM (Denoising Diffusion Bridge Models) setups

## Installation

### Requirements

To create the environment used in this work, follow the installation instructions in the `DiffusionBridge` folder:

```bash
cd DiffusionBridge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install blobfile piq matplotlib opencv-python joblib lmdb scipy clean-fid easydict torchmetrics rich ipdb
```

For I²SB setups, you can also use the conda environment:

```bash
cd I2SB
conda env create --file requirements.yaml python=3
conda activate i2sb
```

## Datasets

We use the same datasets that were used in the I²SB and DDBM/DBIM papers. Instructions for downloading and preparing the datasets can be found in:
- `I2SB/README.md` for I²SB datasets
- `DiffusionBridge/README.md` for DDBM datasets

## Usage

### I²SB Distillation

To reproduce distillation experiments of I²SB models, first load the corresponding data and checkpoints using the README in the `I2SB` folder, then run the following commands depending on the setup.

#### Super-resolution Models (sr4x-bicubic, sr4x-pool)

```
cd I2SB
python train.py --name sr4x-bicubic --n-gpu-per-node 4 --corrupt jpeg-5 --dataset-dir 'path_to_imagenet' --ckpt 'sr4x-bicubic' \
--distillation --batch-size 256 --microbatch 2 --n-bridge-loop 5 --lr 5e-5 --bridge-pretrain-iters 0 --x0-prediction-loss \
--ema 0.99 --beta-max 0.3 --log-writer 'wandb' --num-workers 32 --port '6069' --student-noise-input --init-student-from-ema
```

```
python train.py --name sr4x-pool --n-gpu-per-node 4 --corrupt jpeg-5 --dataset-dir 'path_to_imagenet' --ckpt 'sr4x-pool' \
--distillation --batch-size 256 --microbatch 2 --n-bridge-loop 5 --lr 5e-5 --bridge-pretrain-iters 0 --x0-prediction-loss \
--ema 0.99 --beta-max 0.3 --log-writer 'wandb' --num-workers 32 --port '6069' --student-noise-input --init-student-from-ema
```

#### JPEG Restoration Models (jpeg-5 and jpeg-10)

```
python train.py --name jpeg_5 --n-gpu-per-node 4 --corrupt jpeg-5 --dataset-dir 'path_to_imagenet' --ckpt 'jpeg-5' \
--distillation --batch-size 256 --microbatch 2 --n-bridge-loop 5 --lr 5e-5 --bridge-pretrain-iters 0 --x0-prediction-loss \
--ema 0.99 --beta-max 0.3 --log-writer 'wandb' --num-workers 32 --port '6069' --student-noise-input --init-student-from-ema
```

```
python train.py --name jpeg_10 --n-gpu-per-node 4 --corrupt jpeg-5 --dataset-dir 'path_to_imagenet' --ckpt 'jpeg-10' \
--distillation --batch-size 256 --microbatch 2 --n-bridge-loop 5 --lr 5e-5 --bridge-pretrain-iters 0 --x0-prediction-loss \
--ema 0.99 --beta-max 0.3 --log-writer 'wandb' --num-workers 32 --port '6069' --student-noise-input --init-student-from-ema
```

#### Inpainting Model

```
python train.py --name inpaint-center --n-gpu-per-node 4 --corrupt inpaint-center --dataset-dir 'path_to_imagenet' \
--ckpt 'inpaint-center' --distillation --batch-size 256 --microbatch 2 --n-bridge-loop 5 --lr 5e-5 \
--bridge-pretrain-iters 0 --x0-prediction-loss --ema 0.99 --beta-max 1.0 --log-writer 'wandb' --num-workers 32 \
--normalize-generator-loss-by-t-power-ten --ot-ode --port '6037' --normalize-generator-loss-by-t-power-ten-coeff 2.0 \
--multistep-student --multistep-student-use-fixed-steps --multistep-student-num-fixed-steps 4 --eval-nfe 4 \
--bridge-use-student-intermediate-steps --multistep-student-full-sampling
```

### DDBM Distillation

To reproduce the DDBM model distillation experiments, first load the required data and checkpoints by following the instructions in the README file in the `DiffusionBridge` directory. Make sure to preprocess both the checkpoint and data. Then, run the following commands from the `DiffusionBridge` directory.

#### DIODE-Outdoor Dataset

```bash
cd DiffusionBridge
bash scripts/train.sh diode
```

#### Edges-Handbag Dataset

```bash
bash scripts/train.sh e2h
```

#### ImageNet Inpainting Dataset

```bash
bash scripts/train.sh imagenet_inpaint_center
```

## Acknowledgments

This codebase is built upon:
- [I²SB](https://github.com/NVlabs/I2SB): Image-to-Image Schrödinger Bridge
- [DDBM](https://github.com/alexzhou907/DDBM): Denoising Diffusion Bridge Models

We thank the authors for making their code publicly available.
