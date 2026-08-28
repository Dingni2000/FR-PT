# FR-PT for NAVSIM

This repository contains the reproducible **offline FR-PT** experiments built on top of [NAVSIM](https://github.com/autonomousvision/navsim). The public workflow focuses on two camera-based agents:

| Agent ID | Backbone | Configuration |
| --- | --- | --- |
| `resnet_agent` | ResNet-34 | `navsim/planning/script/config/common/agent/resnet_agent.yaml` |
| `vit_agent` | ViT-B/16 | `navsim/planning/script/config/common/agent/vit_agent.yaml` |

The implementation, configuration, and command-line names use the same public terminology. The ResNet and ViT modules are located under `navsim/agents/resnet_agent/` and `navsim/agents/vit_agent/` respectively.

Only the offline pipeline is documented here. Online FR-PT code is not required for reproducing the paper experiments.

## Contents

- [Installation](#installation)
- [Dataset and experiment layout](#dataset-and-experiment-layout)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Offline FR-PT workflow](#offline-frpt-workflow)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

## Installation

The code targets Python 3.9 and a CUDA-enabled PyTorch environment.

```bash
conda env create -f environment.yml
conda activate navsim

cd /path/to/navsim
pip install -e .
source scripts/env.sh
```

`scripts/env.sh` is the single entry point for path configuration. It derives the repository root from its own location and provides portable defaults for data and experiment outputs.

## Dataset and experiment layout

The default layout places the dataset and experiment directory next to this checkout:

```text
<workspace>/
├── navsim/                         # this repository
├── dataset/
│   ├── navsim_logs/
│   │   ├── trainval/
│   │   └── test/
│   ├── sensor_blobs/
│   │   ├── trainval/
│   │   └── test/
│   └── maps/
└── exp/                            # caches, checkpoints and results
```

When the data is stored elsewhere, override the variables before running any job:

```bash
export OPENSCENE_DATA_ROOT=/path/to/dataset
export NAVSIM_EXP_ROOT=/path/to/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

The split mapping used by NAVSIM is:

- `navtrain` → `trainval`
- `navtest` → `test`

The metric cache is independent of model weights and can be reused across all evaluations.

The public agent/module rename does not change the feature or target cache schema. Existing caches can therefore be reused when their dataset split and preprocessing settings match.

## Configuration

The main path and artifact variables are defined in [`scripts/env.sh`](scripts/env.sh):

```text
FRPT_RESNET34_IMAGENET_CKPT   ImageNet ResNet-34 checkpoint for base training
FRPT_VIT_B16_IMAGENET_CKPT    ImageNet ViT-B/16 checkpoint for base training
FRPT_RESNET34_CHECKPOINT      trained ResNet-34 NAVSIM checkpoint
FRPT_VIT_B16_CHECKPOINT       trained ViT-B/16 NAVSIM checkpoint
FRPT_RESNET34_H5_PATH         offline ResNet-34 feature cache
FRPT_VIT_B16_H5_PATH          offline ViT-B/16 feature cache
FRPT_RESNET34_WORK_DIR        ResNet-34 post-training output directory
FRPT_VIT_B16_WORK_DIR         ViT-B/16 post-training output directory
NAVSIM_METRIC_CACHE           NAVSIM metric cache directory
```

The ImageNet checkpoints are needed only for base model training. Offline FR-PT starts from a complete NAVSIM checkpoint and disables the separate ImageNet load automatically.

## Quick start

From the repository root:

```bash
source scripts/env.sh

# Evaluate one already-trained base checkpoint.
CHECKPOINT=/path/to/resnet_base.ckpt GPU_IDS=0 \
  ./run_resnet_agent_navtest_eval.sh

CHECKPOINT=/path/to/vit_base.ckpt GPU_IDS=0 \
  ./run_vit_agent_navtest_eval.sh
```

Each evaluation wrapper first creates or reuses the `navtest` metric cache and then runs NAVSIM PDM scoring.

## Offline FR-PT workflow

The complete workflow is intentionally split into independent stages so that each stage can be resumed and inspected.

### 1. Train the base agents

Set the ImageNet checkpoint variables first:

```bash
export FRPT_RESNET34_IMAGENET_CKPT=/path/to/resnet34_imagenet.pth
export FRPT_VIT_B16_IMAGENET_CKPT=/path/to/vit_b16_imagenet.pth
```

Train ResNet-34:

```bash
CUDA_VISIBLE_DEVICES=0 python navsim/planning/script/run_training.py \
  agent=resnet_agent \
  experiment_name=training_resnet_agent \
  train_test_split=navtrain \
  cache_path=$NAVSIM_EXP_ROOT/training_cache_resnet_navtrain \
  use_cache_without_dataset=false \
  force_cache_computation=false \
  trainer.params.accelerator=gpu \
  +trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.max_epochs=100 \
  dataloader.params.batch_size=128 \
  hydra.run.dir=$NAVSIM_EXP_ROOT/ckpts/resnet_agent
```

Train ViT-B/16:

```bash
CUDA_VISIBLE_DEVICES=0 python navsim/planning/script/run_training.py \
  agent=vit_agent \
  experiment_name=training_vit_agent \
  train_test_split=navtrain \
  cache_path=$NAVSIM_EXP_ROOT/training_cache_vit_navtrain \
  use_cache_without_dataset=false \
  force_cache_computation=false \
  trainer.params.accelerator=gpu \
  +trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.precision=bf16-mixed \
  trainer.params.max_epochs=100 \
  dataloader.params.batch_size=128 \
  hydra.run.dir=$NAVSIM_EXP_ROOT/ckpts/vit_agent
```

Choose the resulting complete NAVSIM checkpoint as `BASE_CKPT` for the remaining stages.

### 2. Generate the offline H5 feature cache

ResNet-34:

```bash
export BASE_CKPT=/path/to/resnet_base.ckpt

python navsim/planning/script/run_frpt_resnet.py \
  agent=resnet_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  train_test_split=navtrain \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  cache_path=$NAVSIM_EXP_ROOT/training_cache_resnet_navtrain \
  +frpt.mode=cache_h5 \
  +frpt.h5_path=$FRPT_RESNET34_H5_PATH \
  '+frpt.recons_keys=[recons_planning_z2,recons_planning_z1,recons_fusion_z,recons_resnet.z4,recons_resnet.z3]' \
  +frpt.cache_batch_size=128 \
  +frpt.cache_num_workers=4
```

ViT-B/16:

```bash
export BASE_CKPT=/path/to/vit_base.ckpt

python navsim/planning/script/run_frpt_vit.py \
  agent=vit_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  train_test_split=navtrain \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  cache_path=$NAVSIM_EXP_ROOT/training_cache_vit_navtrain \
  +frpt.mode=cache_h5 \
  +frpt.h5_path=$FRPT_VIT_B16_H5_PATH \
  '+frpt.recons_keys=[recons_planning_z2,recons_planning_z1,recons_fusion_z,recons_image_embedding,recons_vit.final_cls_token,recons_vit.transformer.encoder_layers.10.output]' \
  +frpt.cache_batch_size=128 \
  +frpt.cache_num_workers=4
```

The H5 cache is generated from `navtrain` and is reused by all subsequent post-training tasks.

### 3. Run offline post-training

ResNet-34:

```bash
python navsim/planning/script/run_frpt_resnet.py \
  agent=resnet_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  train_test_split=navtrain \
  +frpt.mode=post_train \
  +frpt.work_dir=$FRPT_RESNET34_WORK_DIR \
  +frpt.h5_path=$FRPT_RESNET34_H5_PATH \
  '+frpt.alphas=[0,0.02,0.1,0.3]' \
  '+frpt.seeds=[0]' \
  '+frpt.recons_keys=[recons_resnet.z3]' \
  +frpt.epochs=5 \
  +frpt.post_train_batch_size=256 \
  +frpt.post_train_num_workers=4
```

ViT-B/16:

```bash
python navsim/planning/script/run_frpt_vit.py \
  agent=vit_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  train_test_split=navtrain \
  +frpt.mode=post_train \
  +frpt.work_dir=$FRPT_VIT_B16_WORK_DIR \
  +frpt.h5_path=$FRPT_VIT_B16_H5_PATH \
  '+frpt.alphas=[0.02,0.1,0.3]' \
  '+frpt.seeds=[1]' \
  '+frpt.recons_keys=[recons_planning_z1]' \
  +frpt.epochs=5 \
  +frpt.post_train_batch_size=256 \
  +frpt.post_train_num_workers=4
```

Use `+frpt.mode=all` only when you explicitly want to regenerate the H5 file and run post-training in one job. Separate stages are recommended for long experiments.

### 4. Evaluate all offline checkpoints

```bash
CKPT_DIR=$FRPT_RESNET34_CKPT_DIR GPU_IDS=0 \
  ./run_frpt_resnet_agent_navtest_eval.sh

CKPT_DIR=$FRPT_VIT_B16_CKPT_DIR GPU_IDS=0 \
  ./run_frpt_vit_agent_navtest_eval.sh
```

Set `SKIP_METRIC_CACHE=0` on either batch script if the metric cache has not been generated yet. By default, already evaluated experiment directories are skipped.

## Outputs

The default artifact layout is:

```text
$NAVSIM_EXP_ROOT/
├── training_cache_resnet_navtrain/
├── training_cache_vit_navtrain/
├── metric_cache/
├── frpt_resnet34_output/*.h5
├── frpt_vit_output/*.h5
├── frpt_resnet_post_train/ckpts/
├── frpt_vit_post_train/ckpts/
└── eval_*/
```

Hydra also stores the resolved configuration and logs under each experiment directory.

## Troubleshooting

- **Checkpoint not found:** pass the complete NAVSIM checkpoint with `agent.checkpoint_path` or set the corresponding `FRPT_*_CHECKPOINT` variable.
- **ImageNet checkpoint error during base training:** set `FRPT_RESNET34_IMAGENET_CKPT` or `FRPT_VIT_B16_IMAGENET_CKPT`. These variables are not needed for offline FR-PT.
- **H5 generation fails with missing cached samples:** either generate the training cache first or set `use_cache_without_dataset=false` to build the dataset directly.
- **Metric cache and split do not match:** use one cache path per split; the offline evaluation wrappers use `navtest` by default.
- **GPU memory errors:** reduce `dataloader.params.batch_size`, `+frpt.cache_batch_size`, or `+frpt.post_train_batch_size`; for ViT-B/16 also reduce `vit_backbone_chunk_size`.
- **BF16 is unsupported:** replace `trainer.params.precision=bf16-mixed` with `16-mixed` on GPUs without BF16 support.

The code is distributed under the [Apache 2.0 License](LICENSE). Dataset licenses are inherited from their respective providers.
