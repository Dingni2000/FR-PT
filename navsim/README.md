# FR-PT NAVSIM

This repository contains the NAVSIM implementation and the FR-PT offline
post-training pipeline for trajectory agents. Two agents are
provided:

| Agent | Backbone | NAVSIM config | FR-PT entry point |
| --- | --- | --- | --- |
| resnet_agent | ResNet-34 | navsim/planning/script/config/common/agent/resnet_agent.yaml | navsim/planning/script/run_frpt_resnet.py |
| vit_agent | ViT-B/16 | navsim/planning/script/config/common/agent/vit_agent.yaml | navsim/planning/script/run_frpt_vit.py |

The recommended validation path is:

~~~text
install environment
    -> configure exports and data
    -> verify imports and paths
    -> build NAVSIM feature cache
    -> train a complete NAVSIM checkpoint
    -> export the training split to H5
    -> run FR-PT post-training
    -> build/reuse the navtest metric cache
    -> evaluate the resulting checkpoint
~~~

The repository contains code and configuration only. Dataset files, ImageNet
weights, NAVSIM checkpoints, H5 files and experiment outputs should stay
outside Git.

## 1. Requirements and installation

The provided environment is pinned around Python 3.9, PyTorch 2.0.1,
torchvision 0.15.2 and PyTorch Lightning 2.2.1.
Clone the repository and install the conda environment:

~~~bash
git clone https://github.com/Dingni2000/FR-PT.git
cd FR-PT/navsim

conda env create -f environment.yml
conda activate navsim
pip install -e .
~~~

requirements.txt installs `nuplan-devkit` from the pinned GitHub tag
`nuplan-devkit-v1.2` instead of PyPI. The unmodified environment-creation
command therefore requires network access to GitHub. Without that access,
use a local checkout or mirror of the same version and update the dependency
source before creating the environment. If the environment already exists,
update it with:

~~~bash
conda activate navsim
conda env update --file environment.yml --prune
pip install -e .
~~~

Verify the environment before downloading or caching the full dataset:

~~~bash
python -c 'import torch, h5py, navsim; print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available()); print("navsim:", navsim.__file__)'
~~~

## 2. Dataset and directory layout

### Locate the repository

This project keeps the runnable NAVSIM code in the `navsim/` subdirectory of
the [FR-PT repository](https://github.com/Dingni2000/FR-PT). After cloning,
`NAVSIM_DEVKIT_ROOT` should point to that directory, which contains this
README, `environment.yml` and the `navsim/` Python package. The original
[NAVSIM repository](https://github.com/autonomousvision/navsim) is provided
only as an upstream reference.

Set `NAVSIM_DEVKIT_ROOT` together with the external data and experiment
directories before running the commands below.

~~~bash
# [NAVSIM source root: directory containing this README and environment.yml]
export NAVSIM_DEVKIT_ROOT="/path/to/FR-PT/navsim"
# [OpenScene dataset root: contains navsim_logs/, sensor_blobs/ and maps/]
export OPENSCENE_DATA_ROOT="/path/to/dataset"
# [Experiment root: caches, checkpoints and evaluation outputs]
export NAVSIM_EXP_ROOT="/path/to/exp"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
~~~

All source and script paths in this README are relative to
NAVSIM_DEVKIT_ROOT. Replace each `/path/to/...` placeholder with the
corresponding absolute path on your machine. The data and experiment roots do
not have to be siblings of the repository.

Only the dataset root, experiment root, ImageNet weights and model checkpoints
need separate external paths. Source the environment file after setting the
variables:

~~~bash
source "$NAVSIM_DEVKIT_ROOT/scripts/env.sh"
~~~

Please obtain NAVSIM/OpenScene data according to the applicable dataset
license and terms. The included download helpers are in download/. They are
staging scripts: run them from the directory where the data should be
downloaded, not from inside the Git checkout.

Run the following after setting the variables above and sourcing
scripts/env.sh:

~~~bash
mkdir -p "$OPENSCENE_DATA_ROOT"

(cd "$OPENSCENE_DATA_ROOT" && bash "$NAVSIM_DEVKIT_ROOT/download/download_maps.sh")
(cd "$OPENSCENE_DATA_ROOT" && bash "$NAVSIM_DEVKIT_ROOT/download/download_navtrain.sh")
(cd "$OPENSCENE_DATA_ROOT" && bash "$NAVSIM_DEVKIT_ROOT/download/download_test.sh")
~~~

The download scripts create temporary names such as
trainval_navsim_logs, trainval_sensor_blobs, test_navsim_logs and
test_sensor_blobs. Normalize them to the paths used by scripts/env.sh:

~~~bash
mkdir -p "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs"
mv "$OPENSCENE_DATA_ROOT/trainval_navsim_logs" "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
mv "$OPENSCENE_DATA_ROOT/trainval_sensor_blobs/trainval" "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
mv "$OPENSCENE_DATA_ROOT/test_navsim_logs" "$OPENSCENE_DATA_ROOT/navsim_logs/test"
mv "$OPENSCENE_DATA_ROOT/test_sensor_blobs" "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
~~~

download_navtrain.sh downloads the filtered sensor subset needed by navtrain
but still needs the matching trainval logs. If full trainval sensors are
required, use download_trainval.sh and normalize its output in the same way.

The required directory layout is:

~~~text
<your workspace>/
├── <FR-PT checkout>/navsim/        # NAVSIM_DEVKIT_ROOT
├── <dataset root>/                 # OPENSCENE_DATA_ROOT
│   ├── maps/
│   ├── navsim_logs/
│   │   ├── trainval/
│   │   └── test/
│   └── sensor_blobs/
│       ├── trainval/
│       └── test/
└── <experiment root>/              # NAVSIM_EXP_ROOT
~~~

The split names in this code map to the data folders as follows:

| Config split | Data folder | Typical use |
| --- | --- | --- |
| navtrain | trainval | training, validation and FR-PT H5 export |
| navtest | test | metric caching and final NAVSIM evaluation |
| navmini | mini | small local checks, if mini data is available |

The values already exported in the shell take precedence over the defaults in
scripts/env.sh.

## 3. Environment variables and export

Always source the project environment file in a new shell before launching a
job:

~~~bash
# [NAVSIM source root]
export NAVSIM_DEVKIT_ROOT="/path/to/FR-PT/navsim"
# [OpenScene dataset root]
export OPENSCENE_DATA_ROOT="/path/to/dataset"
# [Experiment root]
export NAVSIM_EXP_ROOT="/path/to/exp"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# Required only when starting base training from ImageNet initialization.
# Download sources:
#   ResNet-34: https://download.pytorch.org/models/resnet34-b627a593.pth
#   ViT-B/16:  https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view
# [Local ResNet-34 checkpoint file]
export FRPT_RESNET34_IMAGENET_CKPT="/path/to/pretrained/resnet34-b627a593.pth"
# [Local ViT-B/16 checkpoint file]
export FRPT_VIT_B16_IMAGENET_CKPT="/path/to/pretrained/imagenet21k+imagenet2012_ViT-B_16-224.pth"

source "$NAVSIM_DEVKIT_ROOT/scripts/env.sh"
~~~

The path variables above are intended to be edited by the user. The ImageNet
paths can also be exported after sourcing because they are read when the agent
is constructed. The checkpoint files may live in any local directory; they do
not need to be copied into the Git checkout.

| Variable | Meaning |
| --- | --- |
| NAVSIM_DEVKIT_ROOT | Absolute path to this checkout; also added to PYTHONPATH. |
| OPENSCENE_DATA_ROOT | Parent directory containing navsim_logs/, sensor_blobs/ and usually maps/. |
| NAVSIM_EXP_ROOT | Root directory for feature caches, H5 files, checkpoints and evaluation outputs. |
| NUPLAN_MAPS_ROOT | Directory containing the nuPlan/NAVSIM map files. |
| NUPLAN_MAP_VERSION | Map version, normally nuplan-maps-v1.0. |
| FRPT_RESNET34_IMAGENET_CKPT | Local ResNet-34 ImageNet initialization checkpoint. |
| FRPT_VIT_B16_IMAGENET_CKPT | Local ViT-B/16 ImageNet initialization checkpoint. |
| FRPT_RESNET34_CHECKPOINT / FRPT_VIT_B16_CHECKPOINT | Default complete NAVSIM checkpoint used by the single-checkpoint evaluation wrappers. |
| FRPT_RESNET34_H5_PATH / FRPT_VIT_B16_H5_PATH | H5 output used by the corresponding FR-PT job. |
| FRPT_RESNET34_CKPT_DIR / FRPT_VIT_B16_CKPT_DIR | Directory containing FR-PT post-training checkpoints. |
| NAVSIM_NAVTEST_METRIC_CACHE | Reusable metric cache for navtest; it is independent of model weights. |

### Pretrained ImageNet weights

The ImageNet weights are required only when starting a new NAVSIM base-training
run. They are not the complete NAVSIM checkpoint used by H5 export, FR-PT
post-training or evaluation.

#### ResNet-34

The current ResNet loader expects the official torchvision state-dict format.
Download the exact compatible file from the official PyTorch model URL, then
save it with the filename shown below:

- [ResNet-34 official weights](https://download.pytorch.org/models/resnet34-b627a593.pth)
- [Torchvision model documentation](https://pytorch.org/vision/stable/models.html)

~~~bash
WEIGHTS_DIR="/path/to/pretrained"
mkdir -p "$WEIGHTS_DIR"
wget -O "$WEIGHTS_DIR/resnet34-b627a593.pth" \
  "https://download.pytorch.org/models/resnet34-b627a593.pth"
export FRPT_RESNET34_IMAGENET_CKPT="$WEIGHTS_DIR/resnet34-b627a593.pth"
~~~

The [Hugging Face `microsoft/resnet-34` page](https://huggingface.co/microsoft/resnet-34)
is useful as a model reference, but its Transformers checkpoint is not a
drop-in replacement for the torchvision state dict expected by this code.

#### ViT-B/16

The current ViT implementation uses the ASYML-converted PyTorch checkpoint
with the ImageNet-21k + ImageNet-1k fine-tuned weights. The exact compatible
file is available from the [official ASYML checkpoint page](https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view?usp=sharing).
The file used by this repository is named
`imagenet21k+imagenet2012_ViT-B_16-224.pth`. Save the downloaded file with
that name and set:

~~~bash
python -m pip install gdown
gdown --id 1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx \
  -O "$WEIGHTS_DIR/imagenet21k+imagenet2012_ViT-B_16-224.pth"
export FRPT_VIT_B16_IMAGENET_CKPT="$WEIGHTS_DIR/imagenet21k+imagenet2012_ViT-B_16-224.pth"
~~~

The corresponding official Google model is also available on Hugging Face:
[google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224).
However, the Hugging Face `pytorch_model.bin` uses Transformers parameter
names and cannot be passed directly to `FRPT_VIT_B16_IMAGENET_CKPT`; use the
ASyML-converted `.pth` checkpoint above unless a conversion step is added.

The two sources above produce the formats expected by the current loaders:
ResNet loads the canonical torchvision state dict, while ViT loads the
ASyML-converted state dict and resizes its 224x224 position embedding to the
configured NAVSIM image grid. Do not rename a Hugging Face Transformers
checkpoint and expect it to load as an equivalent `.pth` file.

Check the resolved paths and pretrained files before starting a long job. This
is only an environment, directory and file check; it does not build a model or
run training.

~~~bash
set -euo pipefail

printf 'NAVSIM_DEVKIT_ROOT=%s\n' "$NAVSIM_DEVKIT_ROOT"
printf 'OPENSCENE_DATA_ROOT=%s\n' "$OPENSCENE_DATA_ROOT"
printf 'NAVSIM_EXP_ROOT=%s\n' "$NAVSIM_EXP_ROOT"
printf 'NUPLAN_MAPS_ROOT=%s\n' "$NUPLAN_MAPS_ROOT"

for path in \
  "$NAVSIM_DEVKIT_ROOT" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  "$NUPLAN_MAPS_ROOT"; do
  test -d "$path" || { echo "Missing directory: $path" >&2; exit 1; }
done

for path in \
  "$FRPT_RESNET34_IMAGENET_CKPT" \
  "$FRPT_VIT_B16_IMAGENET_CKPT"; do
  test -f "$path" || { echo "Missing checkpoint: $path" >&2; exit 1; }
done

echo "Path and checkpoint checks passed."
~~~

## 4. Build the NAVSIM feature cache

This is the first data-processing step and does not require a pre-existing
feature cache. It reads the raw NAVSIM data, instantiates the selected agent
and writes the feature/target cache used by training and FR-PT. Therefore this
step also verifies that the corresponding ImageNet checkpoint can be loaded.
Generate one cache per agent because the ResNet and ViT preprocessors and
feature shapes are different.

### ResNet-34 cache

~~~bash
python navsim/planning/script/run_dataset_caching.py \
  train_test_split=navtrain \
  agent=resnet_agent \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_resnet_navtrain" \
  experiment_name=cache_resnet_features \
  worker=single_machine_thread_pool \
  worker.max_workers=8 \
  worker.use_process_pool=true
~~~

### ViT-B/16 cache

~~~bash
python navsim/planning/script/run_dataset_caching.py \
  train_test_split=navtrain \
  agent=vit_agent \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_vit_navtrain" \
  experiment_name=cache_vit_features \
  worker=single_machine_thread_pool \
  worker.max_workers=8 \
  worker.use_process_pool=true
~~~

On restricted machines, use `worker=sequential`. The cache step can still be
long because it processes the selected `navtrain` scenes. During the later
training and FR-PT commands, `use_cache_without_dataset=true` requires the
corresponding cache directory to exist and requires
`force_cache_computation=false`.

## 5. Train a complete NAVSIM checkpoint

ImageNet initialization weights are needed only when constructing a new base
agent. They are not the checkpoint used by the offline FR-PT stages. After
base training, Lightning writes a complete NAVSIM checkpoint containing the
backbone, feature projection, status branch and planning head. This complete
training output is called the base checkpoint below.

### ResNet-34

~~~bash
python navsim/planning/script/run_training.py \
  train_test_split=navtrain \
  agent=resnet_agent \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_resnet_navtrain" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  experiment_name=frpt_resnet_base \
  trainer.params.accelerator=gpu \
  trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.max_epochs=100 \
  dataloader.params.batch_size=128 \
  dataloader.params.num_workers=4
~~~

### ViT-B/16

~~~bash
python navsim/planning/script/run_training.py \
  train_test_split=navtrain \
  agent=vit_agent \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_vit_navtrain" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  experiment_name=frpt_vit_base \
  trainer.params.accelerator=gpu \
  trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.precision=bf16-mixed \
  trainer.params.max_epochs=100 \
  dataloader.params.batch_size=128 \
  dataloader.params.num_workers=4
~~~

The base checkpoint is written under the corresponding experiment directory:

~~~text
$NAVSIM_EXP_ROOT/frpt_resnet_base/<timestamp>/lightning_logs/version_0/checkpoints/epoch=*.ckpt
$NAVSIM_EXP_ROOT/frpt_vit_base/<timestamp>/lightning_logs/version_0/checkpoints/epoch=*.ckpt
~~~

Replace the wildcard with the concrete `.ckpt` produced by the command you
ran, then verify it before continuing:

~~~bash
# Complete NAVSIM checkpoint produced by the preceding base-training command.
BASE_CKPT="/path/to/the/complete/navsim/base-training.ckpt"
test -f "$BASE_CKPT" || { echo "Missing base checkpoint: $BASE_CKPT" >&2; exit 1; }
~~~

`BASE_CKPT` is only a shell variable used to avoid repeating a long path. It
can be replaced with the actual checkpoint path directly in the commands
below. It must point to the complete NAVSIM training checkpoint, not to an
ImageNet-only `.pth` file.

The FR-PT scripts deliberately disable ImageNet loading and restore the
complete NAVSIM checkpoint for offline processing.

## 6. Export the training split to H5

H5 export runs the trained model on navtrain and writes one `.h5` file for the
FR-PT post-training step. Use the same train_test_split, agent, cache and base
checkpoint for H5 export and post-training.

The output path is controlled by `+frpt.h5_path`, and the same HDF5 file is
used by the following post-training command.

The H5 can become very large, especially for ViT-B/16. Keep it on local
storage and do not commit it.

### Reconstruction keys

`+frpt.recons_keys` selects the intermediate feature that FR-PT should
reconstruct. The selected key must be supported by the agent, included during
H5 export and selected again during post-training. The available keys are:

| Agent | Key | Feature represented |
| --- | --- | --- |
| ResNet-34 | `recons_resnet.z3` | Output of the third ResNet stage, used as input to the final stage. |
| ResNet-34 | `recons_resnet.z4` | Output of the final ResNet stage. |
| ResNet-34 | `recons_fusion_z` | Status-conditioned fusion of image and ego-status features. |
| ResNet-34 | `recons_planning_z1` | First hidden feature in the planning head. |
| ResNet-34 | `recons_planning_z2` | Second hidden feature, immediately before the trajectory head. |
| ViT-B/16 | `recons_vit.transformer.encoder_layers.10.output` | Token sequence output by ViT encoder layer 10. |
| ViT-B/16 | `recons_vit.final_cls_token` | Final ViT CLS token before the image projection. |
| ViT-B/16 | `recons_image_embedding` | Projected image feature before status-conditioned fusion. |
| ViT-B/16 | `recons_fusion_z` | Status-conditioned fusion of image and ego-status features. |
| ViT-B/16 | `recons_planning_z1` | First hidden feature in the planning head. |
| ViT-B/16 | `recons_planning_z2` | Second hidden feature, immediately before the trajectory head. |

The current ResNet implementation exposes `z3` and `z4`; the lower ResNet
stages are not enabled as reconstruction targets. The special key `out`
means ordinary task post-training and is not an H5 reconstruction key.

### ResNet-34 H5

~~~bash
BASE_CKPT="/path/to/the/complete/navsim/resnet_base.ckpt"

python navsim/planning/script/run_frpt_resnet.py \
  train_test_split=navtrain \
  agent=resnet_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_resnet_navtrain" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  experiment_name=frpt_resnet_h5 \
  +frpt.mode=cache_h5 \
  +frpt.h5_path="$FRPT_RESNET34_H5_PATH" \
  +frpt.work_dir="$FRPT_RESNET34_WORK_DIR" \
  +frpt.recons_keys='[recons_planning_z2,recons_planning_z1,recons_fusion_z,recons_resnet.z4,recons_resnet.z3]' \
  +frpt.cache_batch_size=128 \
  +frpt.cache_num_workers=4
~~~

### ViT-B/16 H5

~~~bash
BASE_CKPT="/path/to/the/complete/navsim/vit_base.ckpt"

python navsim/planning/script/run_frpt_vit.py \
  train_test_split=navtrain \
  agent=vit_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_vit_navtrain" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  experiment_name=frpt_vit_h5 \
  +frpt.mode=cache_h5 \
  +frpt.h5_path="$FRPT_VIT_B16_H5_PATH" \
  +frpt.work_dir="$FRPT_VIT_B16_WORK_DIR" \
  +frpt.recons_keys='[recons_planning_z2,recons_planning_z1,recons_fusion_z,recons_image_embedding,recons_vit.final_cls_token,recons_vit.transformer.encoder_layers.10.output]' \
  +frpt.cache_batch_size=128 \
  +frpt.cache_num_workers=4
~~~

## 7. Run FR-PT post-training

Post-training consumes the complete NAVSIM checkpoint and the H5 generated in
the previous step. The output is a group of ordinary NAVSIM-compatible
checkpoints, plus post_train_metrics.csv in the configured work directory.

The main FR-PT parameters are:

- `+frpt.recons_keys`: the intermediate feature key to reconstruct. The key
  must be supported by the selected agent and must also be exported to H5.
- `+frpt.alphas`: the reconstruction-loss weight. `0` disables the
  reconstruction constraint; larger values give it more influence during
  post-training.
- `+frpt.seeds`: random seeds used for separate post-training runs.

The script runs every combination of `recons_keys`, `alphas` and `seeds`. For
the first validation, use one supported key, one alpha and one seed.

### ResNet-34

~~~bash
BASE_CKPT="/path/to/the/complete/navsim/resnet_base.ckpt"

python navsim/planning/script/run_frpt_resnet.py \
  train_test_split=navtrain \
  agent=resnet_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_resnet_navtrain" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  experiment_name=frpt_resnet_post_train \
  +frpt.mode=post_train \
  +frpt.h5_path="$FRPT_RESNET34_H5_PATH" \
  +frpt.work_dir="$FRPT_RESNET34_WORK_DIR" \
  +frpt.recons_keys='[recons_resnet.z3]' \
  +frpt.alphas='[0,0.02,0.1,0.3]' \
  +frpt.seeds='[0]' \
  +frpt.epochs=5 \
  +frpt.post_train_batch_size=256 \
  +frpt.post_train_num_workers=4
~~~

### ViT-B/16

~~~bash
BASE_CKPT="/path/to/the/complete/navsim/vit_base.ckpt"

python navsim/planning/script/run_frpt_vit.py \
  train_test_split=navtrain \
  agent=vit_agent \
  agent.checkpoint_path="$BASE_CKPT" \
  cache_path="$NAVSIM_EXP_ROOT/training_cache_vit_navtrain" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  experiment_name=frpt_vit_post_train \
  +frpt.mode=post_train \
  +frpt.h5_path="$FRPT_VIT_B16_H5_PATH" \
  +frpt.work_dir="$FRPT_VIT_B16_WORK_DIR" \
  +frpt.recons_keys='[recons_planning_z1]' \
  +frpt.alphas='[0.02,0.1,0.3]' \
  +frpt.seeds='[1]' \
  +frpt.epochs=5 \
  +frpt.post_train_batch_size=256 \
  +frpt.post_train_num_workers=4
~~~

The number of jobs is approximately:

~~~text
len(alphas) × len(recons_keys) × len(seeds)
~~~

Start with one reconstruction key, one alpha and one seed. Increase the grid
only after one generated checkpoint can pass NAVSIM evaluation.

## 8. Evaluate on navtest

NAVSIM metric caching is separate from model inference. Build the navtest
metric cache once, then reuse it for the base and all FR-PT checkpoints.

Evaluate one complete checkpoint with the single-checkpoint wrapper:

~~~bash
BASE_CKPT="/path/to/the/complete/navsim/resnet_base.ckpt"
CHECKPOINT="$BASE_CKPT" \
GPU_IDS=0 \
WORKER=sequential \
SKIP_METRIC_CACHE=0 \
./run_resnet_agent_navtest_eval.sh
~~~

For ViT, use:

~~~bash
BASE_CKPT="/path/to/the/complete/navsim/vit_base.ckpt"
CHECKPOINT="$BASE_CKPT" \
GPU_IDS=0 \
WORKER=sequential \
SKIP_METRIC_CACHE=0 \
./run_vit_agent_navtest_eval.sh
~~~

Evaluate FR-PT checkpoints in a directory. The batch wrappers skip existing
experiment directories and skip metric caching by default:

~~~bash
CKPT_DIR="$FRPT_RESNET34_CKPT_DIR" \
GPU_IDS=0 \
LIMIT=1 \
SKIP_METRIC_CACHE=0 \
./run_frpt_resnet_agent_navtest_eval.sh
~~~

After the metric cache exists, evaluate the whole directory with
SKIP_METRIC_CACHE=1. For ViT, replace the directory and wrapper with
FRPT_VIT_B16_CKPT_DIR and run_frpt_vit_agent_navtest_eval.sh.

The wrappers support CKPT_GLOB, LIMIT, WORKER, EVAL_TASK_WORKERS, SKIP_EXISTING,
DRY_RUN and CONTINUE_ON_ERROR. Use DRY_RUN=1 to inspect the commands without
running them.

## 9. What counts as a successful end-to-end test?

For either agent, consider the pipeline valid when all of the following are
true:

1. The base training command writes a complete .ckpt and finishes a train and
   validation step.
2. H5 export completes and writes the expected `.h5` file.
3. Post-training writes at least one checkpoint and finite values appear in
   post_train_metrics.csv.
4. NAVSIM navtest evaluation completes with an output CSV and no failed
   checkpoint task.

## 10. Output layout

Unless overridden, generated files live below NAVSIM_EXP_ROOT:

~~~text
exp/
├── training_cache_resnet_navtrain/
├── training_cache_vit_navtrain/
├── frpt_resnet34_output/*.h5
├── frpt_vit_output/*.h5
├── frpt_resnet_post_train/
│   ├── ckpts/*.ckpt
│   └── post_train_metrics.csv
├── frpt_vit_post_train/
│   ├── ckpts/*.ckpt
│   └── post_train_metrics.csv
├── metric_cache/
└── eval_*/
~~~

Hydra and Lightning also create configuration snapshots and logs under the
experiment directory. These are useful for debugging but should not be pushed
to the public repository.

## License

See [LICENSE](LICENSE).
