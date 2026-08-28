#!/usr/bin/env bash
# Single source of truth for NAVSIM data and offline experiment paths.
#
# Expected layout:
#   <workspace>/navsim   (this checkout)
#   <workspace>/dataset  (OpenScene logs, sensor blobs and maps)
#   <workspace>/exp      (caches, checkpoints and evaluation outputs)
#
# Existing exports always take precedence, so this file is safe to source from
# a shell, a batch script, or a job launcher.

ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd -- "$ENV_SCRIPT_DIR/.." && pwd)}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$NAVSIM_DEVKIT_ROOT/../dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/../exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Paper/offline experiment paths.  These are defaults only; individual jobs
# can still override them on the command line or before sourcing this file.
export FRPT_RESNET34_IMAGENET_CKPT="${FRPT_RESNET34_IMAGENET_CKPT:-}"
export FRPT_VIT_B16_IMAGENET_CKPT="${FRPT_VIT_B16_IMAGENET_CKPT:-}"
export FRPT_RESNET34_CHECKPOINT="${FRPT_RESNET34_CHECKPOINT:-$NAVSIM_EXP_ROOT/ckpts/resnet_agent_seed0.ckpt}"
export FRPT_VIT_B16_CHECKPOINT="${FRPT_VIT_B16_CHECKPOINT:-$NAVSIM_EXP_ROOT/ckpts/vit_agent_seed0.ckpt}"
export FRPT_RESNET34_CKPT_DIR="${FRPT_RESNET34_CKPT_DIR:-$NAVSIM_EXP_ROOT/frpt_resnet_post_train/ckpts}"
export FRPT_VIT_B16_CKPT_DIR="${FRPT_VIT_B16_CKPT_DIR:-$NAVSIM_EXP_ROOT/frpt_vit_post_train/ckpts}"
export FRPT_RESNET34_WORK_DIR="${FRPT_RESNET34_WORK_DIR:-$NAVSIM_EXP_ROOT/frpt_resnet_post_train}"
export FRPT_RESNET34_H5_PATH="${FRPT_RESNET34_H5_PATH:-$NAVSIM_EXP_ROOT/frpt_resnet34_output/resnet_agent_seed0_navtrain.h5}"
export FRPT_VIT_B16_WORK_DIR="${FRPT_VIT_B16_WORK_DIR:-$NAVSIM_EXP_ROOT/frpt_vit_post_train}"
export FRPT_VIT_B16_H5_PATH="${FRPT_VIT_B16_H5_PATH:-$NAVSIM_EXP_ROOT/frpt_vit_output/vit_agent_seed0_navtrain.h5}"
export NAVSIM_METRIC_CACHE="${NAVSIM_METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache}"
export NAVSIM_NAVTEST_METRIC_CACHE="${NAVSIM_NAVTEST_METRIC_CACHE:-$NAVSIM_METRIC_CACHE}"
