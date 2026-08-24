#!/usr/bin/env bash
set -euo pipefail

# Evaluate CameraStatusAgent on NAVSIM navtest.
# Re-running this script resumes metric caching by default: existing
# metric_cache.pkl files are skipped.

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/data/wsc/navsim_workspace/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/wsc/navsim_workspace/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/wsc/navsim_workspace/exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
CHECKPOINT="${CHECKPOINT:-$NAVSIM_EXP_ROOT/frpt_resnet_post_train/ckpts}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/transfuser_navtest_metric_cache}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_camera_status_agent_relu0.5_lr}"
GPU_IDS="${GPU_IDS:-3,5}"
WORKER="${WORKER:-sequential}"
MAX_WORKERS="${MAX_WORKERS:-}"
USE_PROCESS_POOL="${USE_PROCESS_POOL:-False}"
CACHE_WORKER="${CACHE_WORKER:-$WORKER}"
CACHE_MAX_WORKERS="${CACHE_MAX_WORKERS:-$MAX_WORKERS}"
CACHE_USE_PROCESS_POOL="${CACHE_USE_PROCESS_POOL:-$USE_PROCESS_POOL}"
SCORE_WORKER="${SCORE_WORKER:-$WORKER}"
SCORE_MAX_WORKERS="${SCORE_MAX_WORKERS:-$MAX_WORKERS}"
SCORE_USE_PROCESS_POOL="${SCORE_USE_PROCESS_POOL:-$USE_PROCESS_POOL}"
FORCE_METRIC_CACHE="${FORCE_METRIC_CACHE:-False}"
SKIP_METRIC_CACHE="${SKIP_METRIC_CACHE:-0}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT"
echo "OPENSCENE_DATA_ROOT=$OPENSCENE_DATA_ROOT"
echo "NAVSIM_EXP_ROOT=$NAVSIM_EXP_ROOT"
echo "NUPLAN_MAPS_ROOT=$NUPLAN_MAPS_ROOT"
echo "TRAIN_TEST_SPLIT=$TRAIN_TEST_SPLIT"
echo "CHECKPOINT=$CHECKPOINT"
echo "METRIC_CACHE_PATH=$METRIC_CACHE_PATH"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "GPU_IDS=$GPU_IDS"
echo "CACHE_WORKER=$CACHE_WORKER"
echo "CACHE_MAX_WORKERS=$CACHE_MAX_WORKERS"
echo "CACHE_USE_PROCESS_POOL=$CACHE_USE_PROCESS_POOL"
echo "SCORE_WORKER=$SCORE_WORKER"
echo "SCORE_MAX_WORKERS=$SCORE_MAX_WORKERS"
echo "SCORE_USE_PROCESS_POOL=$SCORE_USE_PROCESS_POOL"
echo "FORCE_METRIC_CACHE=$FORCE_METRIC_CACHE"

build_worker_args() {
  local worker_name="$1"
  local max_workers="$2"
  local use_process_pool="$3"
  local -n out_args="$4"

  out_args=(worker="$worker_name")
  if [[ "$worker_name" == "single_machine_thread_pool" ]]; then
    out_args+=(worker.use_process_pool="$use_process_pool")
    if [[ -n "$max_workers" ]]; then
      out_args+=(worker.max_workers="$max_workers")
    fi
  fi
}

build_worker_args "$CACHE_WORKER" "$CACHE_MAX_WORKERS" "$CACHE_USE_PROCESS_POOL" CACHE_WORKER_ARGS
build_worker_args "$SCORE_WORKER" "$SCORE_MAX_WORKERS" "$SCORE_USE_PROCESS_POOL" SCORE_WORKER_ARGS

if [[ "$SKIP_METRIC_CACHE" != "1" ]]; then
  echo "[$(date '+%F %T')] Running metric caching"
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    cache.cache_path="$METRIC_CACHE_PATH" \
    "${CACHE_WORKER_ARGS[@]}" \
    cache.force_feature_computation="$FORCE_METRIC_CACHE"
else
  echo "[$(date '+%F %T')] Skipping metric caching"
fi

echo "[$(date '+%F %T')] Running CameraStatusAgent evaluation"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  agent=camera_status_agent \
  agent.checkpoint_path="'$CHECKPOINT'" \
  experiment_name="$EXPERIMENT_NAME" \
  metric_cache_path="$METRIC_CACHE_PATH" \
  "${SCORE_WORKER_ARGS[@]}"

echo "[$(date '+%F %T')] Done"
