#!/usr/bin/env bash
set -euo pipefail

# Evaluate the ViT-B/16 vit_agent on NAVSIM navtest.
# Re-running this script resumes metric caching by default: existing
# metric_cache.pkl files are skipped.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/env.sh
source "$SCRIPT_DIR/scripts/env.sh"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
CHECKPOINT="${CHECKPOINT:-$FRPT_VIT_B16_CHECKPOINT}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_NAVTEST_METRIC_CACHE}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_vit_agent}"
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

echo "[$(date '+%F %T')] Running ViT agent evaluation"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  agent=vit_agent \
  agent.config.load_imagenet_checkpoint=false \
  agent.config.image_checkpoint_path=null \
  agent.checkpoint_path="$CHECKPOINT" \
  experiment_name="$EXPERIMENT_NAME" \
  metric_cache_path="$METRIC_CACHE_PATH" \
  "${SCORE_WORKER_ARGS[@]}"

echo "[$(date '+%F %T')] Done"
