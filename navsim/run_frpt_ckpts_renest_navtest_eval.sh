#!/usr/bin/env bash
set -euo pipefail

# Batch-evaluate FRPT EgoStatusMLP checkpoints on NAVSIM navtest.
# The metric cache is independent of model weights, so generate it once and
# reuse it across all checkpoints.

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/data/wsc/navsim_workspace/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/wsc/navsim_workspace/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/wsc/navsim_workspace/exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
CKPT_DIR="${CKPT_DIR:-$NAVSIM_EXP_ROOT/frpt_resnet_post_train/ckpts}"
CKPT_GLOB="${CKPT_GLOB:-*.ckpt}"
# CKPT_GLOB='resnet34_seed_0_alpha0.02*_ptseed[0].ckpt'
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/transfuser_navtest_metric_cache}"
EXP_PREFIX="${EXP_PREFIX:-eval_frpt_camera_status_resnet}"
GPU_IDS="${GPU_IDS:-7}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EVAL_TASK_WORKERS="${EVAL_TASK_WORKERS:-1}"

WORKER="${WORKER:-sequential}"
MAX_WORKERS="${MAX_WORKERS:-}"
USE_PROCESS_POOL="${USE_PROCESS_POOL:-False}"
SKIP_METRIC_CACHE="${SKIP_METRIC_CACHE:-1}"
FORCE_METRIC_CACHE="${FORCE_METRIC_CACHE:-False}"

CACHE_WORKER="${CACHE_WORKER:-$WORKER}"
CACHE_MAX_WORKERS="${CACHE_MAX_WORKERS:-$MAX_WORKERS}"
CACHE_USE_PROCESS_POOL="${CACHE_USE_PROCESS_POOL:-$USE_PROCESS_POOL}"
SCORE_WORKER="${SCORE_WORKER:-$WORKER}"
SCORE_MAX_WORKERS="${SCORE_MAX_WORKERS:-$MAX_WORKERS}"
SCORE_USE_PROCESS_POOL="${SCORE_USE_PROCESS_POOL:-$USE_PROCESS_POOL}"

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

sanitize_name() {
  local name="$1"
  name="${name%.ckpt}"
  name="${name//./p}"
  name="${name//[^A-Za-z0-9_]/_}"
  echo "$name"
}

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Checkpoint directory not found: $CKPT_DIR" >&2
  exit 1
fi

mapfile -t CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "$CKPT_GLOB" | sort)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "No checkpoints matched: $CKPT_DIR/$CKPT_GLOB" >&2
  exit 1
fi

if [[ "$LIMIT" -gt 0 && "${#CKPTS[@]}" -gt "$LIMIT" ]]; then
  CKPTS=("${CKPTS[@]:0:$LIMIT}")
fi

echo "NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT"
echo "OPENSCENE_DATA_ROOT=$OPENSCENE_DATA_ROOT"
echo "NAVSIM_EXP_ROOT=$NAVSIM_EXP_ROOT"
echo "TRAIN_TEST_SPLIT=$TRAIN_TEST_SPLIT"
echo "CKPT_DIR=$CKPT_DIR"
echo "CKPT_GLOB=$CKPT_GLOB"
echo "NUM_CKPTS=${#CKPTS[@]}"
echo "METRIC_CACHE_PATH=$METRIC_CACHE_PATH"
echo "EXP_PREFIX=$EXP_PREFIX"
echo "GPU_IDS=$GPU_IDS"
echo "EVAL_TASK_WORKERS=$EVAL_TASK_WORKERS"
echo "SKIP_METRIC_CACHE=$SKIP_METRIC_CACHE"
echo "SKIP_EXISTING=$SKIP_EXISTING"
echo "SCORE_WORKER=$SCORE_WORKER"
echo "SCORE_MAX_WORKERS=$SCORE_MAX_WORKERS"
echo "SCORE_USE_PROCESS_POOL=$SCORE_USE_PROCESS_POOL"

build_worker_args "$CACHE_WORKER" "$CACHE_MAX_WORKERS" "$CACHE_USE_PROCESS_POOL" CACHE_WORKER_ARGS
build_worker_args "$SCORE_WORKER" "$SCORE_MAX_WORKERS" "$SCORE_USE_PROCESS_POOL" SCORE_WORKER_ARGS

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  GPU_LIST=("$GPU_IDS")
fi

eval_one_ckpt() {
  local ckpt="$1"
  local task_index="$2"
  local assigned_gpu="$3"
  local ckpt_name
  local exp_name
  local exp_root
  ckpt_name="$(basename "$ckpt")"
  exp_name="${EXP_PREFIX}_$(sanitize_name "$ckpt_name")"
  exp_root="$NAVSIM_EXP_ROOT/$exp_name"

  if [[ "$SKIP_EXISTING" == "1" && -d "$exp_root" ]] && find "$exp_root" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "[$(date '+%F %T')] Skipping already evaluated $ckpt_name"
    echo "existing_experiment_dir=$exp_root"
    return 0
  fi

  echo "[$(date '+%F %T')] Evaluating $ckpt_name"
  echo "experiment_name=$exp_name"
  echo "assigned_gpu=$assigned_gpu"
  echo "task_index=$task_index"

  local cmd=(
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py"
    train_test_split="$TRAIN_TEST_SPLIT"
    agent=camera_status_agent
    agent.checkpoint_path="$ckpt"
    experiment_name="$exp_name"
    metric_cache_path="$METRIC_CACHE_PATH"
    "${SCORE_WORKER_ARGS[@]}"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$assigned_gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  if CUDA_VISIBLE_DEVICES="$assigned_gpu" "${cmd[@]}"; then
    echo "[$(date '+%F %T')] Finished $ckpt_name"
    return 0
  fi

  echo "[$(date '+%F %T')] Failed $ckpt_name" >&2
  return 1
}

if [[ "$SKIP_METRIC_CACHE" != "1" ]]; then
  echo "[$(date '+%F %T')] Running metric caching once"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
      train_test_split="$TRAIN_TEST_SPLIT" \
      cache.cache_path="$METRIC_CACHE_PATH" \
      "${CACHE_WORKER_ARGS[@]}" \
      cache.force_feature_computation="$FORCE_METRIC_CACHE"
  else
    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
      train_test_split="$TRAIN_TEST_SPLIT" \
      cache.cache_path="$METRIC_CACHE_PATH" \
      "${CACHE_WORKER_ARGS[@]}" \
      cache.force_feature_computation="$FORCE_METRIC_CACHE"
  fi
else
  echo "[$(date '+%F %T')] Skipping metric caching"
fi

failures=0
pids=()
for task_index in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$task_index]}"
  assigned_gpu="${GPU_LIST[$((task_index % ${#GPU_LIST[@]}))]}"

  eval_one_ckpt "$ckpt" "$task_index" "$assigned_gpu" &
  pids+=("$!")

  if [[ "${#pids[@]}" -ge "$EVAL_TASK_WORKERS" ]]; then
    if ! wait "${pids[0]}"; then
      failures=$((failures + 1))
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit 1
      fi
    fi
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo "[$(date '+%F %T')] Done with $failures failed checkpoint evaluations" >&2
fi
echo "[$(date '+%F %T')] Done evaluating ${#CKPTS[@]} checkpoints"
