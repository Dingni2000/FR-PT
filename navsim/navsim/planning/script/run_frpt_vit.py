import os
from pathlib import Path
import inspect
import logging
import multiprocessing as mp

import h5py
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from navsim.agents.FRPT import build_post_train_tasks, model_post_train_all
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.script.run_training import build_datasets
from navsim.planning.training.dataset import CacheOnlyDataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _default_exp_root() -> Path:
    configured_root = os.environ.get("NAVSIM_EXP_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()
    # run_frpt_vit.py lives in <workspace>/navsim/navsim/planning/script/.
    return Path(__file__).resolve().parents[3].parent / "exp"


def _frpt_cfg(cfg: DictConfig, key: str, default):
    return OmegaConf.select(cfg, f"frpt.{key}", default=default)


def _move_features_to_device(features, device):
    return {name: tensor.to(device, non_blocking=True) for name, tensor in features.items()}


def _build_train_dataloader(cfg: DictConfig, agent: AbstractAgent) -> DataLoader:
    if cfg.use_cache_without_dataset:
        logger.info("Using cached NAVSIM tensors from %s", cfg.cache_path)
        assert not cfg.force_cache_computation, (
            "force_cache_computation must be False when use_cache_without_dataset=True"
        )
        assert cfg.cache_path is not None, "cache_path must be provided when use_cache_without_dataset=True"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
    else:
        logger.info("Building NAVSIM train/val datasets; only train split is used for FRPT h5")
        train_data, _ = build_datasets(cfg, agent)

    logger.info("Num FRPT h5 samples: %d", len(train_data))
    return DataLoader(train_data, **cfg.dataloader.params, shuffle=False)


def _split_tasks(tasks, num_workers):
    return [tasks[index::num_workers] for index in range(num_workers)]


def _call_get_recons_fea(model, features, trajectory, recons_keys, residual_iter):
    """Call get_recons_fea per requested key to avoid unneeded heavy ViT reversals."""
    signature = inspect.signature(model.get_recons_fea)
    if not recons_keys:
        kwargs = {}
        if "residual_iter" in signature.parameters:
            kwargs["residual_iter"] = residual_iter
        recons = model.get_recons_fea(features, trajectory, **kwargs)
        if not isinstance(recons, dict):
            raise TypeError("get_recons_fea returned a tensor when all reconstruction keys were requested")
        return recons

    recons = {}
    for recons_key in recons_keys:
        kwargs = {}
        if "recons_key" in signature.parameters:
            kwargs["recons_key"] = recons_key
        elif "recons_keys" in signature.parameters:
            kwargs["recons_keys"] = [recons_key]
        if "residual_iter" in signature.parameters:
            kwargs["residual_iter"] = residual_iter
        recons_res = model.get_recons_fea(features, trajectory, **kwargs)
        if isinstance(recons_res, dict):
            if recons_key not in recons_res:
                raise KeyError(f"get_recons_fea did not return {recons_key}; keys={list(recons_res.keys())}")
            recons_res = recons_res[recons_key]
        recons[recons_key] = recons_res
    return recons


def _run_post_train_worker(worker_id, cfg_container, tasks, device_index):
    cfg = OmegaConf.create(cfg_container)
    pl.seed_everything(int(cfg.seed) + worker_id, workers=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    checkpoint_path = cfg.agent.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("agent.checkpoint_path is required for offline FRPT")
    if not Path(checkpoint_path).expanduser().is_file():
        raise FileNotFoundError(f"Offline FRPT checkpoint not found: {checkpoint_path}")
    # Offline FRPT starts from a complete NAVSIM checkpoint.  Do not require
    # the separate ImageNet backbone checkpoint a second time.
    cfg.agent.config.load_imagenet_checkpoint = False
    cfg.agent.config.image_checkpoint_path = None
    h5_path = _frpt_cfg(cfg, "h5_path", os.environ.get("FRPT_VIT_B16_H5_PATH"))
    work_dir = Path(
        _frpt_cfg(
            cfg,
            "work_dir",
            os.environ.get("FRPT_VIT_B16_WORK_DIR", _default_exp_root() / "frpt_vit"),
        )
    )
    epochs = int(_frpt_cfg(cfg, "epochs", 10))
    lr = float(_frpt_cfg(cfg, "lr", cfg.agent.lr))
    alphas = list(_frpt_cfg(cfg, "alphas", [0, 0.02, 0.1]))
    seeds = tuple(int(seed) for seed in _frpt_cfg(cfg, "seeds", [0]))
    recons_keys = list(_frpt_cfg(cfg, "recons_keys", []))
    post_train_batch_size = int(_frpt_cfg(cfg, "post_train_batch_size", cfg.dataloader.params.batch_size))
    post_train_num_workers = int(_frpt_cfg(cfg, "post_train_num_workers", cfg.dataloader.params.num_workers))
    pin_memory = bool(_frpt_cfg(cfg, "pin_memory", cfg.dataloader.params.pin_memory))

    agent_for_keys: AbstractAgent = instantiate(cfg.agent)
    if not recons_keys:
        recons_keys = agent_for_keys.get_fea_name() + ["out"]
    del agent_for_keys
    cfg.agent.config.return_reconstruction_features = True

    logger.info(
        "FRPT ViT-B/16 offline worker %d starts %d tasks on %s: %s",
        worker_id,
        len(tasks),
        device,
        tasks,
    )
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.to(device)
    model_post_train_all(
        model=agent,
        MODELPATH=checkpoint_path,
        ALPHA_ls=alphas,
        EPOCHS=epochs,
        DEVICE=device,
        BATCH_SIZE=post_train_batch_size,
        lr=lr,
        work_dir=work_dir,
        seeds=seeds,
        recons_keys=recons_keys,
        dataloader_num_workers=post_train_num_workers,
        pin_memory=pin_memory,
        h5_path=h5_path,
        tasks=tasks,
    )


def _create_or_resize_dataset(group, name, sample_tensor, total_size, compression):
    sample_shape = tuple(sample_tensor.shape[1:])
    if name not in group:
        group.create_dataset(
            name,
            shape=(0,) + sample_shape,
            maxshape=(None,) + sample_shape,
            dtype=sample_tensor.detach().cpu().numpy().dtype,
            compression=compression,
        )
    dataset = group[name]
    if dataset.shape[1:] != sample_shape:
        raise ValueError(
            f"H5 dataset '{name}' shape mismatch: existing {dataset.shape[1:]}, new {sample_shape}"
        )
    dataset.resize(total_size, axis=0)
    return dataset


def save_vit_recons_h5(
    model: AbstractAgent,
    data_loader: DataLoader,
    h5_path: Path,
    device: torch.device,
    recons_keys,
    residual_iter: int,
    compression,
    empty_cache_every: int,
) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.attrs["format"] = "frpt_vit_b16_v1"
            h5_file.attrs["recons_keys"] = ",".join(recons_keys)
            feature_group = h5_file.create_group("features")
            dsets = {}
            total_size = 0

            for batch_idx, (features_cpu, targets_cpu) in enumerate(data_loader):
                if "trajectory" not in targets_cpu:
                    raise KeyError("FRPT h5 generation expects target key 'trajectory'.")

                batch_size = targets_cpu["trajectory"].shape[0]
                new_total_size = total_size + batch_size
                features = _move_features_to_device(features_cpu, device)
                trajectory = targets_cpu["trajectory"].to(device, non_blocking=True)

                recons = _call_get_recons_fea(model, features, trajectory, recons_keys, residual_iter)
                selected_recons = {
                    key: tensor
                    for key, tensor in recons.items()
                    if key != "gt_trajectory" and (not recons_keys or key in recons_keys)
                }
                missing_recons = set(recons_keys) - set(selected_recons.keys())
                if missing_recons:
                    raise KeyError(
                        f"get_recons_fea did not return requested keys: {sorted(missing_recons)}"
                    )

                for feature_name, feature_tensor in features_cpu.items():
                    dataset = _create_or_resize_dataset(
                        feature_group,
                        feature_name,
                        feature_tensor,
                        new_total_size,
                        compression,
                    )
                    dataset[total_size:new_total_size] = feature_tensor.detach().cpu().numpy()

                target_dataset = _create_or_resize_dataset(
                    h5_file,
                    "gt_trajectory",
                    targets_cpu["trajectory"],
                    new_total_size,
                    compression,
                )
                target_dataset[total_size:new_total_size] = (
                    targets_cpu["trajectory"].detach().cpu().numpy()
                )

                for recons_name, recons_tensor in selected_recons.items():
                    dataset = _create_or_resize_dataset(
                        h5_file,
                        recons_name,
                        recons_tensor,
                        new_total_size,
                        compression,
                    )
                    dataset[total_size:new_total_size] = recons_tensor.detach().cpu().numpy()

                total_size = new_total_size
                h5_file.flush()
                logger.info(
                    "Saved FRPT h5 batch %d, total samples=%d, keys=%s",
                    batch_idx,
                    total_size,
                    sorted(selected_recons.keys()),
                )

                del features, trajectory, recons, selected_recons
                if torch.cuda.is_available() and empty_cache_every and (batch_idx + 1) % empty_cache_every == 0:
                    torch.cuda.empty_cache()

    finally:
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)

    logger.info("Saved ViT-B/16 FRPT h5 to %s", h5_path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_path = cfg.agent.checkpoint_path
    assert checkpoint_path is not None, "agent.checkpoint_path must point to the trained ViT-B/16 checkpoint"
    if not Path(checkpoint_path).expanduser().is_file():
        raise FileNotFoundError(f"Offline FRPT checkpoint not found: {checkpoint_path}")
    # The complete NAVSIM checkpoint supplies the backbone and task weights.
    cfg.agent.config.load_imagenet_checkpoint = False
    cfg.agent.config.image_checkpoint_path = None

    mode = _frpt_cfg(cfg, "mode", "cache_h5")
    if mode not in ("cache_h5", "post_train", "all"):
        raise ValueError(f"Unsupported frpt.mode={mode}; use cache_h5, post_train, or all")

    h5_path = Path(
        _frpt_cfg(
            cfg,
            "h5_path",
            os.environ.get(
                "FRPT_VIT_B16_H5_PATH",
                str(_default_exp_root() / "frpt_vit" / "vit_agent_navtrain_frpt.h5"),
            ),
        )
    )
    work_dir = Path(_frpt_cfg(cfg, "work_dir", h5_path.parent))
    recons_keys = list(_frpt_cfg(cfg, "recons_keys", []))
    cache_batch_size = _frpt_cfg(cfg, "cache_batch_size", None)
    cache_num_workers = _frpt_cfg(cfg, "cache_num_workers", None)
    residual_iter = int(_frpt_cfg(cfg, "residual_iter", 5))
    compression = _frpt_cfg(cfg, "compression", None)
    empty_cache_every = int(_frpt_cfg(cfg, "empty_cache_every", 1))
    epochs = int(_frpt_cfg(cfg, "epochs", 10))
    lr = float(_frpt_cfg(cfg, "lr", cfg.agent.lr))
    alphas = list(_frpt_cfg(cfg, "alphas", [0, 0.02, 0.1]))
    seeds = tuple(int(seed) for seed in _frpt_cfg(cfg, "seeds", [0]))
    post_train_batch_size = int(_frpt_cfg(cfg, "post_train_batch_size", cfg.dataloader.params.batch_size))
    post_train_num_workers = int(_frpt_cfg(cfg, "post_train_num_workers", cfg.dataloader.params.num_workers))
    task_workers = int(_frpt_cfg(cfg, "task_workers", 1))
    pin_memory = bool(_frpt_cfg(cfg, "pin_memory", cfg.dataloader.params.pin_memory))

    agent_for_keys: AbstractAgent = instantiate(cfg.agent)
    if not recons_keys:
        recons_keys = agent_for_keys.get_fea_name() + ["out"]
        logger.warning("frpt.recons_keys is empty; defaulting to %s", recons_keys)
    del agent_for_keys

    cfg.agent.config.return_reconstruction_features = True

    if cache_batch_size is not None:
        cfg.dataloader.params.batch_size = int(cache_batch_size)
    if cache_num_workers is not None:
        cfg.dataloader.params.num_workers = int(cache_num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Building ViT-B/16 agent for FRPT mode=%s on %s", mode, device)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("H5 path: %s", h5_path)
    logger.info("FRPT work_dir: %s", work_dir)
    logger.info("Requested recons_keys: %s", recons_keys)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.to(device)

    if mode in ("cache_h5", "all"):
        agent.initialize()
        h5_recons_keys = [key for key in recons_keys if key != "out"]
        train_dataloader = _build_train_dataloader(cfg, agent)
        save_vit_recons_h5(
            model=agent,
            data_loader=train_dataloader,
            h5_path=h5_path,
            device=device,
            recons_keys=h5_recons_keys,
            residual_iter=residual_iter,
            compression=compression,
            empty_cache_every=empty_cache_every,
        )

    if mode in ("post_train", "all"):
        tasks = build_post_train_tasks(alphas, recons_keys, seeds)
        logger.info("Total FRPT ViT-B/16 offline post-train tasks: %d", len(tasks))
        if task_workers > 1:
            task_workers = min(task_workers, len(tasks))
            if task_workers > 1:
                device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                cfg_container = OmegaConf.to_container(cfg, resolve=True)
                chunks = [chunk for chunk in _split_tasks(tasks, task_workers) if chunk]
                ctx = mp.get_context("spawn")
                processes = []
                for worker_id, chunk in enumerate(chunks):
                    device_index = worker_id % max(device_count, 1)
                    process = ctx.Process(
                        target=_run_post_train_worker,
                        args=(worker_id, cfg_container, chunk, device_index),
                    )
                    process.start()
                    processes.append(process)
                failed_workers = []
                for process in processes:
                    process.join()
                    if process.exitcode != 0:
                        failed_workers.append(process.pid)
                if failed_workers:
                    raise RuntimeError(f"FRPT ViT-B/16 offline worker failed, pids={failed_workers}")
                return

        summary = model_post_train_all(
            model=agent,
            MODELPATH=checkpoint_path,
            ALPHA_ls=alphas,
            EPOCHS=epochs,
            DEVICE=device,
            BATCH_SIZE=post_train_batch_size,
            lr=lr,
            work_dir=work_dir,
            seeds=seeds,
            recons_keys=recons_keys,
            dataloader_num_workers=post_train_num_workers,
            pin_memory=pin_memory,
            h5_path=h5_path,
            tasks=tasks,
        )
        logger.info("FRPT ViT-B/16 finished. Generated checkpoint groups:")
        for row in summary:
            logger.info(
                "alpha=%s recons_key=%s checkpoints=%s",
                row["alpha"],
                row["recons_key"],
                row["checkpoint_paths"],
            )


if __name__ == "__main__":
    main()
