import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
import numpy as np
import gc
from typing import Dict
import csv
import logging
import fcntl
from collections import defaultdict


logger = logging.getLogger(__name__)


def _log(message):
    print(message)
    logger.info(message)


# def get_score(model, data_loader, DEVICE):  # 分类任务测试分数
#     model.eval()  # Set the module in evaluation mode.
#     correct = 0.0   # 正确率
#     with torch.no_grad():  # 不会进行计算梯度，也不会进行反向传播
#         for data, target in data_loader:
#             data, target = data.to(DEVICE), target.to(DEVICE)
#             output = model(data)['out']
#             pred = output.argmax(dim=1)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     return correct/len(data_loader.dataset)    


class HDF5Dataset(Dataset):
    """HDF5 dataset for FRPT post-training on NAVSIM cached tensors."""

    def __init__(self, file_path, recons_key=None, input_key="ego_status"):
        self.file_path = str(file_path)
        self.recons_key = recons_key
        self.input_key = input_key
        with h5py.File(self.file_path, "r") as f:
            if "gt_trajectory" not in f:
                raise KeyError(f"{self.file_path} does not contain required dataset 'gt_trajectory'")
            self._feature_group_mode = "features" in f
            if self._feature_group_mode:
                self._feature_names = list(f["features"].keys())
                if not self._feature_names:
                    raise KeyError(f"{self.file_path} contains an empty 'features' group")
            else:
                if "input" not in f:
                    raise KeyError(f"{self.file_path} does not contain required dataset 'input' or group 'features'")
                self._feature_names = [self.input_key]
            if self.recons_key not in (None, "out") and self.recons_key not in f:
                raise KeyError(f"{self.file_path} does not contain required dataset '{self.recons_key}'")
            self._length = len(f["gt_trajectory"])

    def __len__(self):
        return self._length

    def __getitem__(self,idx):
        with h5py.File(self.file_path, "r") as f:
            if self._feature_group_mode:
                features = {
                    name: torch.tensor(f["features"][name][idx], dtype=torch.float32)
                    for name in self._feature_names
                }
            else:
                x = torch.tensor(f["input"][idx], dtype=torch.float32)
                features = {self.input_key: x}
            y = torch.tensor(f["gt_trajectory"][idx], dtype=torch.float32)
            batch = {"input": features, "gt_trajectory": y}
            if self.recons_key not in (None, "out"):
                batch[self.recons_key] = torch.tensor(f[self.recons_key][idx], dtype=torch.float32)
        return batch
    

def _nanmean_or_nan(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


def _move_features_to_device(features: Dict[str, torch.Tensor], device):
    return {k: v.to(device, non_blocking=True) for k, v in features.items()}


def _unpack_navsim_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        features, targets = batch
        return features, targets["trajectory"]
    if isinstance(batch, dict):
        return batch["input"], batch["gt_trajectory"]
    raise TypeError(f"Unsupported batch type for FRPT: {type(batch)}")


def _resolve_forward_key(predictions, recons_key):
    if recons_key.startswith("recons_"):
        stripped_key = recons_key[len("recons_"):]
        if stripped_key in predictions:
            return stripped_key
    if recons_key in predictions:
        return recons_key
    raise KeyError(
        f"Cannot find forward feature for recons_key={recons_key}. "
        f"Tried '{recons_key[len('recons_'):] if recons_key.startswith('recons_') else recons_key}' "
        f"and '{recons_key}'. Available keys={list(predictions.keys())}"
    )


def _load_agent_state_dict(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})


def _save_navsim_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {f"agent.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict}, checkpoint_path)


def _new_feature_stats():
    return {
        name: {"forward_sum": 0.0, "forward_absmax": 0.0, "recons_sum": 0.0, "recons_absmax": 0.0, "count": 0}
        for name in ("z1", "z2", "z3")
    }


def _update_feature_stats(stats, forward_res, recons_res):
    for forward_key in ("z1", "z2", "z3"):
        recons_key = f"recons_{forward_key}"
        if forward_key not in forward_res or recons_key not in recons_res:
            continue
        forward_fea = forward_res[forward_key].detach()
        recons_fea = recons_res[recons_key].detach()
        stats[forward_key]["forward_sum"] += forward_fea.sum().item()
        stats[forward_key]["forward_absmax"] = max(
            stats[forward_key]["forward_absmax"], forward_fea.abs().max().item()
        )
        stats[forward_key]["recons_sum"] += recons_fea.sum().item()
        stats[forward_key]["recons_absmax"] = max(
            stats[forward_key]["recons_absmax"], recons_fea.abs().max().item()
        )
        stats[forward_key]["count"] += forward_fea.numel()


def _print_feature_stats(stats):
    lines = []
    for forward_key in ("z1", "z2", "z3"):
        stat = stats[forward_key]
        if stat["count"] == 0:
            continue
        lines.append(
            f"{forward_key}: "
            f"forward_mean={stat['forward_sum'] / stat['count']:.6f}, "
            f"forward_absmax={stat['forward_absmax']:.6f}, "
            f"recons_mean={stat['recons_sum'] / stat['count']:.6f}, "
            f"recons_absmax={stat['recons_absmax']:.6f}"
        )
    if lines:
        _log("[FRPT h5 feature stats] " + " | ".join(lines))


def save_recons_fea_to_h5(
    model,
    data_loader,
    MODELPATH,
    device,
    empty_cache_every=1,
    output_dir="/data/wsc/navsim_workspace/exp/frpt_dn",
    h5_path=None,
    print_feature_stats=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = Path(h5_path) if h5_path is not None else output_dir / (Path(MODELPATH).stem + ".h5")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    flag = True
    dsets = {}
    param_requires_grad = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    feature_stats = _new_feature_stats() if print_feature_stats else None

    try:
        with h5py.File(filepath, "w") as f: 
            for batch_idx, batch in enumerate(data_loader):
                features, y = _unpack_navsim_batch(batch)
                x = features["ego_status"]
                x_cpu = x.detach().cpu()
                y_cpu = y.detach().cpu()
                features = _move_features_to_device(features, device)
                y = y.to(device, non_blocking=True)
                bs = y.shape[0]

                res = model.get_recons_fea(features, y)
                if feature_stats is not None:
                    with torch.no_grad():
                        forward_res = model(features)
                    _update_feature_stats(feature_stats, forward_res, res)

                if flag:
                    flag = False
                    xshape = tuple(x_cpu.shape[1:])
                    dsets["input"] = f.create_dataset(
                        "input", shape=(0,) + xshape, maxshape=(None,) + xshape, dtype=np.float32
                    )
                    dsets["gt_trajectory"] = f.create_dataset(
                        "gt_trajectory", shape=(0,) + tuple(y_cpu.shape[1:]), maxshape=(None,) + tuple(y_cpu.shape[1:]), dtype=np.float32
                    )

                    for name, fea in res.items():
                        if name == "gt_trajectory":
                            continue
                        fea_shape = tuple(fea.shape[1:])
                        dsets[name] = f.create_dataset(
                            name, shape=(0,) + fea_shape, maxshape=(None,) + fea_shape, dtype=np.float32
                        )

                new_size = dsets["input"].shape[0] + bs
                for dset in dsets.values():
                    dset.resize(new_size, axis=0)

                dsets["input"][-bs:] = x_cpu.numpy()
                dsets["gt_trajectory"][-bs:] = y_cpu.numpy()
                del x_cpu, y_cpu

                for name in list(res.keys()):
                    if name == "gt_trajectory":
                        continue
                    fea_cpu = res[name].detach().cpu().numpy()
                    dsets[name][-bs:] = fea_cpu
                    del fea_cpu
                    del res[name]

                f.flush()
                del res, features, x, y

                if torch.cuda.is_available() and empty_cache_every and (batch_idx + 1) % empty_cache_every == 0:
                    torch.cuda.empty_cache()
                gc.collect()
    finally:
        for p, req_grad in zip(model.parameters(), param_requires_grad):
            p.requires_grad_(req_grad)
    if feature_stats is not None:
        _print_feature_stats(feature_stats)
    _log(f"saved recons data to {filepath}")  # NOTE dict [str, list] {'input':[], 'gt_trajectory':[], 'recons_zx':[]}


def frpt(model, trainloader, recons_key, alpha, epochs, device, lr=3e-5):
    if hasattr(model, "set_recons_param"):
        model.set_recons_param(recons_key)
    else:
        for p in model.parameters():
            p.requires_grad = True
        freeze_from = model.get_fea_id(recons_key)
        if freeze_from is None:
            raise ValueError(f"Unknown recons_key={recons_key}; available keys={model.get_fea_name()}")
        for _, layer in list(model._mlp.named_children())[freeze_from:]:  # NOTE ego_status_mlp_agent use '_mlp'
            for p in layer.parameters():
                p.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    taskloss_ls = []
    reconsloss_ls = []
    loss_ls = []
    epoch_metrics = []
    taskcrit = nn.MSELoss()  # NOTE 轨迹规划任务的loss
    reconscrit = nn.MSELoss()
    ema_beta = 0.9
    ema_cls = None
    ema_recons = None
    forward_key = None
        
    for epoch in range(epochs):
        model.train()  # 训练模式
        epoch_forward_sum = 0.0
        epoch_forward_absmax = 0.0
        epoch_recons_sum = 0.0
        epoch_recons_absmax = 0.0
        epoch_feature_count = 0
        for batch in trainloader:
            input_data = _move_features_to_device(batch["input"], device)
            label = batch["gt_trajectory"].to(device)
            recons_feature = batch[recons_key].to(device)

            res = model(input_data)
            clsloss = taskcrit(res['trajectory'], label)
            if forward_key is None:
                forward_key = _resolve_forward_key(res, recons_key)
            forward_feature = res[forward_key]
            reconsloss = reconscrit(forward_feature, recons_feature)
            with torch.no_grad():
                epoch_forward_sum += forward_feature.detach().sum().item()
                epoch_forward_absmax = max(epoch_forward_absmax, forward_feature.detach().abs().max().item())
                epoch_recons_sum += recons_feature.detach().sum().item()
                epoch_recons_absmax = max(epoch_recons_absmax, recons_feature.detach().abs().max().item())
                epoch_feature_count += forward_feature.numel()
            # loss = alpha * reconsloss + (1-alpha) * clsloss  # ORIGINAL
            if alpha == 0:
                loss_weight = torch.tensor(0.0, device=device)
            else:
                with torch.no_grad():
                    if ema_cls is None:
                        ema_cls = clsloss.detach()
                        ema_recons = reconsloss.detach()
                    else:
                        ema_cls = ema_beta * ema_cls + (1 - ema_beta) * clsloss.detach()
                        ema_recons = ema_beta * ema_recons + (1 - ema_beta) * reconsloss.detach()
                    loss_weight = alpha * ema_cls / (ema_recons + 1e-8)
                    loss_weight = loss_weight.clamp(min=0.00001, max=10.0)
            loss = loss_weight * reconsloss + clsloss  # 0706

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        
        forward_mean = epoch_forward_sum / max(epoch_feature_count, 1)
        recons_mean = epoch_recons_sum / max(epoch_feature_count, 1)
        epoch_metric = {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "reconsloss": reconsloss.item(),
            "taskloss": clsloss.item(),
            "lossweight": loss_weight.item(),
            "forward_key": forward_key,
            "forward_mean": forward_mean,
            "forward_absmax": epoch_forward_absmax,
            "recons_key": recons_key,
            "recons_mean": recons_mean,
            "recons_absmax": epoch_recons_absmax,
        }
        epoch_metrics.append(epoch_metric)
        _log(
            f"[{epoch+1}]  loss={epoch_metric['loss']:4f}, reconsloss={epoch_metric['reconsloss']:4f}, "
            f"clsloss={epoch_metric['taskloss']:4f}, lossweight={epoch_metric['lossweight']:4f}, "
            f"{forward_key}_mean={forward_mean:.6f}, {forward_key}_absmax={epoch_forward_absmax:.6f}, "
            f"{recons_key}_mean={recons_mean:.6f}, {recons_key}_absmax={epoch_recons_absmax:.6f}"
        )
        reconsloss_ls.append(reconsloss.item())
        taskloss_ls.append(clsloss.item())  # pred
        loss_ls.append(loss.item())
    _log(f'[RECORD] alpha={alpha}, recons_key={recons_key}, best taskloss={min(taskloss_ls)}\n')
    return {'reconsloss_ls':reconsloss_ls, 'taskloss_ls': taskloss_ls, 'loss_ls':loss_ls, 'epoch_metrics': epoch_metrics}



def normal_pt(model, trainloader, epochs, device, lr=3e-5):
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    taskloss_ls = []
    loss_ls = []
    epoch_metrics = []
    taskcrit = nn.MSELoss()  # NOTE 轨迹规划任务的loss
        
    for epoch in range(epochs):
        model.train()  # 训练模式
        for batch in trainloader:
            input_data = _move_features_to_device(batch["input"], device)
            label = batch["gt_trajectory"].to(device)

            res = model(input_data)
            loss = taskcrit(res['trajectory'], label)

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        
        epoch_metric = {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "reconsloss": np.nan,
            "taskloss": loss.item(),
            "lossweight": 0.0,
            "forward_key": "trajectory",
            "forward_mean": np.nan,
            "forward_absmax": np.nan,
            "recons_key": "out",
            "recons_mean": np.nan,
            "recons_absmax": np.nan,
        }
        epoch_metrics.append(epoch_metric)
        _log(f"[{epoch+1}]  loss={loss.item():4f}, ")

        taskloss_ls.append(loss.item())
    _log(f'[RECORD] best taskloss={min(taskloss_ls)}\n')
    return {'taskloss_ls':taskloss_ls, 'loss_ls': taskloss_ls, 'epoch_metrics': epoch_metrics}


def _append_epoch_metrics_csv(csv_path, config_dict, epoch_metrics):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint_path",
        "alpha",
        "recons_key",
        "seed",
        "epoch",
        "loss",
        "reconsloss",
        "taskloss",
        "lossweight",
        "forward_key",
        "forward_mean",
        "forward_absmax",
        "recons_mean",
        "recons_absmax",
    ]
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for metric in epoch_metrics:
                row = {key: config_dict.get(key, "") for key in ("model", "checkpoint_path", "alpha", "recons_key", "seed")}
                row.update({key: metric.get(key, "") for key in fieldnames if key not in row})
                writer.writerow(row)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def build_post_train_tasks(alpha_ls, recons_keys, seeds):
    tasks = []
    for alpha in alpha_ls:
        for recons_key in recons_keys:
            if recons_key == "out" and alpha > 0:
                continue
            for seed in seeds:
                tasks.append({"alpha": alpha, "recons_key": recons_key, "seed": int(seed)})
    return tasks


def _append_result_pt(result_path, records):
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = result_path.with_suffix(result_path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            datafile_test = torch.load(result_path)
        except (FileNotFoundError, EOFError):
            datafile_test = []
        datafile_test.extend(records)
        torch.save(datafile_test, result_path)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def _summarize_records(task_records):
    grouped = defaultdict(
        lambda: {
            "checkpoint_paths": [],
            "best_loss": [],
            "mean_loss": [],
            "epoch1_loss": [],
            "epoch5_loss": [],
            "epoch10_loss": [],
        }
    )
    for record in task_records:
        key = (record["model_stem"], record["alpha"], record["recons_key"])
        group = grouped[key]
        group["checkpoint_paths"].append(record["checkpoint_path"])
        group["best_loss"].append(record["best_loss"])
        group["mean_loss"].append(record["mean_loss"])
        group["epoch1_loss"].append(record["epoch1_loss"])
        group["epoch5_loss"].append(record["epoch5_loss"])
        group["epoch10_loss"].append(record["epoch10_loss"])
    summary_res = []
    for model_stem, alpha, recons_key in sorted(grouped.keys(), key=lambda item: (str(item[0]), float(item[1]), str(item[2]))):
        group = grouped[(model_stem, alpha, recons_key)]
        summary_res.append({"model": model_stem, "alpha": alpha, "recons_key": recons_key, **group})
    return summary_res


def model_post_train_all(model, MODELPATH, ALPHA_ls, EPOCHS, 
                         DEVICE, BATCH_SIZE, lr=3e-5, save_ckpt=True, save_res=True,
                         work_dir="/data/wsc/navsim_workspace/exp/frpt_dn", seeds=(0, 1, 2),
                         recons_keys=None, dataloader_num_workers=4, pin_memory=True, h5_path=None,
                         tasks=None):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    FILEPATH = Path(h5_path) if h5_path is not None else work_dir / (Path(MODELPATH).stem + ".h5")
    SAVEPATH = work_dir / Path(MODELPATH).stem
    CKPTDIR = work_dir / "ckpts"
    METRICSPATH = work_dir / "post_train_metrics.csv"
    CKPTDIR.mkdir(parents=True, exist_ok=True)
    if not FILEPATH.is_file():
        raise FileNotFoundError(
            f"FRPT feature file not found: {FILEPATH}. "
            "Run save_recons_fea_to_h5(model, train_dataloader, MODELPATH, DEVICE) first."
        )
    model.to(DEVICE)
    if recons_keys is None:
        recons_keys = model.get_fea_name() + ['out']
    available_keys = set(model.get_fea_name() + ['out'])
    unknown_keys = set(recons_keys) - available_keys
    if unknown_keys:
        raise ValueError(f"Unknown recons_keys={sorted(unknown_keys)}; available keys={sorted(available_keys)}")

    if tasks is None:
        tasks = build_post_train_tasks(ALPHA_ls, recons_keys, seeds)

    task_records = []
    for task in tasks:
        ALPHA = task["alpha"]
        RECONS_KEY = task["recons_key"]
        seed = int(task["seed"])
        _log(f"[TASK] start alpha={ALPHA}, recons_key={RECONS_KEY}, seed={seed}, device={DEVICE}")
        trainset_fr = HDF5Dataset(FILEPATH, RECONS_KEY)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        g = torch.Generator().manual_seed(seed)
        train_loader_fr = DataLoader(
            trainset_fr,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            generator=g,
        )
        _load_agent_state_dict(model, MODELPATH, DEVICE)
        if RECONS_KEY == 'out':
            res = normal_pt(model, train_loader_fr, epochs=EPOCHS, device=DEVICE, lr=lr)
        else:
            res = frpt(model, train_loader_fr, RECONS_KEY, ALPHA, epochs=EPOCHS, device=DEVICE, lr=lr)
        ckpt_model = Path(MODELPATH).stem + f"_alpha{ALPHA}_" + RECONS_KEY + f"_ptseed{seed}"
        ckpt_path = CKPTDIR / f"{ckpt_model}.ckpt"
        if save_ckpt:
            _save_navsim_checkpoint(model, ckpt_path)
        config_dict = {'model':ckpt_model, 'checkpoint_path': str(ckpt_path), 'alpha':ALPHA, 'recons_key': RECONS_KEY, 'seed':seed}
        if save_res:
            _append_result_pt(str(SAVEPATH)+f"_{EPOCHS}e.pt", [config_dict | res])
        _append_epoch_metrics_csv(METRICSPATH, config_dict, res.get("epoch_metrics", []))
        task_records.append(
            {
                "model_stem": Path(MODELPATH).stem,
                "checkpoint_path": str(ckpt_path),
                "alpha": ALPHA,
                "recons_key": RECONS_KEY,
                "best_loss": min(res["taskloss_ls"]),
                "mean_loss": np.array(res["taskloss_ls"]).mean(),
                "epoch1_loss": res["taskloss_ls"][0] if len(res["taskloss_ls"]) > 0 else np.nan,
                "epoch5_loss": res["taskloss_ls"][4] if len(res["taskloss_ls"]) > 4 else np.nan,
                "epoch10_loss": res["taskloss_ls"][9] if len(res["taskloss_ls"]) > 9 else np.nan,
            }
        )

    summary_res = _summarize_records(task_records)

    _log(f"{Path(MODELPATH).stem} {'-'*10} result-summary")
    _log(
        f"{'alpha':>7} | {'recons_key':>10} | {'best_mean':>8} | "
        f"{'best_std':>8} | {'best_max':>8} | {'all_mean':>8} | "
        f"{'e1_mean':>8} | {'e5_mean':>8} | {'e10_mean':>8}"
    )
    _log("-" * 96)
    for ele in summary_res:
        best_arr = np.array(ele['best_loss'], dtype=np.float64)
        mean_arr = np.array(ele['mean_loss'], dtype=np.float64)
        _log(
            f"{ele['alpha']:>7} | {ele['recons_key']:>10} | "
            f"{best_arr.mean():>8.4f} | {best_arr.std():>8.4f} | "
            f"{best_arr.max():>8.4f} | {mean_arr.mean():>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch1_loss']):>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch5_loss']):>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch10_loss']):>8.4f} "
        )
    return summary_res


if __name__ == '__main__':
    import argparse
    import os, datetime
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.ego_status_mlp_agent import EgoStatusMLPAgent

    parser = argparse.ArgumentParser(description="Feature-reverse post-training for EgoStatusMLPAgent.")
    parser.add_argument("--modelpath", "--model-path", required=True, help="Initial NAVSIM/Lightning checkpoint path.")
    parser.add_argument("--work-dir", default="/data/wsc/navsim_workspace/exp/frpt_dn")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--alphas", default="0,0.1,0.02,0.3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--hidden-layer-dim", type=int, default=512)
    parser.add_argument("--time-horizon", type=float, default=4)
    parser.add_argument("--interval-length", type=float, default=0.5)
    args = parser.parse_args()

    print('pid:', os.getpid())
    print(datetime.datetime.now())

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    SEED_ls = tuple(int(seed) for seed in args.seeds.split(",") if seed)
    ALPHA_ls = [float(alpha) for alpha in args.alphas.split(",") if alpha]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'BATCHSIZE={BATCH_SIZE}, SEED_ls={SEED_ls}, ALPHA_ls={ALPHA_ls}, DEVICE={DEVICE}')

    MODELPATH = args.modelpath
    h5_path = Path(args.work_dir) / (Path(MODELPATH).stem + ".h5")
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Missing reconstructed feature cache: {h5_path}. "
            "Build it first with save_recons_fea_to_h5(model, navsim_train_loader, MODELPATH, DEVICE)."
        )

    trajectory_sampling = TrajectorySampling(time_horizon=args.time_horizon, interval_length=args.interval_length)
    model = EgoStatusMLPAgent(trajectory_sampling, args.hidden_layer_dim, args.lr).to(DEVICE)

    print('model path', MODELPATH, '\n')
    model_post_train_all(
        model, MODELPATH, ALPHA_ls, EPOCHS, DEVICE, BATCH_SIZE,
        lr=args.lr, work_dir=args.work_dir, seeds=SEED_ls,
    )

    print(datetime.datetime.now())
