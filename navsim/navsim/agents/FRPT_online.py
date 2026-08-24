import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from typing import Dict
import csv
import logging
import inspect
import fcntl
from collections import defaultdict


logger = logging.getLogger(__name__)


def _log(message):
    print(message)
    logger.info(message)

def _nanmean_or_nan(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


def _move_features_to_device(features: Dict[str, torch.Tensor], device):
    moved_features = {}
    for key, value in features.items():
        value = value.to(device, non_blocking=True)
        if torch.is_floating_point(value):
            value = value.float()
        moved_features[key] = value
    return moved_features


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
        f"Available keys={list(predictions.keys())}"
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


def _compute_online_recons_target(model, input_data, label, recons_key, bs_recons=None):
    def _get_recons(input_chunk, label_chunk, recons_key):
        # signature = inspect.signature(model.get_recons_fea)
        # if "recons_key" in signature.parameters:
        recons_res = model.get_recons_fea(input_chunk, label_chunk, recons_key=recons_key)
        # else:
        #     recons_res = model.get_recons_fea(input_chunk, label_chunk)
        if isinstance(recons_res, dict):
            if recons_key not in recons_res:
                raise KeyError(f"get_recons_fea did not return {recons_key}; keys={list(recons_res.keys())}")
            recons_res = recons_res[recons_key]
        return recons_res

    was_training = model.training
    param_requires_grad = [p.requires_grad for p in model.parameters()]
    model.eval()
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.enable_grad():
            batch_size = label.shape[0]
            if bs_recons is None or bs_recons <= 0 or bs_recons >= batch_size:
                recons_res = _get_recons(input_data, label, recons_key)
                return recons_res.detach()

            recons_chunks = []
            for start in range(0, batch_size, bs_recons):
                end = min(start + bs_recons, batch_size)
                input_chunk = {key: value[start:end] for key, value in input_data.items()}
                recons_res = _get_recons(input_chunk, label[start:end], recons_key)
                recons_chunks.append(recons_res.detach())
            return torch.cat(recons_chunks, dim=0)
    finally:
        for p, req_grad in zip(model.parameters(), param_requires_grad):
            p.requires_grad_(req_grad)
        if was_training:
            model.train()


def frpt(model, trainloader, recons_key, alpha, epochs, device, lr=3e-5, bs_recons=None):
    model.set_recons_param(recons_key)
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
            input_data, label = _unpack_navsim_batch(batch)
            input_data = _move_features_to_device(input_data, device)
            label = label.to(device, non_blocking=True).float()
            recons_feature = _compute_online_recons_target(model, input_data, label, recons_key, bs_recons=bs_recons)

            res = model(input_data)
            if forward_key is None:
                forward_key = _resolve_forward_key(res, recons_key)
            forward_feature = res[forward_key]
            recons_feature = recons_feature.to(device=forward_feature.device, dtype=forward_feature.dtype)
            trajectory = res['trajectory']
            label = label.to(dtype=trajectory.dtype)
            reconsloss = reconscrit(forward_feature, recons_feature)
            clsloss = taskcrit(trajectory, label)
            with torch.no_grad():
                epoch_forward_sum += forward_feature.detach().sum().item()
                epoch_forward_absmax = max(epoch_forward_absmax, forward_feature.detach().abs().max().item())
                epoch_recons_sum += recons_feature.detach().sum().item()
                epoch_recons_absmax = max(epoch_recons_absmax, recons_feature.detach().abs().max().item())
                epoch_feature_count += forward_feature.numel()
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
            loss = loss_weight * reconsloss + clsloss 

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
            input_data, label = _unpack_navsim_batch(batch)
            input_data = _move_features_to_device(input_data, device)
            label = label.to(device, non_blocking=True).float()

            res = model(input_data)
            trajectory = res['trajectory']
            label = label.to(dtype=trajectory.dtype)
            loss = taskcrit(trajectory, label)

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
        key = (record["alpha"], record["recons_key"])
        group = grouped[key]
        group["checkpoint_paths"].append(record["checkpoint_path"])
        group["best_loss"].append(record["best_loss"])
        group["mean_loss"].append(record["mean_loss"])
        group["epoch1_loss"].append(record["epoch1_loss"])
        group["epoch5_loss"].append(record["epoch5_loss"])
        group["epoch10_loss"].append(record["epoch10_loss"])
    summary_res = []
    for alpha, recons_key in sorted(grouped.keys(), key=lambda item: (float(item[0]), str(item[1]))):
        group = grouped[(alpha, recons_key)]
        summary_res.append({"alpha": alpha, "recons_key": recons_key, **group})
    return summary_res


def model_post_train_all(model, MODELPATH, ALPHA_ls, EPOCHS, 
                         DEVICE, lr=3e-5, save_ckpt=True, save_res=True,
                         work_dir="/data/wsc/navsim_workspace/exp/frpt_dn", seeds=(0, 1, 2),
                         trainloader=None, bs_recons=None, recons_keys=None, tasks=None):
    """ ego status mlp: bs_recons=None  
        resnet34-based: bs_recon = 32 or 64
        vit-based: bs_recons = 32 or 64
    """
    # TODO 把这个函数搞成多进程的,把任务按照（alpha，recons_key, seed)这个三元组设置分成若干个独立的小任务，
    # 按照传入的TASK_WORKERS来分配这些小任务，每个进程在初始化的时候分别生成互不影响的model_template，trainloader等等，这些东西给该进程的每个小任务应该是相同的的。
    # 同时做好实验结果打印和保存的管理，不要互相影响。保证实验结果的保存和原来一样，不要因为多进程的事情互相影响。
    # 在确定trainloader的加载方式之后，可以【酌情】参考/data/dn/FRTP_revision1/imagecls/FRPT_ood_online.py的多进程管理方式，但一定要适配轨迹规划任务这边的情况
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    SAVEPATH = work_dir / Path(MODELPATH).stem
    CKPTDIR = work_dir / "ckpts"
    METRICSPATH = work_dir / "post_train_metrics.csv"
    CKPTDIR.mkdir(parents=True, exist_ok=True)
    assert trainloader is not None  
    model.to(DEVICE)
    if recons_keys is None:
        recons_keys = model.get_fea_name() + ['out']
    
    if tasks is None:
        tasks = build_post_train_tasks(ALPHA_ls, recons_keys, seeds)

    task_records = []
    for task in tasks:
        ALPHA = task["alpha"]
        RECONS_KEY = task["recons_key"]
        seed = int(task["seed"])
        _log(f"[TASK] start alpha={ALPHA}, recons_key={RECONS_KEY}, seed={seed}, device={DEVICE}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        _load_agent_state_dict(model, MODELPATH, DEVICE)
        if RECONS_KEY == 'out':
            res = normal_pt(model, trainloader, epochs=EPOCHS, device=DEVICE, lr=lr)
        else:
            res = frpt(
                model,
                trainloader,
                RECONS_KEY,
                ALPHA,
                epochs=EPOCHS,
                device=DEVICE,
                lr=lr,
                bs_recons=bs_recons,
            )
        ckpt_model = Path(MODELPATH).stem + f"_alpha{ALPHA}_" + RECONS_KEY + f"_ptseed{seed}"
        ckpt_path = CKPTDIR / f"{ckpt_model}.ckpt"
        if save_ckpt:
            _save_navsim_checkpoint(model, ckpt_path)
        config_dict = {'model':ckpt_model, 'checkpoint_path': str(ckpt_path), 'alpha':ALPHA, 'recons_key': RECONS_KEY, 'seed':seed}
        result_record = config_dict | res
        if save_res:
            _append_result_pt(str(SAVEPATH)+f"_{EPOCHS}e.pt", [result_record])
        _append_epoch_metrics_csv(METRICSPATH, config_dict, res.get("epoch_metrics", []))

        record = {
            "checkpoint_path": str(ckpt_path),
            "alpha": ALPHA,
            "recons_key": RECONS_KEY,
            "best_loss": min(res["taskloss_ls"]),
            "mean_loss": np.array(res["taskloss_ls"]).mean(),
            "epoch1_loss": res["taskloss_ls"][0] if len(res["taskloss_ls"]) > 0 else np.nan,
            "epoch5_loss": res["taskloss_ls"][4] if len(res["taskloss_ls"]) > 4 else np.nan,
            "epoch10_loss": res["taskloss_ls"][9] if len(res["taskloss_ls"]) > 9 else np.nan,
        }
        task_records.append(record)

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
    trajectory_sampling = TrajectorySampling(time_horizon=args.time_horizon, interval_length=args.interval_length)
    model = EgoStatusMLPAgent(trajectory_sampling, args.hidden_layer_dim, args.lr).to(DEVICE)

    print('model path', MODELPATH, '\n')
    model_post_train_all(
        model, MODELPATH, ALPHA_ls, EPOCHS, DEVICE, BATCH_SIZE,
        lr=args.lr, work_dir=args.work_dir, seeds=SEED_ls, bs_recons=None
    )

    print(datetime.datetime.now())
