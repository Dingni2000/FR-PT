import os

os.environ.setdefault("FRPT_CPU_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", os.environ["FRPT_CPU_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["FRPT_CPU_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["FRPT_CPU_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["FRPT_CPU_THREADS"])

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
from torchvision import datasets, transforms
import numpy as np
import gc
import copy
import contextlib
import io
import sys
import shutil
import datetime
import argparse
import runpy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def _configure_cpu_threads():
    num_threads = int(os.environ.get("FRPT_CPU_THREADS", "1"))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        pass
    return num_threads


_configure_cpu_threads()


def _reexec_from_runtime_snapshot():
    if os.environ.get("FRPT_RUNTIME_SNAPSHOT") == "1":
        return
    if os.environ.get("FRPT_DISABLE_RUNTIME_SNAPSHOT") == "1":
        return

    source = Path(__file__).resolve()
    run_dir = source.parent / ".frpt_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot = run_dir / f"FRPT_run_{timestamp}_pid{os.getpid()}.py"
    shutil.copy2(source, snapshot)

    env = os.environ.copy()
    env["FRPT_RUNTIME_SNAPSHOT"] = "1"
    env["FRPT_RUNTIME_SOURCE"] = str(source)
    env["FRPT_RUNTIME_SCRIPT"] = str(snapshot)

    imagecls_dir = str(source.parent)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = imagecls_dir if not pythonpath else imagecls_dir + os.pathsep + pythonpath

    print(f"[INFO] FRPT runtime snapshot: {snapshot}", flush=True)
    os.execvpe(sys.executable, [sys.executable, str(snapshot), *sys.argv[1:]], env)


def _run_external_script_if_requested():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-script", type=str, default=None)
    args, _ = parser.parse_known_args()
    if not args.run_script:
        return False

    run_script = Path(args.run_script).expanduser().resolve()
    if not run_script.exists():
        raise FileNotFoundError(f"run script not found: {run_script}")

    runtime_script = Path(os.environ.get("FRPT_RUNTIME_SCRIPT", __file__)).resolve()
    run_dir = runtime_script.parent
    run_snapshot = run_dir / f"{runtime_script.stem}_{run_script.stem}.py"
    shutil.copy2(run_script, run_snapshot)
    print(f"[INFO] FRPT external run script: {run_snapshot}", flush=True)
    sys.modules["FRPT"] = sys.modules[__name__]

    run_globals = {
        "torch": torch,
        "nn": nn,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "h5py": h5py,
        "Path": Path,
        "datasets": datasets,
        "transforms": transforms,
        "np": np,
        "get_score": get_score,
        "HDF5Dataset": HDF5Dataset,
        "InMemoryFeatureDataset": InMemoryFeatureDataset,
        "_make_feature_dataset": _make_feature_dataset,
        "save_recons_fea_to_h5": save_recons_fea_to_h5,
        "frpt": frpt,
        "model_post_train_all": model_post_train_all,
        "ablation_all": ablation_all,
        "load_model_checkpoint": load_model_checkpoint,
        "prepare_online_experiment": prepare_online_experiment,
        "RUN_SCRIPT": run_snapshot,
        "RUN_SCRIPT_DIR": run_script.parent,
        "FRPT_RUNTIME_SCRIPT": runtime_script,
    }
    runpy.run_path(str(run_snapshot), init_globals=run_globals)
    return True


def get_score(model, data_loader, DEVICE):
    model.eval()  # Set the module in evaluation mode.
    correct = 0.0   # 正确率
    with torch.no_grad():  # 不会进行计算梯度，也不会进行反向传播
        for data, target in data_loader:
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            output = model(data)['out']
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct/len(data_loader.dataset)    


class HDF5Dataset(Dataset):
    def __init__(self, file_path, recons_key):
        self.file_path = file_path
        self.recons_key = recons_key
        self._file = None
        self._length = None

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __len__(self):
        if self._length is None:
            with h5py.File(self.file_path, "r") as f:
                self._length = len(f["input"])
        return self._length

    def __getitem__(self,idx):
        f = self._get_file()
        x = torch.from_numpy(np.asarray(f["input"][idx], dtype=np.float32))
        recons_fea = torch.from_numpy(np.asarray(f[self.recons_key][idx], dtype=np.float32))
        y = torch.as_tensor(f["label"][idx], dtype=torch.long)
        return {'input':x, self.recons_key:recons_fea, "label":y}

    def __getitems__(self, indices):
        f = self._get_file()
        indices = np.asarray(indices)
        order = np.argsort(indices)
        sorted_indices = indices[order]
        restore = np.empty_like(order)
        restore[order] = np.arange(len(order))

        x = np.asarray(f["input"][sorted_indices], dtype=np.float32)[restore]
        recons_fea = np.asarray(f[self.recons_key][sorted_indices], dtype=np.float32)[restore]
        y = np.asarray(f["label"][sorted_indices], dtype=np.int64)[restore]
        return [
            {
                "input": torch.from_numpy(x[i]),
                self.recons_key: torch.from_numpy(recons_fea[i]),
                "label": torch.as_tensor(y[i], dtype=torch.long),
            }
            for i in range(len(indices))
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()
    

class InMemoryFeatureDataset(Dataset):
    def __init__(self, file_path, recons_key):
        self.file_path = file_path
        self.recons_key = recons_key
        with h5py.File(self.file_path, "r") as f:
            self.input = torch.from_numpy(np.asarray(f["input"][:], dtype=np.float32))
            self.recons_fea = torch.from_numpy(np.asarray(f[self.recons_key][:], dtype=np.float32))
            self.label = torch.from_numpy(np.asarray(f["label"][:], dtype=np.int64))

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            self.recons_key: self.recons_fea[idx],
            "label": self.label[idx],
        }


def _dataset_key_nbytes(file_path, recons_key):
    with h5py.File(file_path, "r") as f:
        return f["input"].nbytes + f["label"].nbytes + f[recons_key].nbytes


def _choose_feature_backend(file_path, recons_keys, num_processes):
    backend = os.environ.get("FRPT_DATA_BACKEND", "memory").lower()
    if backend not in {"auto", "hdf5", "memory"}:
        raise ValueError("FRPT_DATA_BACKEND must be one of: auto, hdf5, memory")
    if backend != "auto":
        return backend

    max_gb = float(os.environ.get("FRPT_MEMORY_DATASET_MAX_GB", "96"))
    max_key_bytes = max(_dataset_key_nbytes(file_path, key) for key in recons_keys)
    estimated_gb = max_key_bytes * max(1, int(num_processes)) / (1024 ** 3)
    return "memory" if estimated_gb <= max_gb else "hdf5"


def _make_feature_dataset(file_path, recons_key, backend):
    if backend == "memory":
        return InMemoryFeatureDataset(file_path, recons_key)
    if backend == "hdf5":
        return HDF5Dataset(file_path, recons_key)
    raise ValueError(f"Unknown feature dataset backend: {backend}")


def _nanmean_or_nan(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


def _result_key(model_name, alpha, recons_key, seed):
    return (str(model_name), float(alpha), str(recons_key), int(seed))


def _is_complete_result(res, epochs):
    if not isinstance(res, dict):
        return False
    test_acc_ls = res.get("test_acc_ls")
    return isinstance(test_acc_ls, (list, tuple)) and len(test_acc_ls) >= epochs


def _save_results_with_backup(results, save_file):
    save_path = Path(save_file)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    if save_path.exists():
        shutil.copy2(save_path, save_path.with_suffix(save_path.suffix + ".bak"))
    torch.save(results, tmp_path)
    os.replace(tmp_path, save_path)


def _print_frpt_done(done, total, res):
    best = max(res["test_acc_ls"]) if res.get("test_acc_ls") else float("nan")
    print(
        f"[DONE] {done}/{total} alpha={res['alpha']}, "
        f"recons_key={res['recons_key']}, seed={res['seed']}, best={best:.4f}",
        flush=True,
    )


def _print_ablation_done(done, total, res):
    best = max(res["test_acc_ls"]) if res.get("test_acc_ls") else float("nan")
    print(
        f"[DONE] {done}/{total} alpha={res['alpha']}, "
        f"oe={res['oe']}, seed={res['seed']}, best={best:.4f}",
        flush=True,
    )


def _dataloader_kwargs(num_workers):
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers and num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def _check_recons_file(filepath, required_keys):
    recons_file = Path(filepath)
    if not recons_file.exists():
        available_files = sorted(p.name for p in recons_file.parent.glob("*.h5"))
        available_msg = "\n  ".join(available_files[:30]) if available_files else "(none)"
        if len(available_files) > 30:
            available_msg += f"\n  ... ({len(available_files) - 30} more)"
        raise FileNotFoundError(
            "Missing reconstruction feature file.\n"
            f"Expected: {recons_file}\n"
            "Please generate it with save_recons_fea_to_h5(...) using the same checkpoint.\n"
            f"Available .h5 files in {recons_file.parent}:\n  {available_msg}"
        )
    with h5py.File(recons_file, "r") as f:
        missing_keys = [key for key in required_keys if key not in f]
    if missing_keys:
        raise KeyError(
            f"Reconstruction feature file is missing required keys: {missing_keys}\n"
            f"File: {recons_file}"
        )


def _load_model_checkpoint(
    model,
    modelpath,
    device,
    use_vit_load=False,
    vit_checkpoint_key="teacher",
    verbose=False,
):
    if not use_vit_load:
        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
        return {"loader": "torch.load_state_dict", "modelpath": str(modelpath)}

    try:
        from models import load_small_dataset_vit_ssl_checkpoint
    except ImportError:
        from imagecls.models import load_small_dataset_vit_ssl_checkpoint

    load_info = load_small_dataset_vit_ssl_checkpoint(
        model,
        modelpath,
        checkpoint_key=vit_checkpoint_key,
        verbose=verbose,
    )
    load_info = dict(load_info)
    load_info["loader"] = "load_small_dataset_vit_ssl_checkpoint"
    load_info["modelpath"] = str(modelpath)
    load_info["checkpoint_key"] = vit_checkpoint_key
    return load_info


def load_model_checkpoint(
    model,
    modelpath,
    device,
    use_vit_load=False,
    vit_checkpoint_key="teacher",
    verbose=False,
):
    return _load_model_checkpoint(
        model,
        modelpath,
        device,
        use_vit_load=use_vit_load,
        vit_checkpoint_key=vit_checkpoint_key,
        verbose=verbose,
    )


def save_recons_fea_to_h5(model, data_loader, filepath, device, empty_cache_every=1):
    flag = True
    dsets = {}
    param_requires_grad = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        with h5py.File(filepath, "w") as f:
            for batch_idx, (x, y) in enumerate(data_loader):
                x_cpu = x.detach().cpu()
                y_cpu = y.detach().cpu()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                bs = x.shape[0]

                res = model.get_recons_fea(x, y)

                if flag:
                    flag = False
                    xshape = tuple(x_cpu.shape[1:])
                    dsets["input"] = f.create_dataset(
                        "input", shape=(0,) + xshape, maxshape=(None,) + xshape, dtype=np.float32
                    )
                    dsets["label"] = f.create_dataset(
                        "label", shape=(0,), maxshape=(None,), dtype=np.int32
                    )

                    for name, fea in res.items():
                        fea_shape = tuple(fea.shape[1:])
                        dsets[name] = f.create_dataset(
                            name, shape=(0,) + fea_shape, maxshape=(None,) + fea_shape, dtype=np.float32
                        )

                new_size = dsets["input"].shape[0] + bs
                for dset in dsets.values():
                    dset.resize(new_size, axis=0)

                dsets["input"][-bs:] = x_cpu.numpy()
                dsets["label"][-bs:] = y_cpu.numpy()
                del x_cpu, y_cpu

                for name in list(res.keys()):
                    fea_cpu = res[name].detach().cpu().numpy()
                    dsets[name][-bs:] = fea_cpu
                    del fea_cpu
                    del res[name]

                f.flush()
                del res, x, y

                if torch.cuda.is_available() and empty_cache_every and (batch_idx + 1) % empty_cache_every == 0:
                    torch.cuda.empty_cache()
                gc.collect()
    finally:
        for p, req_grad in zip(model.parameters(), param_requires_grad):
            p.requires_grad_(req_grad)
    print(f"saved recons data to {filepath}")


def frpt(model, trainloader, testloader, recons_key, alpha, epochs, device, lr=1e-4, get_train=False):
    for p in model.parameters():
        p.requires_grad = True
    freeze_from = model.get_fea_id(recons_key)
    if freeze_from is None:
        raise ValueError(f"Unknown recons_key={recons_key}; available keys={model.get_fea_name()}")
    for _, layer in list(model.named_children())[freeze_from:]:
        for p in layer.parameters():
            p.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=5e-2)
    testacc_frpt = []
    trainacc_frpt = []
    clsloss_ls = []
    reconsloss_ls = []
    loss_ls = []
    clscrit = nn.CrossEntropyLoss()
    reconscrit = nn.MSELoss()
    ema_beta = 0.9
    ema_cls = None
    ema_recons = None
        
    for epoch in range(epochs):
        model.train()  # 训练模式
        for batch in trainloader:
            input_data = batch["input"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            recons_feature = batch[recons_key].to(device, non_blocking=True)

            res = model(input_data)
            clsloss = clscrit(res['out'], label)
            reconsloss = reconscrit(res[recons_key[7:]], recons_feature)
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

        test_acc = get_score(model, testloader, device)
        
        if get_train:
            train_acc = get_score(model, trainloader)
            trainacc_frpt.append(train_acc)
            print(f"[{epoch+1}] trainacc={train_acc:4f}, testacc={test_acc:4f}, loss={loss.item():4f}, reconsloss={reconsloss.item():4f}, clsloss={clsloss.item():4f}, clsloss={clsloss.item():4f}, lossweight={loss_weight.item():4f}")
        else:
            print(f"[{epoch+1}] testacc={test_acc:4f}, loss={loss.item():4f}, reconsloss={reconsloss.item():4f}, clsloss={clsloss.item():4f}, lossweight={loss_weight.item():4f}")
        testacc_frpt.append(test_acc)
        reconsloss_ls.append(reconsloss.item())
        clsloss_ls.append(clsloss.item())  # pred
        loss_ls.append(loss.item())
    print(f'[RECORD] alpha={alpha}, recons_key={recons_key}, best testacc={max(testacc_frpt)}\n')
    return {'reconsloss_ls':reconsloss_ls, 'clsloss_ls': clsloss_ls, 'loss_ls':loss_ls,
             'test_acc_ls': testacc_frpt, 'train_acc_ls':trainacc_frpt}


_FRPT_WORKER_CTX = {}


def _init_frpt_worker(modelpath, filepath, model_name, epochs, device_str, batch_size,
                      lr, num_workers, use_vit_load, vit_checkpoint_key, data_backend):
    _configure_cpu_threads()
    model_template, _, test_loader = prepare_online_experiment(
        modelpath,
        batch_size=batch_size,
        num_workers=num_workers,
        device=torch.device("cpu"),
    )
    model_template = model_template.to("cpu")
    _FRPT_WORKER_CTX.update({
        "model_template": model_template,
        "modelpath": modelpath,
        "filepath": filepath,
        "test_loader": test_loader,
        "model_name": model_name,
        "epochs": epochs,
        "device_str": device_str,
        "batch_size": batch_size,
        "lr": lr,
        "num_workers": num_workers,
        "use_vit_load": use_vit_load,
        "vit_checkpoint_key": vit_checkpoint_key,
        "data_backend": data_backend,
    })


def _frpt_one_seed_worker(args):
    task_idx, alpha, recons_key, seed = args
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        ctx = _FRPT_WORKER_CTX
        device = torch.device(ctx["device_str"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        g = torch.Generator().manual_seed(seed)

        trainset_fr = _make_feature_dataset(ctx["filepath"], recons_key, ctx["data_backend"])
        train_loader_fr = DataLoader(
            trainset_fr,
            batch_size=ctx["batch_size"],
            shuffle=True,
            generator=g,
            **_dataloader_kwargs(ctx["num_workers"]),
        )

        worker_model = copy.deepcopy(ctx["model_template"]).to(device)
        _load_model_checkpoint(
            worker_model,
            ctx["modelpath"],
            device,
            use_vit_load=ctx["use_vit_load"],
            vit_checkpoint_key=ctx["vit_checkpoint_key"],
            verbose=False,
        )
        res = frpt(worker_model, train_loader_fr, ctx["test_loader"], recons_key, alpha,
                   epochs=ctx["epochs"], device=device, lr=ctx["lr"])
        config_dict = {'model':ctx["model_name"], 'alpha':alpha, 'recons_key':recons_key, 'seed':seed}

        del worker_model, train_loader_fr, trainset_fr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return task_idx, alpha, recons_key, seed, (config_dict | res), log_buffer.getvalue()


def model_post_train_all(MODELPATH, BATCH_SIZE, NUM_WORKERS, DEVICE,
                         ALPHA_ls, SEED_ls, EPOCHS, lr=1e-4, save_res=True,
                         NUM_PROCESSES=4, version=None):
    model, train_loader, test_loader = prepare_online_experiment(
        MODELPATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    if Path(MODELPATH).stem in ['vit_timnet_patch8_input64', 'vit_cifar100_patch4_input32']:
        use_vit_load = True
    else: use_vit_load = False
    vit_checkpoint_key = 'teacher'
    imgclsdata = Path(MODELPATH).parent.name
    if version == 'new_pooling':
        FILEPATH = f"/data/dn/FRTP_revision1/imagecls/recons_data_new/{imgclsdata}/" + Path(MODELPATH).stem + ".h5"
        SAVEPATH = f"/data/dn/FRTP_revision1/imagecls/logs_new/{imgclsdata}/"+Path(MODELPATH).stem
    else:
        FILEPATH = f"/data/dn/FRTP_revision1/imagecls/recons_data/{imgclsdata}/" + Path(MODELPATH).stem + ".h5"
        SAVEPATH = f"/data/dn/FRTP_revision1/imagecls/logs/{imgclsdata}/"+Path(MODELPATH).stem
    print('recons_data_path',FILEPATH)
    print('save_path',SAVEPATH)
    Path(SAVEPATH).parent.mkdir(parents=True, exist_ok=True)
    model_name = Path(MODELPATH).stem
    recons_keys = list(model.get_fea_name())
    _check_recons_file(FILEPATH, ["input", "label", *recons_keys])
    save_file = SAVEPATH+f"_{EPOCHS}e.pt" 
    if save_res:
        try:
            datafile_test = torch.load(save_file, weights_only=False)
        except (FileNotFoundError, EOFError):
            datafile_test = []
    else:
        datafile_test = []
    completed_results = {}
    duplicate_existing = 0
    for old_res in datafile_test:
        if not _is_complete_result(old_res, EPOCHS):
            continue
        try:
            key = _result_key(old_res["model"], old_res["alpha"], old_res["recons_key"], old_res["seed"])
        except KeyError:
            continue
        if key in completed_results:
            duplicate_existing += 1
        completed_results[key] = old_res

    tasks = []
    task_idx = 0
    skipped_existing = 0
    for ALPHA in ALPHA_ls:
        for RECONS_KEY in recons_keys:
            for seed in SEED_ls:
                key = _result_key(model_name, ALPHA, RECONS_KEY, seed)
                if key in completed_results:
                    skipped_existing += 1
                    continue
                tasks.append((task_idx, ALPHA, RECONS_KEY, seed))
                task_idx += 1

    if NUM_PROCESSES is None:
        NUM_PROCESSES = int(os.environ.get("FRPT_NUM_PROCESSES", min(4, len(tasks), os.cpu_count() or 1)))
    NUM_PROCESSES = max(1, min(int(NUM_PROCESSES), len(tasks) if tasks else 1))
    print(f"NUM PROCESSES: {NUM_PROCESSES}")
    data_backend = _choose_feature_backend(FILEPATH, recons_keys, NUM_PROCESSES)
    print(f"FRPT DATA BACKEND: {data_backend}")
    if save_res:
        print(f"[INFO] loaded {len(datafile_test)} existing results from {save_file}")
        if skipped_existing:
            print(f"[INFO] skipping {skipped_existing} completed FRPT experiment(s)")
        if duplicate_existing:
            print(f"[WARN] found {duplicate_existing} duplicate completed result key(s); using the latest entry for summary")

    worker_args = tasks
    results_by_idx = {}
    new_results = []

    if worker_args:
        print(f"[INFO] running {len(worker_args)} FRPT experiments with {NUM_PROCESSES} process(es)")
    total_tasks = len(worker_args)
    done_tasks = 0
    if NUM_PROCESSES == 1:
        _init_frpt_worker(str(MODELPATH), FILEPATH, model_name, EPOCHS, str(DEVICE),
                          BATCH_SIZE, lr, NUM_WORKERS, use_vit_load,
                          vit_checkpoint_key, data_backend)
        for args in worker_args:
            idx, _, _, _, exp_res, log_text = _frpt_one_seed_worker(args)
            print(log_text, end="")
            results_by_idx[idx] = exp_res
            done_tasks += 1
            _print_frpt_done(done_tasks, total_tasks, exp_res)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=NUM_PROCESSES,
            mp_context=ctx,
            initializer=_init_frpt_worker,
            initargs=(
                str(MODELPATH), FILEPATH, model_name, EPOCHS, str(DEVICE),
                BATCH_SIZE, lr, NUM_WORKERS, use_vit_load,
                vit_checkpoint_key, data_backend,
            ),
        ) as executor:
            futures = [executor.submit(_frpt_one_seed_worker, args) for args in worker_args]
            for future in as_completed(futures):
                idx, _, _, _, exp_res, log_text = future.result()
                results_by_idx[idx] = (exp_res, log_text)
                results_key = _result_key(exp_res["model"], exp_res["alpha"], exp_res["recons_key"], exp_res["seed"])
                completed_results[results_key] = exp_res
                new_results.append(exp_res)
                done_tasks += 1
                _print_frpt_done(done_tasks, total_tasks, exp_res)
                if save_res:
                    datafile_test.append(exp_res)
                    _save_results_with_backup(datafile_test, save_file)
        for idx in range(len(worker_args)):
            exp_res, log_text = results_by_idx[idx]
            print(log_text, end="")
            results_by_idx[idx] = exp_res
    for idx in range(len(worker_args)):
        res = results_by_idx[idx]
        results_key = _result_key(res["model"], res["alpha"], res["recons_key"], res["seed"])
        if results_key not in completed_results:
            completed_results[results_key] = res
        if res not in new_results:
            new_results.append(res)

    summary_res = []
    for ALPHA in ALPHA_ls:
        for RECONS_KEY in model.get_fea_name():
            one_exp = []
            summary_res.append({'model':Path(MODELPATH).stem, 'alpha':ALPHA,
                                 'recons_key': RECONS_KEY, 'best_testacc':[], 'mean_testacc':[],
                                 'epoch1_testacc': [], 'epoch5_testacc': [], 'epoch10_testacc': []})
            for seed in SEED_ls:
                key = _result_key(model_name, ALPHA, RECONS_KEY, seed)
                if key not in completed_results:
                    print(f"[WARN] missing result for alpha={ALPHA}, recons_key={RECONS_KEY}, seed={seed}; skip summary entry")
                    continue
                res = completed_results[key]
                one_exp.append(res)
                summary_res[-1]['best_testacc'].append(max(res['test_acc_ls']))
                summary_res[-1]['mean_testacc'].append(np.array(res['test_acc_ls']).mean())
                for epoch_idx, summary_key in [(0, 'epoch1_testacc'), (4, 'epoch5_testacc'), (9, 'epoch10_testacc')]:
                    if len(res['test_acc_ls']) > epoch_idx:
                        summary_res[-1][summary_key].append(res['test_acc_ls'][epoch_idx])
                    else:
                        summary_res[-1][summary_key].append(np.nan)

    if save_res and new_results and NUM_PROCESSES == 1:
        datafile_test.extend(new_results)
        _save_results_with_backup(datafile_test, save_file)
        print(f"[INFO] appended {len(new_results)} new result(s) to {save_file}")
    elif save_res and new_results:
        print(f"[INFO] saved {len(new_results)} new result(s) incrementally to {save_file}")
    elif save_res:
        print(f"[INFO] no new FRPT results to save; all requested experiments already completed")

    print(Path(MODELPATH).stem, '-'*10, 'result-summary')
    print(
        f"{'alpha':>7} | {'recons_key':>10} | {'best_mean':>8} | "
        f"{'best_std':>8} | {'best_max':>8} | {'all_mean':>8} | "
        f"{'e1_mean':>8} | {'e5_mean':>8} | {'e10_mean':>8}"
    )
    print("-" * 96)
    for ele in summary_res:
        best_arr = np.array(ele['best_testacc'], dtype=np.float64)
        mean_arr = np.array(ele['mean_testacc'], dtype=np.float64)
        print(
            f"{ele['alpha']:>7} | {ele['recons_key']:>10} | "
            f"{best_arr.mean():>8.4f} | {best_arr.std():>8.4f} | "
            f"{best_arr.max():>8.4f} | {mean_arr.mean():>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch1_testacc']):>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch5_testacc']):>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch10_testacc']):>8.4f} "
        )
    return summary_res


_ABLATION_WORKER_CTX = {}


def _init_ablation_worker(modelpath, filepath, model_name, epochs, device_str,
                          batch_size, num_workers, use_vit_load, vit_checkpoint_key, data_backend):
    _configure_cpu_threads()
    model_template, _, test_loader = prepare_online_experiment(
        modelpath,
        batch_size=batch_size,
        num_workers=num_workers,
        device=torch.device("cpu"),
    )
    model_template = model_template.to("cpu")
    _ABLATION_WORKER_CTX.update({
        "model_template": model_template,
        "modelpath": modelpath,
        "filepath": filepath,
        "test_loader": test_loader,
        "model_name": model_name,
        "epochs": epochs,
        "device_str": device_str,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "use_vit_load": use_vit_load,
        "vit_checkpoint_key": vit_checkpoint_key,
        "data_backend": data_backend,
    })


def _ablation_one_seed_worker(args):
    task_idx, alpha, oe, seed = args
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        ctx = _ABLATION_WORKER_CTX
        device = torch.device(ctx["device_str"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        g = torch.Generator().manual_seed(seed)

        trainset_fr = _make_feature_dataset(ctx["filepath"], 'recons_out', ctx["data_backend"])
        train_loader_fr = DataLoader(
            trainset_fr,
            batch_size=ctx["batch_size"],
            shuffle=True,
            generator=g,
            **_dataloader_kwargs(ctx["num_workers"]),
        )

        worker_model = copy.deepcopy(ctx["model_template"]).to(device)
        _load_model_checkpoint(
            worker_model,
            ctx["modelpath"],
            device,
            use_vit_load=ctx["use_vit_load"],
            vit_checkpoint_key=ctx["vit_checkpoint_key"],
            verbose=False,
        )
        for p in worker_model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(worker_model.parameters(), lr=0.001)
        testacc_frpt = []
        clsloss_ls, reconsloss_ls, loss_ls = [], [], []
        clscrit = nn.CrossEntropyLoss()
        reconscrit = nn.MSELoss()

        for epoch in range(ctx["epochs"]):
            worker_model.train()
            for batch in train_loader_fr:
                input_data = batch["input"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                res = worker_model(input_data)
                num_classes = res['out'].shape[1]
                if oe == 'nearest-embed':
                    recons_feature = batch['recons_out'].to(device, non_blocking=True)
                elif oe == 'max-assign':
                    recons_feature = res['out'].detach().clone()
                    row_idx = torch.arange(recons_feature.shape[0], device=device)
                    recons_feature[row_idx, label] = torch.max(recons_feature, dim=1)[0]
                    recons_feature[recons_feature>10.0]=10.0  # NOTE new add
                    recons_feature[recons_feature<-10.0]=-10.0  # NOTE new add
                elif oe == 'one-hot':
                    recons_feature = nn.functional.one_hot(label, num_classes=num_classes).float()
                else:
                    raise ValueError

                reconsloss = reconscrit(res['out'], recons_feature)
                clsloss = clscrit(res['out'], label)
                loss = alpha * reconsloss + clsloss  # 两个矩阵差的范数

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_acc = get_score(worker_model, ctx["test_loader"], device)

            print(f"[{epoch+1}] testacc={test_acc:4f}, loss={loss.item():4f}, reconsloss={reconsloss.item():4f}, clsloss={clsloss.item():4f}")
            testacc_frpt.append(test_acc)
            reconsloss_ls.append(reconsloss.item())
            clsloss_ls.append(clsloss.item())
            loss_ls.append(loss.item())

        oneset_res = {'reconsloss_ls':reconsloss_ls, 'clsloss_ls': clsloss_ls,
                      'loss_ls':loss_ls, 'test_acc_ls': testacc_frpt}
        config_dict = {'model':ctx["model_name"], 'alpha':alpha, 'recons_key': 'recons_out',
                       'oe':oe, 'seed':seed}
        print(f'[RECORD] alpha={alpha}, oe={oe}, seed={seed}, best testacc={max(testacc_frpt)}\n')

        del worker_model, train_loader_fr, trainset_fr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return task_idx, alpha, oe, seed, (config_dict | oneset_res), log_buffer.getvalue()


def ablation_all(MODELPATH, BATCH_SIZE, NUM_WORKERS, DEVICE,
                 ALPHA_ls, SEED_ls, EPOCHS, save_res=True, NUM_PROCESSES=4):
    model, train_loader, test_loader = prepare_online_experiment(
        MODELPATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    if Path(MODELPATH).stem in ['vit_timnet_patch8_input64', 'vit_cifar100_patch4_input32']:
        use_vit_load = True
    else: use_vit_load = False
    vit_checkpoint_key = 'teacher'
    MODELPATH = Path(MODELPATH)
    imgclsdata = MODELPATH.parent.name
    FILEPATH = f"/data/dn/FRTP_revision1/imagecls/recons_data/{imgclsdata}/" + MODELPATH.stem + ".h5"
    SAVEPATH = f"/data/dn/FRTP_revision1/imagecls/logs/ablation/ab_"+MODELPATH.stem
    Path(SAVEPATH).parent.mkdir(parents=True, exist_ok=True)
    recons_file = Path(FILEPATH)
    if not recons_file.exists():
        available_files = sorted(p.name for p in recons_file.parent.glob("*.h5"))
        available_msg = "\n  ".join(available_files[:30]) if available_files else "(none)"
        if len(available_files) > 30:
            available_msg += f"\n  ... ({len(available_files) - 30} more)"
        raise FileNotFoundError(
            "Missing reconstruction feature file for ablation_all.\n"
            f"Expected: {recons_file}\n"
            "Please generate it with save_recons_fea_to_h5(...) using the same checkpoint.\n"
            f"Available .h5 files in {recons_file.parent}:\n  {available_msg}"
        )
    with h5py.File(recons_file, "r") as f:
        missing_keys = [key for key in ["input", "label", "recons_out"] if key not in f]
    if missing_keys:
        raise KeyError(
            f"Reconstruction feature file is missing required keys: {missing_keys}\n"
            f"File: {recons_file}"
        )

    oe_ls = ['one-hot', 'max-assign', 'nearest-embed']
    tasks = []
    task_idx = 0
    for ALPHA in ALPHA_ls:
        for OE in oe_ls:
            for seed in SEED_ls:
                tasks.append((task_idx, ALPHA, OE, seed))
                task_idx += 1

    if NUM_PROCESSES is None:
        NUM_PROCESSES = int(os.environ.get("FRPT_ABLATION_NUM_PROCESSES", min(4, len(tasks), os.cpu_count() or 1)))
    NUM_PROCESSES = max(1, min(int(NUM_PROCESSES), len(tasks) if tasks else 1))
    print(f"ABLATION NUM PROCESSES: {NUM_PROCESSES}")
    data_backend = _choose_feature_backend(FILEPATH, ["recons_out"], NUM_PROCESSES)
    print(f"ABLATION DATA BACKEND: {data_backend}")

    results_by_idx = {}

    if tasks:
        print(f"[INFO] running {len(tasks)} ablation experiments with {NUM_PROCESSES} process(es)")
    total_tasks = len(tasks)
    done_tasks = 0
    if NUM_PROCESSES == 1:
        _init_ablation_worker(str(MODELPATH), FILEPATH, MODELPATH.stem, EPOCHS, str(DEVICE),
                              BATCH_SIZE, NUM_WORKERS, use_vit_load,
                              vit_checkpoint_key, data_backend)
        for args in tasks:
            idx, _, _, _, exp_res, log_text = _ablation_one_seed_worker(args)
            print(log_text, end="")
            results_by_idx[idx] = exp_res
            done_tasks += 1
            _print_ablation_done(done_tasks, total_tasks, exp_res)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=NUM_PROCESSES,
            mp_context=ctx,
            initializer=_init_ablation_worker,
            initargs=(
                str(MODELPATH), FILEPATH, MODELPATH.stem, EPOCHS, str(DEVICE),
                BATCH_SIZE, NUM_WORKERS, use_vit_load,
                vit_checkpoint_key, data_backend,
            ),
        ) as executor:
            futures = [executor.submit(_ablation_one_seed_worker, args) for args in tasks]
            for future in as_completed(futures):
                idx, _, _, _, exp_res, log_text = future.result()
                results_by_idx[idx] = (exp_res, log_text)
                done_tasks += 1
                _print_ablation_done(done_tasks, total_tasks, exp_res)
        for idx in range(len(tasks)):
            exp_res, log_text = results_by_idx[idx]
            print(log_text, end="")
            results_by_idx[idx] = exp_res

    summary_res = []
    summary_idx = 0
    for ALPHA in ALPHA_ls:
        for OE in oe_ls:
            one_exp = []
            summary_res.append({'model':MODELPATH.stem, 'alpha':ALPHA,
                            'oe': OE, 'best_testacc':[], 'mean_testacc':[],
                            'epoch1_testacc': [], 'epoch5_testacc': [], 'epoch10_testacc': []})
            for seed in SEED_ls:
                res = results_by_idx[summary_idx]
                summary_idx += 1
                one_exp.append(res)
                summary_res[-1]['best_testacc'].append(max(res['test_acc_ls']))
                summary_res[-1]['mean_testacc'].append(np.array(res['test_acc_ls']).mean())
                for epoch_idx, summary_key in [(0, 'epoch1_testacc'), (4, 'epoch5_testacc'), (9, 'epoch10_testacc')]:
                    if len(res['test_acc_ls']) > epoch_idx:
                        summary_res[-1][summary_key].append(res['test_acc_ls'][epoch_idx])
                    else:
                        summary_res[-1][summary_key].append(np.nan)
            if save_res:
                try:
                    datafile_test = torch.load(SAVEPATH+f"_{EPOCHS}e.pt", weights_only=False)
                except (FileNotFoundError, EOFError):
                    datafile_test = []
                datafile_test.extend(one_exp)
                torch.save(datafile_test, SAVEPATH+f"_{EPOCHS}e.pt")

    print(MODELPATH.stem, '-'*10, 'result-summary')
    print(
        f"{'alpha':>6} | {'OE':>15} | {'best_mean':>8} | "
        f"{'best_std':>8} | {'best_max':>8} | {'all_mean':>8} | "
        f"{'e1_mean':>8} | {'e5_mean':>8} | {'e10_mean':>8}"
    )
    print("-" * 96)
    for ele in summary_res:
        best_arr = np.array(ele['best_testacc'], dtype=np.float64)
        mean_arr = np.array(ele['mean_testacc'], dtype=np.float64)
        print(
            f"{ele['alpha']:>6} | {ele['oe']:>15} | "
            f"{best_arr.mean():>8.4f} | {best_arr.std():>8.4f} | "
            f"{best_arr.max():>8.4f} | {mean_arr.mean():>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch1_testacc']):>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch5_testacc']):>8.4f} | "
            f"{_nanmean_or_nan(ele['epoch10_testacc']):>8.4f} "
        )
    return summary_res


def prepare_online_experiment(MODELPATH, batch_size, num_workers, device):
    import re

    MODELPATH = Path(MODELPATH)
    stem = MODELPATH.stem.lower()
    dataset_name = MODELPATH.parent.name.lower()

    from custom_datasets import (
        get_mnist_dataloaders,
        get_cifar10_dataloaders,
        get_cifar100_dataloaders,
        get_imagenette_dataloaders,
        get_imagewoof_dataloaders,
        get_tinyimagenet_dataloaders,
    )
    from models import (
        SimpleCNN_mn,
        SimpleCNN_ci10, ResNet_ci10,
        SimpleCNN_ci100, ResNet_ci100, SimpleViT_ci100, ViTCIFAR100Patch4,
        SimpleCNN_nette, ResNet_nette,
        SimpleCNN_woof, ResNet_woof,
        SimpleCNN_tin, ResNet_tin, SimpleViT_tin, ViTTinyImageNetPatch8,
    )

    dataset_builders = {
        "mnist": get_mnist_dataloaders,
        "cifar10": get_cifar10_dataloaders,
        "everyci10": get_cifar10_dataloaders,
        "cifar100": get_cifar100_dataloaders,
        "nette": get_imagenette_dataloaders,
        "imagenette": get_imagenette_dataloaders,
        "woof": get_imagewoof_dataloaders,
        "imagewoof": get_imagewoof_dataloaders,
        "tin": get_tinyimagenet_dataloaders,
        "tinyimagenet": get_tinyimagenet_dataloaders,
        "tiny-imagenet": get_tinyimagenet_dataloaders,
    }
    if dataset_name not in dataset_builders:
        raise ValueError(f"Cannot infer dataset from checkpoint parent: {MODELPATH.parent}")
    train_loader, test_loader = dataset_builders[dataset_name](batch_size=batch_size, num_workers=num_workers)

    def _simplecnn_version(default="v2"):
        match = re.search(r"simplecnnv?(\d+)", stem)
        return match.group(1) if match else default

    def _resnet_version(default="18"):
        match = re.search(r"resnet(\d+)", stem)
        return match.group(1) if match else default

    if dataset_name in {"mnist", "mn"}:
        if "simplecnn" in stem:
            activate = torch.relu if "relu" in stem else torch.tanh
            model = SimpleCNN_mn(activate=activate, version=_simplecnn_version("v2"))
        else:
            raise ValueError(f"Cannot infer MNIST model architecture from {MODELPATH.name}")
    elif dataset_name in {"cifar10", "ci10", "everyci10"}:
        if "simplecnn" in stem:
            activate = torch.relu if "relu" in stem else torch.tanh
            model = SimpleCNN_ci10(activate=activate, version=_simplecnn_version("v3"))
        elif "resnet" in stem:
            model = ResNet_ci10(version=_resnet_version(), pretrain=False)
        else:
            raise ValueError(f"Cannot infer CIFAR10 model architecture from {MODELPATH.name}")
    elif dataset_name in {"cifar100", "ci100"}:
        if "vit_cifar100_patch4_input32" in stem:
            model = ViTCIFAR100Patch4()
        elif "simplevit" in stem:
            model = SimpleViT_ci100()
        elif "simplecnn" in stem:
            activate = torch.relu if "relu" in stem else torch.tanh
            model = SimpleCNN_ci100(activate=activate, version=_simplecnn_version("v2"))
        elif "resnet" in stem:
            model = ResNet_ci100(version=_resnet_version(), pretrain=False)
        else:
            raise ValueError(f"Cannot infer CIFAR100 model architecture from {MODELPATH.name}")
    elif dataset_name in {"nette", "imagenette"}:
        if "simplecnn" in stem:
            activate = torch.relu if "relu" in stem else torch.tanh
            model = SimpleCNN_nette(activate=activate, version=_simplecnn_version("v2"))
        elif "resnet" in stem:
            model = ResNet_nette(version=_resnet_version(), pretrain=False)
        else:
            raise ValueError(f"Cannot infer Imagenette model architecture from {MODELPATH.name}")
    elif dataset_name in {"woof", "imagewoof"}:
        if "simplecnn" in stem:
            activate = torch.relu if "relu" in stem else torch.tanh
            model = SimpleCNN_woof(activate=activate, version=_simplecnn_version("v2"))
        elif "resnet" in stem:
            model = ResNet_woof(version=_resnet_version(), pretrain=False)
        else:
            raise ValueError(f"Cannot infer Imagewoof model architecture from {MODELPATH.name}")
    elif dataset_name in {"tin", "tinyimagenet", "tiny-imagenet"}:
        if "vit_timnet_patch8_input64" in stem:
            model = ViTTinyImageNetPatch8()
        elif "simplevit" in stem:
            model = SimpleViT_tin()
        elif "simplecnn" in stem:
            activate = torch.relu if "relu" in stem else torch.tanh
            model = SimpleCNN_tin(activate=activate, version=_simplecnn_version("v2"))
        elif "resnet" in stem:
            model = ResNet_tin(version=_resnet_version(), pretrain=False)
        else:
            raise ValueError(f"Cannot infer TinyImageNet model architecture from {MODELPATH.name}")
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    return model.to(device), train_loader, test_loader


if __name__ == '__main__':
    cpu_threads = _configure_cpu_threads()
    _reexec_from_runtime_snapshot()
    print('pid:', os.getpid())
    print('FRPT_CPU_THREADS:', cpu_threads)
    if os.environ.get("FRPT_RUNTIME_SCRIPT"):
        print('runtime script:', os.environ["FRPT_RUNTIME_SCRIPT"])
    if os.environ.get("FRPT_RUNTIME_SOURCE"):
        print('source script:', os.environ["FRPT_RUNTIME_SOURCE"])
    print(datetime.datetime.now())
    if _run_external_script_if_requested():
        print(datetime.datetime.now())
        sys.exit(0)

    BATCH_SIZE = 256
    NUM_WORKERS = 0
    SEED_ls = [0,1,2,3,4,5,6,7,8,9]
    DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/mnist/mn_simplecnnv2_relu_0.9771.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar10/ci10_simplecnnv2_relu_0.6738.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar10/ci10_simplecnnv2_relu_0.6663.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar10/ci10_resnet18_e5_0.9493.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar10/ci10_simplecnnv4_relu_e75_0.6102.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar10/ci10_simplecnnv5_relu_e30_0.5813.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/nette/nette_simplecnnv2_relu_0.6005.pth"
    MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/nette/nette_simplecnnv2_relu_epoch10_0.5934.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/nette/nette_resnet18_epoch5_0.9814.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/woof/woof_simplecnnv2_relu_e20_0.3321.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/woof/woof_simplecnnv2_relu_e10_0.3479.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/woof/woof_simplecnnv2_relu_e2_0.3421.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/woof/woof_resnet18_epoch20_0.9458.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar100/ci100_simplecnnv2_relu_0.4411.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar100/ci100_simplecnnv2_relu_epoch10_0.4386.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar100/ci100_simplecnnv2_relu_e15_0.4461.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar100/ci100_resnet18_epoch200_0.7926.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/cifar100/ci100_simplevit_p4_d192_l9_e30_0.6320.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/tin/tin_simplecnnv2_relu_0.3215.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/tin/tin_resnet34_epoch20_0.5872.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/tin/tin_simplecnnv2_relu_0.3233.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/tin/tin_simplecnnv2_relu_e16_0.3282.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/tin/tin_simplecnnv2_relu_e20_0.3324.pth"
    # MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/tin/tin_simplevit_p8_d192_l9_e60_0.4491.pth"
    LR = 5e-5

    ALPHA_ls = [0, 0.02, 0.1, 0.3]
    # ALPHA_ls = [0.5]
    EPOCHS = 10
    # for filename in os.listdir("/data/dn/FRTP_revision1/imagecls/ckpts/everyci10"):
    #     if int(filename.split('e')[-1].split('_')[0]) not in [2,4,6,8,10]:
    #         continue
    #     if filename == 'cifar10': continue
    #     MODELPATH = "/data/dn/FRTP_revision1/imagecls/ckpts/everyci10/"+filename
    print(f'BATCHSIZE={BATCH_SIZE}, SEED_ls={SEED_ls}, ALPHA_ls={ALPHA_ls}, DEVICE={DEVICE}, LR={LR}')
    print('model path', MODELPATH, '\n')
    model_post_train_all( MODELPATH, BATCH_SIZE, NUM_WORKERS,
        DEVICE, ALPHA_ls, SEED_ls, EPOCHS, lr=LR, NUM_PROCESSES=8,
        version="new_pooling")
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")

    # ALPHA_ls = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    # EPOCHS = 10
    # print(f'BATCHSIZE={BATCH_SIZE}, SEED_ls={SEED_ls}, ALPHA_ls={ALPHA_ls}, DEVICE={DEVICE}, LR={LR}')
    # print('model path', MODELPATH, '\n')
    # ablation_all(
    #     MODELPATH, BATCH_SIZE, NUM_WORKERS,
    #     DEVICE, ALPHA_ls,  SEED_ls, EPOCHS,
    #     NUM_PROCESSES=4, save_res=True)

    print(datetime.datetime.now())
