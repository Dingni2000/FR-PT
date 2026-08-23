from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path("/data/dn/FRTP_revision1")
IMAGECLS_ROOT = PROJECT_ROOT / "imagecls"
DATA_ROOT = PROJECT_ROOT / "mydata"
LOGS_STAGE_ROOT = IMAGECLS_ROOT / "logs_stage"

if str(IMAGECLS_ROOT) not in sys.path:
    sys.path.insert(0, str(IMAGECLS_ROOT))


from models import (  # noqa: E402
    SimpleCNN_mn,
    SimpleCNN_ci10,
    ResNet_ci10,
    SimpleCNN_ci100,
    ResNet_ci100,
    SimpleViT_ci100,
    SimpleCNN_nette,
    ResNet_nette,
    SimpleCNN_woof,
    ResNet_woof,
    SimpleCNN_tin,
    ResNet_tin,
    SimpleViT_tin,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_name: str
    modelpath: str | None
    lr: float
    epochs: int
    every_epoch: int
    train_record: bool
    batch_size: int
    device_spec: str
    make_data: Callable[[], tuple[object, object]]
    model_builders: dict[str, Callable[[], nn.Module]]


def resolve_device(device_spec: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(device_spec)
    return torch.device("cpu")


def load_optional_ckpt(model: nn.Module, modelpath: str | None, device: torch.device) -> None:
    if modelpath is None:
        return
    ckpt = torch.load(modelpath, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)


def get_score(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)["out"]
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(data_loader.dataset)


RECONS_METRICS = ("relative_diff", "absolute_diff", "forward_norm")


def init_recons_history(recons_keys: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        recons_key: {metric: [] for metric in RECONS_METRICS}
        for recons_key in recons_keys
    }


def append_recons_history(
    history: dict[str, dict[str, list[float]]],
    values: dict[str, dict[str, float]],
) -> None:
    for recons_key, metric_values in values.items():
        for metric in RECONS_METRICS:
            history[recons_key][metric].append(metric_values[metric])


def extend_recons_history(
    history: dict[str, dict[str, list[float]]],
    new_history: dict[str, dict[str, list[float]]],
) -> None:
    for recons_key, metric_values in new_history.items():
        history.setdefault(recons_key, {metric: [] for metric in RECONS_METRICS})
        for metric in RECONS_METRICS:
            history[recons_key].setdefault(metric, [])
            history[recons_key][metric].extend(metric_values.get(metric, []))


def append_existing_results(
    save_path: Path,
    results: dict[str, object],
) -> dict[str, object]:
    if not save_path.exists():
        return results

    print(f"appending existing results from {save_path}")
    existing_results = torch.load(save_path, map_location="cpu", weights_only=False)
    if not isinstance(existing_results, dict):
        raise TypeError(f"existing results at {save_path} is not a dict")

    merged_results = dict(existing_results)
    for key in ("model", "lr", "recons_keys", "recons_metrics"):
        merged_results[key] = results[key]

    merged_results["num_epochs"] = (
        existing_results.get("num_epochs", 0) + results["num_epochs"]
    )
    for key in ("train_score_list", "test_score_list"):
        merged_results.setdefault(key, [])
        merged_results[key].extend(results[key])

    for key in ("train_recons_loss_list", "test_recons_loss_list"):
        merged_results.setdefault(key, init_recons_history(results["recons_keys"]))
        extend_recons_history(merged_results[key], results[key])

    return merged_results


def get_recons_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Return sample-averaged reconstruction diffs/norms for each recons_key."""
    model.eval()
    sample_num = 0
    eps = 1e-12
    recons_keys =  model.get_fea_name()  # NOTE
    relative_diff_dict = {recons_key: 0.0 for recons_key in recons_keys}
    diff_norm_dict = {recons_key: 0.0 for recons_key in recons_keys}
    forward_norm_dict = {recons_key: 0.0 for recons_key in recons_keys}
    # recons_norm_dict = {recons_key: 0.0 for recons_key in recons_keys}
    param_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)
    try:
        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.enable_grad():
                recons_feature = model.get_recons_fea(input.detach(), target, recons_key=None)  # NOTE
            # recons_feature = {'recons_out':recons_feature.detach()}# NOTE
            recons_feature = {key: value.detach() for key, value in recons_feature.items()}
            with torch.no_grad():
                res = model(input)
            bs = input.size(0)
            sample_num += bs
            for recons_key in recons_keys:
                forward_feature = res[recons_key[7:]]
                diff = forward_feature - recons_feature[recons_key]
                diff_norm = diff.flatten(1).pow(2).sum(dim=1).sqrt()
                diff_norm_dict[recons_key] += diff_norm.sum().item()
                feature_norm = forward_feature.flatten(1).pow(2).sum(dim=1).sqrt()
                # recons_norm = recons_feature.flatten(1).pow(2).sum(dim=1).sqrt()
                forward_norm_dict[recons_key] += feature_norm.sum().item()
                # recons_norm_dict[recons_key] += recons_norm.sum().item()
                relative_diff_dict[recons_key] += (diff_norm / (feature_norm + eps)).sum().item()
    finally:
        for param, requires_grad in zip(model.parameters(), param_requires_grad):
            param.requires_grad_(requires_grad)
    if sample_num == 0:
        raise ValueError("data_loader produced no samples")
    return {
        recons_key: {
            "relative_diff": relative_diff_dict[recons_key] / sample_num,
            "absolute_diff": diff_norm_dict[recons_key] / sample_num,
            "forward_norm": forward_norm_dict[recons_key] / sample_num,
            # "recons_norm": recon_norm_dict[recons_key] / sample_num
        }
        for recons_key in recons_keys
    }


def train_stage_recons(
    model: nn.Module,
    lr: float,
    num_epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    train_record: bool = True,
    every_epoch: int = 5,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    clscrit = nn.CrossEntropyLoss()
    recons_keys = model.get_fea_name() # NOTE

    train_score_list, test_score_list = [], []
    train_recons_loss_list = init_recons_history(recons_keys)
    test_recons_loss_list = init_recons_history(recons_keys)

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = clscrit(model(data)["out"], target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % every_epoch == 0:
            print("************* ", end="")
            test_score = get_score(model, test_loader, device)
            test_score_list.append(test_score)
            test_recons_loss = get_recons_loss(model, test_loader, device)
            append_recons_history(test_recons_loss_list, test_recons_loss)

            if train_record:
                train_score = get_score(model, train_loader, device)
                train_score_list.append(train_score)
                train_recons_loss = get_recons_loss(model, train_loader, device)
                append_recons_history(train_recons_loss_list, train_recons_loss)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"train_score={train_score:.4f}, test_score={test_score:.4f}, "
                    f"train_recons_stats={train_recons_loss}, test_recons_stats={test_recons_loss}"
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] test_score={test_score:.4f}, "
                    f"test_recons_stats={test_recons_loss}"
                )


    results = {
        "model": model.name,
        "lr": lr,
        "num_epochs": num_epochs,
        "recons_keys": recons_keys,
        "recons_metrics": RECONS_METRICS,
        "train_score_list": train_score_list,
        "test_score_list": test_score_list,
        "train_recons_loss_list": train_recons_loss_list,
        "test_recons_loss_list": test_recons_loss_list,
    }

    ckpt_path = LOGS_STAGE_ROOT / f"{model.name}_{lr}lr.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ckpt_path)
    print(f"saved model ckpt to {ckpt_path}")

    save_path = LOGS_STAGE_ROOT / f"{model.name}_{lr}lr.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results = append_existing_results(save_path, results)
    torch.save(results, save_path)
    print(f"saved results to {save_path}")
    print("train_score_list =", results["train_score_list"])
    print("test_score_list =", results["test_score_list"])
    print("train_recons_loss_list =", results["train_recons_loss_list"])
    print("test_recons_loss_list =", results["test_recons_loss_list"])
    return results


def make_mn_data() -> tuple[object, object]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    train_set = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=transform)
    print(len(train_set), len(test_set))
    return train_set, test_set


def make_ci10_data() -> tuple[object, object]:
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
            transforms.RandomErasing(p=0.25),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )
    trainset = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=transform_test)
    print(len(trainset), len(testset))
    return trainset, testset


def make_ci100_data() -> tuple[object, object]:
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
            transforms.RandomErasing(p=0.25),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )
    trainset = datasets.CIFAR100(root=str(DATA_ROOT), train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=str(DATA_ROOT), train=False, download=True, transform=transform_test)
    print(len(trainset), len(testset))
    return trainset, testset


def make_nette_data() -> tuple[object, object]:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            transforms.RandomErasing(p=0.15),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    dataset_dir = DATA_ROOT / "imagenette2"
    trainset = datasets.ImageFolder(str(dataset_dir / "train"), transform=transform_train)
    testset = datasets.ImageFolder(str(dataset_dir / "val"), transform=transform_test)
    print(len(trainset), len(testset))
    return trainset, testset


def make_woof_data() -> tuple[object, object]:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            transforms.RandomErasing(p=0.15),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    dataset_dir = DATA_ROOT / "imagewoof2"
    trainset = datasets.ImageFolder(str(dataset_dir / "train"), transform=transform_train)
    testset = datasets.ImageFolder(str(dataset_dir / "val"), transform=transform_test)
    print(len(trainset), len(testset))
    return trainset, testset


def make_tin_data() -> tuple[object, object]:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            transforms.RandomErasing(p=0.25),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    dataset_dir = DATA_ROOT / "tiny-imagenet-200"
    trainset = datasets.ImageFolder(str(dataset_dir / "train"), transform=transform_train)
    testset = datasets.ImageFolder(str(dataset_dir / "val"), transform=transform_test)
    print(len(trainset), len(testset))
    print(trainset.classes)
    return trainset, testset


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "mn": ExperimentConfig(
        name="mn",
        model_name="simplecnnv2",
        modelpath=None,
        lr=1e-5,
        epochs=100,
        every_epoch=5,
        train_record=True,
        batch_size=256,
        device_spec="cuda:0",
        make_data=make_mn_data,
        model_builders={
            "simplecnnv2": lambda: SimpleCNN_mn(activate=torch.relu, version="v2"),
        },
    ),
    "ci10": ExperimentConfig(
        name="ci10",
        model_name="resnet18",
        modelpath=None,
        lr=1e-5,
        epochs=100,
        every_epoch=5,
        train_record=True,
        batch_size=256,
        device_spec="cuda:0",
        make_data=make_ci10_data,
        model_builders={
            "simplecnnv2": lambda: SimpleCNN_ci10(activate=torch.relu, version="v2"),
            "resnet18": lambda: ResNet_ci10(version="18", pretrain=False),
        },
    ),
    "ci100": ExperimentConfig(
        name="ci100",
        model_name="resnet18",
        modelpath=None,
        lr=1e-5,
        epochs=100,
        every_epoch=5,
        train_record=True,
        batch_size=256,
        device_spec="cuda:2",
        make_data=make_ci100_data,
        model_builders={
            "simplecnnv1": lambda: SimpleCNN_ci100(activate=torch.relu, version="v1"),
            "simplecnnv2": lambda: SimpleCNN_ci100(activate=torch.relu, version="v2"),
            "resnet18": lambda: ResNet_ci100(version="18", pretrain=False),
            "simplevit": lambda: SimpleViT_ci100(),
        },
    ),
    "nette": ExperimentConfig(
        name="nette",
        model_name="resnet18",
        modelpath=None,
        lr=1e-5,
        epochs=100,
        every_epoch=5,
        train_record=True,
        batch_size=256,
        device_spec="cuda:1",
        make_data=make_nette_data,
        model_builders={
            "simplecnnv1": lambda: SimpleCNN_nette(activate=torch.relu, version="v1"),
            "simplecnnv2": lambda: SimpleCNN_nette(activate=torch.relu, version="v2"),
            "resnet18": lambda: ResNet_nette(version="18", pretrain=False),
            "resnet34": lambda: ResNet_nette(version="34", pretrain=False),
        },
    ),
    "woof": ExperimentConfig(
        name="woof",
        model_name="resnet18",
        modelpath=None,
        lr=1e-5,
        epochs=100,
        every_epoch=5,
        train_record=True,
        batch_size=256,
        device_spec="cuda:2",
        make_data=make_woof_data,
        model_builders={
            "simplecnnv1": lambda: SimpleCNN_woof(activate=torch.relu, version="v1"),
            "simplecnnv2": lambda: SimpleCNN_woof(activate=torch.relu, version="v2"),
            "resnet18": lambda: ResNet_woof(version="18", pretrain=False),
            "resnet34": lambda: ResNet_woof(version="34", pretrain=False),
        },
    ),
    "tin": ExperimentConfig(
        name="tin",
        model_name="resnet34",
        modelpath=None,
        lr=1e-5,
        epochs=100,
        every_epoch=5,
        train_record=True,
        batch_size=256,
        device_spec="cuda:6",
        make_data=make_tin_data,
        model_builders={
            "simplecnnv1": lambda: SimpleCNN_tin(activate=torch.relu, version="v1"),
            "simplecnnv2": lambda: SimpleCNN_tin(activate=torch.relu, version="v2"),
            "resnet34": lambda: ResNet_tin(version="34", pretrain=False),
            "simplevit": lambda: SimpleViT_tin(),
        },
    ),
}


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    print(f"===== {config.name} / {config.model_name} =====")
    device = resolve_device(config.device_spec)
    print("DEVICE:", device)
    trainset, testset = config.make_data()
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    print(len(train_loader), len(test_loader))

    model = config.model_builders[config.model_name]().to(device)
    load_optional_ckpt(model, config.modelpath, device)
    return train_stage_recons(
        model,
        lr=config.lr,
        num_epochs=config.epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        train_record=config.train_record,
        every_epoch=config.every_epoch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run different-stage vs reconstruction-loss experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS.keys()),
        default=tuple(EXPERIMENTS.keys()),
        help="Experiments to run. Defaults to all experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.experiments:
        run_experiment(EXPERIMENTS[name])


if __name__ == "__main__":
    main()
