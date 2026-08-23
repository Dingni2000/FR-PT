from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path("/data/dn/FRTP_revision1")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "mydata"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LabelLike = Union[int, str]


def mnist_transforms() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), 
                               transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])


def cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform_train, transform_test


def cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
    )
    return transform_train, transform_test


def imagenet_subset_transforms(
    image_size: int,
    train_crop_scale: Tuple[float, float],
    randaugment_magnitude: int,
    random_erasing_p: float,
    resize_size: Optional[int] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=train_crop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=random_erasing_p),
        ]
    )

    test_ops = []
    if resize_size is not None:
        test_ops.extend([transforms.Resize(resize_size), transforms.CenterCrop(image_size)])
    test_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform_train, transforms.Compose(test_ops)


def _resolve_target_label(dataset: Any, target_label: LabelLike) -> int:
    if isinstance(target_label, int):
        return target_label

    classes = getattr(dataset, "classes", None)
    if classes is not None:
        if target_label in classes:
            return int(classes.index(target_label))
        class_to_idx = getattr(dataset, "class_to_idx", {})
        if target_label in class_to_idx:
            return int(class_to_idx[target_label])

    try:
        return int(target_label)
    except ValueError as exc:
        raise ValueError(
            f"target_label={target_label!r} is not in dataset.classes and is not an int label"
        ) from exc


def _dataset_targets(dataset: Any) -> torch.Tensor:
    if not hasattr(dataset, "targets"):
        raise TypeError("dataset must expose a 'targets' attribute to use label dropping")
    return torch.as_tensor(dataset.targets, dtype=torch.long)


def make_label_drop_subset(
    dataset: Any,
    target_label: LabelLike,
    drop_ratio: float = 0.90,
    seed: int = 42,
    min_keep: int = 1,
) -> Tuple[Subset, Dict[str, Any]]:
    """Return a Subset where a fraction of one class is removed."""
    if not 0.0 <= drop_ratio < 1.0:
        raise ValueError("drop_ratio must be in [0.0, 1.0)")

    resolved_label = _resolve_target_label(dataset, target_label)
    targets = _dataset_targets(dataset)
    target_idx = torch.where(targets == resolved_label)[0]
    other_idx = torch.where(targets != resolved_label)[0]
    if target_idx.numel() == 0:
        raise ValueError(f"no samples found for target_label={target_label!r} ({resolved_label})")

    keep_num = int(round(target_idx.numel() * (1.0 - drop_ratio)))
    keep_num = max(min_keep, keep_num)
    keep_num = min(keep_num, target_idx.numel())

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(target_idx.numel(), generator=generator)
    kept_target_idx = target_idx[perm[:keep_num]]
    dropped_target_idx = target_idx[perm[keep_num:]]
    keep_idx = torch.cat([other_idx, kept_target_idx])

    subset = Subset(dataset, keep_idx.tolist())
    info = {
        "target_label": resolved_label,
        "target_name": _label_name(dataset, resolved_label),
        "drop_ratio": drop_ratio,
        "seed": seed,
        "kept_target": keep_num,
        "total_target": int(target_idx.numel()),
        "original_size": int(targets.numel()),
        "subset_size": len(subset),
    }
    subset.drop_info = info
    subset.kept_indices = keep_idx.tolist()
    subset.kept_target_indices = kept_target_idx.tolist()
    subset.dropped_target_indices = dropped_target_idx.tolist()
    return subset, info


def make_recovered_target_replay_subset(
    dataset: Any,
    target_label: LabelLike,
    recovered_target_indices: Optional[Sequence[int]] = None,
    replay_ratio: float = 0.50,
    max_recovered: Optional[int] = None,
    seed: int = 43,
) -> Tuple[Subset, Dict[str, Any]]:
    """Build post-training data from recovered target samples plus non-target replay."""
    if replay_ratio < 0.0:
        raise ValueError("replay_ratio must be non-negative")

    resolved_label = _resolve_target_label(dataset, target_label)
    targets = _dataset_targets(dataset)
    generator = torch.Generator().manual_seed(seed)

    if recovered_target_indices is None:
        recovered_idx = torch.where(targets == resolved_label)[0]
    else:
        recovered_idx = torch.as_tensor(list(recovered_target_indices), dtype=torch.long)
        if recovered_idx.numel() == 0:
            raise ValueError("recovered_target_indices is empty")
        if torch.any(targets[recovered_idx] != resolved_label):
            raise ValueError("recovered_target_indices contains non-target samples")

    if max_recovered is not None:
        if max_recovered <= 0:
            raise ValueError("max_recovered must be positive when provided")
        perm = torch.randperm(recovered_idx.numel(), generator=generator)
        recovered_idx = recovered_idx[perm[: min(max_recovered, recovered_idx.numel())]]

    replay_pool = torch.where(targets != resolved_label)[0]
    replay_num = int(round(recovered_idx.numel() * replay_ratio))
    replay_num = min(replay_num, replay_pool.numel())
    replay_perm = torch.randperm(replay_pool.numel(), generator=generator)
    replay_idx = replay_pool[replay_perm[:replay_num]]

    post_idx = torch.cat([recovered_idx, replay_idx])
    post_perm = torch.randperm(post_idx.numel(), generator=generator)
    post_idx = post_idx[post_perm]

    subset = Subset(dataset, post_idx.tolist())
    info = {
        "target_label": resolved_label,
        "target_name": _label_name(dataset, resolved_label),
        "seed": seed,
        "replay_ratio": replay_ratio,
        "recovered_target": int(recovered_idx.numel()),
        "replay": int(replay_idx.numel()),
        "subset_size": len(subset),
    }
    subset.posttrain_info = info
    subset.recovered_target_indices = recovered_idx.tolist()
    subset.replay_indices = replay_idx.tolist()
    return subset, info


def _label_name(dataset: Any, label: int) -> str:
    classes = getattr(dataset, "classes", None)
    if classes is not None and 0 <= label < len(classes):
        return str(classes[label])
    return str(label)


def _maybe_drop_train_label(
    trainset: Any,
    drop_label: Optional[LabelLike],
    drop_ratio: float,
    seed: int,
) -> Any:
    if drop_label is None:
        return trainset
    subset, _ = make_label_drop_subset(trainset, drop_label, drop_ratio=drop_ratio, seed=seed)
    return subset


def _make_loader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: Optional[bool],
    generator: Optional[torch.Generator] = None,
    **loader_kwargs: Any,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **loader_kwargs,
    )


def get_mnist_dataloaders(
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 0,
    download: bool = True,
    drop_label: Optional[LabelLike] = None,
    drop_ratio: float = 0.90,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    get_testset: bool = False,
    **loader_kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    transform = mnist_transforms()
    trainset = datasets.MNIST(str(data_root), train=True, download=download, transform=transform)
    testset = datasets.MNIST(str(data_root), train=False, download=download, transform=transform)
    trainset = _maybe_drop_train_label(trainset, drop_label, drop_ratio, seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        trainset, batch_size, True, num_workers, pin_memory, generator=generator, **loader_kwargs
    )
    test_loader = _make_loader(testset, batch_size, False, num_workers, pin_memory, **loader_kwargs)
    if get_testset: return train_loader, test_loader, testset
    return train_loader, test_loader


def get_cifar10_dataloaders(
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
    download: bool = True,
    drop_label: Optional[LabelLike] = None,
    drop_ratio: float = 0.90,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    get_testset: bool = False,
    **loader_kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    transform_train, transform_test = cifar10_transforms()
    trainset = datasets.CIFAR10(
        root=str(data_root), train=True, download=download, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root=str(data_root), train=False, download=download, transform=transform_test
    )
    trainset = _maybe_drop_train_label(trainset, drop_label, drop_ratio, seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        trainset, batch_size, True, num_workers, pin_memory, generator=generator, **loader_kwargs
    )
    test_loader = _make_loader(testset, batch_size, False, num_workers, pin_memory, **loader_kwargs)
    if get_testset: return train_loader, test_loader, testset
    return train_loader, test_loader


def get_cifar100_dataloaders(
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    batch_size: int = 512,
    num_workers: int = 4,
    download: bool = True,
    drop_label: Optional[LabelLike] = None,
    drop_ratio: float = 0.90,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    get_testset:bool = False,
    **loader_kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    transform_train, transform_test = cifar100_transforms()
    trainset = datasets.CIFAR100(
        root=str(data_root), train=True, download=download, transform=transform_train
    )
    testset = datasets.CIFAR100(
        root=str(data_root), train=False, download=download, transform=transform_test
    )
    trainset = _maybe_drop_train_label(trainset, drop_label, drop_ratio, seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        trainset, batch_size, True, num_workers, pin_memory, generator=generator, **loader_kwargs
    )
    test_loader = _make_loader(testset, batch_size, False, num_workers, pin_memory, **loader_kwargs)
    if get_testset: return train_loader, test_loader, testset
    return train_loader, test_loader


def _get_imagefolder_dataloaders(
    dataset_dir: Union[str, Path],
    transform_train: transforms.Compose,
    transform_test: transforms.Compose,
    batch_size: int,
    num_workers: int,
    drop_label: Optional[LabelLike],
    drop_ratio: float,
    seed: int,
    pin_memory: Optional[bool],
    get_testset: bool = False,
    **loader_kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    dataset_dir = Path(dataset_dir)
    trainset = datasets.ImageFolder(str(dataset_dir / "train"), transform=transform_train)
    testset = datasets.ImageFolder(str(dataset_dir / "val"), transform=transform_test)
    trainset = _maybe_drop_train_label(trainset, drop_label, drop_ratio, seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        trainset, batch_size, True, num_workers, pin_memory, generator=generator, **loader_kwargs
    )
    test_loader = _make_loader(testset, batch_size, False, num_workers, pin_memory, **loader_kwargs)
    if get_testset: return train_loader, test_loader, testset
    return train_loader, test_loader


def get_imagenette_dataloaders(
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
    drop_label: Optional[LabelLike] = None,
    drop_ratio: float = 0.90,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    get_testset: bool = False,
    **loader_kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    transform_train, transform_test = imagenet_subset_transforms(
        image_size=224,
        train_crop_scale=(0.65, 1.0),
        randaugment_magnitude=7,
        random_erasing_p=0.15,
        resize_size=256,
    )
    return _get_imagefolder_dataloaders(
        Path(data_root) / "imagenette2",
        transform_train,
        transform_test,
        batch_size,
        num_workers,
        drop_label,
        drop_ratio,
        seed,
        pin_memory,
        get_testset,
        **loader_kwargs,
    )


def get_imagewoof_dataloaders(
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
    drop_label: Optional[LabelLike] = None,
    drop_ratio: float = 0.90,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    get_testset: bool =False,
    **loader_kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    transform_train, transform_test = imagenet_subset_transforms(
        image_size=224,
        train_crop_scale=(0.65, 1.0),
        randaugment_magnitude=7,
        random_erasing_p=0.15,
        resize_size=256,
    )
    return _get_imagefolder_dataloaders(
        Path(data_root) / "imagewoof2",
        transform_train,
        transform_test,
        batch_size,
        num_workers,
        drop_label,
        drop_ratio,
        seed,
        pin_memory,
        get_testset,
        **loader_kwargs,
    )


def get_tinyimagenet_dataloaders(
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
    drop_label: Optional[LabelLike] = None,
    drop_ratio: float = 0.90,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    return_train_eval_loader: bool = False,
    get_testset: bool = False,
    **loader_kwargs: Any,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    transform_train, transform_test = imagenet_subset_transforms(
        image_size=64,
        train_crop_scale=(0.6, 1.0),
        randaugment_magnitude=9,
        random_erasing_p=0.25,
    )
    dataset_dir = Path(data_root) / "tiny-imagenet-200"
    return _get_imagefolder_dataloaders(
        dataset_dir,
        transform_train,
        transform_test,
        batch_size,
        num_workers,
        drop_label,
        drop_ratio,
        seed,
        pin_memory,
        get_testset,
        **loader_kwargs,
    )

    # if not return_train_eval_loader:
    #     return train_loader, test_loader

    # train_evalset = datasets.ImageFolder(str(dataset_dir / "train"), transform=transform_test)
    # train_eval_loader = _make_loader(
    #     train_evalset, batch_size, False, num_workers, pin_memory, **loader_kwargs
    # )
    # return train_loader, train_eval_loader, test_loader


def get_tinyimageent_dataloaders(*args: Any, **kwargs: Any) -> Any:
    """Alias kept for the common tinyimageent typo in experiment notes."""
    return get_tinyimagenet_dataloaders(*args, **kwargs)


DATASET_LOADERS = {
    "mnist": get_mnist_dataloaders,
    "cifar10": get_cifar10_dataloaders,
    "cifar100": get_cifar100_dataloaders,
    "imagenette": get_imagenette_dataloaders,
    "imagewoof": get_imagewoof_dataloaders,
    "tinyimagenet": get_tinyimagenet_dataloaders,
    "tiny-imagenet": get_tinyimagenet_dataloaders,
    "tinyimageent": get_tinyimageent_dataloaders,
}


def get_dataloaders(dataset_name: str, *args: Any, **kwargs: Any) -> Any:
    key = dataset_name.lower().replace("_", "").replace("-", "")
    aliases = {
        "tinyimagenet200": "tinyimagenet",
        "tinyimagenet": "tinyimagenet",
        "tinyimageent": "tinyimageent",
    }
    key = aliases.get(key, key)
    if key not in DATASET_LOADERS:
        supported = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"unknown dataset_name={dataset_name!r}; supported: {supported}")
    return DATASET_LOADERS[key](*args, **kwargs)


if __name__ == "__main__":
    batch_size = 256
    num_workers = 0
    target_label = 4
    drop_ratio = 0.90
    replay_ratio = 0.50
    seed = 42

    transform = mnist_transforms()
    train_set = datasets.MNIST(
        str(DEFAULT_DATA_ROOT), train=True, download=False, transform=transform
    )
    test_set = datasets.MNIST(
        str(DEFAULT_DATA_ROOT), train=False, download=False, transform=transform
    )

    pretrain_train_set, pretrain_info = make_label_drop_subset(
        train_set, target_label=target_label, drop_ratio=drop_ratio, seed=seed
    )
    posttrain_train_set, posttrain_info = make_recovered_target_replay_subset(
        train_set,
        target_label=target_label,
        recovered_target_indices=pretrain_train_set.dropped_target_indices,
        replay_ratio=replay_ratio,
        seed=seed + 1,
    )

    pretrain_train_loader = _make_loader(
        pretrain_train_set, batch_size, True, num_workers, pin_memory=None
    )
    pretrain_test_loader = _make_loader(
        test_set, batch_size, False, num_workers, pin_memory=None
    )
    posttrain_train_loader = _make_loader(
        posttrain_train_set, batch_size, True, num_workers, pin_memory=None
    )
    posttrain_test_loader = _make_loader(
        test_set, batch_size, False, num_workers, pin_memory=None
    )

    print("MNIST long-tail continual post-training example")
    print(f"pretrain_train_loader: {len(pretrain_train_loader.dataset)} samples")
    print(f"pretrain_test_loader:  {len(pretrain_test_loader.dataset)} samples")
    print(f"posttrain_train_loader: {len(posttrain_train_loader.dataset)} samples")
    print(f"posttrain_test_loader:  {len(posttrain_test_loader.dataset)} samples")
    print(f"pretrain_info: {pretrain_info}")
    print(f"posttrain_info: {posttrain_info}")
