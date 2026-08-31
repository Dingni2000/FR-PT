from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor, nn


CaptureMode = Literal["none", "stages", "blocks", "full"]
_VALID_CAPTURE_MODES = {"none", "stages", "blocks", "full"}
_LOCAL_RESNET34_CHECKPOINT = Path(__file__).resolve().parent / "resnet34-b627a593.pth"


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """3x3 convolution used by ResNet34 BasicBlock."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """1x1 projection used by a downsampling shortcut."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Two-convolution residual block used by ResNet18 and ResNet34.

    The trainable module names intentionally match the canonical PyTorch
    ResNet implementation: conv1, bn1, conv2, bn2 and downsample. Therefore,
    the official ImageNet ResNet34 state_dict can be loaded strictly.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=False)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    """Self-contained ResNet implementation with canonical parameter names."""

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: Sequence[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError(f"layers must contain four stage depths, got {layers}.")

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)

        # Keep the ImageNet classification modules so the official checkpoint,
        # including fc.weight/fc.bias, can be loaded with strict=True.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_parameters(zero_init_residual=zero_init_residual)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        if blocks <= 0:
            raise ValueError(f"blocks must be positive, got {blocks}.")

        downsample: Optional[nn.Module] = None
        output_channels = planes * block.expansion
        if stride != 1 or self.inplanes != output_channels:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, output_channels, stride=stride),
                nn.BatchNorm2d(output_channels),
            )

        modules: List[nn.Module] = [
            block(
                in_channels=self.inplanes,
                out_channels=planes,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = output_channels

        for _ in range(1, blocks):
            modules.append(
                block(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    stride=1,
                    downsample=None,
                )
            )

        return nn.Sequential(*modules)

    def _initialize_parameters(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_stage_features(self, x: Tensor) -> Dict[str, Tensor]:
        """Return ResNet stage outputs in the same z0-z4 style as resnet_recons."""
        z_stem = self.relu(self.bn1(self.conv1(x)))
        z0 = self.maxpool(z_stem)
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return {
            # "z0": z0,
            # "z1": z1,
            # "z2": z2,
            "z3": z3,
            "z4": z4,
        }

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet34(num_classes: int = 1000) -> ResNet:
    """Build ResNet34 with stage depths [3, 4, 6, 3]."""
    return ResNet(
        block=BasicBlock,
        layers=(3, 4, 6, 3),
        num_classes=num_classes,
    )


def _safe_torch_load(checkpoint_path: Path) -> Any:
    """Load a trusted local checkpoint without network access."""
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with PyTorch versions that predate weights_only.
        return torch.load(checkpoint_path, map_location="cpu")


def _extract_tensor_state_dict(checkpoint: Any) -> Dict[str, Tensor]:
    """Extract and normalize a tensor state_dict from common containers."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "The ImageNet checkpoint must be a mapping, "
            f"but received {type(checkpoint)}."
        )

    state: Mapping[str, Any] = checkpoint
    for container_key in ("state_dict", "model_state_dict", "model"):
        candidate = state.get(container_key)
        if isinstance(candidate, Mapping):
            state = candidate
            break

    removable_prefixes = (
        "module.",
        "model.",
        "backbone.",
        "image_backbone.",
        "resnet.",
    )
    tensor_state: Dict[str, Tensor] = {}

    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            continue

        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in removable_prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break

        if key in tensor_state:
            raise RuntimeError(
                f"Duplicate parameter key '{key}' after prefix normalization."
            )
        tensor_state[key] = value

    if not tensor_state:
        raise ValueError("No tensor parameters were found in the checkpoint.")

    return tensor_state


def load_local_imagenet_resnet34(
    model: ResNet,
    checkpoint_path: str,
) -> None:
    """Strictly load a local official-compatible ResNet34 ImageNet state_dict."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ImageNet checkpoint does not exist: {path}")

    checkpoint = _safe_torch_load(path)
    state_dict = _extract_tensor_state_dict(checkpoint)
    model_state = model.state_dict()
    for key, value in model_state.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value

    missing_keys = sorted(set(model_state) - set(state_dict))
    unexpected_keys = sorted(set(state_dict) - set(model_state))
    shape_mismatches: List[str] = []

    for key in sorted(set(model_state) & set(state_dict)):
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(state_dict[key].shape)}, "
                f"model={tuple(model_state[key].shape)}"
            )

    if missing_keys or unexpected_keys or shape_mismatches:
        message = [
            "The checkpoint is not strictly compatible with this ResNet34.",
            f"checkpoint: {path}",
            f"missing keys: {len(missing_keys)}",
            f"unexpected keys: {len(unexpected_keys)}",
            f"shape mismatches: {len(shape_mismatches)}",
        ]
        if missing_keys:
            message.append(f"missing examples: {missing_keys[:20]}")
        if unexpected_keys:
            message.append(f"unexpected examples: {unexpected_keys[:20]}")
        if shape_mismatches:
            message.append(
                "shape mismatch examples: " + "; ".join(shape_mismatches[:20])
            )
        raise RuntimeError("\n".join(message))

    model.load_state_dict(state_dict, strict=True)
    print(
        "[ResNet34] Loaded local ImageNet checkpoint\n"
        f"  path: {path}\n"
        f"  tensors: {len(state_dict)}"
    )


class ResNet34FeatureBackbone(nn.Module):
    """Explicit ResNet34 image backbone with optional stage feature capture.

    Capture modes:
      none: no intermediate backbone tensors.
      stages/blocks/full: return stage outputs z0-z4.

    The public capture_mode values remain compatible with older configs, but
    this backbone intentionally records only ResNet stage outputs.  That keeps
    the trajectory model output compact while matching the feature naming used
    by resnet_recons.py.
    """

    output_channels: int = 512

    def __init__(
        self,
        checkpoint_path: Optional[str],
        load_imagenet_checkpoint: bool,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.resnet = resnet34(num_classes=1000)

        if load_imagenet_checkpoint:
            if not checkpoint_path:
                checkpoint_path = str(_LOCAL_RESNET34_CHECKPOINT)
            load_local_imagenet_resnet34(
                model=self.resnet,
                checkpoint_path=checkpoint_path,
            )

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _validate_capture_mode(capture_mode: str) -> CaptureMode:
        if capture_mode not in _VALID_CAPTURE_MODES:
            raise ValueError(
                f"capture_mode must be one of {sorted(_VALID_CAPTURE_MODES)}, "
                f"but received '{capture_mode}'."
            )
        return capture_mode  # type: ignore[return-value]

    @staticmethod
    def _record(
        feature_dict: Dict[str, Tensor],
        name: str,
        tensor: Tensor,
        retain_gradients: bool,
    ) -> Tensor:
        if retain_gradients and tensor.requires_grad:
            tensor.retain_grad()
        feature_dict[name] = tensor
        return tensor

    def forward(
        self,
        image: Tensor,
        capture_mode: str = "stages",
        retain_gradients: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        mode = self._validate_capture_mode(capture_mode)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "Expected image shape [B, 3, H, W], "
                f"got {tuple(image.shape)}."
            )
        if retain_gradients and mode == "none":
            raise ValueError(
                "retain_gradients=True requires capture_mode other than 'none'."
            )

        stage_features = self.resnet.forward_stage_features(image)
        image_feature_map = stage_features["z4"]

        # if mode == "none":
        #     return image_feature_map, {}

        features: Dict[str, Tensor] = {}
        for feature_name, feature_tensor in stage_features.items():
            self._record(features, feature_name, feature_tensor, retain_gradients)
        return image_feature_map, features
