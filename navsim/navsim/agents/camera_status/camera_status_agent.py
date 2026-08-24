from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.camera_status.camera_status_config import CameraStatusConfig
from navsim.agents.camera_status.camera_status_features import (
    CameraStatusFeatureBuilder,
    CameraStatusTrajectoryTargetBuilder,
)
from navsim.agents.camera_status.camera_status_model import CameraStatusTrajectoryModel
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
import argparse
from pathlib import Path
from pathlib import Path


    
class CameraStatusTrajectoryAgent(AbstractAgent):
    """NAVSIM v1.1 agent using stitched cameras and ego status to predict trajectory."""
    def __init__(
        self,
        config: CameraStatusConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._model = CameraStatusTrajectoryModel(config)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """Load either a Lightning agent checkpoint or a model-only state dict."""
        if self._checkpoint_path is None:
            return

        checkpoint: Dict[str, Any] = torch.load(
            self._checkpoint_path,
            map_location=torch.device("cpu"),
        )
        state_dict: Dict[str, torch.Tensor] = checkpoint.get("state_dict", checkpoint)

        # Lightning checkpoints normally store keys such as "agent._model...".
        cleaned_state_dict = {
            key[len("agent.") :] if key.startswith("agent.") else key: value
            for key, value in state_dict.items()
        }

        if cleaned_state_dict and all(
            key.startswith("_model.") for key in cleaned_state_dict
        ):
            self.load_state_dict(cleaned_state_dict, strict=True)
        else:
            # Also support model-only checkpoints with keys like "image_backbone...".
            model_state_dict = {
                key[len("_model.") :] if key.startswith("_model.") else key: value
                for key, value in cleaned_state_dict.items()
            }
            self._model.load_state_dict(model_state_dict, strict=True)

    def get_sensor_config(self) -> SensorConfig:
        """Load only the current-frame front/left/right cameras; no LiDAR."""
        current_frame = [self._config.camera_history_index]
        return SensorConfig(
            cam_f0=current_frame,
            cam_l0=current_frame,
            cam_l1=False,
            cam_l2=False,
            cam_r0=current_frame,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [CameraStatusTrajectoryTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [CameraStatusFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._model(features)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """GPU-safe inference implementation for NAVSIM evaluation."""
        self.eval()
        device = next(self.parameters()).device

        features: Dict[str, torch.Tensor] = {}#获取输入特征
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features = {
            name: tensor.unsqueeze(0).to(device=device, non_blocking=True)
            for name, tensor in features.items()
        }

        with torch.no_grad():
            predictions = self.forward(features)

        poses = predictions["trajectory"].squeeze(0).detach().cpu().numpy()
        return Trajectory(
            poses=poses,
            trajectory_sampling=self._config.trajectory_sampling,
        )

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del features  # Only trajectory supervision is used in V1.
        return torch.nn.functional.l1_loss(
            predictions["trajectory"],
            targets["trajectory"].to(dtype=torch.float32),
        )

    # def get_optimizers(
    #     self,
    # ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
    #     trainable_parameters = [
    #         parameter for parameter in self._model.parameters() if parameter.requires_grad
    #     ]
    #     if not trainable_parameters:
    #         raise RuntimeError("The model has no trainable parameters.")

    #     return torch.optim.AdamW(
    #         trainable_parameters,
    #         lr=self._lr,
    #         weight_decay=self._config.weight_decay,
    #     )
    def get_optimizers(self,) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """
        Use fixed differential learning rates:

        - pretrained explicit ResNet34: lr * backbone_lr_scale
        - newly initialized task modules: lr

        No warmup and no learning-rate scheduler are used.
        """

        backbone_parameters: List[torch.nn.Parameter] = []
        task_parameters: List[torch.nn.Parameter] = []

        backbone_parameter_names: List[str] = []
        task_parameter_names: List[str] = []

        for name, parameter in self._model.named_parameters():
            if not parameter.requires_grad:
                continue

            # The explicit ResNet retains the ImageNet classification head
            # only for loading the complete official checkpoint.
            #
            # This head is not used in the NAVSIM trajectory forward path,
            # so it should not be optimized.
            if name.startswith("image_backbone.resnet.fc."):
                continue

            # This is the exact parameter path of the explicit ResNet34
            # implementation used in the current version.
            if name.startswith("image_backbone.resnet."):
                backbone_parameters.append(parameter)
                backbone_parameter_names.append(name)
            else:
                task_parameters.append(parameter)
                task_parameter_names.append(name)

        if not backbone_parameters:
            raise RuntimeError(
                "No trainable explicit ResNet34 parameters were found under "
                "'image_backbone.resnet.'."
            )

        if not task_parameters:
            raise RuntimeError(
                "No trainable Camera-Status task parameters were found."
            )

        backbone_lr = self._lr * self._config.backbone_lr_scale
        task_lr = self._lr

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": backbone_lr,
                    "group_name": "resnet34_backbone",
                },
                {
                    "params": task_parameters,
                    "lr": task_lr,
                    "group_name": "camera_status_task",
                },
            ],
            weight_decay=self._config.weight_decay,
        )

        backbone_numel = sum(
            parameter.numel()
            for parameter in backbone_parameters
        )
        task_numel = sum(
            parameter.numel()
            for parameter in task_parameters
        )

        print("=" * 80)
        print("[Optimizer] Fixed differential learning rates")
        print("=" * 80)
        print(
            f"ResNet34 backbone:"
            f"\n  tensor count : {len(backbone_parameters)}"
            f"\n  parameter num: {backbone_numel:,}"
            f"\n  learning rate: {backbone_lr:.2e}"
        )
        print(
            f"Camera-Status task modules:"
            f"\n  tensor count : {len(task_parameters)}"
            f"\n  parameter num: {task_numel:,}"
            f"\n  learning rate: {task_lr:.2e}"
        )
        print(
            f"Weight decay: {self._config.weight_decay:.2e}"
        )
        print("Warmup: disabled")
        print("LR scheduler: disabled")
        print("=" * 80)

        return optimizer

    def get_recons_fea(self, input, gt, recons_key=None, residual_iter=20):
        return self._model.get_recons_fea(input, gt, recons_key=recons_key, residual_iter=residual_iter)
    
    def set_recons_param(self, recons_key):
        return self._model.set_recons_param(recons_key)
    
    def get_fea_name(self):
        return list(self._model.recons_map.keys())
    
    

if __name__ == "__main__":
    # 已训练完成的旧版 Camera-Status ResNet34 权重
    checkpoint_path = Path(
        "/data/wsc/navsim_workspace/exp/ckpts/"
        "camera_status_agent/camera_resnet34_seed_0.ckpt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    print("=" * 80)
    print("CameraStatus explicit ResNet34 checkpoint loading test")
    print("=" * 80)
    print(f"Old checkpoint: {checkpoint_path}")

    # 构建当前修改后的显式 ResNet34 模型。
    # 不把 checkpoint_path 传入 Agent，因为原 initialize() 是 strict=True，
    # 且不会自动处理旧、新 ResNet 参数名称的差异。
    config = CameraStatusConfig()

    agent = CameraStatusTrajectoryAgent(
        config=config,
        lr=1e-4,
        checkpoint_path=None,
    )

    model = agent._model
    model.eval()

    checkpoint: Dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
    )

    old_state_dict = checkpoint.get("state_dict", checkpoint)

    if not isinstance(old_state_dict, dict):
        raise TypeError(
            f"Invalid checkpoint state_dict type: {type(old_state_dict)}"
        )

    new_state_dict = model.state_dict()

    def remove_checkpoint_prefix(key: str) -> str:
        """
        将 Lightning/DDP checkpoint 参数名转换为 Model 级参数名。

        例如：

        agent._model.image_backbone.layer1.0.conv1.weight
        ->
        image_backbone.layer1.0.conv1.weight
        """

        prefixes = (
            "module.",
            "agent.",
            "_model.",
            "model.",
        )

        changed = True
        while changed:
            changed = False

            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break

        return key

    def convert_old_resnet_key(key: str) -> str:
        """
        旧模型可能是：

        image_backbone.conv1.weight
        image_backbone.layer1.0.conv1.weight

        新显式模型是：

        image_backbone.resnet.conv1.weight
        image_backbone.resnet.layer1.0.conv1.weight
        """

        # 已经是新显式 ResNet 参数名，不需要转换。
        if key.startswith("image_backbone.resnet."):
            return key

        # 兼容旧模型中可能存在的 model/backbone 包装层。
        old_wrapper_prefixes = (
            "image_backbone.model.",
            "image_backbone.backbone.",
        )

        for prefix in old_wrapper_prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                return f"image_backbone.resnet.{suffix}"

        # 只转换属于 ResNet 主体的参数，避免误伤其他融合模块。
        resnet_module_prefixes = (
            "conv1.",
            "bn1.",
            "layer1.",
            "layer2.",
            "layer3.",
            "layer4.",
            "fc.",
        )

        if key.startswith("image_backbone."):
            suffix = key[len("image_backbone."):]

            if suffix.startswith(resnet_module_prefixes):
                return f"image_backbone.resnet.{suffix}"

        # status encoder、modulation、planning head 等保持原参数名。
        return key

    compatible_state_dict: Dict[str, torch.Tensor] = {}

    unexpected_keys = []
    shape_mismatches = []
    duplicate_mappings = []

    for old_key, old_tensor in old_state_dict.items():
        if not isinstance(old_key, str):
            continue

        if not torch.is_tensor(old_tensor):
            continue

        model_key = remove_checkpoint_prefix(old_key)
        model_key = convert_old_resnet_key(model_key)

        if model_key in compatible_state_dict:
            duplicate_mappings.append(
                (old_key, model_key)
            )
            continue

        if model_key not in new_state_dict:
            unexpected_keys.append(
                (old_key, model_key)
            )
            continue

        old_shape = tuple(old_tensor.shape)
        new_shape = tuple(new_state_dict[model_key].shape)

        if old_shape != new_shape:
            shape_mismatches.append(
                (model_key, old_shape, new_shape)
            )
            continue

        compatible_state_dict[model_key] = old_tensor

    missing_keys = [
        key
        for key in new_state_dict
        if key not in compatible_state_dict
    ]

    def is_harmless_missing_key(key: str) -> bool:
        """
        显式 ResNet 保留的 ImageNet 分类头不参与 NAVSIM forward，
        BatchNorm 的 num_batches_tracked 也不是可训练参数。
        """

        if key in {
            "image_backbone.resnet.fc.weight",
            "image_backbone.resnet.fc.bias",
        }:
            return True

        if key.endswith("num_batches_tracked"):
            return True

        return False

    harmless_missing_keys = [
        key
        for key in missing_keys
        if is_harmless_missing_key(key)
    ]

    critical_missing_keys = [
        key
        for key in missing_keys
        if not is_harmless_missing_key(key)
    ]

    print()
    print("-" * 80)
    print("Checkpoint compatibility result")
    print("-" * 80)
    print(f"Old checkpoint tensors : {len(old_state_dict)}")
    print(f"New model tensors      : {len(new_state_dict)}")
    print(f"Compatible tensors     : {len(compatible_state_dict)}")
    print(f"Critical missing keys  : {len(critical_missing_keys)}")
    print(f"Harmless missing keys  : {len(harmless_missing_keys)}")
    print(f"Unexpected old keys    : {len(unexpected_keys)}")
    print(f"Shape mismatches       : {len(shape_mismatches)}")
    print(f"Duplicate mappings     : {len(duplicate_mappings)}")

    if critical_missing_keys:
        print()
        print("Critical missing keys:")

        for key in critical_missing_keys:
            print(f"  - {key}")

    if harmless_missing_keys:
        print()
        print("Harmless missing keys:")

        for key in harmless_missing_keys:
            print(f"  - {key}")

    if unexpected_keys:
        print()
        print("Unexpected checkpoint keys:")

        for old_key, converted_key in unexpected_keys[:100]:
            print(f"  - {old_key}")
            print(f"    converted to: {converted_key}")

        if len(unexpected_keys) > 100:
            print(f"  ... and {len(unexpected_keys) - 100} more")

    if shape_mismatches:
        print()
        print("Shape mismatches:")

        for key, old_shape, new_shape in shape_mismatches:
            print(
                f"  - {key}: "
                f"checkpoint={old_shape}, "
                f"current_model={new_shape}"
            )

    if duplicate_mappings:
        print()
        print("Duplicate mappings:")

        for old_key, new_key in duplicate_mappings:
            print(f"  - {old_key} -> {new_key}")

    print()
    print("-" * 80)
    print("Actual load_state_dict test")
    print("-" * 80)

    load_result = model.load_state_dict(
        compatible_state_dict,
        strict=False,
    )

    print(f"load_state_dict missing    : {len(load_result.missing_keys)}")
    print(f"load_state_dict unexpected : {len(load_result.unexpected_keys)}")

    can_reuse_checkpoint = (
        len(critical_missing_keys) == 0
        and len(shape_mismatches) == 0
        and len(duplicate_mappings) == 0
    )

    if not can_reuse_checkpoint:
        print()
        print("=" * 80)
        print("[FAIL] The old checkpoint cannot fully initialize the new model.")
        print("=" * 80)
        print(
            "当前存在关键参数缺失或参数维度冲突，"
            "需要继续修改参数名称映射，或者重新训练模型。"
        )
        raise RuntimeError(
            "Old checkpoint is not fully compatible with the explicit ResNet model."
        )

    print()
    print("=" * 80)
    print("[PASS] The old checkpoint can initialize the new explicit ResNet model.")
    print("=" * 80)
    print(
        "所有参与 Camera-Status 轨迹预测的关键参数均已成功加载。"
    )

    if harmless_missing_keys:
        print(
            "缺失项仅为未参与轨迹预测的 ImageNet fc 分类头，"
            "或 BatchNorm 的统计 buffer。"
        )

    print(
        "因此，单纯将 ResNet34 重写为显式结构，"
        "不需要重新训练。"
    )

    # --------------------------------------------------------------
    # 完整 forward 冒烟测试
    # --------------------------------------------------------------
    print()
    print("-" * 80)
    print("Forward smoke test")
    print("-" * 80)

    dummy_features = {
        "camera_feature": torch.randn(
            1,
            3,
            config.image_height,
            config.image_width,
            dtype=torch.float32,
        ),
        "status_feature": torch.randn(
            1,
            config.status_input_dim,
            dtype=torch.float32,
        ),
    }

    with torch.no_grad():
        predictions = model(dummy_features)

    if "trajectory" not in predictions:
        raise KeyError(
            "Model output does not contain the key 'trajectory'."
        )

    trajectory = predictions["trajectory"]

    print(f"trajectory shape : {tuple(trajectory.shape)}")
    print(
        "trajectory finite:",
        bool(torch.isfinite(trajectory).all().item()),
    )

    expected_shape = (
        1,
        config.trajectory_sampling.num_poses,
        3,
    )

    if tuple(trajectory.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected trajectory shape: {tuple(trajectory.shape)}, "
            f"expected: {expected_shape}"
        )

    if not torch.isfinite(trajectory).all():
        raise RuntimeError(
            "Trajectory contains NaN or Inf."
        )

    print()
    print("=" * 80)
    print("Forward smoke test: PASS")
    print("The explicit ResNet model can reuse the old trained checkpoint.")
    print("=" * 80)
