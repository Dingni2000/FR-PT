from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.resnet_agent.resnet_config import ResNetConfig
from navsim.agents.resnet_agent.resnet_features import (
    ResNetFeatureBuilder,
    ResNetTrajectoryTargetBuilder,
)
from navsim.agents.resnet_agent.resnet_model import ResNetTrajectoryModel
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


    
class ResNetTrajectoryAgent(AbstractAgent):
    """NAVSIM v1.1 agent using stitched cameras and ego status to predict trajectory."""
    def __init__(
        self,
        config: ResNetConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._model = ResNetTrajectoryModel(config)

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
        return [ResNetTrajectoryTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ResNetFeatureBuilder(config=self._config)]

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
                "No trainable ResNet task parameters were found."
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
                    "group_name": "resnet_task",
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
            f"ResNet task modules:"
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
