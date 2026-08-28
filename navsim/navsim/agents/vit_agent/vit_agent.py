from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # PyTorch < 2.0 compatibility
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.resnet_agent.resnet_features import (
    ResNetFeatureBuilder,
    ResNetTrajectoryTargetBuilder,
)
from navsim.agents.vit_agent.vit_config import (
    ViTConfig,
)
from navsim.agents.vit_agent.vit_model import (
    ViTTrajectoryModel,
)
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class ViTTrajectoryAgent(AbstractAgent):
    """NAVSIM ViT-B/16 trajectory agent.

    The model keeps the V1 status-modulation and planning path while replacing
    ResNet34 with the explicit ViT-B/16 backbone. The agent also preserves the
    reconstruction-adapter interface used by the post-training pipeline.

    Checkpoint roles
    ----------------
    ``config.image_checkpoint_path``
        Local ImageNet21k+ImageNet2012 ViT-B/16 pretraining checkpoint. It is
        loaded by ``ExplicitViTB16Backbone`` during model construction.

    ``checkpoint_path``
        Optional complete NAVSIM training checkpoint. It is loaded by
        ``initialize`` after model construction and therefore overrides the
        corresponding ImageNet-initialized and randomly initialized weights.
    """

    def __init__(
        self,
        config: ViTConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}.")
        if config.backbone_lr <= 0.0:
            raise ValueError(
                f"config.backbone_lr must be positive, got {config.backbone_lr}."
            )

        self._config = config
        self._lr = float(lr)
        self._checkpoint_path = checkpoint_path
        self._model = ViTTrajectoryModel(config)

    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _safe_torch_load(path: Path) -> Any:
        """Load a checkpoint on CPU across old and new PyTorch versions."""

        try:
            return torch.load(
                str(path),
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        except TypeError:
            return torch.load(
                str(path),
                map_location=torch.device("cpu"),
            )

    @staticmethod
    def _unwrap_state_dict(checkpoint: Any) -> Dict[str, Tensor]:
        """Extract tensor parameters from common Lightning/PyTorch formats."""

        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "NAVSIM checkpoint must be a mapping, got "
                f"{type(checkpoint)}."
            )

        state_like: Mapping[str, Any] = checkpoint
        for key in ("state_dict", "model_state_dict", "model", "agent"):
            nested = state_like.get(key)
            if isinstance(nested, Mapping):
                state_like = nested
                break

        state_dict = {
            key: value
            for key, value in state_like.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        if not state_dict:
            raise RuntimeError("No tensor parameters were found in the checkpoint.")
        return state_dict

    @staticmethod
    def _strip_wrapper_prefixes(key: str) -> str:
        """Remove only known training-wrapper prefixes from a parameter key."""

        prefixes = ("module.", "agent.", "model.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    @staticmethod
    def _shape_mismatches(
        candidate: Mapping[str, Tensor],
        expected: Mapping[str, Tensor],
    ) -> List[str]:
        mismatches: List[str] = []
        for key in sorted(set(candidate).intersection(expected)):
            if tuple(candidate[key].shape) != tuple(expected[key].shape):
                mismatches.append(
                    f"{key}: checkpoint={tuple(candidate[key].shape)}, "
                    f"model={tuple(expected[key].shape)}"
                )
        return mismatches

    def _load_complete_checkpoint(self, raw_state_dict: Dict[str, Tensor]) -> str:
        """Load either an agent-level or model-only checkpoint strictly.

        Unrelated Lightning state entries are ignored, but every parameter of
        the selected target (agent or model) must be present and shape-compatible.
        """

        normalized = {
            self._strip_wrapper_prefixes(key): value
            for key, value in raw_state_dict.items()
        }

        agent_expected = self.state_dict()
        model_expected = self._model.state_dict()

        candidates: List[Tuple[str, Dict[str, Tensor], Mapping[str, Tensor]]] = []

        # Candidate A: checkpoint already contains agent keys such as
        # ``_model.image_backbone...``.
        candidates.append(("agent", normalized, agent_expected))

        # Candidate B: model-only checkpoint. Also accept agent keys by removing
        # one leading ``_model.``.
        model_candidate = {
            key[len("_model.") :] if key.startswith("_model.") else key: value
            for key, value in normalized.items()
        }
        candidates.append(("model", model_candidate, model_expected))

        diagnostics: List[str] = []
        for target_name, candidate_all, expected in candidates:
            candidate = {
                key: value for key, value in candidate_all.items() if key in expected
            }
            missing = sorted(set(expected) - set(candidate))
            mismatched = self._shape_mismatches(candidate, expected)

            if not missing and not mismatched:
                if target_name == "agent":
                    self.load_state_dict(candidate, strict=True)
                else:
                    self._model.load_state_dict(candidate, strict=True)
                return target_name

            diagnostics.append(
                f"{target_name}: missing={missing[:20]}, "
                f"shape_mismatch={mismatched[:20]}"
            )

        raise RuntimeError(
            "The complete NAVSIM checkpoint is incompatible with the current "
            "ViT-B/16 architecture. This optimized version expects "
            "the configured token grid and exactly matching module shapes.\n"
            + "\n".join(diagnostics)
        )

    def initialize(self) -> None:
        """Optionally restore a complete trained NAVSIM checkpoint."""

        if self._checkpoint_path is None:
            return

        checkpoint_path = Path(self._checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"NAVSIM checkpoint was not found: {checkpoint_path}"
            )

        checkpoint = self._safe_torch_load(checkpoint_path)
        raw_state_dict = self._unwrap_state_dict(checkpoint)
        loaded_target = self._load_complete_checkpoint(raw_state_dict)
        print(
            "[ViTTrajectoryAgent] Loaded complete NAVSIM "
            f"checkpoint into {loaded_target}: {checkpoint_path}"
        )

    def get_sensor_config(self) -> SensorConfig:
        """Use only the current cam_l0/cam_f0/cam_r0 frame, as in V1."""

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
        """Reuse the V1 trajectory-target builder."""

        return [ResNetTrajectoryTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Reuse the V1 crop/stitch/status feature builder and cache format."""
        return [ResNetFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Run the model and return trajectory plus configured reconstruction tensors."""
        return self._model(features)

    # ------------------------------------------------------------------
    # Reconstruction/post-training adapter interface
    # ------------------------------------------------------------------
    def get_recons_fea(
        self,
        input_data: Dict[str, Tensor],
        gt_data: Tensor,
        recons_key: Optional[str] = None,
        residual_iter: int = 20,
    ):
        """Delegate feature reconstruction to the integrated ViT model."""

        return self._model.get_recons_fea(
            input_data=input_data,
            gt_data=gt_data,
            residual_iter=residual_iter,
            recons_key=recons_key,
        )

    def set_recons_param(self, recons_key: str) -> None:
        """Enable only parameters upstream of the selected reconstruction node."""
        self._model.set_recons_param(recons_key)

    def get_fea_name(self) -> List[str]:
        """Return the six supported reconstruction feature names in order."""
        return self._model.get_fea_name()

    # ------------------------------------------------------------------
    # Closed-loop inference and training interface
    # ------------------------------------------------------------------
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Build one sample, run inference, and convert it to a NAVSIM trajectory."""

        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            features: Dict[str, Tensor] = {}
            for builder in self.get_feature_builders():
                built = builder.compute_features(agent_input)
                features.update(built)

            batched_features: Dict[str, Tensor] = {}
            for name, value in features.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"Feature {name!r} must be a tensor, got {type(value)}."
                    )
                batched_features[name] = value.unsqueeze(0).to(
                    device=device,
                    non_blocking=True,
                )

            with torch.inference_mode():
                predictions = self.forward(batched_features)

            poses = predictions["trajectory"].squeeze(0).float().cpu().numpy()
            return Trajectory(
                poses=poses,
                trajectory_sampling=self._config.trajectory_sampling,
            )
        finally:
            self.train(was_training)

    def compute_loss(
        self,
        features: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
    ) -> Tensor:
        """V1 trajectory L1 loss; reconstruction tensors are not loss terms here."""

        del features
        if "trajectory" not in predictions or "trajectory" not in targets:
            raise KeyError(
                "Both predictions and targets must contain 'trajectory'."
            )

        predicted_trajectory = predictions["trajectory"]
        target_trajectory = targets["trajectory"].to(
            device=predicted_trajectory.device,
            dtype=predicted_trajectory.dtype,
            non_blocking=True,
        )
        if predicted_trajectory.shape != target_trajectory.shape:
            raise ValueError(
                "Trajectory prediction/target shapes differ: "
                f"prediction={tuple(predicted_trajectory.shape)}, "
                f"target={tuple(target_trajectory.shape)}."
            )

        return torch.nn.functional.l1_loss(
            predicted_trajectory,
            target_trajectory,
        )

    def _optimizer_parameter_groups(self) -> List[Dict[str, Any]]:
        """Build disjoint LR groups for pretrained and random parameters."""

        trainable = [
            parameter
            for parameter in self._model.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("The model has no trainable parameters.")

        # Only treat the backbone as pretrained when the local ImageNet
        # checkpoint was actually loaded. Otherwise all trainable parameters are
        # random initialization and use the main agent learning rate.
        pretrained: List[Tensor] = []
        if self._config.load_imagenet_checkpoint:
            pretrained = [
                parameter
                for parameter in self._model.image_backbone.parameters()
                if parameter.requires_grad
            ]

        pretrained_ids = {id(parameter) for parameter in pretrained}
        random_init = [
            parameter
            for parameter in trainable
            if id(parameter) not in pretrained_ids
        ]

        groups: List[Dict[str, Any]] = []
        if pretrained:
            groups.append(
                {
                    "params": pretrained,
                    "lr": float(self._config.backbone_lr),
                    "name": "pretrained_vit_backbone",
                }
            )
        if random_init:
            groups.append(
                {
                    "params": random_init,
                    "lr": self._lr,
                    "name": "randomly_initialized_v5_modules",
                }
            )

        grouped_ids = [
            id(parameter)
            for group in groups
            for parameter in group["params"]
        ]
        trainable_ids = {id(parameter) for parameter in trainable}
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("A parameter appears in more than one optimizer group.")
        if set(grouped_ids) != trainable_ids:
            raise RuntimeError(
                "Optimizer groups do not cover exactly all trainable parameters."
            )
        return groups

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Create AdamW with differential learning rates.

        Loaded ViT-B/16 parameters use ``config.backbone_lr``. The new
        768->512 bridge, status encoder, gamma/beta heads, planning layers and
        trajectory head use the agent-level ``lr``. After ``set_recons_param``,
        only currently enabled parameters are included.
        """

        return torch.optim.AdamW(
            self._optimizer_parameter_groups(),
            weight_decay=float(self._config.weight_decay),
        )
