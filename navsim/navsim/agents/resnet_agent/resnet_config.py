from dataclasses import dataclass, field
from typing import Optional, Sequence

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class ResNetConfig:
    """Configuration for the lightweight ResNet trajectory agent."""

    trajectory_sampling: TrajectorySampling = field(
        default_factory=lambda: TrajectorySampling(
            time_horizon=4,
            interval_length=0.5,
        )
    )

    # Camera preprocessing. These crop values follow NAVSIM v1.1 Transfuser.
    camera_history_index: int = 3
    image_height: int = 256
    image_width: int = 1024
    side_crop_height: int = 28
    side_crop_width: int = 416
    front_crop_height: int = 28
    feature_cache_name: str = "camera_status_feature_v1"

    # Image encoder.
    image_architecture: str = "resnet34"
    # The ImageNet weights are intentionally kept outside the repository.
    # Set FRPT_RESNET34_IMAGENET_CKPT (or pass this field explicitly) when
    # load_imagenet_checkpoint=True.
    image_checkpoint_path: Optional[str] = None
    load_imagenet_checkpoint: bool = True
    freeze_image_encoder: bool = False
    normalize_image: bool = True
    image_mean: Sequence[float] = (0.485, 0.456, 0.406)
    image_std: Sequence[float] = (0.229, 0.224, 0.225)
    spatial_pool_height: int = 1
    spatial_pool_width: int = 4
    image_embedding_dim: int = 512

    # Backbone feature capture:
    #   none              -> normal training/inference, no ResNet feature maps returned
    #   stages/blocks/full -> return compact ResNet stage outputs z0-z4
    backbone_feature_capture: str = "stages"
    retain_backbone_feature_gradients: bool = False

    # Ego status and vector modulation.
    status_input_dim: int = 8
    status_intermediate_dim: int = 64
    status_context_dim: int = 128
    modulation_scale: float = 0.1

    # Planning head.
    planning_hidden_dim: int = 256
    leaky_relu_slope: float = 0.5

    # Optimization.
    weight_decay: float = 1.0e-4
    backbone_lr_scale: float = 0.1

    # Keep compact vector intermediates in predictions.
    return_intermediate_features: bool = True

    def __post_init__(self) -> None:
        if self.camera_history_index < 0:
            raise ValueError("camera_history_index must be non-negative.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive.")
        if self.spatial_pool_height <= 0 or self.spatial_pool_width <= 0:
            raise ValueError("Spatial pooling dimensions must be positive.")
        if self.image_architecture != "resnet34":
            raise ValueError("This implementation currently supports only resnet34.")
        if self.load_imagenet_checkpoint and not self.image_checkpoint_path:
            raise ValueError(
                "image_checkpoint_path is required when "
                "load_imagenet_checkpoint=True. Set "
                "FRPT_RESNET34_IMAGENET_CKPT to the local ImageNet checkpoint."
            )
        if not 0.0 < self.backbone_lr_scale <= 1.0:
            raise ValueError(
                "backbone_lr_scale must be in the interval (0, 1]."
            )
        if self.backbone_feature_capture not in {
            "none",
            "stages",
            "blocks",
            "full",
        }:
            raise ValueError(
                "backbone_feature_capture must be one of: "
                "none, stages, blocks, full."
            )
        if (
            self.retain_backbone_feature_gradients
            and self.backbone_feature_capture == "none"
        ):
            raise ValueError(
                "retain_backbone_feature_gradients=True requires "
                "backbone_feature_capture other than 'none'."
            )
        if self.status_input_dim != 8:
            raise ValueError(
                "The current FeatureBuilder returns 8 ego-status values; "
                "status_input_dim must therefore be 8."
            )
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must each contain 3 values.")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("All image_std values must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
