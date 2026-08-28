from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)


@dataclass
class ViTConfig:
    """Configuration for V1 with only the image backbone replaced by ViT-B/16.

    The external dataloader batch size is not changed. Memory is controlled by:
    1. reducing the patch-token grid before the Transformer;
    2. chunking the ViT branch internally;
    3. BF16 autocast on CUDA;
    4. memory-efficient scaled-dot-product attention;
    5. Transformer activation checkpointing.
    """

    trajectory_sampling: TrajectorySampling = field(
        default_factory=lambda: TrajectorySampling(
            time_horizon=4,
            interval_length=0.5,
        )
    )

    # Reuse V1 feature/target builders and camera preprocessing.
    camera_history_index: int = 3
    image_height: int = 256
    image_width: int = 1024
    side_crop_height: int = 28
    side_crop_width: int = 416
    front_crop_height: int = 28
    feature_cache_name: str = "camera_status_feature_v1"

    # Local ViT ImageNet pretraining checkpoint loaded by torch.load.
    image_architecture: str = "vit_b16_google_i21k_in1k_224"
    image_checkpoint_path: Optional[str] = None
    load_imagenet_checkpoint: bool = True
    freeze_image_encoder: bool = False

    normalize_image: bool = True
    image_mean: Sequence[float] = (0.5, 0.5, 0.5)
    image_std: Sequence[float] = (0.5, 0.5, 0.5)

    # Checkpoint-fixed ViT-B/16 architecture.
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_mlp_dim: int = 3072
    vit_depth: int = 12
    vit_num_heads: int = 12
    vit_dropout_rate: float = 0.1
    vit_attention_dropout: float = 0.0
    vit_layer_norm_eps: float = 1.0e-5

    # The patch embedding first produces 16x64 tokens. Before Transformer,
    # adaptive average pooling reduces this grid to 8x32 = 256 tokens.
    # Sequence length becomes 257 after adding CLS, instead of 1025.
    vit_token_grid_height: int = 8
    vit_token_grid_width: int = 32

    # Internal ViT micro-batch. Dataloader batch remains unchanged (e.g. 128).
    # The outputs are concatenated before the status branch and trajectory loss.
    vit_backbone_chunk_size: int = 16

    # CUDA memory controls.
    force_backbone_bf16: bool = True
    strict_memory_efficient_attention: bool = True
    use_gradient_checkpointing: bool = True

    # V1-compatible downstream dimensions and modules.
    image_embedding_dim: int = 512
    status_input_dim: int = 8
    status_intermediate_dim: int = 64
    status_context_dim: int = 128
    modulation_scale: float = 0.1
    planning_hidden_dim: int = 256
    leaky_relu_slope: float = 0.5

    # Differential learning rates.
    backbone_lr: float = 1.0e-5
    weight_decay: float = 1.0e-4

    # Only the six requested reconstruction tensors are returned.
    return_reconstruction_features: bool = True
    retain_reconstruction_feature_gradients: bool = False
    reconstruction_vit_layer_index: int = 10

    def __post_init__(self) -> None:
        if self.image_architecture != "vit_b16_google_i21k_in1k_224":
            raise ValueError(
                "ViT agent is fixed to explicit ViT-B/16, got "
                f"{self.image_architecture!r}."
            )
        if self.camera_history_index < 0:
            raise ValueError("camera_history_index must be non-negative.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive.")
        if (
            self.image_height % self.vit_patch_size != 0
            or self.image_width % self.vit_patch_size != 0
        ):
            raise ValueError(
                "image size must be divisible by vit_patch_size: "
                f"image=({self.image_height}, {self.image_width}), "
                f"patch={self.vit_patch_size}."
            )

        required = {
            "vit_patch_size": (self.vit_patch_size, 16),
            "vit_embed_dim": (self.vit_embed_dim, 768),
            "vit_mlp_dim": (self.vit_mlp_dim, 3072),
            "vit_depth": (self.vit_depth, 12),
            "vit_num_heads": (self.vit_num_heads, 12),
        }
        invalid = [
            f"{name}={actual} (required {expected})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if invalid:
            raise ValueError(
                "The local ViT-B/16 checkpoint requires " + ", ".join(invalid)
            )
        if self.load_imagenet_checkpoint and not self.image_checkpoint_path:
            raise ValueError(
                "image_checkpoint_path is required when "
                "load_imagenet_checkpoint=True."
            )
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must each have 3 values.")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("All image_std values must be positive.")

        raw_grid_h = self.image_height // self.vit_patch_size
        raw_grid_w = self.image_width // self.vit_patch_size
        if not 1 <= self.vit_token_grid_height <= raw_grid_h:
            raise ValueError(
                "vit_token_grid_height must be in "
                f"[1, {raw_grid_h}], got {self.vit_token_grid_height}."
            )
        if not 1 <= self.vit_token_grid_width <= raw_grid_w:
            raise ValueError(
                "vit_token_grid_width must be in "
                f"[1, {raw_grid_w}], got {self.vit_token_grid_width}."
            )
        if self.vit_backbone_chunk_size <= 0:
            raise ValueError("vit_backbone_chunk_size must be positive.")
        if self.status_input_dim != 8:
            raise ValueError("status_input_dim must remain 8.")
        if self.image_embedding_dim != 512:
            raise ValueError("image_embedding_dim must remain 512.")
        if not 0 <= self.reconstruction_vit_layer_index < self.vit_depth:
            raise ValueError(
                "reconstruction_vit_layer_index must be in "
                f"[0, {self.vit_depth - 1}]."
            )
        if self.backbone_lr <= 0.0:
            raise ValueError("backbone_lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 <= self.vit_dropout_rate < 1.0:
            raise ValueError("vit_dropout_rate must be in [0, 1).")
        if not 0.0 <= self.vit_attention_dropout < 1.0:
            raise ValueError("vit_attention_dropout must be in [0, 1).")
