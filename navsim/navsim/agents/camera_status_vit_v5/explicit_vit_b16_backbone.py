from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class PositionEmbs(nn.Module):
    """Learned absolute position embedding; names match the ASYML checkpoint."""

    def __init__(self, num_patches: int, embed_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1:] != self.pos_embedding.shape[1:]:
            raise ValueError(
                "Token and position-embedding shapes differ: "
                f"tokens={tuple(tokens.shape)}, "
                f"position={tuple(self.pos_embedding.shape)}."
            )
        return self.dropout(tokens + self.pos_embedding)


class MlpBlock(nn.Module):
    """ViT feed-forward block; parameter names match the checkpoint."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim, bias=True)
        self.fc2 = nn.Linear(mlp_dim, in_dim, bias=True)
        self.act = nn.GELU()
        self.dropout1 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.fc1(tokens)
        tokens = self.act(tokens)
        tokens = self.dropout1(tokens)
        tokens = self.fc2(tokens)
        tokens = self.dropout2(tokens)
        return tokens


class LinearGeneral(nn.Module):
    """Google/ASYML-style tensor-layout-preserving linear projection."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*input_shape, *output_shape))
        self.bias = nn.Parameter(torch.zeros(*output_shape))

    def forward(
        self,
        tensor: Tensor,
        dims: Tuple[list[int], list[int]],
    ) -> Tensor:
        return torch.tensordot(tensor, self.weight, dims=dims) + self.bias


@contextmanager
def _sdp_kernel_context(strict_memory_efficient: bool) -> Iterator[None]:
    """Prefer Flash/memory-efficient SDPA and optionally forbid math fallback."""

    if not torch.cuda.is_available() or not hasattr(torch.backends.cuda, "sdp_kernel"):
        yield
        return

    # PyTorch 2.0-compatible API. On A100 + BF16 + head_dim=64, Flash SDPA is
    # supported. In strict mode, a non-fused fallback raises instead of silently
    # materializing a large [B, heads, N, N] attention matrix.
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_mem_efficient=True,
        enable_math=not strict_memory_efficient,
    ):
        yield


class SelfAttention(nn.Module):
    """Explicit Q/K/V projections with memory-efficient SDPA."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float,
        strict_memory_efficient: bool,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by heads={num_heads}."
            )
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "PyTorch >= 2.0 is required for scaled_dot_product_attention."
            )

        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.attention_dropout = float(attention_dropout)
        self.strict_memory_efficient = bool(strict_memory_efficient)

        self.query = LinearGeneral(
            input_shape=(embed_dim,),
            output_shape=(self.num_heads, self.head_dim),
        )
        self.key = LinearGeneral(
            input_shape=(embed_dim,),
            output_shape=(self.num_heads, self.head_dim),
        )
        self.value = LinearGeneral(
            input_shape=(embed_dim,),
            output_shape=(self.num_heads, self.head_dim),
        )
        self.out = LinearGeneral(
            input_shape=(self.num_heads, self.head_dim),
            output_shape=(embed_dim,),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        # [B,N,D] -> [B,H,N,Dh]
        query = self.query(tokens, dims=([2], [0])).permute(0, 2, 1, 3).contiguous()
        key = self.key(tokens, dims=([2], [0])).permute(0, 2, 1, 3).contiguous()
        value = self.value(tokens, dims=([2], [0])).permute(0, 2, 1, 3).contiguous()

        dropout_p = self.attention_dropout if self.training else 0.0
        with _sdp_kernel_context(self.strict_memory_efficient):
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )

        # [B,H,N,Dh] -> [B,N,H,Dh] -> [B,N,D]
        context = context.permute(0, 2, 1, 3).contiguous()
        return self.out(context, dims=([2, 3], [0, 1]))


class EncoderBlock(nn.Module):
    """Pre-LN ViT encoder block with checkpoint-compatible names."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout: float,
        layer_norm_eps: float,
        strict_memory_efficient_attention: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            strict_memory_efficient=strict_memory_efficient_attention,
        )
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = MlpBlock(
            in_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.dropout(self.attn(self.norm1(tokens)))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class Encoder(nn.Module):
    """Position embedding, 12 encoder blocks, and final LayerNorm."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout: float,
        layer_norm_eps: float,
        strict_memory_efficient_attention: bool,
    ) -> None:
        super().__init__()
        self.pos_embedding = PositionEmbs(
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout=attention_dropout,
                    layer_norm_eps=layer_norm_eps,
                    strict_memory_efficient_attention=(
                        strict_memory_efficient_attention
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(
        self,
        tokens: Tensor,
        capture_layer_index: Optional[int],
        use_gradient_checkpointing: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tokens = self.pos_embedding(tokens)
        captured: Optional[Tensor] = None

        for index, block in enumerate(self.encoder_layers):
            if use_gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)

            if index == capture_layer_index:
                captured = tokens

        return self.norm(tokens), captured


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint_data: Any) -> Dict[str, Tensor]:
    if not isinstance(checkpoint_data, Mapping):
        raise TypeError(
            f"Checkpoint must be a mapping, got {type(checkpoint_data)}."
        )

    for key in ("state_dict", "model_state_dict", "model"):
        nested = checkpoint_data.get(key)
        if isinstance(nested, Mapping):
            checkpoint_data = nested
            break

    state_dict: Dict[str, Tensor] = {}
    removable_prefixes = ("module.", "model.", "backbone.", "image_backbone.")
    for raw_key, value in checkpoint_data.items():
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
        state_dict[key] = value

    if not state_dict:
        raise RuntimeError("No tensor state_dict entries found in checkpoint.")
    return state_dict


def _infer_square_grid(num_patch_tokens: int) -> Tuple[int, int]:
    side = int(math.sqrt(num_patch_tokens))
    if side * side != num_patch_tokens:
        raise ValueError(
            "Expected square pretrained patch grid, got "
            f"{num_patch_tokens} patch positions."
        )
    return side, side


def _resize_absolute_position_embedding(
    position_embedding: Tensor,
    target_grid_size: Tuple[int, int],
) -> Tensor:
    if position_embedding.ndim != 3 or position_embedding.shape[0] != 1:
        raise ValueError(
            "Expected position embedding [1, tokens, dim], got "
            f"{tuple(position_embedding.shape)}."
        )

    cls_position = position_embedding[:, :1]
    patch_position = position_embedding[:, 1:]
    source_h, source_w = _infer_square_grid(patch_position.shape[1])
    target_h, target_w = target_grid_size
    embed_dim = patch_position.shape[-1]
    original_dtype = patch_position.dtype

    patch_position = patch_position.reshape(
        1, source_h, source_w, embed_dim
    ).permute(0, 3, 1, 2)
    patch_position = F.interpolate(
        patch_position.float(),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=original_dtype)
    patch_position = patch_position.permute(0, 2, 3, 1).reshape(
        1, target_h * target_w, embed_dim
    )
    return torch.cat((cls_position, patch_position), dim=1)


class ExplicitViTB16Backbone(nn.Module):
    """Explicit ViT-B/16 with token-grid reduction and chunked execution."""

    FINAL_CLS_KEY = "recons_vit.final_cls_token"

    def __init__(
        self,
        checkpoint_path: Optional[str],
        load_imagenet_checkpoint: bool,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        token_grid_size: Tuple[int, int],
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
        attention_dropout: float,
        layer_norm_eps: float,
        freeze: bool,
        capture_layer_index: int,
        use_gradient_checkpointing: bool,
        retain_selected_gradients: bool,
        backbone_chunk_size: int,
        force_bf16: bool,
        strict_memory_efficient_attention: bool,
    ) -> None:
        super().__init__()

        required = {
            "patch_size": (tuple(patch_size), (16, 16)),
            "embed_dim": (embed_dim, 768),
            "mlp_dim": (mlp_dim, 3072),
            "num_heads": (num_heads, 12),
            "num_layers": (num_layers, 12),
            "num_classes": (num_classes, 1000),
        }
        invalid = [
            f"{name}={actual} (required {expected})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if invalid:
            raise ValueError(
                "The selected ViT-B/16 checkpoint requires " + ", ".join(invalid)
            )
        if not 0 <= capture_layer_index < num_layers:
            raise ValueError(
                f"capture_layer_index must be in [0,{num_layers - 1}]."
            )
        if backbone_chunk_size <= 0:
            raise ValueError("backbone_chunk_size must be positive.")

        image_h, image_w = image_size
        patch_h, patch_w = patch_size
        if image_h % patch_h != 0 or image_w % patch_w != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        raw_grid = (image_h // patch_h, image_w // patch_w)
        token_h, token_w = token_grid_size
        if not (1 <= token_h <= raw_grid[0] and 1 <= token_w <= raw_grid[1]):
            raise ValueError(
                f"token_grid_size={token_grid_size} must not exceed "
                f"raw_grid={raw_grid}."
            )

        self.image_size = (int(image_h), int(image_w))
        self.patch_size = (int(patch_h), int(patch_w))
        self.raw_grid_size = raw_grid
        self.token_grid_size = (int(token_h), int(token_w))
        self.num_patches = token_h * token_w
        self.embed_dim = int(embed_dim)
        self.capture_layer_index = int(capture_layer_index)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.retain_selected_gradients = bool(retain_selected_gradients)
        self.backbone_chunk_size = int(backbone_chunk_size)
        self.force_bf16 = bool(force_bf16)

        # Names intentionally match the converted local checkpoint.
        self.embedding = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer = Encoder(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            strict_memory_efficient_attention=(
                strict_memory_efficient_attention
            ),
        )
        self.classifier = nn.Linear(embed_dim, num_classes, bias=True)

        self._initialize_parameters()
        if load_imagenet_checkpoint:
            if not checkpoint_path:
                raise ValueError(
                    "checkpoint_path is required when loading pretrained weights."
                )
            self.load_local_imagenet_checkpoint(checkpoint_path)

        for parameter in self.classifier.parameters():
            parameter.requires_grad = False
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    @property
    def sequence_length(self) -> int:
        return self.num_patches + 1

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(
            self.transformer.pos_embedding.pos_embedding,
            std=0.02,
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LinearGeneral):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_local_imagenet_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ViT checkpoint not found: {path}")

        state_dict = _extract_state_dict(_safe_torch_load(path))
        position_key = "transformer.pos_embedding.pos_embedding"
        if position_key not in state_dict:
            raise RuntimeError(
                f"Checkpoint is missing required key {position_key!r}."
            )

        old_shape = tuple(state_dict[position_key].shape)
        state_dict[position_key] = _resize_absolute_position_embedding(
            state_dict[position_key],
            target_grid_size=self.token_grid_size,
        )

        model_state = self.state_dict()
        missing = sorted(set(model_state) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(model_state))
        mismatched = [
            key
            for key in sorted(set(model_state) & set(state_dict))
            if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
        ]
        if missing or unexpected or mismatched:
            details = [
                (key, tuple(state_dict[key].shape), tuple(model_state[key].shape))
                for key in mismatched[:20]
            ]
            raise RuntimeError(
                "Checkpoint is incompatible with ExplicitViTB16Backbone.\n"
                f"path={path}\n"
                f"missing={missing[:20]}\n"
                f"unexpected={unexpected[:20]}\n"
                f"shape_mismatch={details}"
            )

        self.load_state_dict(state_dict, strict=True)
        print(
            "[ViT-B/16] Loaded local ImageNet checkpoint\n"
            f"  path: {path}\n"
            f"  image_size: {self.image_size}\n"
            f"  raw_patch_grid: {self.raw_grid_size}\n"
            f"  transformer_token_grid: {self.token_grid_size}\n"
            f"  sequence_length: {self.sequence_length}\n"
            f"  position_embedding: {old_shape} -> "
            f"{tuple(state_dict[position_key].shape)}"
        )

    def _retain_grad_if_requested(self, tensor: Tensor) -> None:
        if self.retain_selected_gradients and tensor.requires_grad:
            tensor.retain_grad()

    def _autocast_context(self, images: Tensor):
        enabled = self.force_bf16 and images.is_cuda
        if not enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        )

    def _forward_one_chunk(
        self,
        images: Tensor,
        return_reconstruction_features: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        with self._autocast_context(images):
            # [b,3,256,1024] -> [b,768,16,64]
            embedding_map = self.embedding(images)

            # [b,768,16,64] -> [b,768,8,32]. This is the decisive memory
            # reduction: 1024 patch tokens become 256 patch tokens.
            if tuple(embedding_map.shape[-2:]) != self.token_grid_size:
                embedding_map = F.adaptive_avg_pool2d(
                    embedding_map,
                    output_size=self.token_grid_size,
                )

            patch_tokens = embedding_map.flatten(2).transpose(1, 2).contiguous()
            cls_token = self.cls_token.expand(images.shape[0], -1, -1)
            tokens = torch.cat((cls_token, patch_tokens), dim=1)

            capture_index = (
                self.capture_layer_index
                if return_reconstruction_features
                else None
            )
            tokens, layer_output = self.transformer(
                tokens,
                capture_layer_index=capture_index,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )
            final_cls = tokens[:, 0]
        return final_cls, layer_output

    def forward(
        self,
        images: Tensor,
        return_reconstruction_features: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if images.ndim != 4:
            raise ValueError(
                f"Expected images [B,3,H,W], got {tuple(images.shape)}."
            )
        if images.shape[1] != 3 or tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected [B,3,{self.image_size[0]},{self.image_size[1]}], "
                f"got {tuple(images.shape)}."
            )

        cls_chunks = []
        layer_chunks = []
        for start in range(0, images.shape[0], self.backbone_chunk_size):
            end = min(start + self.backbone_chunk_size, images.shape[0])
            final_cls, layer_output = self._forward_one_chunk(
                images[start:end],
                return_reconstruction_features=return_reconstruction_features,
            )
            cls_chunks.append(final_cls)
            if return_reconstruction_features:
                if layer_output is None:
                    raise RuntimeError(
                        f"Encoder layer {self.capture_layer_index} was not captured."
                    )
                layer_chunks.append(layer_output)

        final_cls_token = torch.cat(cls_chunks, dim=0)
        selected: Dict[str, Tensor] = {}
        if return_reconstruction_features:
            layer_output = torch.cat(layer_chunks, dim=0)
            self._retain_grad_if_requested(layer_output)
            self._retain_grad_if_requested(final_cls_token)
            layer_key = (
                "recons_vit.transformer.encoder_layers."
                f"{self.capture_layer_index}.output"
            )
            selected[layer_key] = layer_output
            selected[self.FINAL_CLS_KEY] = final_cls_token

        return final_cls_token, selected
