import math
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..rvs_cpt import composite_reverseCom, linear_reverseCom, optimal_embedding
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from rvs_cpt import composite_reverseCom, linear_reverseCom, optimal_embedding


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Stochastic depth applied per sample.

    Args:
        x:
            Tensor with arbitrary trailing dimensions.
        drop_prob:
            Probability of dropping the residual branch.
        training:
            Whether the module is in training mode.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device,
    )
    random_tensor.floor_()

    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(
            x,
            drop_prob=self.drop_prob,
            training=self.training,
        )


class Mlp(nn.Module):
    """
    ViT feed-forward network:

        Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # Keep these names unchanged for checkpoint compatibility.
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    """
    Standard multi-head self-attention used by the released models.

    The original implementation uses one fused qkv linear layer:
        qkv: D -> 3D
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={dim} must be divisible by "
                f"num_heads={num_heads}."
            )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (
            qk_scale
            if qk_scale is not None
            else self.head_dim ** -0.5
        )

        all_head_dim = self.head_dim * num_heads

        # Names must remain qkv/proj for loading official weights.
        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_qkv: bool = False,
    ):
        """
        Args:
            x:
                Token tensor [B, N, D].
            return_attention:
                Return the attention matrix.
            return_qkv:
                Also return q, k and v for reconstruction analysis.

        Returns:
            By default:
                output tensor [B, N, D]

            If return_attention=True:
                output, attention

            If return_qkv=True:
                output, attention, q, k, v
        """
        batch_size, num_tokens, dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)

        attention_logits = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        attention = attention_logits.softmax(dim=-1)
        attention = self.attn_drop(attention)

        output = attention @ v
        output = output.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            dim,
        )

        output = self.proj(output)
        output = self.proj_drop(output)

        if return_qkv:
            return output, attention, q, k, v

        if return_attention:
            return output, attention

        return output


class Block(nn.Module):
    """
    Pre-LayerNorm ViT encoder block:

        y = x + Attention(LN1(x))
        z = y + MLP(LN2(y))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        # Keep module names identical to the official implementation.
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ):
        """
        return_details=True is useful for feature reconstruction.

        It exposes:
            block_input
            norm1_output
            attention_output
            after_attention_residual
            norm2_output
            mlp_output
            block_output
            attention_matrix
        """
        block_input = x

        norm1_output = self.norm1(block_input)

        attention_output, attention_matrix = self.attn(
            norm1_output,
            return_attention=True,
        )

        after_attention_residual = (
            block_input + self.drop_path(attention_output)
        )

        norm2_output = self.norm2(
            after_attention_residual
        )

        mlp_output = self.mlp(norm2_output)

        block_output = (
            after_attention_residual
            + self.drop_path(mlp_output)
        )

        if not return_details:
            return block_output

        details = {
            "block_input": block_input,
            "norm1_output": norm1_output,
            "attention_output": attention_output,
            "attention_matrix": attention_matrix,
            "after_attention_residual":
                after_attention_residual,
            "norm2_output": norm2_output,
            "mlp_output": mlp_output,
            "block_output": block_output,
        }

        return block_output, details


class PatchEmbed(nn.Module):
    """
    Image-to-patch embedding implemented using Conv2d.

    CIFAR-100:
        [B, 3, 32, 32]
        -> [B, 192, 8, 8]
        -> [B, 64, 192]

    Tiny ImageNet:
        [B, 3, 64, 64]
        -> [B, 192, 8, 8]
        -> [B, 64, 192]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size={img_size} must be divisible by "
                f"patch_size={patch_size}."
            )

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size,
        )
        self.num_patches = (
            self.grid_size[0] * self.grid_size[1]
        )

        # Keep name "proj" for checkpoint compatibility.
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_spatial: bool = False,
    ):
        batch_size, channels, height, width = x.shape

        if (
            height % self.patch_size != 0
            or width % self.patch_size != 0
        ):
            raise ValueError(
                f"Input spatial size {(height, width)} must be "
                f"divisible by patch_size={self.patch_size}."
            )

        spatial_features = self.proj(x)
        tokens = spatial_features.flatten(2).transpose(1, 2)

        if return_spatial:
            return tokens, spatial_features

        return tokens


class SmallDatasetVisionTransformer(nn.Module):
    """
    Standalone implementation matching the ViT architecture used in:

        How to Train Vision Transformer on Small-scale Datasets?
        BMVC 2022

    Official configuration:
        embed_dim = 192
        depth = 9
        num_heads = 12
        mlp_ratio = 2
        qkv_bias = True
        LayerNorm eps = 1e-6

    This class deliberately exposes intermediate representations to make
    feature reconstruction easier.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_classes: int,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 9,
        num_heads: int = 12,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        pre_train: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.in_chans = in_chans

        norm_layer = partial(
            nn.LayerNorm,
            eps=norm_eps,
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.num_patches = self.patch_embed.num_patches

        # Both inputs produce an 8x8 token grid:
        # 64 patch tokens + 1 CLS token.
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + 1,
                embed_dim,
            )
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_path_values = torch.linspace(
            0,
            drop_path_rate,
            depth,
        ).tolist()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_values[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.cls_key = f"z{depth + 1}"
        self.recons_map = {
            "recons_out": depth + 4,
            f"recons_{self.cls_key}": depth + 3,
        }
        for idx in range(depth, -1, -1):
            self.recons_map[f"recons_z{idx}"] = idx + 2
        self._initialize_weights()

    def named_children(self):
        yield "patch_embed", self.patch_embed
        yield "pos_drop", self.pos_drop
        for idx, block in enumerate(self.blocks, start=1):
            yield f"block{idx}", block
        yield "norm", self.norm
        yield "head", self.head

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )
        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def interpolate_pos_encoding(
        self,
        x: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """
        Bicubically interpolate patch positional embeddings when input
        resolution differs from the training resolution.

        This follows the original repository implementation.
        """
        num_input_patches = x.shape[1] - 1
        num_pretrained_patches = (
            self.pos_embed.shape[1] - 1
        )

        if (
            num_input_patches == num_pretrained_patches
            and width == height
        ):
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]

        original_grid_size = int(
            math.sqrt(num_pretrained_patches)
        )

        if (
            original_grid_size * original_grid_size
            != num_pretrained_patches
        ):
            raise RuntimeError(
                "The number of pretrained patch tokens is not "
                "a perfect square."
            )

        new_width = width // self.patch_size
        new_height = height // self.patch_size

        patch_pos_embed = patch_pos_embed.reshape(
            1,
            original_grid_size,
            original_grid_size,
            dim,
        )
        patch_pos_embed = patch_pos_embed.permute(
            0, 3, 1, 2
        )

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_width, new_height),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        patch_pos_embed = patch_pos_embed.reshape(
            1,
            new_width * new_height,
            dim,
        )

        return torch.cat([class_pos_embed, patch_pos_embed],dim=1,)

    def prepare_tokens(
        self,
        x: torch.Tensor,
        return_spatial_patch_features: bool = False,
    ):
        batch_size, channels, width, height = x.shape

        patch_tokens, spatial_patch_features = (
            self.patch_embed(
                x,
                return_spatial=True,
            )
        )

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            [cls_tokens, patch_tokens],
            dim=1,
        )

        position_embedding = (
            self.interpolate_pos_encoding(
                tokens,
                width,
                height,
            )
        )

        tokens_before_dropout = (
            tokens + position_embedding
        )
        tokens_after_dropout = self.pos_drop(
            tokens_before_dropout
        )

        if not return_spatial_patch_features:
            return tokens_after_dropout

        details = {
            "spatial_patch_features":
                spatial_patch_features,
            "patch_tokens": patch_tokens,
            "cls_tokens": cls_tokens,
            "position_embedding": position_embedding,
            "tokens_before_pos_dropout":
                tokens_before_dropout,
            "tokens": tokens_after_dropout,
        }

        return tokens_after_dropout, details

    def forward_features(
        self,
        x: torch.Tensor,
        return_all_features: bool = False,
    ):
        """
        Args:
            return_all_features:
                If True, return all intermediate block features.

        Returns:
            Default:
                normalized CLS feature [B, 192]

            return_all_features=True:
                cls_feature and a dictionary containing every block's
                intermediate tensors.
        """
        tokens, token_details = self.prepare_tokens(
            x,
            return_spatial_patch_features=True,
        )

        block_outputs: List[torch.Tensor] = []
        block_details: List[Dict[str, torch.Tensor]] = []

        for block in self.blocks:
            if return_all_features:
                tokens, details = block(
                    tokens,
                    return_details=True,
                )
                block_details.append(details)
            else:
                tokens = block(tokens)

            block_outputs.append(tokens)

        normalized_tokens = self.norm(tokens)
        cls_feature = normalized_tokens[:, 0]

        if not return_all_features:
            return cls_feature

        features = {
            **token_details,
            "block_outputs": block_outputs,
            "block_details": block_details,
            "tokens_before_final_norm": tokens,
            "normalized_tokens": normalized_tokens,
            "cls_feature": cls_feature,
        }

        return cls_feature, features

    def forward(
        self,
        x: torch.Tensor,
        return_all_features: bool = False,
    ):
        if not return_all_features:
            return self.forward_recons_features(x)

        # Keep the old explicit feature API available for code that needs the
        # detailed block internals rather than the FRPT-style feature dict.
        cls_feature, features = self.forward_features(
            x,
            return_all_features=True,
        )

        logits = self.head(cls_feature)
        features["logits"] = logits

        return logits, features

    def forward_recons_features(
        self,
        x: torch.Tensor,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        tokens = self.prepare_tokens(x)
        res: Dict[str, torch.Tensor] = {"z0": tokens}

        for idx, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            res[f"z{idx}"] = tokens

        normalized_tokens = self.norm(tokens)
        cls_feature = normalized_tokens[:, 0]
        res[self.cls_key] = cls_feature
        res["out"] = self.head(cls_feature)
        return res

    @staticmethod
    def _replace_cls_token(
        tokens: torch.Tensor,
        cls_feature: torch.Tensor,
    ) -> torch.Tensor:
        out = tokens.detach().clone()
        out[:, 0] = cls_feature
        return out

    @staticmethod
    def _module_requires_grad(module: nn.Module) -> List[bool]:
        return [param.requires_grad for param in module.parameters()]

    @staticmethod
    def _restore_requires_grad(
        module: nn.Module,
        requires_grad: List[bool],
    ) -> None:
        for param, req_grad in zip(module.parameters(), requires_grad):
            param.requires_grad_(req_grad)

    def _reverse_final_norm_cls(
        self,
        tokens_before_norm: torch.Tensor,
        recons_cls_after_norm: torch.Tensor,
        residual_iter: int,
        damping: float,
        max_step_norm: float,
        tol: float,
        reverse_method: str,
        jacobian_elem_limit: float,
    ) -> torch.Tensor:
        init_cls = tokens_before_norm[:, 0].detach()
        recons_cls_before_norm = composite_reverseCom(
            init_cls,
            recons_cls_after_norm,
            self.norm,
            iter=residual_iter,
            damping=damping,
            max_step_norm=max_step_norm,
            tol=tol,
            method=reverse_method,
            jacobian_elem_limit=jacobian_elem_limit,
        )

        return self._replace_cls_token(tokens_before_norm, recons_cls_before_norm)

    def _reverse_block(
        self,
        block: Block,
        front_fea: torch.Tensor,
        back_fea_star: torch.Tensor,
        residual_iter: int,
        damping: float,
        max_step_norm: float,
        tol: float,
        reverse_method: str,
        jacobian_elem_limit: float,
    ) -> torch.Tensor:
        front_fea_star = composite_reverseCom(
            front_fea,
            back_fea_star,
            block,
            iter=residual_iter,
            damping=damping,
            max_step_norm=max_step_norm,
            tol=tol,
            method=reverse_method,
            jacobian_elem_limit=jacobian_elem_limit,
        )
        return front_fea_star

    def get_recons_fea(
        self,
        input_data: torch.Tensor,
        label_data: torch.Tensor,
        residual_iter: int = 5,
        damping: float = 0.8,
        max_step_norm: float = 10.0,
        tol: float = 1e-6,
        reverse_method: str = "auto",
        jacobian_elem_limit: float = 5e7,
        recons_key: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        res = self(input_data)
        recons_out = optimal_embedding(res["out"], label_data)
        if recons_key == "recons_out": return recons_out

        if isinstance(self.head, nn.Identity):
            recons_cls = recons_out
        else:
            recons_cls = linear_reverseCom(res[self.cls_key], recons_out, self.head)
        if recons_key == f"recons_{self.cls_key}": return recons_cls

        current = self._reverse_final_norm_cls(
            res[f"z{self.depth}"],
            recons_cls,
            residual_iter=residual_iter,
            damping=damping,
            max_step_norm=max_step_norm,
            tol=tol,
            reverse_method=reverse_method,
            jacobian_elem_limit=jacobian_elem_limit,
        )
        if recons_key == f"recons_z{self.depth}": return current

        recons = {
            "recons_out": recons_out,
            f"recons_{self.cls_key}": recons_cls,
            f"recons_z{self.depth}": current,
        }

        for idx in reversed(range(1, self.depth + 1)):
            current = self._reverse_block(
                self.blocks[idx - 1],
                res[f"z{idx - 1}"],
                current,
                residual_iter=residual_iter,
                damping=damping,
                max_step_norm=max_step_norm,
                tol=tol,
                reverse_method=reverse_method,
                jacobian_elem_limit=jacobian_elem_limit,
            )
            recons[f"recons_z{idx - 1}"] = current
            if recons_key == f"recons_z{idx - 1}": return current

        return recons

    def check_cond(self):
        for name, param in self.named_parameters():
            if "weight" not in name:
                continue
            print(name, param.size(), "-" * 20)
            if param.ndim == 2:
                print(torch.linalg.matrix_rank(param.data))
                print(torch.linalg.cond(param.data))
            elif param.ndim == 4:
                weight_mat = param.data.detach().reshape(param.data.shape[0], -1).to(torch.float64)
                print("weight matrix rank:", torch.linalg.matrix_rank(weight_mat))
                print("weight matrix cond:", torch.linalg.cond(weight_mat))

    def get_last_selfattention(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.prepare_tokens(x)

        for block in self.blocks[:-1]:
            tokens = block(tokens)

        norm1_output = self.blocks[-1].norm1(tokens)
        _, attention = self.blocks[-1].attn(
            norm1_output,
            return_attention=True,
        )

        return attention

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
    ) -> List[torch.Tensor]:
        if n <= 0 or n > len(self.blocks):
            raise ValueError(
                f"n must lie in [1, {len(self.blocks)}], "
                f"but received {n}."
            )

        tokens = self.prepare_tokens(x)
        outputs = []

        for index, block in enumerate(self.blocks):
            tokens = block(tokens)

            if len(self.blocks) - index <= n:
                outputs.append(self.norm(tokens))

        return outputs

    def get_fea_id(self, recons_key):
        return self.recons_map.get(recons_key, None)

    def get_fea_name(self):
        return list(self.recons_map.keys())

    def forward_logits(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        cls_feature = self.forward_features(
            x,
            return_all_features=False,
        )
        return self.head(cls_feature)
    

def load_small_dataset_vit_ssl_checkpoint(
    model: SmallDatasetVisionTransformer,
    checkpoint_path: Union[str, Path],
    checkpoint_key: str = "teacher",
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Load the official DINO-style self-supervised checkpoint.

    Only the ViT backbone is loaded. The supervised classification head
    remains randomly initialized because the released checkpoint is an
    SSL checkpoint rather than a final supervised classifier checkpoint.

    Args:
        model:
            ViTCIFAR100Patch4 or ViTTinyImageNetPatch8.
        checkpoint_path:
            Path to the downloaded .pth file.
        checkpoint_key:
            Normally "teacher". "student" may also exist.
        verbose:
            Print loading statistics.

    Returns:
        Dictionary with loaded, missing, unexpected and skipped keys.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (TypeError, pickle.UnpicklingError):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to be a dictionary, "
            f"but received {type(checkpoint)}."
        )

    if checkpoint_key in checkpoint:
        source_state = checkpoint[checkpoint_key]
    elif "state_dict" in checkpoint:
        source_state = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        source_state = checkpoint["model_state_dict"]
    else:
        source_state = checkpoint

    if not isinstance(source_state, dict):
        raise TypeError(
            "The selected checkpoint state is not a dictionary."
        )

    model_state = model.state_dict()
    compatible_state = {}

    skipped_keys = []
    unexpected_source_keys = []

    for original_key, value in source_state.items():
        key = original_key

        # DistributedDataParallel prefix.
        if key.startswith("module."):
            key = key[len("module."):]

        # DINO MultiCropWrapper:
        # load only backbone.* and exclude projection-head parameters.
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        elif key.startswith("head."):
            # SSL projection head: not the classification head.
            skipped_keys.append(original_key)
            continue

        # Additional possible wrapper name.
        if key.startswith("encoder."):
            key = key[len("encoder."):]

        if key not in model_state:
            unexpected_source_keys.append(original_key)
            continue

        if model_state[key].shape != value.shape:
            skipped_keys.append(
                (
                    original_key,
                    tuple(value.shape),
                    tuple(model_state[key].shape),
                )
            )
            continue

        compatible_state[key] = value

    load_result = model.load_state_dict(
        compatible_state,
        strict=False,
    )

    loaded_keys = sorted(compatible_state.keys())
    missing_keys = sorted(load_result.missing_keys)
    unexpected_keys = sorted(load_result.unexpected_keys)

    if verbose:
        print("=" * 72)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Checkpoint key: {checkpoint_key}")
        print(f"Loaded tensors: {len(loaded_keys)}")
        print(f"Missing model tensors: {len(missing_keys)}")
        print(
            f"Unused source tensors: "
            f"{len(unexpected_source_keys)}"
        )
        print(f"Skipped tensors: {len(skipped_keys)}")

        if "head.weight" in missing_keys:
            print(
                "Classification head was not loaded, which is expected "
                "for the official SSL checkpoint."
            )

        print("=" * 72)

    return {
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "unused_source_keys": unexpected_source_keys,
        "skipped_keys": skipped_keys,
    }


def print_simplevit_smoke(model, input_shape):
    children = list(model.named_children())
    child_names = [name for name, _ in children]
    model_name = getattr(model, "name", model.__class__.__name__)
    print(f"\n[pretrain vit check] {model_name}")
    print("named_children:", child_names)
    for recons_key in model.get_fea_name():
        freeze_from = model.get_fea_id(recons_key)
        print(
            f"{recons_key:>12} | idx={freeze_from:>2} | "
            f"train={child_names[:freeze_from]} | freeze={child_names[freeze_from:]}"
        )

    device = next(model.parameters()).device
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        res = model(x)
        logits, features = model(x, return_all_features=True)
    print("forward shapes:", {key: tuple(value.shape) for key, value in res.items()})
    print("legacy logits shape:", tuple(logits.shape))
    print("legacy feature keys:", list(features.keys()))


if __name__ == "__main__":
    model = SmallDatasetVisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=100,
        embed_dim=192,
        depth=9,
        num_heads=12,
        mlp_ratio=2.0,
    )
    print_simplevit_smoke(model, (2, 3, 32, 32))
