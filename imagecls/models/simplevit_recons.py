import torch
import torch.nn as nn

try:
    from ..rvs_cpt import composite_reverseCom, linear_reverseCom, optimal_embedding
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from rvs_cpt import composite_reverseCom, linear_reverseCom, optimal_embedding


class ViTTokenEmbed(nn.Module):
    def __init__(self, image_size, patch_size, input_channels, embed_dim):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls, x), dim=1) + self.pos_embed


class SimpleViTBlock(nn.Module):
    """No-norm Transformer block compatible with composite_reverseCom."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        y = x + self.drop(attn_out)
        return y + self.drop(self.mlp(y))


class SimpleViTBlockV2(nn.Module):
    """Standard Pre-LayerNorm Transformer block."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.0, norm_eps=1e-6):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        normalized = self.norm1(x)
        attn_out, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        y = x + self.drop(attn_out)
        return y + self.drop(self.mlp(self.norm2(y)))


class ReconstructableSimpleViT(nn.Module):
    def __init__(
        self,
        task_name,
        image_size,
        patch_size,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=2.0,
        input_channels=3,
        dropout=0.0,
    ):
        super().__init__()
        self.task_name = task_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.name = f"{task_name}_simplevit_p{patch_size}_d{embed_dim}_l{depth}"

        self.token_embed = ViTTokenEmbed(image_size, patch_size, input_channels, embed_dim)
        for idx in range(1, depth + 1):
            self.add_module(
                f"block{idx}",
                SimpleViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout),)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.cls_key = f"z{depth + 1}"
        self.recons_map = {"recons_out": depth + 2, f"recons_{self.cls_key}": depth + 1}
        for idx in range(depth, -1, -1):
            self.recons_map[f"recons_z{idx}"] = idx + 1

    def get_fea_id(self, recons_key):
        return self.recons_map.get(recons_key, None)

    def get_fea_name(self):
        return list(self.recons_map.keys())

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

    def _block(self, idx):
        return getattr(self, f"block{idx}")

    def forward(self, x):
        tokens = self.token_embed(x)
        res = {"z0": tokens}
        for idx in range(1, self.depth + 1):
            tokens = self._block(idx)(tokens)
            res[f"z{idx}"] = tokens
        cls_fea = tokens[:, 0]
        res[self.cls_key] = cls_fea
        res["out"] = self.head(cls_fea)
        return res

    @staticmethod
    def _replace_cls(tokens, cls_fea):
        out = tokens.detach().clone()
        out[:, 0] = cls_fea
        return out

    def get_recons_fea(
        self,
        input_data,
        label_data,
        residual_iter=20,
        damping=0.8,
        max_step_norm=10.0,
        tol=1e-6,
        reverse_method="auto",
        jacobian_elem_limit=5e7,
        recons_key=None,
    ):
        res = self(input_data)
        recons_out = optimal_embedding(res["out"], label_data)
        if recons_key == "recons_out": return recons_out

        recons_cls = linear_reverseCom(res[self.cls_key], recons_out, self.head)
        if recons_key == f"recons_{self.cls_key}": return recons_cls
        current = self._replace_cls(res[f"z{self.depth}"], recons_cls)
        if recons_key == f"recons_z{self.depth}": return current

        recons = {
            "recons_out": recons_out,
            f"recons_{self.cls_key}": recons_cls,
            f"recons_z{self.depth}": current,
        }
        for idx in reversed(range(1, self.depth + 1)):
            current = composite_reverseCom(
                res[f"z{idx - 1}"],
                current,
                self._block(idx),
                iter=residual_iter,
                damping=damping,
                max_step_norm=max_step_norm,
                tol=tol,
                method=reverse_method,
                jacobian_elem_limit=jacobian_elem_limit,
            )
            recons[f"recons_z{idx - 1}"] = current
            if recons_key == f"recons_z{idx - 1}": return current
        return recons


# class ReconstructableSimpleViTv2(ReconstructableSimpleViT):
#     """Simple ViT with standard Pre-LayerNorm blocks and final LayerNorm."""
#     def __init__(self, *args, norm_eps=1e-6, **kwargs):
#         super().__init__(*args, **kwargs)
#         # self.norm = nn.LayerNorm(self.embed_dim, eps=norm_eps)
#         for idx in range(1, self.depth + 1):
#             old_block = self._block(idx)
#             setattr(
#                 self,
#                 f"block{idx}",
#                 SimpleViTBlockV2(
#                     self.embed_dim,
#                     self.num_heads,
#                     mlp_ratio=old_block.mlp[0].out_features / self.embed_dim,
#                     dropout=old_block.drop.p,
#                     norm_eps=norm_eps))
#         self.name = self.name.replace("simplevit_", "simplevitv2_")
#         head = self._modules.pop("head")
#         norm = self._modules.pop("norm")
#         self._modules["norm"] = norm
#         self._modules["head"] = head
#         self.recons_map = {"recons_out": self.depth + 3, f"recons_{self.cls_key}": self.depth + 2}
#         for idx in range(self.depth, -1, -1):
#             self.recons_map[f"recons_z{idx}"] = idx + 1

#     def forward(self, x):
#         tokens = self.token_embed(x)
#         res = {"z0": tokens}
#         for idx in range(1, self.depth + 1):
#             tokens = self._block(idx)(tokens)
#             res[f"z{idx}"] = tokens
#         # normalized = self.norm(tokens)
#         res[self.cls_key] = tokens[:, 0]
#         res["out"] = self.head(res[self.cls_key])
#         return res

#     @staticmethod
#     def _reverse_module(front_fea, back_fea_star, module, **kwargs):
#         return composite_reverseCom(front_fea, back_fea_star, module, **kwargs)

#     def get_recons_fea(
#         self,
#         input_data,
#         label_data,
#         residual_iter=20,
#         damping=0.8,
#         max_step_norm=10.0,
#         tol=1e-6,
#         reverse_method="auto",
#         jacobian_elem_limit=5e7,
#         recons_key=None,
#     ):
#         res = self(input_data)
#         recons_out = optimal_embedding(res["out"], label_data)
#         if recons_key == "recons_out":
#             return recons_out

#         recons_cls = linear_reverseCom(res[self.cls_key], recons_out, self.head)
#         if recons_key == f"recons_{self.cls_key}":
#             return recons_cls

#         reverse_args = dict(
#             iter=residual_iter,
#             damping=damping,
#             max_step_norm=max_step_norm,
#             tol=tol,
#             method=reverse_method,
#             jacobian_elem_limit=jacobian_elem_limit,
#         )
#         normalized = self.norm(res[f"z{self.depth}"])
#         normalized = self._replace_cls(normalized, recons_cls)
#         current = self._reverse_module(res[f"z{self.depth}"], normalized, self.norm, **reverse_args)
#         if recons_key == f"recons_z{self.depth}":
#             return current

#         recons = {
#             "recons_out": recons_out,
#             f"recons_{self.cls_key}": recons_cls,
#             f"recons_z{self.depth}": current,
#         }
#         for idx in reversed(range(1, self.depth + 1)):
#             current = composite_reverseCom(
#                 res[f"z{idx - 1}"],
#                 current,
#                 self._block(idx),
#                 iter=residual_iter,
#                 damping=damping,
#                 max_step_norm=max_step_norm,
#                 tol=tol,
#                 method=reverse_method,
#                 jacobian_elem_limit=jacobian_elem_limit,
#             )
#             recons[f"recons_z{idx - 1}"] = current
#             if recons_key == f"recons_z{idx - 1}":
#                 return current
#         return recons


class ReconstructableSimpleViTv2(ReconstructableSimpleViT):
    """Simple ViT with Pre-LayerNorm Transformer blocks."""

    def __init__(self, *args, norm_eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)

        for idx in range(1, self.depth + 1):
            old_block = self._block(idx)
            setattr(
                self,
                f"block{idx}",
                SimpleViTBlockV2(
                    self.embed_dim,
                    self.num_heads,
                    mlp_ratio=old_block.mlp[0].out_features / self.embed_dim,
                    dropout=old_block.drop.p,
                    norm_eps=norm_eps,
                ),
            )

        self.name = self.name.replace("simplevit_", "simplevitv2_")


def print_simplevit_smoke(model, input_shape):
    children = list(model.named_children())
    child_names = [name for name, _ in children]
    print(f"\n[simplevit check] {model.name}")
    print("named_children:", child_names)
    for recons_key in model.get_fea_name():
        freeze_from = model.get_fea_id(recons_key)
        print(
            f"{recons_key:>10} | idx={freeze_from:>2} | "
            f"train={child_names[:freeze_from]} | freeze={child_names[freeze_from:]}"
        )

    device = next(model.parameters()).device
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        res = model(x)
    print("forward shapes:", {key: tuple(value.shape) for key, value in res.items()})


if __name__ == "__main__":
    model = ReconstructableSimpleViT(
        task_name="test",
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )
    print_simplevit_smoke(model, (2, 3, 32, 32))
