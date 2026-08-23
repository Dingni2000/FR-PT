import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision.models.resnet import BasicBlock, ResNet

try:
    from ..rvs_cpt import composite_reverseCom, linear_reverseCom, optimal_embedding
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from rvs_cpt import composite_reverseCom, linear_reverseCom, optimal_embedding


class ReconstructableResNet(ResNet):
    _ARCH_LAYERS = {
        "18": [2, 2, 2, 2],
        "34": [3, 4, 6, 3],
    }
    _PRETRAIN_PATHS = {
        "18": Path(__file__).resolve().parent / "resnet18-f37072fd.pth",
        "34": Path(__file__).resolve().parent / "resnet34-b627a593.pth",
    }

    def __init__(
        self,
        task_name,
        num_classes,
        arch="18",
        input_channels=3,
        pretrain=False,
        cifar_stem=False,
    ):
        arch = str(arch).lower().replace("resnet", "")
        if arch not in self._ARCH_LAYERS:
            raise ValueError("arch must be one of: '18', '34', 'resnet18', 'resnet34'")
        super().__init__(BasicBlock, self._ARCH_LAYERS[arch], num_classes=num_classes)
        if cifar_stem:
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        elif input_channels != 3:
            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.task_name = task_name
        self.arch = arch
        self.input_channels = input_channels
        self.cifar_stem = cifar_stem
        self.name = f"{task_name}_resnet{arch}"
        self.pretrain_info = None
        self.recons_map = {
            "recons_out": 10,
            "recons_z5": 9,
            "recons_z4": 8,
            "recons_z3": 7,
            "recons_z2": 6,
            "recons_z1": 5,
            # "recons_z0": 4,
        }
        if pretrain:
            self.pretrain_info = self.load_pretrained_weights()
            print(self.pretrain_info)

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

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Accept torchvision ImageNet checkpoints even when task-specific conv1/fc
        shapes differ. Matched keys are loaded normally; mismatched keys are
        skipped and reported in the returned IncompatibleKeys.
        """
        own_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if key in own_state and tuple(own_state[key].shape) == tuple(value.shape):
                filtered[key] = value
            elif key in own_state:
                skipped.append(key)
        incompatible = super().load_state_dict(filtered, strict=False)
        missing = list(dict.fromkeys(incompatible.missing_keys))
        unexpected = list(dict.fromkeys(incompatible.unexpected_keys))
        if strict:
            missing = [k for k in missing if k not in skipped]
            if missing or unexpected:
                raise RuntimeError(
                    "Error(s) in loading state_dict for "
                    f"{self.__class__.__name__}: missing_keys={missing}, "
                    f"unexpected_keys={unexpected}, skipped_shape_mismatch={skipped}"
                )
        return torch.nn.modules.module._IncompatibleKeys(
            list(dict.fromkeys(missing + skipped)),
            unexpected,
        )

    def load_pretrained_weights(self, map_location="cpu"):
        """Load the local official torchvision ResNet18/34 checkpoint."""
        ckpt_path = self._PRETRAIN_PATHS[self.arch]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        return self.load_state_dict(state_dict, strict=True)

    def _forward_impl_with_blocks(self, x, collect_block_inputs=False):
        z_stem = self.relu(self.bn1(self.conv1(x)))
        z0 = self.maxpool(z_stem)

        out = z0
        block_inputs = {}
        layer_outputs = {}
        for layer_idx, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4), start=1):
            for block_idx, block in enumerate(layer):
                if collect_block_inputs:
                    block_inputs[(layer_idx, block_idx)] = out
                out = block(out)
            layer_outputs[f"z{layer_idx}"] = out

        z5_map = self.avgpool(out)
        z5 = torch.flatten(z5_map, 1)
        out = self.fc(z5)
        res = {
            "z0": z0,
            "z1": layer_outputs["z1"],
            "z2": layer_outputs["z2"],
            "z3": layer_outputs["z3"],
            "z4": layer_outputs["z4"],
            "z5": z5,
            "out": out,
        }
        if collect_block_inputs:
            return res, block_inputs
        return res

    def forward(self, x):
        return self._forward_impl_with_blocks(x, collect_block_inputs=False)

    @staticmethod
    def _avgpool_reverse(pooled, ref_feature):
        b, c, h, w = ref_feature.shape
        return pooled.view(b, c, 1, 1).expand(b, c, h, w).contiguous()

    def _reverse_layer(self, layer, layer_idx, block_inputs, back_star, residual_iter):
        current = back_star
        for block_idx in reversed(range(len(layer))):
            block = layer[block_idx]
            front_ref = block_inputs[(layer_idx, block_idx)]
            current = composite_reverseCom(
                front_ref,
                current,
                block,
                iter=residual_iter,
                damping=0.9,
                max_step_norm=10.0,
                tol=1e-6,
            )
        return current

    def get_recons_fea(self, input_data, label_data, residual_iter=20, recons_key=None):
        res, block_inputs = self._forward_impl_with_blocks(input_data, collect_block_inputs=True)
        recons_out = optimal_embedding(res["out"], label_data)
        if recons_key == "recons_out": return recons_out

        recons_z5 = linear_reverseCom(res["z5"], recons_out, self.fc)
        if recons_key == "recons_z5": return recons_z5
        recons_z4 = self._avgpool_reverse(recons_z5, res["z4"])
        if recons_key == "recons_z4": return recons_z4

        recons_z3 = self._reverse_layer(self.layer4, 4, block_inputs, recons_z4, residual_iter)
        if recons_key == "recons_z3": return recons_z3
        recons_z2 = self._reverse_layer(self.layer3, 3, block_inputs, recons_z3, residual_iter)
        if recons_key == "recons_z2": return recons_z2
        recons_z1 = self._reverse_layer(self.layer2, 2, block_inputs, recons_z2, residual_iter)
        if recons_key == "recons_z1": return recons_z1
        recons_z0 = self._reverse_layer(self.layer1, 1, block_inputs, recons_z1, residual_iter)
        if recons_key == "recons_z0": return recons_z0

        return {
            "recons_out": recons_out,
            "recons_z5": recons_z5,
            "recons_z4": recons_z4,
            "recons_z3": recons_z3,
            "recons_z2": recons_z2,
            "recons_z1": recons_z1,
            "recons_z0": recons_z0,
        }


def make_resnet_model(
    task_name,
    num_classes,
    arch="18",
    input_channels=3,
    pretrain=False,
    cifar_stem=False,
):
    return ReconstructableResNet(
        task_name=task_name,
        num_classes=num_classes,
        arch=arch,
        input_channels=input_channels,
        pretrain=pretrain,
        cifar_stem=cifar_stem,
    )

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def check_recons_map(model):
        children = list(model.named_children())
        child_names = [name for name, _ in children]
        expected_freeze_from = {
            "recons_out": [],
            "recons_z5": ["fc"],
            "recons_z4": ["avgpool", "fc"],
            "recons_z3": ["layer4", "avgpool", "fc"],
            "recons_z2": ["layer3", "layer4", "avgpool", "fc"],
            "recons_z1": ["layer2", "layer3", "layer4", "avgpool", "fc"],
            "recons_z0": ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
        }

        print(f"\n[recons_map check] {model.name}")
        print("named_children:", child_names)
        for recons_key, expected_frozen in expected_freeze_from.items():
            freeze_from = model.get_fea_id(recons_key)
            frozen_names = child_names[freeze_from:]
            trainable_names = child_names[:freeze_from]
            ok = frozen_names == expected_frozen
            print(
                f"{recons_key:>10} | idx={freeze_from:>2} | "
                f"train={trainable_names} | freeze={frozen_names} | "
                f"{'OK' if ok else 'MISMATCH'}"
            )

    for arch in ("18", "34"):
        model = make_resnet_model(
            task_name="ci10",
            num_classes=10,
            arch=arch,
            input_channels=3,
            pretrain=False,
            cifar_stem=True,
        ).to(DEVICE)
        check_recons_map(model)

    # tests = [
    #     ("ci10", 10, "18", 3, (1, 3, 32, 32)),
    #     ("ci10", 10, "34", 3, (1, 3, 32, 32)),
    #     ("mn", 10, "18", 1, (1, 1, 28, 28)),
    #     ("mn", 10, "34", 1, (1, 1, 28, 28)),
    # ]
    # for task_name, num_classes, arch, input_channels, shape in tests:
    #     model = make_resnet_model(
    #         task_name=task_name,
    #         num_classes=num_classes,
    #         arch=arch,
    #         input_channels=input_channels,
    #         pretrain=True,
    #     ).to(DEVICE)
    #     model.eval()
    #     with torch.no_grad():
    #         res = model(torch.randn(*shape, device=DEVICE))
    #     print(model.name)
    #     print("pretrain_info:", model.pretrain_info)
    #     print({key: tuple(value.shape) for key, value in res.items()})
