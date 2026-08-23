from abc import ABC, abstractmethod
import torch
import torch.nn as nn

try:
    from ...rvs_cpt import convo_reverseCom, linear_reverseCom, optimal_embedding, pooling_reverseCom, route_rvs_act
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from rvs_cpt import convo_reverseCom, linear_reverseCom, optimal_embedding, pooling_reverseCom, route_rvs_act


class SimpleCNN_tin_Base(nn.Module, ABC):
    def __init__(self, activate=torch.tanh, version="base"):
        super().__init__()
        self.activate = activate
        self._version = version
        self.name = f"tin_simplecnn{version}_{self.activate.__name__}"
        self.recons_map = {}

    def get_fea_id(self, recons_key):
        return self.recons_map.get(recons_key, None)

    def get_fea_name(self):
        return list(self.recons_map.keys())

    def check_cond(self):
        for name, param in list(self.named_parameters()):
            if "weight" in name:
                print(name, param.size(), '-' * 20)
                print(torch.linalg.matrix_rank(param.data))
                print(torch.linalg.cond(param.data))
                if "conv" in name:
                    weight_mat = param.data.detach().reshape(param.data.shape[0], -1).to(torch.float64)
                    print('weight matrix rank:', torch.linalg.matrix_rank(weight_mat))
                    print('weight matrix cond:', torch.linalg.cond(weight_mat))

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        pass


class SimpleCNN_tin_v1(SimpleCNN_tin_Base):
    def __init__(self, activate=torch.relu):
        super().__init__(activate=activate, version="v1")
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 25, 3)
        self.conv3 = nn.Conv2d(25, 35, 3)
        self.conv4 = nn.Conv2d(35, 45, 3)
        self.fc1 = nn.Linear(45 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 200)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.recons_map = { 'recons_out': 6, 'recons_z5': 5,'recons_z4': 4, 'recons_z3': 3,
            'recons_z2': 2, 'recons_z1': 1,}

    def forward(self, x):
        """x size: batch_size, channel=3, height=64, width=64"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        z3 = self.conv3(a2_p)
        a3 = self.activate(z3)
        a3_p = self.pooling(a3)
        z4 = self.conv4(a3_p)
        a4 = self.activate(z4)
        a4_v = a4.view(a4.size(0), -1)
        z5 = self.fc1(a4_v)
        a5 = self.activate(z5)
        out = self.fc2(a5)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a5 = self.activate(res['z5'])
        recons_a5 = linear_reverseCom(a5, recons_out, self.fc2)
        recons_z5 = route_rvs_act(self.activate, res['z5'], recons_a5)
        if recons_key == 'recons_z5': return recons_z5

        a4 = self.activate(res['z4'])
        a4_v = a4.view(input_data.size(0), -1)
        recons_a4 = linear_reverseCom(a4_v, recons_z5, self.fc1).view_as(res['z4'])
        recons_z4 = route_rvs_act(self.activate, res['z4'], recons_a4)
        if recons_key == 'recons_z4': return recons_z4

        a3_p = self.pooling(self.activate(res['z3']))
        recons_a3_p = convo_reverseCom(a3_p, recons_z4, self.conv4, method=conv_method)
        recons_a3 = pooling_reverseCom(recons_a3_p, self.activate(res['z3']), self.pooling)
        recons_z3 = route_rvs_act(self.activate, res['z3'], recons_a3)
        if recons_key == 'recons_z3': return recons_z3

        a2_p = self.pooling(self.activate(res['z2']))
        recons_a2_p = convo_reverseCom(a2_p, recons_z3, self.conv3, method=conv_method)
        recons_a2 = pooling_reverseCom(recons_a2_p, self.activate(res['z2']), self.pooling)
        recons_z2 = route_rvs_act(self.activate, res['z2'], recons_a2)
        if recons_key == 'recons_z2': return recons_z2

        a1_p = self.pooling(self.activate(res['z1']))
        recons_a1_p = convo_reverseCom(a1_p, recons_z2, self.conv2, method=conv_method)
        recons_a1 = pooling_reverseCom(recons_a1_p, self.activate(res['z1']), self.pooling)
        recons_z1 = route_rvs_act(self.activate, res['z1'], recons_a1)
        if recons_key == 'recons_z1': return recons_z1
        return {'recons_out': recons_out, 'recons_z5': recons_z5, 'recons_z4': recons_z4,
            'recons_z3': recons_z3, 'recons_z2': recons_z2, 'recons_z1': recons_z1,}


class SimpleCNN_tin_v2(SimpleCNN_tin_Base):
    def __init__(self, activate=torch.relu):
        super().__init__(activate=activate, version="v2")
        self.conv1 = nn.Conv2d(3, 45, 5)
        self.conv2 = nn.Conv2d(45, 45, 3)
        self.conv3 = nn.Conv2d(45, 45, 3)
        self.conv4 = nn.Conv2d(45, 45, 3)
        self.fc1 = nn.Linear(45 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 200)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.recons_map = {'recons_out': 6, 'recons_z5': 5, 'recons_z4': 4, 'recons_z3': 3,
                           'recons_z2': 2, 'recons_z1': 1}

    def forward(self, x):
        """x size: batch_size, channel=3, height=64, width=64"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        z3 = self.conv3(a2_p)
        a3 = self.activate(z3)
        a3_p = self.pooling(a3)
        z4 = self.conv4(a3_p)
        a4 = self.activate(z4)
        a4_v = a4.view(a4.size(0), -1)
        z5 = self.fc1(a4_v)
        a5 = self.activate(z5)
        out = self.fc2(a5)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a5 = self.activate(res['z5'])
        recons_a5 = linear_reverseCom(a5, recons_out, self.fc2)
        recons_z5 = route_rvs_act(self.activate, res['z5'], recons_a5)
        if recons_key == 'recons_z5': return recons_z5

        a4 = self.activate(res['z4'])
        a4_v = a4.view(input_data.size(0), -1)
        recons_a4 = linear_reverseCom(a4_v, recons_z5, self.fc1).view_as(res['z4'])
        recons_z4 = route_rvs_act(self.activate, res['z4'], recons_a4)
        if recons_key == 'recons_z4': return recons_z4

        a3_p = self.pooling(self.activate(res['z3']))
        recons_a3_p = convo_reverseCom(a3_p, recons_z4, self.conv4, method=conv_method)
        recons_a3 = pooling_reverseCom(recons_a3_p, self.activate(res['z3']), self.pooling)
        recons_z3 = route_rvs_act(self.activate, res['z3'], recons_a3)
        if recons_key == 'recons_z3': return recons_z3

        a2_p = self.pooling(self.activate(res['z2']))
        recons_a2_p = convo_reverseCom(a2_p, recons_z3, self.conv3, method=conv_method)
        recons_a2 = pooling_reverseCom(recons_a2_p, self.activate(res['z2']), self.pooling)
        recons_z2 = route_rvs_act(self.activate, res['z2'], recons_a2)
        if recons_key == 'recons_z2': return recons_z2

        a1_p = self.pooling(self.activate(res['z1']))
        recons_a1_p = convo_reverseCom(a1_p, recons_z2, self.conv2, method=conv_method)
        recons_a1 = pooling_reverseCom(recons_a1_p, self.activate(res['z1']), self.pooling)
        recons_z1 = route_rvs_act(self.activate, res['z1'], recons_a1)
        if recons_key == 'recons_z1': return recons_z1
        return {'recons_out': recons_out, 'recons_z5': recons_z5, 'recons_z4': recons_z4,
                'recons_z3': recons_z3, 'recons_z2': recons_z2, 'recons_z1': recons_z1}


def SimpleCNN_tin(activate=torch.tanh, version="v1"):
    version = str(version).lower()
    if version in ("1", "v1", "simplecnnv1"):
        return SimpleCNN_tin_v1(activate=activate)
    if version in ("2", "v2", "simplecnnv2"):
        return SimpleCNN_tin_v2(activate=activate)
    raise ValueError("version must be one of: 'v1', 'v2', 1, 2")



try:
    from ..resnet_recons import ReconstructableResNet
    from ..simplevit_recons import ReconstructableSimpleViT, ReconstructableSimpleViTv2, print_simplevit_smoke
    from ..pretrain_vit_recons import SmallDatasetVisionTransformer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from resnet_recons import ReconstructableResNet
    from simplevit_recons import ReconstructableSimpleViT, ReconstructableSimpleViTv2, print_simplevit_smoke
    from pretrain_vit_recons import SmallDatasetVisionTransformer


class ResNet18_tin(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(task_name="tin", num_classes=200, arch="18", input_channels=3, pretrain=pretrain)


class ResNet34_tin(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(task_name="tin", num_classes=200, arch="34", input_channels=3, pretrain=pretrain)


def ResNet_tin(version="18", pretrain=False):
    version = str(version).lower().replace("resnet", "")
    if version in (18, "18", "v18"):
        return ResNet18_tin(pretrain=pretrain)
    if version in (34, "34", "v34"):
        return ResNet34_tin(pretrain=pretrain)
    raise ValueError("version must be one of: '18', '34', 'resnet18', 'resnet34'")


class SimpleViT_tin(ReconstructableSimpleViT):
    def __init__(
        self,
        patch_size=8,
        embed_dim=192,
        depth=9,
        num_heads=12,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__(
            task_name="tin",
            image_size=64,
            patch_size=patch_size,
            num_classes=200,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            input_channels=3,
            dropout=dropout,
        )


class SimpleViTv2_tin(ReconstructableSimpleViTv2):
    def __init__(
        self,
        patch_size=8,
        embed_dim=192,
        depth=9,
        num_heads=12,
        mlp_ratio=2.0,
        dropout=0.0,
        norm_eps=1e-6,
    ):
        super().__init__(
            task_name="tin",
            image_size=64,
            patch_size=patch_size,
            num_classes=200,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            input_channels=3,
            dropout=dropout,
            norm_eps=norm_eps,
        )


class ViTTinyImageNetPatch8(SmallDatasetVisionTransformer):
    """
    Architecture corresponding to:
        vit_timnet_patch8_input64.pth

    Input:
        [B, 3, 64, 64]

    Tokens:
        8 x 8 patches = 64 patch tokens
        + 1 CLS token
        = 65 tokens

    Output:
        [B, 200]
    """

    def __init__(
        self,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__(
            img_size=64,
            patch_size=8,
            num_classes=200,
            in_chans=3,
            embed_dim=192,
            depth=9,
            num_heads=12,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=1e-6,
        )




if __name__ == '__main__':
    for version in (1, 2):
        model = SimpleCNN_tin(torch.tanh, version=version)
        x = torch.randn(2, 3, 64, 64)
        res = model(x)
        print(model.name)
        print({k: tuple(v.shape) for k, v in res.items()})
        print(model.get_fea_name())

    vit = SimpleViT_tin()
    print_simplevit_smoke(vit, (2, 3, 64, 64))
