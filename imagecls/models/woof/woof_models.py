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


class SimpleCNN_woof_Base(nn.Module, ABC):
    def __init__(self, activate=torch.tanh, version="base"):
        super().__init__()
        self.activate = activate
        self._version = version
        self.name = f"woof_simplecnn{version}_{self.activate.__name__}"
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
    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None, device='cuda:0'):
        pass


class SimpleCNN_woof_v1(SimpleCNN_woof_Base):
    def __init__(self, activate=torch.tanh):
        super().__init__(activate=activate, version="v1")
        self.conv1 = nn.Conv2d(3, 10, 25)
        self.conv2 = nn.Conv2d(10, 15, 17)
        self.conv3 = nn.Conv2d(15, 20, 11)
        self.conv4 = nn.Conv2d(20, 25, 5)
        self.conv5 = nn.Conv2d(25, 30, 3)
        self.fc1 = nn.Linear(30 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.recons_map = {
            'recons_out': 7,
            'recons_z6': 6,
            'recons_z5': 5,
            'recons_z4': 4,
            'recons_z3': 3,
            'recons_z2': 2,
            'recons_z1': 1,
        }

    def forward(self, x):
        """x size: batch_size, channel=3, height=224, width=224"""
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
        a4_p = self.pooling(a4)
        z5 = self.conv5(a4_p)
        a5 = self.activate(z5)
        a5_v = a5.view(a5.size(0), -1)
        z6 = self.fc1(a5_v)
        a6 = self.activate(z6)
        out = self.fc2(a6)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5, 'z6': z6, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None, device='cuda:0'):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a6 = self.activate(res['z6'])
        recons_a6 = linear_reverseCom(a6, recons_out, self.fc2)
        recons_z6 = route_rvs_act(self.activate, res['z6'], recons_a6)
        if recons_key == 'recons_z6': return recons_z6

        a5 = self.activate(res['z5'])
        a5_v = a5.view(input_data.size(0), -1)
        recons_a5 = linear_reverseCom(a5_v, recons_z6, self.fc1).view_as(res['z5'])
        recons_z5 = route_rvs_act(self.activate, res['z5'], recons_a5)
        if recons_key == 'recons_z5': return recons_z5

        a4_p = self.pooling(self.activate(res['z4']))
        recons_a4_p = convo_reverseCom(a4_p, recons_z5, self.conv5, method=conv_method)
        recons_a4 = pooling_reverseCom(recons_a4_p, self.activate(res['z4']), self.pooling)
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
        return {
            'recons_out': recons_out,
            'recons_z6': recons_z6,
            'recons_z5': recons_z5,
            'recons_z4': recons_z4,
            'recons_z3': recons_z3,
            'recons_z2': recons_z2,
            'recons_z1': recons_z1,
        }


class SimpleCNN_woof_v2(SimpleCNN_woof_Base):
    def __init__(self, activate=torch.tanh):
        super().__init__(activate=activate, version="v2")
        self.conv1 = nn.Conv2d(3, 16, 25)
        self.conv2 = nn.Conv2d(16, 16, 17)
        self.conv3 = nn.Conv2d(16, 16, 11)
        self.conv4 = nn.Conv2d(16, 16, 5)
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.recons_map = {
            'recons_out': 7,
            'recons_z6': 6,
            'recons_z5': 5,
            'recons_z4': 4,
            'recons_z3': 3,
            'recons_z2': 2,
            'recons_z1': 1,
        }

    def forward(self, x):
        """x size: batch_size, channel=3, height=224, width=224"""
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
        a4_p = self.pooling(a4)
        z5 = self.conv5(a4_p)
        a5 = self.activate(z5)
        a5_v = a5.view(a5.size(0), -1)
        z6 = self.fc1(a5_v)
        a6 = self.activate(z6)
        out = self.fc2(a6)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5, 'z6': z6, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None, device='cuda:0'):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a6 = self.activate(res['z6'])
        recons_a6 = linear_reverseCom(a6, recons_out, self.fc2)
        recons_z6 = route_rvs_act(self.activate, res['z6'], recons_a6)
        if recons_key == 'recons_z6': return recons_z6

        a5 = self.activate(res['z5'])
        a5_v = a5.view(input_data.size(0), -1)
        recons_a5 = linear_reverseCom(a5_v, recons_z6, self.fc1).view_as(res['z5'])
        recons_z5 = route_rvs_act(self.activate, res['z5'], recons_a5)
        if recons_key == 'recons_z5': return recons_z5

        a4_p = self.pooling(self.activate(res['z4']))
        recons_a4_p = convo_reverseCom(a4_p, recons_z5, self.conv5, method=conv_method)
        recons_a4 = pooling_reverseCom(recons_a4_p, self.activate(res['z4']), self.pooling)
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
        return {
            'recons_out': recons_out,
            'recons_z6': recons_z6,
            'recons_z5': recons_z5,
            'recons_z4': recons_z4,
            'recons_z3': recons_z3,
            'recons_z2': recons_z2,
            'recons_z1': recons_z1,
        }


def SimpleCNN_woof(activate=torch.tanh, version="v1"):
    version = str(version).lower()
    if version in ("1", "v1", "simplecnnv1"):
        return SimpleCNN_woof_v1(activate=activate)
    if version in ("2", "v2", "simplecnnv2"):
        return SimpleCNN_woof_v2(activate=activate)
    raise ValueError("version must be one of: 'v1', 'v2', 1, 2")



try:
    from ..resnet_recons import ReconstructableResNet
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from resnet_recons import ReconstructableResNet


class ResNet18_woof(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(task_name="woof", num_classes=10, arch="18", input_channels=3, pretrain=pretrain)


class ResNet34_woof(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(task_name="woof", num_classes=10, arch="34", input_channels=3, pretrain=pretrain)


def ResNet_woof(version="18", pretrain=False):
    version = str(version).lower().replace("resnet", "")
    if version in (18, "18", "v18"):
        return ResNet18_woof(pretrain=pretrain)
    if version in ("34", "v34"):
        return ResNet34_woof(pretrain=pretrain)
    raise ValueError("version must be one of: '18', '34', 'resnet18', 'resnet34'")


if __name__ == '__main__':
    for version in (1, 2):
        model = SimpleCNN_woof(torch.tanh, version=version)
        x = torch.randn(2, 3, 224, 224)
        res = model(x)
        print(model.name)
        print({k: tuple(v.shape) for k, v in res.items()})
        print(model.get_fea_name())
