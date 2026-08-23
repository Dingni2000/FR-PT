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


class SimpleCNN_mn_Base(nn.Module, ABC):
    def __init__(self, activate=torch.tanh, version="base"):
        super().__init__()
        self.activate = activate
        self._version = version
        self.name = f"mn_simplecnn{version}_{self.activate.__name__}"
        self.recons_map = {}

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

    def get_fea_id(self, recons_key):
        return self.recons_map.get(recons_key, None)
    
    def get_fea_name(self):
        return list(self.recons_map.keys())
    

class SimpleCNN_mn_v1(SimpleCNN_mn_Base):
    def __init__(self, activate=torch.tanh):
        super().__init__(activate=activate, version="v1")
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.fc1 = nn.Linear(4 * 4 * 4, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.recons_map = {'recons_out': 3, 'recons_z2': 2, 'recons_z1': 1}

    def forward(self, x):
        """x size: batch_size=250, channel=1, height=28, width=28"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        a2_v = a2_p.view(a2_p.size(0), -1)
        out = self.fc1(a2_v)
        return {'z1': z1, 'z2': z2, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a2 = self.activate(res['z2'])
        a2_v = self.pooling(a2).view(input_data.size(0), -1)
        recons_a2_p = linear_reverseCom(a2_v, recons_out, self.fc1).view(input_data.size(0), 4, 4, 4)
        recons_a2 = pooling_reverseCom(recons_a2_p, self.activate(res['z2']), self.pooling)
        recons_z2 = route_rvs_act(self.activate, res['z2'], recons_a2)
        if recons_key == 'recons_z2': return recons_z2

        a1_p = self.pooling(self.activate(res['z1']))
        recons_a1_p = convo_reverseCom(a1_p, recons_z2, self.conv2, method=conv_method)
        recons_a1 = pooling_reverseCom(recons_a1_p, self.activate(res['z1']), self.pooling)
        recons_z1 = route_rvs_act(self.activate, res['z1'], recons_a1)
        if recons_key == 'recons_z1': return recons_z1
        return {'recons_out': recons_out, 'recons_z2': recons_z2, 'recons_z1': recons_z1}


class SimpleCNN_mn_v2(SimpleCNN_mn_Base):
    def __init__(self, activate=torch.tanh):
        super().__init__(activate=activate, version="v2")
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 3, 5)
        self.fc1 = nn.Linear(3 * 4 * 4, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.recons_map = {'recons_out':3, 'recons_z2':2, 'recons_z1':1}

    def forward(self, x):
        """x size: batch_size=250, channel=1, height=28, width=28"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        a2_v = a2_p.view(a2_p.size(0), -1)
        out = self.fc1(a2_v)
        return {'z1': z1, 'z2': z2, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a2 = self.activate(res['z2'])
        a2_v = self.pooling(a2).view(input_data.size(0), -1)
        recons_a2_p = linear_reverseCom(a2_v, recons_out, self.fc1).view(input_data.size(0), 3, 4, 4)
        recons_a2 = pooling_reverseCom(recons_a2_p, self.activate(res['z2']), self.pooling)
        recons_z2 = route_rvs_act(self.activate, res['z2'], recons_a2)
        if recons_key == 'recons_z2': return recons_z2

        a1_p = self.pooling(self.activate(res['z1']))
        recons_a1_p = convo_reverseCom(a1_p, recons_z2, self.conv2, method=conv_method)
        recons_a1 = pooling_reverseCom(recons_a1_p, self.activate(res['z1']), self.pooling)
        recons_z1 = route_rvs_act(self.activate, res['z1'], recons_a1)
        if recons_key == 'recons_z1': return recons_z1
        return {'recons_out': recons_out, 'recons_z2': recons_z2, 'recons_z1': recons_z1}




def SimpleCNN_mn(activate=torch.tanh, version="v2"):
    version = str(version).lower()
    if version in (1, "1", "v1", "simplecnnv1"):
        return SimpleCNN_mn_v1(activate=activate)
    if version in (2, "2", "v2", "simplecnnv2"):
        return SimpleCNN_mn_v2(activate=activate)
    raise ValueError("version must be one of: 'v1', 'v2', 1, 2")




try:
    from ..resnet_recons import ReconstructableResNet
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from resnet_recons import ReconstructableResNet


class ResNet18_mn(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(task_name="mn", num_classes=10, arch="18", input_channels=1, pretrain=pretrain)


class ResNet34_mn(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(task_name="mn", num_classes=10, arch="34", input_channels=1, pretrain=pretrain)


def ResNet_mn(version="18", pretrain=False):
    version = str(version).lower().replace("resnet", "")
    if version in (18, "18", "v18"):
        return ResNet18_mn(pretrain=pretrain)
    if version in (34, "34", "v34"):
        return ResNet34_mn(pretrain=pretrain)
    raise ValueError("version must be one of: '18', '34', 'resnet18', 'resnet34'")


if __name__ == '__main__':
    model = SimpleCNN_mn(torch.relu, version=2)
    for idx, (n, layer) in enumerate(list(model.named_children())):
        print(idx, n, layer)
    print(list(model.named_children())[1])
