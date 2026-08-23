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


class SimpleCNN_ci10_Base(nn.Module, ABC):
    def __init__(self, activate=torch.tanh, version="base"):
        super().__init__()
        self.activate = activate
        self._version = version
        self.name = f"ci10_simplecnn{version}_{self.activate.__name__}"
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

    

class SimpleCNN_ci10_v1(SimpleCNN_ci10_Base):
    def __init__(self, activate=torch.relu):
        super().__init__(activate=activate, version="v1")
        self.conv1 = nn.Conv2d(3, 5, 5) # 1: 灰度图片的通道，10：输出通道，5：kernel
        self.conv2 = nn.Conv2d(5, 10, 3) # 10: 输入通道，20: 输出通道，3: kernel
        self.conv3 = nn.Conv2d(10, 15, 3)
        self.fc1 = nn.Linear(15*4*4, 128) # 5*6*6: 输入维度， 500: 输出维度
        self.fc2 = nn.Linear(128, 10)
        self.activate = activate
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)  # 使用 Xavier 初始化权重
        nn.init.xavier_uniform_(self.fc2.weight)  # 使用 Xavier 初始化权重
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.name = 'ci10_simplecnnv1_'+self.activate.__name__
        self.recons_map = {'recons_out':5, 'recons_z4':4, 'recons_z3':3, 'recons_z2':2, 'recons_z1':1}

        
    def forward(self, x):
        """x size: batch_size=250, channel=1, height=28, width=28 """
        z1 = self.conv1(x)
        a1 = self.activate(z1) # 输入: （batch_size，1，28，28）, 输出: （batch_size，5，24，24） （28-5+1）
        a1_p = self.pooling(a1) # 输入:(batch_size,5, 24, 24), 输出: (batch_size, 5, 12, 12)（24/2=12）
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2) # 输入: (batch_size,5, 12,12), 输出: (batch_size，7，10，10) （12-3+1=10）
        a2_p = self.pooling(a2) # 输入:(batch_size, 7, 10,10), 输出: （batch_size,7,5,5） （10/2=5）
        z3 = self.conv3(a2_p)
        a3 = self.activate(z3)
        a3_v = a3.view(a3.size(0), -1) # 拉平以供全连接层使用，-1 自动计算维度， size就是（batch_size，7*5*5=175
        z4 = self.fc1(a3_v)
        a4 = self.activate(z4) # 输入: （batch_size，175）, 输出: （batch_size, 10）
        out = self.fc2(a4) # 输入: （batch_size，10）, 输出: （batch_size, 10）
        return {'z1':z1, 'z2':z2, "z3":z3, "z4":z4, "out":out} # 输出三个值，分别是卷积层的输出，全连接层的输出，以及最终的分类结果
    
    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a4 = self.activate(res['z4'])
        recons_a4 = linear_reverseCom(a4, recons_out, self.fc2)
        recons_z4 = route_rvs_act(self.activate, res['z4'], recons_a4)
        if recons_key == 'recons_z4': return recons_z4

        a3 = self.activate(res['z3'])
        a3_v = a3.view(input_data.size(0), -1)
        recons_a3 = linear_reverseCom(a3_v, recons_z4, self.fc1).view_as(res['z3'])
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
        return {'recons_out':recons_out, 'recons_z4':recons_z4, 'recons_z3':recons_z3, 'recons_z2':recons_z2, 'recons_z1':recons_z1}


class SimpleCNN_ci10_v2(SimpleCNN_ci10_Base):
    def __init__(self, activate=torch.relu):
        super().__init__(activate=activate, version="v2")
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.conv3 = nn.Conv2d(10, 10, 3)
        self.fc1 = nn.Linear(10 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.recons_map = {'recons_out': 5, 'recons_z4': 4, 'recons_z3': 3, 'recons_z2': 2, 'recons_z1': 1}

    def forward(self, x):
        """x size: batch_size, channel=3, height=32, width=32"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        z3 = self.conv3(a2_p)
        a3 = self.activate(z3)
        a3_v = a3.view(a3.size(0), -1)
        z4 = self.fc1(a3_v)
        a4 = self.activate(z4)
        out = self.fc2(a4)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a4 = self.activate(res['z4'])
        recons_a4 = linear_reverseCom(a4, recons_out, self.fc2)
        recons_z4 = route_rvs_act(self.activate, res['z4'], recons_a4)
        if recons_key == 'recons_z4': return recons_z4

        a3 = self.activate(res['z3'])
        a3_v = a3.view(input_data.size(0), -1)
        recons_a3 = linear_reverseCom(a3_v, recons_z4, self.fc1).view_as(res['z3'])
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
        return {'recons_out': recons_out, 'recons_z4': recons_z4, 'recons_z3': recons_z3,
                'recons_z2': recons_z2, 'recons_z1': recons_z1}


class SimpleCNN_ci10_v3(SimpleCNN_ci10_Base):
    def __init__(self, activate=torch.relu):
        super().__init__(activate=activate, version="v3")
        self.conv1 = nn.Conv2d(3, 15, 5)
        self.conv2 = nn.Conv2d(15, 10, 3)
        self.conv3 = nn.Conv2d(10, 5, 3)
        self.fc1 = nn.Linear(5 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.recons_map = {'recons_out': 5, 'recons_z4': 4, 'recons_z3': 3, 'recons_z2': 2, 'recons_z1': 1}

    def forward(self, x):
        """x size: batch_size, channel=3, height=32, width=32"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        z3 = self.conv3(a2_p)
        a3 = self.activate(z3)
        a3_v = a3.view(a3.size(0), -1)
        z4 = self.fc1(a3_v)
        a4 = self.activate(z4)
        out = self.fc2(a4)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a4 = self.activate(res['z4'])
        recons_a4 = linear_reverseCom(a4, recons_out, self.fc2)
        recons_z4 = route_rvs_act(self.activate, res['z4'], recons_a4)
        if recons_key == 'recons_z4': return recons_z4

        a3 = self.activate(res['z3'])
        a3_v = a3.view(input_data.size(0), -1)
        recons_a3 = linear_reverseCom(a3_v, recons_z4, self.fc1).view_as(res['z3'])
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
        return {'recons_out': recons_out, 'recons_z4': recons_z4, 'recons_z3': recons_z3,
                'recons_z2': recons_z2, 'recons_z1': recons_z1}

 

class SimpleCNN_ci10_v4(SimpleCNN_ci10_Base):
    def __init__(self, activate=torch.relu):
        super().__init__(activate=activate, version="v4")
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.conv3 = nn.Conv2d(10, 15, 3)
        self.fc1 = nn.Linear(15 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.recons_map = {'recons_out': 5, 'recons_z4': 4, 'recons_z3': 3, 'recons_z2': 2, 'recons_z1': 1}

    def forward(self, x):
        """x size: batch_size, channel=3, height=32, width=32"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        z3 = self.conv3(a2_p)
        a3 = self.activate(z3)
        a3_v = a3.view(a3.size(0), -1)
        z4 = self.fc1(a3_v)
        a4 = self.activate(z4)
        out = self.fc2(a4)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a4 = self.activate(res['z4'])
        recons_a4 = linear_reverseCom(a4, recons_out, self.fc2)
        recons_z4 = route_rvs_act(self.activate, res['z4'], recons_a4)
        if recons_key == 'recons_z4': return recons_z4

        a3 = self.activate(res['z3'])
        a3_v = a3.view(input_data.size(0), -1)
        recons_a3 = linear_reverseCom(a3_v, recons_z4, self.fc1).view_as(res['z3'])
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
        return {'recons_out': recons_out, 'recons_z4': recons_z4, 'recons_z3': recons_z3,
                'recons_z2': recons_z2, 'recons_z1': recons_z1}


class SimpleCNN_ci10_v5(SimpleCNN_ci10_Base):
    def __init__(self, activate=torch.relu):
        super().__init__(activate=activate, version="v5")
        self.conv1 = nn.Conv2d(3, 15, 5)
        self.conv2 = nn.Conv2d(15, 10, 3)
        self.conv3 = nn.Conv2d(10, 5, 3)
        self.fc1 = nn.Linear(5 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.recons_map = {'recons_out': 5, 'recons_z4': 4, 'recons_z3': 3, 'recons_z2': 2, 'recons_z1': 1}

    def forward(self, x):
        """x size: batch_size, channel=3, height=32, width=32"""
        z1 = self.conv1(x)
        a1 = self.activate(z1)
        a1_p = self.pooling(a1)
        z2 = self.conv2(a1_p)
        a2 = self.activate(z2)
        a2_p = self.pooling(a2)
        z3 = self.conv3(a2_p)
        a3 = self.activate(z3)
        a3_v = a3.view(a3.size(0), -1)
        z4 = self.fc1(a3_v)
        a4 = self.activate(z4)
        out = self.fc2(a4)
        return {'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'out': out}

    def get_recons_fea(self, input_data, label_data, conv_method='fft_pad', recons_key=None):
        res = self(input_data)
        recons_out = optimal_embedding(res['out'], label_data)
        if recons_key == 'recons_out': return recons_out

        a4 = self.activate(res['z4'])
        recons_a4 = linear_reverseCom(a4, recons_out, self.fc2)
        recons_z4 = route_rvs_act(self.activate, res['z4'], recons_a4)
        if recons_key == 'recons_z4': return recons_z4

        a3 = self.activate(res['z3'])
        a3_v = a3.view(input_data.size(0), -1)
        recons_a3 = linear_reverseCom(a3_v, recons_z4, self.fc1).view_as(res['z3'])
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
        return {'recons_out': recons_out, 'recons_z4': recons_z4, 'recons_z3': recons_z3,
                'recons_z2': recons_z2, 'recons_z1': recons_z1}

 


def SimpleCNN_ci10(activate=torch.tanh, version="v1"):
    version = str(version).lower()
    if version in (1, "1", "v1", "simplecnnv1"):
        return SimpleCNN_ci10_v1(activate=activate)
    if version in (2, "2", "v2", "simplecnnv2"):
        return SimpleCNN_ci10_v2(activate=activate)
    if version in (3, '3', 'v3', 'simplecnnv3'):
        return SimpleCNN_ci10_v3(activate=activate)
    if version in (4, "4", "v4", "simplecnnv2"):
        return SimpleCNN_ci10_v4(activate=activate)
    if version in (5, '5', 'v5', 'simplecnnv5'):
        return SimpleCNN_ci10_v5(activate=activate)
    raise ValueError("version must be one of: 'v1', 'v2', 1, 2,3,'v3'")



try:
    from ..resnet_recons import ReconstructableResNet
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from resnet_recons import ReconstructableResNet


class ResNet18_ci10(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(
            task_name="ci10",
            num_classes=10,
            arch="18",
            input_channels=3,
            pretrain=pretrain,
            cifar_stem=True,
        )


class ResNet34_ci10(ReconstructableResNet):
    def __init__(self, pretrain=False):
        super().__init__(
            task_name="ci10",
            num_classes=10,
            arch="34",
            input_channels=3,
            pretrain=pretrain,
            cifar_stem=True,
        )


def ResNet_ci10(version="18", pretrain=False):
    version = str(version).lower().replace("resnet", "")
    if version in (18, "18", "v18"):
        return ResNet18_ci10(pretrain=pretrain)
    if version in (34, "34", "v34"):
        return ResNet34_ci10(pretrain=pretrain)
    raise ValueError("version must be one of: '18', '34', 'resnet18', 'resnet34'")


if __name__ == '__main__':
    for version in (1, 2):
        model = SimpleCNN_ci10(torch.relu, version=version)
        x = torch.randn(2, 3, 32, 32)
        res = model(x)
        print(model.name)
        print({k: tuple(v.shape) for k, v in res.items()})
