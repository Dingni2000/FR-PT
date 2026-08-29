import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

COND_TIKHONOV = 1e4
COND_TRUNCATE = 1e8
TIKHONOV_REL = 1e-6
SVD_RTOL = 1e-8
_STABILITY_LOGGED = set()



def optimal_embedding(out, label):
    """ 输出向量out:bs*cls,  标签label:bs
        输出recons_out:bs*cls,它的第label个分量最大,离out尽量近"""
    bs = len(label)
    diff = out - out[torch.arange(bs), label].unsqueeze(-1)  # (bs, cls)
    sorted_indices = torch.argsort(diff, dim=-1, descending=True)  # (bs, cls)
    diff_sorted = torch.gather(diff, dim=-1, index=sorted_indices)  # (bs, cls)
    mask0 = diff_sorted>torch.tensor(0, device=out.device)  
    mul_left_sorted_ori = diff_sorted * (torch.arange(1, out.shape[1]+1, device=out.device).unsqueeze(0))
    mul_left_sorted = torch.zeros_like(mul_left_sorted_ori)
    mul_left_sorted[:,:-1] = mul_left_sorted_ori[:,1:]
    mul_right_sorted = diff_sorted * (torch.arange(2, out.shape[1]+2, device=out.device).unsqueeze(0))
    cumsum = torch.cumsum(diff_sorted, dim=-1)  # (bs, cls)
    mask1 = torch.zeros_like(mask0, dtype=torch.bool, device=out.device)
    mask1[mask0] = (mul_left_sorted[mask0] <= cumsum[mask0]-5e-6) & (cumsum[mask0] <= mul_right_sorted[mask0]-5e-6)
    # assert torch.all(torch.sum(mask1, dim=-1)<=1), mask1
    has_true = torch.any(mask1, dim=-1).to(torch.bool)  # (bs,)
    count = - torch.ones(bs, device=out.device, dtype=torch.long)
    count[has_true] = torch.nonzero(mask1, as_tuple=True)[1]  # (bs,)
    mask2 = torch.arange(out.shape[1], device=out.device).unsqueeze(0).expand(bs, -1) <= count.unsqueeze(-1)  # (bs, cls)
    mask = mask2.scatter(dim=-1, index=sorted_indices, src=mask2)  # (bs, cls)

    A = torch.eye(out.shape[1], device=out.device).unsqueeze(0).repeat(bs, 1, 1)
    idens = torch.ones(out.shape[0], out.shape[1], out.shape[1], device=out.device)
    A -= (idens/(count+2).unsqueeze(-1).unsqueeze(-1).expand(-1, out.shape[1], out.shape[1]))
    A[~mask] = 0
    A.transpose(-1, -2)[~mask] = 0
    diff[~mask] = 0
    muhalf = torch.matmul(A, diff.unsqueeze(-1)).squeeze(-1)  # (bs, cls)
    # assert torch.all(muhalf>=-1e-6), muhalf
    recons_out = out - muhalf
    recons_out[torch.arange(bs), label] += torch.sum(muhalf, dim=-1) 
    return recons_out


def linear_reverseCom(front_fea, back_fea, _layer_params):
    """back_fea: (batchsize, back_fea_dim)"""
    if front_fea.shape[-1] >= back_fea.shape[-1]:
        return linear_reverseCom_shrink(front_fea, back_fea, _layer_params)
    else:
        return linear_reverseCom_expand(back_fea, _layer_params)


def linear_reverseCom_shrink(front_fea, back_fea, _layer_params):
    """back_fea: (batchsize, back_fea_dim)"""
    DEVICE = front_fea.device
    front_fea_dim = front_fea.shape[-1]
    A = torch.zeros(front_fea_dim + back_fea.shape[-1], front_fea_dim + back_fea.shape[-1]).to(DEVICE) 
    A[torch.arange(front_fea_dim), torch.arange(front_fea_dim)] = 1.0
    A[front_fea_dim:, :front_fea_dim] = _layer_params.weight.data.detach() #  NOTE torch.linalg.solve needs weights full-rank
    A[:front_fea_dim, front_fea_dim:] = A[front_fea_dim:, :front_fea_dim].transpose(-1, -2)  
    B = torch.cat((front_fea, back_fea - _layer_params.bias.data.detach().unsqueeze(0).to(DEVICE)), dim=1).unsqueeze(-1)  # type: ignore
    # assert torch.all(torch.linalg.eigvals(A).real > 0) 
    # front_star_ = torch.linalg.solve(A.unsqueeze(0).expand(back_fea.shape[0], -1, -1), B).squeeze(-1)[:,:front_fea_dim]
    _, L, U = torch.linalg.lu(A, pivot=False)  # NOTE A只能用LU分解，先拆一次更快一点
    Y = torch.linalg.solve_triangular(L.unsqueeze(0).expand(B.shape[0], -1, -1), B, upper=False)    
    X = torch.linalg.solve_triangular(U.unsqueeze(0).expand(B.shape[0], -1, -1), Y, upper=True)
    return X.squeeze(-1)[:, :front_fea_dim]  # (batchsize, front_fea_dim)


def linear_reverseCom_expand(back_fea, _layer_params):
    """n_l < n_{l+1}"""
    res = torch.linalg.lstsq(_layer_params.weight.data.detach().unsqueeze(0).expand(back_fea.shape[0], -1, -1), 
                (back_fea - _layer_params.bias.data.detach()).unsqueeze(-1), driver='gels').solution.squeeze(-1)
    return res



def residual_tanh_reverseCom(front_fea, back_fea, _steps, _residual_layer, _bias=True):
    DEVICE = front_fea.device
    identity_matrix = torch.eye(front_fea.shape[1], device=DEVICE)   # (fea_dim, fea_dim)
    activate_fea = torch.tanh(_residual_layer.fc1(front_fea))
    each_dlt_back_fea = (back_fea - activate_fea - front_fea)/_steps   # (bs, fea_dim)
    for i in range(_steps):
        df = (1-torch.pow(activate_fea, 2)).unsqueeze(1) * _residual_layer.fc1.weight.data.detach()  # (bs, fea_dim, fea_dim)
        dlt_front_fea = torch.linalg.solve(_residual_layer.fc2.weight.data.detach().unsqueeze(0)@df+identity_matrix, each_dlt_back_fea.unsqueeze(-1))  # (bs, fea_dim, 1)
        front_fea = front_fea + dlt_front_fea.squeeze(-1)
        activate_fea = torch.tanh(_residual_layer.fc1(front_fea))
    return front_fea


def residual_relu_reverseCom(front_fea, back_fea, _steps, _residual_layer, _bias=True):
    DEVICE = front_fea.device
    identity_matrix = torch.eye(front_fea.shape[1], device=DEVICE)   # (fea_dim, fea_dim)
    activate_fea = torch.relu(_residual_layer.fc1(front_fea))  # (bs, fea_dim)
    each_dlt_back_fea = (back_fea - activate_fea - front_fea)/_steps  # (bs, fea_dim)
    for i in range(_steps):
        mask = (activate_fea > 0).int()
        df = _residual_layer.fc1.weight.data.detach().unsqueeze(0)*mask.unsqueeze(-1)  # (bs, fea_dim, fea_dim)
        dlt_front_fea = torch.linalg.solve(_residual_layer.fc2.weight.data.detach().unsqueeze(0)@df+identity_matrix, each_dlt_back_fea.unsqueeze(-1))
        front_fea = front_fea + dlt_front_fea.squeeze(-1)
        activate_fea = torch.relu(_residual_layer.fc1(front_fea))
    return front_fea


def bn_reverseCom(y, bn_layer):
    weight = bn_layer.weight.detach().view(1, -1, 1, 1)
    bias = bn_layer.bias.detach().view(1, -1, 1, 1)
    running_mean = bn_layer.running_mean.detach().view(1, -1, 1, 1)
    running_var = bn_layer.running_var.detach().view(1, -1, 1, 1)
    eps = bn_layer.eps
    
    x = (y - bias) / weight * torch.sqrt(running_var + eps) + running_mean
    return x


def pooling_reverseCom(y, x_shape, pooling_layer):
    """
    使用插值快速还原 MaxPool 的输出
    
    Args:
        y: MaxPool 的输出，shape (batch, channels, h', w')
        x_shape: 原始输入 x 的 shape
        pool_layer: nn.MaxPool2d 层
    
    Returns:
        x_reconstructed: 与原始输入 x 相同大小的张量
    """
    assert pooling_layer.kernel_size == pooling_layer.stride, "Only support stride == kernel_size"
    if x_shape[-1] % pooling_layer.kernel_size != 0 or x_shape[-2] % pooling_layer.kernel_size != 0:
        print("[WARNING] The input size of Pool is not a multiple of the kernel size, which may introduce errors.")
    
    x_reconstructed = F.interpolate(
        y,
        size=x_shape[-2:],
        mode='nearest',  # 最近邻插值
        align_corners=None
    )
    return x_reconstructed


if __name__ == "__main__":
    pass
