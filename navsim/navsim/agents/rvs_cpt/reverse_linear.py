import torch
try: 
    from utils import check_reliability, mdp_tikhonov_solve, remove_bias
except ImportError:
    from .utils import check_reliability, mdp_tikhonov_solve, remove_bias

def linear_reverseCom_shrink(front_fea, back_fea, _layer_params):
    """back_fea: (batchsize, back_fea_dim)"""
    W = _layer_params.weight.detach()
    y = remove_bias(back_fea, _layer_params)
    residual = y - front_fea @ W.mT
    delta = W.mT @ torch.linalg.solve(W @ W.mT, residual.mT)
    x = front_fea + delta.mT
    reliable = check_reliability(x, W, y, "solve", x_ref=front_fea)
    if reliable.all(): return x
    failed = ~reliable
    # x = x.clone()
    x[failed] = mdp_tikhonov_solve(W, y[failed], front_fea[failed])[0]
    return x


def linear_reverseCom_expand(front_fea, back_fea, _layer_params):
    """ n_l < n_{l+1} """
    A = _layer_params.weight.data.detach()
    b = remove_bias(back_fea, _layer_params)
    res = torch.linalg.lstsq(A, b.mT, driver="gels").solution.mT
    reliable = check_reliability(res, A, b, "lstsq", x_ref=front_fea)
    if reliable.all():
        return res
    failed = ~reliable
    # res = res.clone()
    res[failed] = mdp_tikhonov_solve(A, b[failed], front_fea[failed])[0]
    return res


def linear_reverseCom(front_fea, back_fea, _layer_params):
    """back_fea: (batchsize, back_fea_dim)"""
    if front_fea.shape[-1] >= back_fea.shape[-1]:
        return linear_reverseCom_shrink(front_fea, back_fea, _layer_params)
    else:
        return linear_reverseCom_expand(front_fea, back_fea, _layer_params)
