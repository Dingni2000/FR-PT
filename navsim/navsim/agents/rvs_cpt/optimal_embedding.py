import torch
import torch.nn.functional as F

def optimal_embedding(out, label, method='nearest'):
    """Construct a constrained output embedding for each label."""
    if method == 'nearest':
        return nearest_embedding(out, label)
    elif method == 'max_assign':
        return max_assign(out, label, scale=1)
    elif method == 'one_hot':
        return F.one_hot(label, num_classes=out.shape[1]).to(device=out.device, dtype=out.dtype)
    else:
        raise NotImplementedError


def nearest_embedding(out, label):
    """Project each output to the nearest vector where its label is maximal."""
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
    mask1[mask0] = (mul_left_sorted[mask0] <= cumsum[mask0]+5e-6) & (cumsum[mask0] <= mul_right_sorted[mask0]+5e-6)
    # assert torch.all(torch.sum(mask1, dim=-1)<=1), mask1
    has_true = torch.any(mask1, dim=-1).to(torch.bool)  # (bs,)
    count = - torch.ones(bs, device=out.device, dtype=torch.long)
    first_true = torch.argmax(mask1.to(torch.long), dim=-1)
    count[has_true] = first_true[has_true]  # (bs,)
    mask2 = torch.arange(out.shape[1], device=out.device).unsqueeze(0).expand(bs, -1) <= count.unsqueeze(-1)  # (bs, cls)
    mask = mask2.scatter(dim=-1, index=sorted_indices, src=mask2)  # (bs, cls)

    A = torch.eye(out.shape[1], device=out.device).unsqueeze(0).repeat(bs, 1, 1)
    idens = torch.ones(out.shape[0], out.shape[1], out.shape[1], device=out.device)
    A -= (idens/(count+2).unsqueeze(-1).unsqueeze(-1).expand(-1, out.shape[1], out.shape[1]))
    A[~mask] = 0
    A.transpose(-1, -2)[~mask] = 0
    diff[~mask] = 0
    muhalf = torch.matmul(A.to(out.dtype), diff.unsqueeze(-1)).squeeze(-1)  # (bs, cls)
    recons_out = out - muhalf
    recons_out[torch.arange(bs), label] += torch.sum(muhalf, dim=-1) 
    return recons_out


def max_assign(out, label, scale=1):
    """Set each label component to a scaled version of the row maximum."""
    bs = len(label)
    max_val, _ = torch.max(out, dim=-1)  # (bs,)
    recons_out = out.clone()
    recons_out[torch.arange(bs, device=out.device), label] = max_val * scale
    return recons_out
