import torch
import torch.nn.functional as F

def optimal_embedding(out, label, method='nearest'):
    """ 输出向量out:bs*cls,  标签label: bs 
        输出recons_out:bs*cls """
    if method == 'nearest':
        return nearest_embedding(out, label)
    elif method == 'max_assign':
        return max_assign(out, label, scale=1)
    elif method == 'one_hot':
        return F.one_hot(label, num_classes=out.shape[1]).to(device=out.device, dtype=out.dtype)
    else:
        raise NotImplementedError


def nearest_embedding(out, label):
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
    """ 输出向量out:bs*cls,  标签label: bs 
        输出recons_out:bs*cls,它的第label个分量就是最大分量的scale倍"""
    bs = len(label)
    max_val, _ = torch.max(out, dim=-1)  # (bs,)
    recons_out = out.clone()
    recons_out[torch.arange(bs, device=out.device), label] = max_val * scale
    return recons_out


if __name__ == '__main__':
    def assert_close(actual, expected, message):
        """给出比裸 assert 更容易定位问题的报错信息。"""
        if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise AssertionError(
                f"{message}\nactual:\n{actual}\nexpected:\n{expected}"
            )


    def test_nearest_embedding():
        # 覆盖：目标类别最小、已经最大、与最大值并列，以及多个分量需要调整。
        out = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [3.0, 3.0, 1.0],
            [0.0, 4.0, 2.0],
        ])
        label = torch.tensor([0, 2, 0, 2], dtype=torch.long)
        expected = torch.tensor([
            [2.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
            [3.0, 3.0, 1.0],
            [0.0, 3.0, 3.0],
        ])

        result = optimal_embedding(out, label, method='nearest')
        assert_close(result, expected, 'nearest 输出不符合预期')

        # nearest 是到约束集合的投影：目标分量应为最大值，且每行总和不变。
        target_values = result.gather(1, label.unsqueeze(1))
        assert torch.all(target_values >= result - 1e-6), '目标类别不是最大分量'
        assert_close(result.sum(dim=1), out.sum(dim=1), 'nearest 改变了行元素之和')

        random_out = torch.randn(256, 8, dtype=torch.float64)
        random_label = torch.randint(0, 8, (256,))
        random_result = nearest_embedding(random_out, random_label)
        random_targets = random_result.gather(1, random_label.unsqueeze(1))
        assert torch.all(random_targets >= random_result - 1e-6), (
            '随机输入中目标类别不是最大分量'
        )
        assert random_result.dtype == random_out.dtype, 'nearest 应保持输入的数据类型'


    def test_max_assign():
        out = torch.tensor([
            [1.0, 2.0, 3.0],
            [-4.0, -2.0, -3.0],
        ])
        label = torch.tensor([0, 2], dtype=torch.long)

        result = optimal_embedding(out, label, method='max_assign')
        expected = torch.tensor([
            [3.0, 2.0, 3.0],
            [-4.0, -2.0, -2.0],
        ])
        assert_close(result, expected, 'max_assign 输出不符合预期')

        scaled = max_assign(out, label, scale=2)
        expected_scaled = torch.tensor([
            [6.0, 2.0, 3.0],
            [-4.0, -2.0, -4.0],
        ])
        assert_close(scaled, expected_scaled, 'max_assign 的 scale 参数未正确生效')

        random_out = torch.randn(128, 8, dtype=torch.float64)
        random_label = torch.randint(0, 8, (128,))
        random_result = nearest_embedding(random_out, random_label)
        random_targets = random_result.gather(1, random_label.unsqueeze(1))
        assert torch.all(random_targets >= random_result - 1e-6), (
            '随机输入中目标类别不是最大分量'
        )
        assert random_result.dtype == random_out.dtype, 'nearest 应保持输入的数据类型'
        
    def test_one_hot():
        out = torch.tensor([
            [1.0, 2.0, 3.0],
            [-4.0, -2.0, -3.0],
        ])
        label = torch.tensor([0, 2], dtype=torch.long)
        result = optimal_embedding(out, label, method='one_hot')
        expected = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert_close(result, expected, 'one_hot 输出不符合预期')
        assert result.dtype == out.dtype, 'one_hot 应保持输入的数据类型'
        assert result.device == out.device, 'one_hot 应与输入位于同一设备'

    tests = [
        test_nearest_embedding,
        test_max_assign,
        test_one_hot
    ]
    for test in tests:
        test()
        print(f'[PASS] {test.__name__}')
