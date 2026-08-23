# `_linearized_reverse_map` 为什么更像 Gauss-Newton，而不是欧拉法

本文解释 `imagecls/reverse_compute.py` 里的两个反向求解思路：

- `_linearized_reverse_map`: 显式构造 Jacobian，并解局部线性最小二乘。
- `_vjp_reverse_map`: 不显式构造 Jacobian，直接对输入 `z` 做梯度优化。

核心结论：

> `_linearized_reverse_map` 更像阻尼 Gauss-Newton 法；`_vjp_reverse_map` 更像用反向传播实现的梯度流/欧拉式优化。

---

## 1. 反向求解问题

代码里要解的问题可以写成：

```text
target ~= func(z)
```

也就是给定目标输出 `target`，找一个输入特征 `z`，使得前向函数 `func(z)` 尽量接近 `target`。

定义残差：

```text
r(z) = func(z) - target
```

目标函数就是：

```text
min_z  1/2 ||r(z)||^2
```

或者等价地：

```text
min_z  1/2 ||func(z) - target||^2
```

这就是一个非线性最小二乘问题。

---

## 2. Gauss-Newton 推导

设当前迭代点是 `z_k`，希望找一个增量 `delta`：

```text
z_{k+1} = z_k + delta
```

由于 `func` 可能是非线性的，直接求最优 `delta` 很难。Gauss-Newton 的做法是在 `z_k` 附近做一阶线性化：

```text
func(z_k + delta) ~= func(z_k) + J_k delta
```

其中：

```text
J_k = d func(z) / d z |_{z = z_k}
```

也就是 `func` 对输入 `z` 的 Jacobian 矩阵。

我们希望：

```text
func(z_k + delta) ~= target
```

代入线性化：

```text
func(z_k) + J_k delta ~= target
```

移项：

```text
J_k delta ~= target - func(z_k)
```

代码中的 `residual` 定义为：

```python
residual = target.reshape(-1) - y_flat
```

所以局部子问题就是：

```text
min_delta ||J_k delta - residual||^2
```

这正是代码里的：

```python
J = torch.autograd.functional.jacobian(flat_func, z_flat, vectorize=True).detach()
delta = torch.linalg.lstsq(J, residual.unsqueeze(-1)).solution.squeeze(-1)
z = z + damping * delta.reshape(shape)
```

因此 `_linearized_reverse_map` 的结构是：

```text
当前点 z_k
  -> 构造 Jacobian J_k
  -> 解最小二乘 J_k delta ~= residual
  -> 阻尼更新 z_{k+1} = z_k + damping * delta
```

这就是阻尼 Gauss-Newton。

---

## 3. Gauss-Newton 与标准 Newton 法的关系

目标函数：

```text
F(z) = 1/2 ||r(z)||^2
```

它的梯度是：

```text
grad F(z) = J(z)^T r(z)
```

它的 Hessian 是：

```text
H(z) = J(z)^T J(z) + sum_i r_i(z) * Hessian(r_i(z))
```

标准 Newton 法需要解：

```text
H delta = -grad F
```

也就是：

```text
[J^T J + sum_i r_i Hessian(r_i)] delta = -J^T r
```

Gauss-Newton 忽略二阶残差项：

```text
sum_i r_i Hessian(r_i)
```

于是近似为：

```text
J^T J delta = -J^T r
```

由于本文的代码里使用的是：

```text
residual = target - func(z) = -r(z)
```

所以等价写法是：

```text
J^T J delta = J^T residual
```

这又等价于最小二乘：

```text
min_delta ||J delta - residual||^2
```

因此，`torch.linalg.lstsq(J, residual)` 本质上是在求 Gauss-Newton 步。

---

## 4. 为什么 Jacobian 矩阵会很大

假设输入特征 `z` 的 shape 是：

```text
[B, C_in, H_in, W_in]
```

输出 `func(z)` 的 shape 是：

```text
[B, C_out, H_out, W_out]
```

代码里会 flatten：

```python
z_flat = z.reshape(-1)
y_flat = func(z).reshape(-1)
```

所以：

```text
n = z.numel() = B * C_in * H_in * W_in
m = func(z).numel() = B * C_out * H_out * W_out
```

Jacobian 的形状是：

```text
J: [m, n]
```

也就是：

```text
J[i, j] = partial y_i / partial z_j
```

每一个输出元素都要对每一个输入元素求偏导。

举一个中等大小的特征图例子：

```text
B = 1
C_in = 64
H_in = 56
W_in = 56
```

输入维度：

```text
n = 1 * 64 * 56 * 56 = 200704
```

如果输出维度差不多也是：

```text
m = 200704
```

那么 Jacobian 大小是：

```text
m * n = 200704 * 200704 ~= 4.03e10 个元素
```

如果用 float32，每个元素 4 bytes：

```text
4.03e10 * 4 bytes ~= 161 GB
```

这还只是一个 batch、一个中等 feature map 的 Jacobian。实际运行时还要考虑 autograd 中间状态、线性代数工作区、GPU 显存碎片等，开销会更大。

所以对于 ResNet 这种大 feature map，显式构造完整 Jacobian 通常非常贵，甚至不可行。

这就是为什么 `resnet_residual_reverseCom` 默认用：

```python
method="vjp"
```

而不是：

```python
method="linearized"
```

---

## 5. 为什么欧拉式/梯度法不需要显式得到 J

考虑同一个目标函数：

```text
F(z) = 1/2 ||func(z) - target||^2
```

令：

```text
r(z) = func(z) - target
```

那么梯度是：

```text
grad F(z) = J(z)^T r(z)
```

注意这里确实出现了 `J^T r`，但我们不一定要显式构造完整的 `J`。

深度学习框架的反向传播擅长计算的是：

```text
J^T v
```

这叫 vector-Jacobian product，简称 VJP。

如果取：

```text
v = r(z)
```

反向传播就能直接得到：

```text
J^T r(z)
```

也就是损失函数对输入 `z` 的梯度：

```text
grad_z F(z)
```

代码里 `_vjp_reverse_map` 做的是：

```python
y = func(z)
residual = y - target
loss = (rel ** 2).mean()
loss.backward()
opt.step()
```

`loss.backward()` 并不会生成一个完整的 `[m, n]` Jacobian 矩阵。它只沿计算图从输出往输入传播一个向量，直接算出 `z.grad`。

这就是为什么 VJP 方法省内存：

```text
显式 Jacobian:
  需要保存 J，大小是 m * n

VJP/backward:
  只需要算 J^T v，结果大小是 n
```

对于上面的例子：

```text
n = 200704
m = 200704
```

显式 Jacobian 需要大约：

```text
200704 * 200704 个数
```

而 VJP 的结果 `z.grad` 只需要：

```text
200704 个数
```

这就是数量级上的差别。

---

## 6. 为什么说它更像欧拉式更新

连续时间的梯度流可以写成：

```text
dz/dt = -grad F(z)
```

用显式欧拉法离散化：

```text
z_{k+1} = z_k - lr * grad F(z_k)
```

也就是：

```text
z_{k+1} = z_k - lr * J_k^T r(z_k)
```

这类方法只需要梯度 `grad F(z_k)`，而梯度可以由 VJP 得到，不需要显式构造完整 `J_k`。

`_vjp_reverse_map` 用的是 Adam，不是最朴素的 SGD 欧拉步，但思想更接近这一类一阶优化：

```text
根据 loss.backward() 得到的梯度更新 z
```

因此可以粗略理解为：

```text
_linearized_reverse_map:
  显式 Jacobian + 局部最小二乘，Gauss-Newton 风格

_vjp_reverse_map:
  VJP/backward + Adam 更新，一阶梯度优化风格
```

---

## 7. 为什么 `lstsq` 而不是 `solve`

`solve(A, b)` 适合解：

```text
A x = b
```

并且通常要求 `A` 是方阵、满秩、条件不要太差。

但这里的局部线性系统是：

```text
J delta ~= residual
```

其中：

```text
J: [m, n]
delta: [n]
residual: [m]
```

`m` 是输出维度，`n` 是输入维度。实际中经常出现：

- `m > n`: 超定系统，一般没有精确解。
- `m < n`: 欠定系统，解不唯一。
- `m == n`: 也可能秩亏或病态。

因此它不是一个适合直接 `solve` 的普通方阵线性方程。

Gauss-Newton 的局部子问题本来就是：

```text
min_delta ||J delta - residual||^2
```

所以用：

```python
torch.linalg.lstsq(J, residual.unsqueeze(-1))
```

比用 `solve` 更合理。

如果从正规方程角度写，最小二乘满足：

```text
J^T J delta = J^T residual
```

看起来可以用 `solve(J^T J, J^T residual)`。但实际中通常不推荐直接这么做，因为：

- `J^T J` 会放大条件数，数值稳定性更差。
- `J` 秩亏时，`J^T J` 奇异。
- 显式形成 `J^T J` 也有额外开销。

所以直接调用 `lstsq` 通常更稳。

---

## 8. 收敛性和稳定性对比

### Gauss-Newton 风格

优点：

- 接近解时通常收敛很快。
- 每一步利用局部线性结构，比普通梯度方向更有信息量。
- 对非线性最小二乘问题很自然。

缺点：

- 需要构造或隐式处理 Jacobian。
- 显式 Jacobian 对大 feature map 非常贵。
- 如果初始点离解太远，线性化不可靠，可能走出很差的步子。
- 如果 `J` 病态或秩亏，`delta` 可能异常大。

代码中的稳定化手段：

```python
damping
max_step_norm
_check_linearized_step_reliability
_regularized_lstsq
```

也就是阻尼、步长裁剪、可靠性检查、Tikhonov 正则 fallback。

### 欧拉式/VJP 梯度优化

优点：

- 不需要显式存储完整 Jacobian。
- 每步显存开销更接近普通训练的 backward。
- 更适合大 feature map。
- 实现上对任意 PyTorch module 更通用。

缺点：

- 通常比 Gauss-Newton 慢。
- 对学习率、优化器、迭代次数敏感。
- 可能在平坦区域慢慢挪，或者在非凸问题中停在较差位置。
- Adam 的更新方向不是精确的反问题线性解。

代码中的稳定化手段：

```python
lr
anchor_weight
max_step_norm
finite check
```

其中 `anchor_weight` 会让解不要离初始 `init` 太远：

```text
loss = data_loss + anchor_weight * ||z - init||^2
```

这有助于稳定，但也意味着结果不一定是纯粹最小化 `||func(z)-target||^2` 的解。

---

## 9. 一句话总结

`_linearized_reverse_map` 每一步显式构造：

```text
J = d func / d z
```

然后解：

```text
min_delta ||J delta - (target - func(z))||^2
```

所以它是 Gauss-Newton 风格。

`_vjp_reverse_map` 不构造完整 `J`，而是通过：

```text
loss.backward()
```

直接得到：

```text
J^T residual
```

再用 Adam 更新 `z`，所以它是 VJP/一阶梯度优化风格。

显式 Jacobian 大，是因为它要保存每个输出元素对每个输入元素的偏导，大小是：

```text
输出元素数 * 输入元素数
```

而 VJP 只计算 `J^T v`，结果大小只有：

```text
输入元素数
```

这就是二者在大特征图上内存差异巨大的根本原因。
