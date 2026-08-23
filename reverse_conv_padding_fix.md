# Padding-aware Reverse Convolution 说明

本文说明 `test_reverse_conv.ipynb` 中新增的反向卷积求解逻辑，以及为什么原来的 FFT-based 方法在 `padding=0` 时可以验证成功，但在 `padding>0` 时会出现“只有中间区域正确，边界一圈不正确”的现象。

## 1. 问题设定

设第 `l` 层卷积的前向计算为

```text
y = Conv2d(x; W, b)
```

其中：

- `x` 是第 `l` 层特征图，shape 为 `(BS, C_l, H_l, W_l)`；
- `y` 是第 `l+1` 层 pre-activation 特征图，shape 为 `(BS, C_{l+1}, H_{l+1}, W_{l+1})`；
- `W` 和 `b` 是冻结的卷积核和 bias；
- stride 和 dilation 暂按 notebook 中的设定处理，即常规 `stride=1, dilation=1`。

目标是：给定原始前向特征 `x0` 和目标输出 `y_target`，求一个新的 `x*`，使得

```text
Conv2d(x*; W, b) = y_target
```

并且在所有满足约束的解中，选择离原始特征 `x0` 最近的解：

```text
min ||x - x0||_2^2
s.t. Conv2d(x; W, b) = y_target
```

当 `C_l >= C_{l+1}` 时，通常未知量更多，存在无穷多解，所以上面这个“最近解”问题是合适的。

当 `C_l < C_{l+1}` 时，约束通常过定，目标输出不一定能被某个 `x` 精确满足，因此改成最小二乘问题：

```text
min ||Conv2d(x; W, b) - y_target||_2^2
```

## 2. 原 FFT 方法的核心思路

卷积层里的 `conv2d` 实际做的是 cross-correlation，不是数学定义中的 convolution。为了使用卷积定理，需要先把 kernel 在空间维度上翻转：

```python
reverse_weight = torch.flip(weight, [-2, -1])
```

这样可以把 PyTorch 的 cross-correlation 写成数学卷积形式，再通过 FFT 转化为频域乘法：

```text
FFT(y) = FFT(x) * FFT(flip(W)) - G(x)
```

这里的 `G(x)` 是边界修正项。它表示：有限尺寸 feature map 上的真实卷积并不是循环卷积，FFT 乘法默认会引入循环边界贡献，所以要把真实卷积没有使用到的边界贡献扣掉。

在原代码里，`G(x)` 使用的是原始前向特征 `front_fea`，即近似为：

```text
G(x*) ≈ G(x0)
```

于是每个频率点 `(u, v)` 可以独立求一个小的线性系统：

```text
sum_m FFT(x*_m)(u,v) FFT(W_nm)(u,v) - G_nm(x0)(u,v)
    = FFT(y_target_n - b_n)(u,v)
```

这个设计可以把大规模空间域问题拆成很多小规模频域问题，因此速度很快。

## 3. 为什么 padding=0 时问题不明显

当 `padding=0` 时，输出只来自输入图像中完整覆盖 kernel 的有效区域。此时边界项的影响和输出裁剪方式比较一致，原近似

```text
G(x*) ≈ G(x0)
```

在测试里不会造成明显的外圈错误。尤其是你验证的是真实输出的 valid 区域，因此原 FFT 系统和 PyTorch 前向结果可以对齐。

## 4. 为什么 padding>0 时边界会错

当 `padding>0` 时，PyTorch 的前向卷积等价于先对输入 `x` 做 zero padding，再进行 cross-correlation。此时输出边界位置的计算窗口会包含一部分真实输入、一部分 padding zero。

也就是说，边界输出正是由输入特征的边缘元素决定的：

```text
y_top/bottom/left/right = boundary-dependent function of x
```

在 FFT 建模里，这部分依赖被放进了边界修正项：

```text
G(x)
```

但是原反解代码中，求解时使用的是固定的：

```text
G(x0)
```

所以线性系统内部实际约束的是：

```text
FFT(x*) * FFT(W) - G(x0) = FFT(y_target)
```

而真实前向验证时，PyTorch 计算的是：

```text
FFT(x*) * FFT(W) - G(x*)
```

两者只有在 `G(x*) = G(x0)` 时才一致。padding>0 时，边界输出高度依赖 `x*` 的边缘值，因此这个近似会导致：

- 中间区域正确；
- 外圈边界不正确；
- 频域线性系统的残差看起来很小，但真实 `conv2d(x*)` 的边界误差很大。

这就是 notebook 中观察到的现象。

## 5. 新代码的精确建模方式

为了先保证数学正确性，我新增了一个 padding-aware 的精确线性算子版本。

核心思想是：不要手动近似 `G(x)`，而是直接把 PyTorch 的 `conv2d` 当成一个线性算子 `A`：

```text
vec(y - b) = A vec(x)
```

其中 `A` 完全由真实的 `F.conv2d` 构造，因此自动包含：

- padding；
- kernel 的 cross-correlation 定义；
- 输出空间尺寸；
- 边界行为；
- 多输入通道和多输出通道。

这样就不会出现 FFT 边界项和真实 `conv2d` 边界行为不一致的问题。

## 6. `C_l >= C_{l+1}` 时的最近解推导

先去掉 bias：

```text
y' = y_target - b
```

把 feature map 拉平成向量：

```text
x  = vec(feature)
x0 = vec(original feature)
y' = vec(target pre-activation)
```

问题为：

```text
min_x 1/2 ||x - x0||_2^2
s.t. A x = y'
```

构造 Lagrangian：

```text
L(x, λ) = 1/2 ||x - x0||_2^2 + λ^T (A x - y')
```

对 `x` 求导并令其为 0：

```text
x - x0 + A^T λ = 0
```

因此：

```text
x = x0 - A^T λ
```

代回约束：

```text
A x0 - A A^T λ = y'
```

得到：

```text
A A^T λ = A x0 - y'
```

等价地，可以定义 residual：

```text
r = y' - A x0
```

则求：

```text
A A^T α = r
x* = x0 + A^T α
```

这正是代码中的实现：

```python
y0 = (A @ x0.T).T
residual = target - y0
gram = A @ A.T
lagrange = torch.linalg.lstsq(gram, residual.T).solution
x_new = x0 + (A.T @ lagrange).T
```

这里用 `torch.linalg.lstsq` 而不是直接 `solve`，是为了处理 `A A^T` 可能秩亏的情况。这样可以得到最小范数的乘子解，并给出离 `x0` 最近的可行解。

## 7. `C_l < C_{l+1}` 时的最小二乘推导

当输出通道更多时，约束可能无法精确满足。此时求：

```text
min_x ||A x - y'||_2^2
```

代码中直接调用：

```python
x_new = torch.linalg.lstsq(A, target.T).solution.T
```

这会返回最小二乘意义下尽量满足前向卷积一致性的解。

## 8. 新增函数说明

### `build_conv2d_matrix`

位置：`test_reverse_conv.ipynb`

作用：构造单个样本上的精确卷积线性映射矩阵 `A`。

它的做法是：

1. 构造输入空间的单位基；
2. 把每个 basis vector reshape 成一个输入 feature map；
3. 对所有 basis feature map 一次性调用 `F.conv2d`；
4. 把输出 reshape 成矩阵。

因此得到的 `A` 与 PyTorch `conv2d` 完全一致。

### `_remove_bias`

作用：从目标 `back_fea` 中减去 bias。

因为线性系统只处理：

```text
A x = y - b
```

bias 是仿射项，需要先移到右侧。

### `convo_reverseCom_exact_shrink`

作用：处理 `C_l >= C_{l+1}` 的情况。

它求解：

```text
min ||x - x0||_2^2
s.t. A x = y_target - b
```

对应“所有满足前向一致性的解中，选择离原始 feature 最近的解”。

### `convo_reverseCom_exact_expand`

作用：处理 `C_l < C_{l+1}` 的情况。

它求解：

```text
min ||A x - (y_target - b)||_2^2
```

对应“找一个尽量满足前向卷积一致性的解”。

### `convo_reverseCom_exact`

作用：根据通道数选择 shrink 或 expand 分支。

### `convo_reverseCom`

现在的 wrapper 逻辑是：

```python
if padding > 0:
    return convo_reverseCom_exact(...)
else:
    return original_fft_based_method(...)
```

原因是：

- `padding=0` 时，原 FFT 快速路径仍能工作；
- `padding>0` 时，原 FFT 版本固定 `G(x0)` 会导致边界不一致，因此改走精确线性算子版本。

## 9. 验证结果

在原 notebook 的测试设置中：

```python
X_SHAPE = (1, 2, 5, 7)
KERNEL_SHAPE = (1, X_SHAPE[1], 3, 3)
PADDING = 1
```

修改后重新运行：

```python
x_new = convo_reverseCom(x, y_noise, conv)
print((conv(x_new.to(torch.float32)) - y_noise).abs().max())
```

得到：

```text
tensor(4.7684e-07)
```

这是 float32 的舍入误差级别，说明边界和中间区域都已经符合目标 `y_noise`。

padding=0 的原 FFT 路径也重新验证过：

```text
tensor(1.4305e-06)
```

同样是数值误差级别。

## 10. 关于 FFT 加速版本的后续方向

当前新增的精确版本优先保证正确性，但它显式构造了矩阵 `A`，所以当 feature map 很大时，内存和计算量会明显增加。

如果后续希望继续使用 FFT 加速，同时支持 `padding>0` 的精确边界约束，需要处理一个关键点：

```text
G(x) 不能固定为 G(x0)，而必须作为关于 x 的线性项一起进入求解。
```

但 `G(x)` 包含空间域 mask：

```text
FFT(mask * x)
```

空间域乘 mask 在频域中会造成频率混合，因此问题不再能完全拆成每个 `(u, v)` 独立的小线性系统。可能的改进方向包括：

- 把内部区域继续用 FFT 对角化；
- 把边界区域作为稀疏/低维变量单独建模；
- 使用 matrix-free 的 `A` 和 `A^T`，通过迭代法求解，而不显式构造完整矩阵；
- 用 `F.conv2d` 表示 `A`，用 `F.conv_transpose2d` 表示 `A^T`，实现大规模共轭梯度或 LSQR。

这会比当前显式矩阵版本更适合真实网络中的大 feature map。

## 11. 当前 FFT-based 更新

后续代码已经把 `padding>0` 分支从显式矩阵版本改成了 FFT-based matrix-free 版本。显式矩阵函数仍然保留，用作小规模校验；实际 `convo_reverseCom` 在 `padding>0` 时会调用：

```python
convo_reverseCom_fft_padding(...)
```

新的实现不再构造完整矩阵 `A`，而是只实现两个算子：

```text
A(x)   = Conv2d(x; W, padding)
A^T(v) = A 的伴随算子
```

其中 `A(x)` 和 `A^T(v)` 都通过 FFT 计算。

### 11.1 FFT 前向算子 `A(x)`

PyTorch 的 `conv2d` 是 cross-correlation。为了用数学卷积的 FFT 定理，代码中仍然先翻转 kernel：

```python
kernel_flip = torch.flip(weight, dims=(-2, -1))
```

然后：

1. 对输入 `x` 做 zero padding；
2. 用 FFT 计算 padded input 和 flipped kernel 的 full linear convolution；
3. 从 full convolution 中裁剪出 PyTorch `conv2d` 对应的 valid 输出区域。

对应代码是：

```python
fft_conv2d_forward(x, weight, padding)
```

该函数已经验证过，与 `F.conv2d(x, weight, padding=padding)` 的最大误差为 double 精度舍入量级。

### 11.2 FFT 伴随算子 `A^T(v)`

为了求最近解，需要用到 `A^T`。如果前向是：

```text
y = A x
```

那么伴随算子满足：

```text
<A x, v> = <x, A^T v>
```

代码中 `A^T(v)` 的做法是：

1. 把输出侧梯度 `v` 和未翻转的 `weight` 做 full linear convolution；
2. 得到 padded input 空间上的梯度；
3. 裁掉 padding 区域，只保留原始输入 `x` 的空间范围。

对应代码是：

```python
fft_conv2d_adjoint(y_grad, weight, padding, input_hw)
```

该函数已通过内积关系验证：

```text
<A x, v> - <x, A^T v> ≈ 1e-14
```

### 11.3 Shrink 分支：FFT-CG 最近解

当 `C_l >= C_{l+1}` 时，仍然求：

```text
min ||x - x0||_2^2
s.t. A x = y_target - b
```

投影公式为：

```text
r = y_target - b - A x0
(A A^T) alpha = r
x* = x0 + A^T alpha
```

现在不再显式构造 `A A^T`，而是通过函数组合计算：

```python
def apply_AAT(alpha):
    return A(A_T(alpha))
```

然后用 batched conjugate gradient 求解 `alpha`：

```python
alpha = conjugate_gradient_batched(apply_AAT, residual)
x_new = x0 + A_T(alpha)
```

对应代码：

```python
convo_reverseCom_fft_padding_shrink(...)
```

### 11.4 Expand 分支：FFT-CG 最小二乘

当 `C_l < C_{l+1}` 时，求：

```text
min ||A x - (y_target - b)||_2^2
```

正规方程为：

```text
A^T A x = A^T (y_target - b)
```

同样不显式构造矩阵，只定义：

```python
def apply_ATA(x):
    return A_T(A(x))
```

然后用 CG 求解。

对应代码：

```python
convo_reverseCom_fft_padding_expand(...)
```

### 11.5 当前验证结果

当前 notebook 中的 `padding=1, C_l >= C_{l+1}` 测试：

```text
max |Conv2d(x_new) - y_noise| = 9.5367e-07
```

`padding=0` 的旧 FFT 路径仍然正常：

```text
max error = 1.6689e-06
```

`padding=1, C_l < C_{l+1}` 的 expand 分支与显式矩阵 least-squares 对照：

```text
FFT-CG residual   = 1.3517e-06
matrix residual   = 1.3621e-06
solution diff norm = 1.5017e-08
```

说明新的 FFT-based matrix-free 版本在保持边界正确性的同时，避免了显式构造大矩阵。

## 12. 一步近似解 `fast_approx`

如果希望完全避免迭代，可以使用当前 notebook 中新增的调用方式：

```python
x_new = convo_reverseCom(x, y_noise, conv, method="fast_approx")
```

它会在 `padding>0` 时跳过 FFT-CG，直接复用原来的逐频点闭式求解逻辑。

### 12.1 为什么精确问题不能一步对角化

无 padding 或循环 padding 下，卷积可以被 FFT 完全对角化：

```text
FFT(Ax)(u,v) = K(u,v) FFT(x)(u,v)
```

于是每个频率点 `(u,v)` 都是一个独立的小线性系统。

但 zero padding 的真实 `conv2d` 不是循环卷积。它等价于：

```text
pad -> linear convolution -> crop
```

其中 `pad/crop` 在频域中不是逐点乘法，而会混合不同频率。因此精确的 padding-aware 问题一般不能写成每个频率点独立的一步闭式解。

### 12.2 `fast_approx` 近似了什么

`fast_approx` 采用固定边界项近似：

```text
G(x*) ≈ G(x0)
```

于是约束重新变成逐频点可分的形式：

```text
K(u,v) X*(u,v) = Y_target(u,v) + G(x0)(u,v)
```

当 `C_l >= C_{l+1}` 时，每个频率点求最近解：

```text
min ||X - X0||_2^2
s.t. K X = Y + G(x0)
```

当 `C_l < C_{l+1}` 时，每个频率点求最小二乘解：

```text
min ||K X - (Y + G(x0))||_2^2
```

这就是“一步计算”的来源：不再用 CG，不再反复调用 `A` 和 `A^T`。

### 12.3 速度和误差特征

在当前小测试中：

```text
method="fast_approx":
    center max error ≈ 5.96e-07
    border max error ≈ 1.41

method="fft_cg":
    full-image max error ≈ 4.77e-07
```

这说明 `fast_approx` 的误差主要集中在 padding 边界；中间区域仍然基本正确。

因此可以按需求选择：

```python
# 精确，padding 边界也对，但需要 CG 迭代
x_new = convo_reverseCom(x, y_noise, conv, method="fft_cg")

# 一步近似，更快，但 padding 边界会有误差
x_new = convo_reverseCom(x, y_noise, conv, method="fast_approx")

# 小规模校验用，显式构造矩阵，不适合大 feature map
x_new = convo_reverseCom(x, y_noise, conv, method="exact_matrix")
```

如果后续你的任务更关注 feature map 的内部区域，或者 padding 边界会在后续层中被弱化，`fast_approx` 是一个合理的加速选择。若目标是严格满足整张 `y_noise`，仍应使用 `fft_cg`。

## 13. `fast_approx` 的误差分析

这一节把 `fast_approx` 的误差写成严格的线性算子形式。先忽略 bias，因为 bias 可以并入目标：

```text
y = y_target - b
```

所有范数默认使用 Euclidean/Frobenius 范数。设输入空间为

```text
X = R^{C_in x H x W}
```

输出空间为

```text
Y = R^{C_out x H_out x W_out}
```

为了避免 FFT 归一化常数干扰，下面的算子范数分析默认使用 unitary FFT 记号。若使用 PyTorch 默认的非归一化 `fft2/ifft2`，公式中会出现固定的 `H,W` 缩放常数，但所有误差结构和结论不变。

真实的 PyTorch zero-padding 卷积记为线性算子

```text
A: X -> Y
```

FFT 中使用的循环卷积近似记为

```text
C: X -> Y
```

二者的差记为边界算子

```text
G = C - A
```

于是对任意输入 `x` 都有精确恒等式：

```text
A x = C x - G x
```

文档前面写的

```text
FFT(Ax) = K FFT(x) - G(x)
```

本质上就是上面这个等式的频域表示。

### 13.1 `fast_approx` 解的真实 residual

`fast_approx` 用旧输入 `x0` 固定边界项，求解的不是

```text
C x - G x = y
```

而是近似问题

```text
C x - G x0 = y
```

等价地，

```text
C x = y + G x0
```

设 `fast_approx` 得到的解为

```text
x_f
```

若频域逐点线性系统被精确求解，则

```text
C x_f - G x0 - y = 0
```

现在用真实卷积 `A` 去验证 `x_f`，真实 residual 为

```text
r_f = A x_f - y
```

代入 `A = C - G`：

```text
r_f
= (C - G) x_f - y
= C x_f - G x_f - y
```

再加减 `G x0`：

```text
r_f
= (C x_f - G x0 - y) - (G x_f - G x0)
```

由于第一项正是 `fast_approx` 约束 residual，精确求解时它为零，因此

```text
r_f = -G(x_f - x0)
```

也就是说：

```text
真实输出误差 = 固定边界项产生的边界算子误差
```

如果频域系统存在数值 residual

```text
rho_f = C x_f - G x0 - y
```

则更一般地有

```text
r_f = rho_f - G(x_f - x0)
```

所以：

```text
||A x_f - y||
<= ||rho_f|| + ||G|| ||x_f - x0||
```

这条不等式是 `fast_approx` 最重要的误差公式。它说明两件事：

1. 频域近似系统解得再准，也只能保证 `rho_f` 小；
2. 真实 `conv2d` 的误差还包含 `G(x_f - x0)`，也就是输入改变量在 padding 边界算子上的投影。

因此 `fast_approx` 成功的充分条件不是单纯的 FFT residual 小，而是

```text
||G(x_f - x0)|| 很小
```

### 13.2 边界误差的显式求和形式

下面写出 `G` 的空间域表达式。为简洁起见，先考虑 stride=1、dilation=1、二维 cross-correlation。设 kernel 支持为

```text
S = {0,...,R-1} x {0,...,S_w-1}
```

padding 为

```text
p_h, p_w
```

真实 zero-padding 卷积为

```text
(A x)_{n,i,j}
= sum_m sum_{a=0}^{R-1} sum_{b=0}^{S_w-1}
    W_{n,m,a,b}
    x_{m,i+a-p_h,j+b-p_w}
    1_{0 <= i+a-p_h < H}
    1_{0 <= j+b-p_w < W}
```

循环卷积版本为

```text
(C x)_{n,i,j}
= sum_m sum_{a=0}^{R-1} sum_{b=0}^{S_w-1}
    W_{n,m,a,b}
    x_{m,(i+a-p_h) mod H,(j+b-p_w) mod W}
```

因此

```text
(G x)_{n,i,j}
= (C x - A x)_{n,i,j}
```

即

```text
(G x)_{n,i,j}
= sum_m sum_{a,b}
    W_{n,m,a,b}
    [
      x_{m,(i+a-p_h) mod H,(j+b-p_w) mod W}
      -
      x_{m,i+a-p_h,j+b-p_w}
      1_{0 <= i+a-p_h < H}
      1_{0 <= j+b-p_w < W}
    ]
```

注意第二项在越界时为零。于是只有当卷积窗口越过输入边界时，`G` 才可能非零。定义边界输出集合

```text
B = {
  (i,j):
  exists (a,b) such that
  i+a-p_h notin [0,H-1]
  or
  j+b-p_w notin [0,W-1]
}
```

若 `(i,j) notin B`，则所有 kernel 采样点都在输入内部，因此

```text
(G x)_{n,i,j} = 0
```

从而对 `fast_approx` 的真实 residual 有

```text
(r_f)_{n,i,j}
= - (G(x_f - x0))_{n,i,j}
= 0,    (i,j) notin B
```

这就严格解释了实验现象：

```text
center error 很小，border error 大
```

不是偶然数值现象，而是因为 `G` 的支撑集本来就集中在 padding 边界。

### 13.3 逐点误差上界

令

```text
d_f = x_f - x0
```

由上一节的显式表达式，若忽略数值 residual `rho_f`，则

```text
(r_f)_{n,i,j}
= - sum_m sum_{(a,b) in E(i,j)}
    W_{n,m,a,b}
    d_f[m,(i+a-p_h) mod H,(j+b-p_w) mod W]
```

其中 `E(i,j)` 表示在输出位置 `(i,j)` 处会越过输入边界的 kernel offset 集合：

```text
E(i,j) = {
  (a,b):
  i+a-p_h notin [0,H-1]
  or
  j+b-p_w notin [0,W-1]
}
```

于是有逐点绝对误差界：

```text
|(r_f)_{n,i,j}|
<= sum_m sum_{(a,b) in E(i,j)}
    |W_{n,m,a,b}|
    |d_f[m,(i+a-p_h) mod H,(j+b-p_w) mod W]|
```

进一步得到 infinity norm 上界：

```text
||r_f||_inf
<= L_boundary ||d_f||_inf
```

其中

```text
L_boundary
= max_{n,i,j}
  sum_m sum_{(a,b) in E(i,j)}
    |W_{n,m,a,b}|
```

显然

```text
L_boundary
<= max_n sum_m sum_{a,b} |W_{n,m,a,b}|
```

这个界说明：`fast_approx` 的边界误差由两部分控制：

```text
边界相关 kernel 权重的 l1 大小
```

和

```text
x_f - x0 的幅度
```

如果反解任务只要求对 `x0` 做很小扰动，或者卷积核在越界 offset 上的权重很小，则 `fast_approx` 的真实误差也会小。反过来，如果目标 `y_target` 迫使 `x_f` 在边界相关位置发生较大变化，则即使频域约束完全满足，真实 `conv2d` 的边界误差仍可能很大。

### 13.4 Frobenius 范数误差界

由

```text
r_f = rho_f - G d_f
```

立刻得到

```text
||r_f||_2
<= ||rho_f||_2 + ||G d_f||_2
<= ||rho_f||_2 + ||G||_{2->2} ||d_f||_2
```

这里

```text
||G||_{2->2}
```

是边界算子的谱范数。由于 `G` 只在边界输出集合 `B` 上非零，它通常远小于完整卷积算子的粗糙上界，但它不一定可以忽略。一个可计算的粗上界是

```text
||G||_{2->2}
<= sqrt(||G||_1 ||G||_inf)
```

其中 `||G||_inf` 是矩阵行绝对值和的最大值，正是上一节的 `L_boundary`；`||G||_1` 是矩阵列绝对值和的最大值。若记

```text
M_boundary
= max_{m,h,w}
   sum_{n,i,j,a,b:
        (i+a-p_h,j+b-p_w) wraps to (h,w)}
      |W_{n,m,a,b}|
```

则

```text
||G||_{2->2}
<= sqrt(L_boundary M_boundary)
```

因此

```text
||A x_f - y||_2
<= ||rho_f||_2
   + sqrt(L_boundary M_boundary) ||x_f - x0||_2
```

这是一个完全空间域的、可解释的误差界。

### 13.5 与精确最近解的扰动关系

当 `C_in >= C_out` 时，`fast_approx` 在每个频率点求的是最近解。把所有频率点合起来，它等价于：

```text
min_x 1/2 ||x - x0||_2^2
s.t. C x = y + G x0
```

设精确 padding-aware 最近解为

```text
x_* = argmin_x 1/2 ||x - x0||_2^2
      s.t. A x = y
```

令

```text
d_* = x_* - x0
```

由于 `A = C - G` 且 `A x_* = y`，有

```text
C x_* = y + G x_*
```

而 fast constraint 是

```text
C x_f = y + G x0
```

二者相减：

```text
C(x_f - x_*) = -G(x_* - x0) = -G d_*
```

如果 `C` 在相关约束子空间上满行秩，并记其最小非零奇异值为

```text
sigma_min^+(C)
```

则最小范数扰动满足典型的线性约束扰动界：

```text
dist(x_f - x_*, ker C)
<= ||C^dagger|| ||G d_*||
= ||G d_*|| / sigma_min^+(C)
```

进一步，

```text
dist(x_f - x_*, ker C)
<= (||G||_{2->2} / sigma_min^+(C)) ||x_* - x0||
```

这个式子给出了 `fast_approx` 接近精确解的条件：

```text
||G||_{2->2} << sigma_min^+(C)
```

并且精确解本身离 `x0` 不远：

```text
||x_* - x0|| 小
```

如果某些频率点的 kernel 矩阵病态，即

```text
sigma_min^+(K(u,v)) 很小
```

那么即使边界扰动 `G d_*` 不大，解误差也可能被频域伪逆放大。这是 `fast_approx` 在病态卷积核下可能不稳定的主要理论原因。

更具体地，如果循环卷积算子可以被 unitary FFT 对角化为

```text
C = F_Y^* diag(K(omega)) F_X
```

其中每个 `K(omega)` 是一个 `C_out x C_in` 的复矩阵，则

```text
||C||_{2->2}
= max_omega sigma_max(K(omega))
```

并且在满行秩的 shrink 情况下，

```text
||C^dagger||_{2->2}
= max_omega ||K(omega)^dagger||_{2->2}
= 1 / min_omega sigma_min(K(omega))
```

如果某些频点接近秩亏，则

```text
min_omega sigma_min(K(omega)) approx 0
```

这时固定边界项带来的小扰动 `G d_*` 会经过 `C^dagger` 放大，表现为 `x_f` 本身出现较大偏移，继而真实 residual 中的

```text
G(x_f - x0)
```

也可能被放大。

### 13.6 expand 情况的 residual 解释

当 `C_in < C_out` 时，目标通常过定，代码走 least-squares。此时 `fast_approx` 求的是

```text
min_x ||C x - (y + G x0)||_2^2
```

真实目标却是

```text
min_x ||A x - y||_2^2
= min_x ||C x - G x - y||_2^2
```

设 fast 解为 `x_f`，定义

```text
rho_f = C x_f - G x0 - y
```

真实 residual 仍然满足同一个恒等式：

```text
A x_f - y = rho_f - G(x_f - x0)
```

不同之处只是 `rho_f` 不一定为零，而是近似问题的最小二乘 residual。因此 expand 情况下有

```text
||A x_f - y||_2
<= ||C x_f - G x0 - y||_2
   + ||G||_{2->2} ||x_f - x0||_2
```

第一项是近似 least-squares 本身无法消掉的残差；第二项是固定旧边界项引入的额外误差。

### 13.7 结论

`fast_approx` 的误差不是一个神秘的 FFT 数值误差，而是确定性的建模误差。核心公式是：

```text
A x_f - y
= rho_f - G(x_f - x0)
```

特别地，如果近似频域问题被精确满足：

```text
A x_f - y
= -G(x_f - x0)
```

因此：

```text
fast_approx 准确
<=> x_f - x0 在边界算子 G 上的投影很小
```

对于 `padding=0` 的 valid 输出区域，`G` 在该区域为零，所以旧方法可以精确对齐真实卷积。对于 `padding>0`，`G` 支撑在输出边界；所以误差主要出现在 top/bottom/left/right 的 padding 边界，而中心区域仍然可以非常准确。
