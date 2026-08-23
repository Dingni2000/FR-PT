import numpy as np
import pandas as pd
from scipy import stats

def _bootstrap_ci(diff, n_boot=20000, ci=0.95, random_state=0):
    """Bootstrap CI for mean paired improvement."""
    rng = np.random.default_rng(random_state)
    n = len(diff)

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(sample.mean())

    alpha = 1 - ci
    low, high = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return low, high


def _sign_permutation_test(diff):
    """
    Exact sign-flip permutation test for paired data.
    H1: mean(diff) > 0
    diff = aux - base
    """
    n = len(diff)
    observed = diff.mean()

    # 2^n exact enumeration is fine for n=10
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
    perm_means = (signs * diff).mean(axis=1)

    p_value = np.mean(perm_means >= observed)
    return p_value

def _get_records(data, use_aux_key="use_aux", seed_key="seed", acc_key="testacc"):
    """
    输入格式:
    data = [
        {"use_aux": False, "seed": 0, "testacc": [0.80, 0.82, ..., 0.86]},
        {"use_aux": True,  "seed": 0, "testacc": [0.81, 0.83, ..., 0.88]},
        ...
    ]
    """
    rows = []
    for d in data:
        use_aux = bool(d[use_aux_key])
        seed = d[seed_key]
        accs = d[acc_key]

        for epoch, acc in enumerate(accs, start=1):
            rows.append({
                "use_aux": use_aux,
                "seed": seed,
                "epoch": epoch,
                "testacc": float(acc),
            })

    return pd.DataFrame(rows)


def _paired_arrays(df, metric="final"):
    """
    metric:
        - "final": 每个 seed 最后一个 epoch 的 test accuracy，推荐作为主检验
        - "best": 每个 seed 训练过程中最好的 test accuracy
        - "mean": 每个 seed 所有 epoch 的平均 test accuracy
        - "auc": 每个 seed 的学习曲线面积，等价于平均性能趋势
    """
    if metric == "final":
        tmp = df.sort_values("epoch").groupby(["use_aux", "seed"]).tail(1)

    elif metric == "best":
        tmp = df.groupby(["use_aux", "seed"], as_index=False)["testacc"].max()

    elif metric == "mean":
        tmp = df.groupby(["use_aux", "seed"], as_index=False)["testacc"].mean()

    elif metric == "auc":
        records = []
        for (use_aux, seed), g in df.groupby(["use_aux", "seed"]):
            g = g.sort_values("epoch")
            auc = np.trapz(g["testacc"].values, g["epoch"].values)
            records.append({"use_aux": use_aux, "seed": seed, "testacc": auc})
        tmp = pd.DataFrame(records)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    pivot = tmp.pivot(index="seed", columns="use_aux", values="testacc")

    if False not in pivot.columns or True not in pivot.columns:
        raise ValueError("Both use_aux=False and use_aux=True must exist.")

    pivot = pivot.dropna()
    base = pivot[False].values
    aux = pivot[True].values
    seeds = pivot.index.tolist()

    return base, aux, seeds


def _cohens_d_paired(diff):
    """Paired Cohen's dz."""
    return diff.mean() / diff.std(ddof=1)


def statistical_test_aux_loss(
    data,
    metric="final",
    alpha=0.05,
    n_boot=20000,
    random_state=0,
):
    """
    检验辅助 loss 是否显著提升 test accuracy.

    主假设:
        H0: mean(acc_aux - acc_base) <= 0
        H1: mean(acc_aux - acc_base) > 0

    推荐:
        metric="final" 作为主检验
        metric="mean" 或 "auc" 作为补充检验
    """
    df = _get_records(data)
    base, aux, seeds = _paired_arrays(df, metric=metric)

    diff = aux - base
    n = len(diff)

    if n < 2:
        raise ValueError("Need at least 2 paired seeds.")

    mean_base = base.mean()
    mean_aux = aux.mean()
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)

    # 1-sided paired t-test: aux > base
    t_stat, p_ttest = stats.ttest_rel(aux, base, alternative="greater")

    # 1-sided Wilcoxon signed-rank test: aux > base
    # 对 n=10 更稳健，但如果所有 diff 都为 0 会报错
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(aux, base, alternative="greater")
    except ValueError:
        w_stat, p_wilcoxon = np.nan, np.nan

    # Exact paired permutation test
    p_perm = _sign_permutation_test(diff)

    # Effect size
    dz = _cohens_d_paired(diff)

    # Bootstrap CI for mean improvement
    ci_low, ci_high = _bootstrap_ci(
        diff,
        n_boot=n_boot,
        ci=0.95,
        random_state=random_state,
    )

    result = {
        "metric": metric,
        "n_paired_seeds": n,
        "seeds": seeds,

        "base_mean": mean_base,
        "aux_mean": mean_aux,
        "mean_improvement": mean_diff,
        "std_improvement": std_diff,

        "paired_t_stat": t_stat,
        "paired_t_pvalue_one_sided": p_ttest,

        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue_one_sided": p_wilcoxon,

        "permutation_pvalue_one_sided": p_perm,

        "cohens_dz_paired": dz,
        "bootstrap_95ci_low": ci_low,
        "bootstrap_95ci_high": ci_high,

        "significant_by_ttest": p_ttest < alpha,
        "significant_by_wilcoxon": p_wilcoxon < alpha if not np.isnan(p_wilcoxon) else False,
        "significant_by_permutation": p_perm < alpha,
    }

    return result, df


def print_test_report(result):
    print(f"Metric: {result['metric']}")
    print(f"Number of paired seeds: {result['n_paired_seeds']}")
    print()

    print(f"Base mean acc: {result['base_mean']:.6f}")
    print(f"Aux  mean acc: {result['aux_mean']:.6f}")
    print(f"Mean improvement: {result['mean_improvement']:.6f}")
    print(f"Std improvement:  {result['std_improvement']:.6f}")
    print()

    print("One-sided tests: H1 = aux > base")
    print(f"Paired t-test p-value:     {result['paired_t_pvalue_one_sided']:.6g}")
    print(f"Wilcoxon p-value:          {result['wilcoxon_pvalue_one_sided']:.6g}")
    print(f"Permutation p-value:       {result['permutation_pvalue_one_sided']:.6g}")
    print()

    print(f"Paired Cohen's dz: {result['cohens_dz_paired']:.6f}")
    print(
        "Bootstrap 95% CI of improvement: "
        f"[{result['bootstrap_95ci_low']:.6f}, "
        f"{result['bootstrap_95ci_high']:.6f}]"
    )
    print()

    if result["paired_t_pvalue_one_sided"] < 0.05:
        print("Conclusion by paired t-test: significant improvement.")
    else:
        print("Conclusion by paired t-test: not significant.")
