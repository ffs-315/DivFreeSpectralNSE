import numpy as np
from pathlib import Path
from scipy.sparse import diags
from scipy.linalg import eigh
from shenfun import FunctionSpace, TrialFunction, TestFunction, grad, inner


# ============================================================
# 数据保存路径
# ============================================================
data_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/"
    "Data/preconditioned_spectrum"
)
data_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 参数设置
# ============================================================
gamma = 1.0
N_values = np.array([16, 32, 64, 128], dtype=int)


# 保存公共参数
np.save(data_dir / "N_values.npy", N_values)
np.save(data_dir / "gamma.npy", np.array(gamma, dtype=float))


# 每行依次记录：
# N, 矩阵阶数, 等于1的特征值个数, 大于1的特征值个数,
# 小于1的特征值个数, 最小特征值, 最大特征值
spectrum_summary = []


# ============================================================
# 对不同 N 计算预条件矩阵的广义特征值
# ============================================================
for N in N_values:
    N = int(N)
    ndof = N - 3
    I = np.eye(ndof)

    print("-" * 60)
    print(f"正在计算 N = {N}, 矩阵阶数 = {ndof**2}")

    # phi_{N-4} 含 L_N，因此需要 N+1 个 Legendre 模态。
    V = FunctionSpace(
        N + 1,
        "Legendre",
        bc=(0, 0, 0, 0),
    )

    u = TrialFunction(V)
    v = TestFunction(V)

    # Shenfun 内置双调和基尚未乘论文中的归一化因子 d_k。
    M_shen = inner(u, v).diags(format="csr")
    S_shen = inner(grad(u), grad(v)).diags(format="csr")

    k = np.arange(ndof, dtype=float)
    d = 1.0 / np.sqrt(
        2.0 * (2.0*k + 3.0)**2 * (2.0*k + 5.0)
    )

    D_scale = diags(
        d,
        offsets=0,
        shape=(ndof, ndof),
        format="csr",
    )

    M = (D_scale @ M_shen @ D_scale).toarray()
    S = (D_scale @ S_shen @ D_scale).toarray()

    # Q = S M^{-1} S；使用 solve 代替显式计算 M^{-1}。
    Q = S @ np.linalg.solve(M, S)
    Q = 0.5*(Q + Q.T)

    # --------------------------------------------------------
    # 大 N 算法：避免构造 (N-3)^2 阶的二维稠密矩阵。
    # --------------------------------------------------------
    # 求广义特征分解
    #
    #     S X = M X Lambda,       X^T M X = I.
    #
    # 在 X tensor X 基下，预条件矩阵 P 变成对角矩阵，其对角元为
    #
    #     p_ij = mu_i + mu_j + gamma*(mu_i + mu_j)^2.
    mu, X = eigh(S, M, check_finite=False)

    # D = I - S M^{-1} S 是秩为 4 的半正定矩阵。
    # 只对变换后的 D_hat 做特征分解，矩阵阶数仍然只有 N-3。
    D = I - Q
    D_hat = X.T @ D @ X
    D_hat = 0.5*(D_hat + D_hat.T)

    delta, Z = eigh(D_hat, check_finite=False)
    tolerance_D = 1.0e-10 * max(1.0, np.max(np.abs(delta)))
    positive_D = delta > tolerance_D
    rank_D = int(np.count_nonzero(positive_D))

    if rank_D == 0:
        raise RuntimeError(f"N={N} 时没有检测到 D 的正特征值")

    B = Z[:, positive_D] * np.sqrt(delta[positive_D])[None, :]

    # D_hat = B B^T。预条件后的二维矩阵具有形式
    #
    #     I + W W^T,
    #
    # 其中 W 只有 2*rank(D)*(N-3) 列。非单位特征值可由
    # 小矩阵 G = W^T W 得到。
    mu_sum = mu[:, None] + mu[None, :]
    p = mu_sum + gamma*mu_sum**2
    inverse_p = 1.0/p

    reduced_size = rank_D*ndof
    G11 = np.zeros((reduced_size, reduced_size))
    G22 = np.zeros((reduced_size, reduced_size))
    G12 = np.zeros((reduced_size, reduced_size))

    # (B tensor I)^T diag(1/p_ij) (B tensor I)
    for j in range(ndof):
        index = np.arange(rank_D)*ndof + j
        block = B.T @ (inverse_p[:, j, None]*B)
        G11[np.ix_(index, index)] = gamma*block

    # (I tensor B)^T diag(1/p_ij) (I tensor B)
    for i in range(ndof):
        index = i*rank_D + np.arange(rank_D)
        block = B.T @ (inverse_p[i, :, None]*B)
        G22[np.ix_(index, index)] = gamma*block

    # 两个低秩因子之间的交叉项。
    for a in range(rank_D):
        row_index = a*ndof + np.arange(ndof)
        for b in range(rank_D):
            column_index = np.arange(ndof)*rank_D + b
            block = (
                np.outer(B[:, b], B[:, a])
                * inverse_p.T
            )
            G12[np.ix_(row_index, column_index)] = gamma*block

    G = np.block(
        [
            [G11, G12],
            [G12.T, G22],
        ]
    )
    G = 0.5*(G + G.T)

    theta = np.linalg.eigvalsh(G)

    # 两个低秩子空间的交维数为 rank(D)^2，因此真正的非单位
    # 特征值个数为 2*rank(D)*(N-3)-rank(D)^2。
    number_nonunit = 2*rank_D*ndof - rank_D**2
    theta_nonzero = np.maximum(theta[-number_nonunit:], 0.0)

    number_unit = ndof**2 - number_nonunit
    eigenvalues = np.concatenate(
        (
            np.ones(number_unit),
            1.0 + theta_nonzero,
        )
    )

    # 每个 N 的全部特征值单独保存，避免不同长度数组使用 object 类型。
    np.save(
        data_dir / f"eigenvalues_N_{N}.npy",
        eigenvalues,
    )

    tolerance = 1.0e-10 * max(
        1.0,
        np.max(np.abs(eigenvalues)),
    )

    number_one = int(
        np.count_nonzero(np.abs(eigenvalues - 1.0) <= tolerance)
    )
    number_greater = int(
        np.count_nonzero(eigenvalues > 1.0 + tolerance)
    )
    number_less = int(
        np.count_nonzero(eigenvalues < 1.0 - tolerance)
    )

    spectrum_summary.append(
        [
            N,
            ndof**2,
            number_one,
            number_greater,
            number_less,
            eigenvalues[0],
            eigenvalues[-1],
        ]
    )

    print(f"特征值文件: eigenvalues_N_{N}.npy")
    print(f"lambda = 1 的个数: {number_one}")
    print(f"lambda > 1 的个数: {number_greater}")
    print(f"lambda < 1 的个数: {number_less}")
    print(f"最小特征值: {eigenvalues[0]:.16e}")
    print(f"最大特征值: {eigenvalues[-1]:.16e}")


# 保存汇总数据
spectrum_summary = np.asarray(spectrum_summary, dtype=float)
np.save(
    data_dir / "spectrum_summary.npy",
    spectrum_summary,
)


print("=" * 60)
print("全部谱数据已经保存到:")
print(data_dir)
print("保存的文件包括:")
print("  N_values.npy")
print("  gamma.npy")
print("  spectrum_summary.npy")
for N in N_values:
    print(f"  eigenvalues_N_{int(N)}.npy")
