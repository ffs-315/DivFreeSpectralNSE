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
    "Data/preconditioned_spectrum_gamma"
)
data_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 参数设置：固定 N，改变 gamma
# ============================================================
N = 128

if N < 7:
    raise ValueError("Theorem 4.1 要求 N >= 7")

gamma_exponents = np.array([-3, -2, -1, 0], dtype=int)
gamma_values = 10.0**gamma_exponents

ndof = N - 3
I = np.eye(ndof)


# 保存公共参数
np.save(data_dir / "N.npy", np.array(N, dtype=int))
np.save(data_dir / "gamma_exponents.npy", gamma_exponents)
np.save(data_dir / "gamma_values.npy", gamma_values)


# ============================================================
# 用 Shenfun 组装一维矩阵 M 和 S
# ============================================================
print("=" * 60)
print(f"固定 N = {N}, 二维矩阵阶数 = {ndof**2}")

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


# ============================================================
# 与 gamma 无关的低秩分解，只计算一次
# ============================================================
Q = S @ np.linalg.solve(M, S)
Q = 0.5*(Q + Q.T)

# S X = M X Lambda, X^T M X = I。
mu, X = eigh(S, M, check_finite=False)

# D = I - S M^{-1} S。
D = I - Q
D_hat = X.T @ D @ X
D_hat = 0.5*(D_hat + D_hat.T)

delta, Z = eigh(D_hat, check_finite=False)
tolerance_D = 1.0e-10 * max(1.0, np.max(np.abs(delta)))
positive_D = delta > tolerance_D
rank_D = int(np.count_nonzero(positive_D))

if rank_D == 0:
    raise RuntimeError("没有检测到矩阵 D 的正特征值")

B = Z[:, positive_D] * np.sqrt(delta[positive_D])[None, :]
mu_sum = mu[:, None] + mu[None, :]

reduced_size = rank_D*ndof
number_nonunit = 2*rank_D*ndof - rank_D**2
number_unit = ndof**2 - number_nonunit


# 每行依次记录：
# gamma, 矩阵阶数, 等于1的特征值个数, 大于1的特征值个数,
# 小于1的特征值个数, 最小特征值, 最大特征值
spectrum_summary = []


# ============================================================
# 对四个 gamma 计算预条件矩阵的特征值
# ============================================================
for exponent, gamma in zip(gamma_exponents, gamma_values):
    gamma = float(gamma)
    gamma_tag = f"{gamma:.0e}"

    print("-" * 60)
    print(f"正在计算 gamma = 10^({int(exponent)}) = {gamma_tag}")

    # 在 X tensor X 基下，预条件矩阵的对角部分为
    # p_ij = mu_i + mu_j + gamma*(mu_i + mu_j)^2。
    p = mu_sum + gamma*mu_sum**2
    inverse_p = 1.0/p

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
    theta_nonzero = np.maximum(theta[-number_nonunit:], 0.0)

    eigenvalues = np.concatenate(
        (
            np.ones(number_unit),
            1.0 + theta_nonzero,
        )
    )

    eigenvalue_file = data_dir / f"eigenvalues_gamma_{gamma_tag}.npy"
    np.save(eigenvalue_file, eigenvalues)

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
            gamma,
            ndof**2,
            number_one,
            number_greater,
            number_less,
            eigenvalues[0],
            eigenvalues[-1],
        ]
    )

    print(f"特征值文件: {eigenvalue_file.name}")
    print(f"lambda = 1 的个数: {number_one}")
    print(f"lambda > 1 的个数: {number_greater}")
    print(f"lambda < 1 的个数: {number_less}")
    print(f"最小特征值: {eigenvalues[0]:.16e}")
    print(f"最大特征值: {eigenvalues[-1]:.16e}")


spectrum_summary = np.asarray(spectrum_summary, dtype=float)
np.save(data_dir / "spectrum_summary.npy", spectrum_summary)


print("=" * 60)
print("全部谱数据已经保存到:")
print(data_dir)
print("保存的文件包括:")
print("  N.npy")
print("  gamma_exponents.npy")
print("  gamma_values.npy")
print("  spectrum_summary.npy")
for gamma in gamma_values:
    print(f"  eigenvalues_gamma_{float(gamma):.0e}.npy")
