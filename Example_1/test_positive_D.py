import numpy as np
from scipy.sparse import diags
from shenfun import FunctionSpace, TrialFunction, TestFunction, grad, inner


# 输入论文中的 N；基函数指标为 k = 0, 1, ..., N-4
N = 12

if N < 4:
    raise ValueError("N 必须满足 N >= 4")


# phi_{N-4} 中含有 L_N，因此 Shenfun 正交空间需要 N+1 个模态：
# L_0, L_1, ..., L_N
V = FunctionSpace(
    N + 1,
    "Legendre",
    bc=(0, 0, 0, 0),
)

u = TrialFunction(V)
v = TestFunction(V)


# Shenfun 内置 ShenBiharmonic 基没有论文中的归一化因子 d_k
M_shen = inner(u, v).diags(format="csr")
S_shen = inner(grad(u), grad(v)).diags(format="csr")


# 论文中的归一化因子
# d_k = 1/sqrt(2*(2k+3)^2*(2k+5))
ndof = N - 3
k = np.arange(ndof, dtype=float)
d = 1.0 / np.sqrt(2.0 * (2.0*k + 3.0)**2 * (2.0*k + 5.0))
D_scale = diags(d, offsets=0, shape=(ndof, ndof), format="csr")


# phi_k = d_k * psi_k，因此矩阵需要左右各乘一次 D_scale
M = (D_scale @ M_shen @ D_scale).tocsr()
S = (D_scale @ S_shen @ D_scale).tocsr()

M.eliminate_zeros()
S.eliminate_zeros()


# 转成普通二维数组，便于查看所有矩阵元素
M_array = M.toarray()
S_array = S.toarray()


# 由 I = S M^{-1} S + D 计算引理 4.1 中的矩阵 D。
# 使用 solve(M, S) 代替显式计算 inv(M)，数值上更加稳定。
I = np.eye(ndof)
D_array = I - S_array @ np.linalg.solve(M_array, S_array)

# 理论上 D 是对称半正定矩阵；消除浮点运算产生的微小非对称误差。
D_array = 0.5 * (D_array + D_array.T)

# 对称矩阵使用 eigvalsh，返回从小到大排列的实特征值。
D_eigenvalues = np.linalg.eigvalsh(D_array)

# 使用与矩阵规模和最大特征值相关的容差计算数值秩。
D_tolerance = (
    ndof
    * np.finfo(float).eps
    * max(1.0, np.max(np.abs(D_eigenvalues)))
)
D_rank = int(np.count_nonzero(np.abs(D_eigenvalues) > D_tolerance))

np.set_printoptions(
    precision=12,
    suppress=True,
    linewidth=200,
)

print("\n矩阵维数:", (ndof, ndof))

print("\nS = (phi_j', phi_i'):")
print(S_array)

print("\nM = (phi_j, phi_i):")
print(M_array)

print("\nD = I - S M^(-1) S:")
print(D_array)

print("\nD 的特征值（从小到大）:")
print(D_eigenvalues)

print("\nD 的数值秩:")
print(D_rank)

print("\n理论秩 min(4, N-3):")
print(min(4, N - 3))

print("\n数值秩判定容差:")
print(D_tolerance)

print("\nS 的非零对角线偏移:")
print(sorted(set(S.tocoo().col - S.tocoo().row)))

print("\nM 的非零对角线偏移:")
print(sorted(set(M.tocoo().col - M.tocoo().row)))
