import numpy as np
from scipy.sparse import diags
from shenfun import FunctionSpace, TrialFunction, TestFunction, grad, inner


# 固定 gamma = 1；可以输入多个 N，例如：7 8 10 12 16 24 32
gamma = 1.0
text = input("请输入 N（多个 N 用空格或逗号分隔，且 N >= 7）: ")
N_values = [int(value) for value in text.replace(",", " ").split()]

if len(N_values) == 0:
    raise ValueError("至少需要输入一个 N")

if any(N < 7 for N in N_values):
    raise ValueError("Theorem 4.1 要求 N >= 7")


results = []

for N in N_values:
    ndof = N - 3
    I = np.eye(ndof)

    # phi_{N-4} 含 L_N，因此 Shenfun 空间需要 N+1 个 Legendre 模态。
    V = FunctionSpace(N + 1, "Legendre", bc=(0, 0, 0, 0))
    u = TrialFunction(V)
    v = TestFunction(V)

    M_shen = inner(u, v).diags(format="csr")
    S_shen = inner(grad(u), grad(v)).diags(format="csr")

    k = np.arange(ndof, dtype=float)
    d = 1.0 / np.sqrt(2.0 * (2.0*k + 3.0)**2 * (2.0*k + 5.0))
    D_scale = diags(d, offsets=0, shape=(ndof, ndof), format="csr")

    M = (D_scale @ M_shen @ D_scale).toarray()
    S = (D_scale @ S_shen @ D_scale).toarray()

    # Q = S M^{-1} S；用 solve 代替显式求逆。
    Q = S @ np.linalg.solve(M, S)
    Q = 0.5 * (Q + Q.T)

    Mdiv = np.kron(M, S) + np.kron(S, M)
    Sdiv = np.kron(I, M) + 2.0*np.kron(S, S) + np.kron(M, I)
    Sdiv_tilde = (
        np.kron(Q, M)
        + 2.0*np.kron(S, S)
        + np.kron(M, Q)
    )

    A = Mdiv + gamma*Sdiv
    P = Mdiv + gamma*Sdiv_tilde

    # 求对称广义特征值问题 A x = lambda P x。
    # P = L L^T，将其变换为 L^{-1} A L^{-T} 的普通对称特征值问题。
    L = np.linalg.cholesky(P)
    Y = np.linalg.solve(L, A)
    C = np.linalg.solve(L, Y.T).T
    C = 0.5 * (C + C.T)
    eigenvalues = np.linalg.eigvalsh(C)

    tolerance = 1.0e-10 * max(1.0, np.max(np.abs(eigenvalues)))
    index_one = np.abs(eigenvalues - 1.0) <= tolerance
    index_greater = eigenvalues > 1.0 + tolerance
    index_less = eigenvalues < 1.0 - tolerance

    number_one = int(np.count_nonzero(index_one))
    number_greater = int(np.count_nonzero(index_greater))
    number_less = int(np.count_nonzero(index_less))

    number_one_theoretical = (N - 7)**2
    number_greater_theoretical = 8*N - 40

    if number_one == 0:
        delta_one = np.nan
    else:
        delta_one = float(np.max(np.abs(eigenvalues[index_one] - 1.0)))

    results.append(
        (
            N,
            ndof**2,
            number_one,
            number_one_theoretical,
            number_greater,
            number_greater_theoretical,
            number_less,
            delta_one,
        )
    )


print("\n固定 gamma = 1 时的预条件矩阵谱验证")
print(
    f"{'N':>4} {'(N-3)^2':>10} {'m1_num':>10} {'m1_th':>10} "
    f"{'m>1_num':>10} {'m>1_th':>10} {'m<1':>8} {'delta_1':>14}"
)
print("-" * 96)

for row in results:
    N, size, m1_num, m1_th, mg_num, mg_th, ml_num, delta = row
    delta_text = "--" if np.isnan(delta) else f"{delta:.4e}"
    print(
        f"{N:4d} {size:10d} {m1_num:10d} {m1_th:10d} "
        f"{mg_num:10d} {mg_th:10d} {ml_num:8d} {delta_text:>14}"
    )


print("\nLaTeX 表格数据行")
for row in results:
    N, size, m1_num, m1_th, mg_num, mg_th, ml_num, delta = row
    delta_text = "--" if np.isnan(delta) else f"{delta:.4e}"
    print(
        f"{N} & {size} & {m1_num} & {m1_th} & {mg_num} & "
        f"{mg_th} & {ml_num} & ${delta_text}$ \\\\"
    )
