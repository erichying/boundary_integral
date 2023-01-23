from scipy.integrate import quad
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np

eV = 1.6e-19
hbar = 1.05e-34
mass = 0.067 * 9.1e-31
W = 5e-9
D = 10e-9
L = 10e-9
M = N = K = 10


def calculate_transmission(e):
    energy = e * eV
    beta = np.array([np.sqrt(energy * 2 * mass / hbar ** 2 - ((n + 1) * np.pi / W) ** 2 + 0j) for n in range(N)])
    gamma = np.array([np.sqrt(energy * 2 * mass / hbar ** 2 - ((m + 1) * np.pi / L) ** 2 + 0j) for m in range(M)])
    alpha_d = np.array([np.sqrt(energy * 2 * mass / hbar ** 2 - ((k + 1) * np.pi / D) ** 2 + 0j) for k in range(K)])
    alpha_l = np.array([np.sqrt(energy * 2 * mass / hbar ** 2 - ((k + 1) * np.pi / L) ** 2 + 0j) for k in range(K)])

    b_d = 1 / np.sqrt(alpha_d)
    b_l = 1 / np.sqrt(alpha_l)
    c = 1 / np.sqrt(gamma)
    d = 1 / np.sqrt(beta)

    def f1(n, k):
        def fun(y, n, k):
            return np.sin((n + 1) * np.pi / W * (y + W / 2)) * np.sin((k + 1) * np.pi / D * (y + D / 2))

        def real(y, n, k):
            return np.real(fun(y, n, k))

        def imag(y, n, k):
            return np.imag(fun(y, n, k))

        real_integral = quad(real, -W / 2, W / 2, args=(n, k))
        imag_integral = quad(imag, -W / 2, W / 2, args=(n, k))
        return real_integral[0] + 1j * imag_integral[0]

    M1 = M2 = M3 = M4 = np.zeros((K, N), complex)
    for n in range(0, N):
        for k in range(0, K):
            M1[k, n] = 1j * alpha_d[k] * d[n] * b_d[k] * np.exp(-1j * alpha_d[k] * L / 2) * f1(n, k)
            M2[k, n] = 1j * beta[n] * d[n] * b_d[k] * np.exp(-1j * alpha_d[k] * L / 2) * f1(n, k)
            M3[k, n] = 1j * alpha_d[k] * d[n] * b_d[k] * np.exp(1j * alpha_d[k] * L / 2) * f1(n, k)
            M4[k, n] = 1j * beta[n] * d[n] * b_d[k] * np.exp(1j * alpha_d[k] * L / 2) * f1(n, k)

    def f2(m, k):
        def fun(y, m, k):
            return np.sinh(1j * gamma[m] * (y - D / 2)) * np.sin((k + 1) * np.pi / D * (y + D / 2))

        def real(y, m, k):
            return np.real(fun(y, m, k))

        def imag(y, m, k):
            return np.imag(fun(y, m, k))

        real_integral = quad(real, W / 2, D / 2, args=(m, k))
        imag_integral = quad(imag, W / 2, D / 2, args=(m, k))
        return real_integral[0] + 1j * imag_integral[0]

    M5 = M6 = np.zeros((K, M), complex)
    for m in range(0, M):
        for k in range(0, K):
            M5[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.exp(-1j * alpha_d[k] * L / 2) * f2(m, k)
            M6[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.cos((m + 1) * np.pi) * np.exp(1j * alpha_d[k] * L / 2) * f2(m, k)

    def f3(m, k):
        def fun(y, m, k):
            return np.sinh(1j * gamma[m] * (y + D / 2)) * np.sin((k + 1) * np.pi / D * (y + D / 2))

        def real(y, m, k):
            return np.real(fun(y, m, k))

        def imag(y, m, k):
            return np.imag(fun(y, m, k))

        real_integral = quad(real, -D / 2, -W / 2, args=(m, k))
        imag_integral = quad(imag, -D / 2, -W / 2, args=(m, k))
        return real_integral[0] + 1j * imag_integral[0]

    M7 = M8 = np.zeros((K, M), complex)
    for m in range(0, M):
        for k in range(0, K):
            M7[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.exp(-1j * alpha_d[k] * L / 2) * f3(m, k)
            M8[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.cos((m + 1) * np.pi) * np.exp(1j * alpha_d[k] * L / 2) * f3(m, k)

    M9 = M10 = M11 = M12 = np.zeros((M, K), complex)
    for m in range(0, M):
        for k in range(0, K):
            M9[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.exp(1j * alpha_d[k] * L / 2) * f2(m, k)
            M10[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.cos((m + 1) * np.pi) * np.exp(-1j * alpha_d[k] * L / 2) * f2(m, k)
            M11[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.exp(1j * alpha_d[k] * L / 2) * f3(m, k)
            M12[k, m] = c[m] * b_d[k] * (m + 1) * np.pi / L * np.cos((m + 1) * np.pi) * np.exp(-1j * alpha_d[k] * L / 2) * f3(m, k)

    def g1(n, k):
        def fun(y, n, k):
            return np.sin((n + 1) * np.pi / W * (y + W / 2)) * np.exp(1j * alpha_l[k] * y)

        def real(y, n, k):
            return np.real(fun(y, n, k))

        def imag(y, n, k):
            return np.imag(fun(y, n, k))

        real_integral = quad(real, -W / 2, W / 2, args=(n, k))
        imag_integral = quad(imag, -W / 2, W / 2, args=(n, k))
        return real_integral[0] + 1j * imag_integral[0]

    M13 = np.zeros((K, N), complex)
    M14 = np.zeros((K, N), complex)
    for n in range(0, N):
        for k in range(0, K):
            M13[k, n] = d[n] * b_l[k] * (k + 1) * np.pi / L * g1(n, k)
            M14[k, n] = d[n] * b_l[k] * (k + 1) * np.pi / L * np.cos((k + 1) * np.pi) * g1(n, k)

    def g2(n, k):
        def fun(y, n, k):
            return np.sin((n + 1) * np.pi / W * (y + W / 2)) * np.exp(-1j * alpha_l[k] * y)

        def real(y, n, k):
            return np.real(fun(y, n, k))

        def imag(y, n, k):
            return np.imag(fun(y, n, k))

        real_integral = quad(real, -W / 2, W / 2, args=(n, k))
        imag_integral = quad(imag, -W / 2, W / 2, args=(n, k))
        return real_integral[0] + 1j * imag_integral[0]

    M15 = np.zeros((K, N), complex)
    M16 = np.zeros((K, N), complex)
    for n in range(0, N):
        for k in range(0, K):
            M15[k, n] = d[n] * b_l[k] * (k + 1) * np.pi / L * g2(n, k)
            M16[k, n] = d[n] * b_l[k] * (k + 1) * np.pi / L * np.cos((k + 1) * np.pi) * g2(n, k)

    row1 = np.concatenate((M1 + M2, M3 - M4, -M5 - M6, -M7 - M8), axis=1)
    row2 = np.concatenate((M4 - M3, -M1 - M2, -M9 - M10, -M11 - M12), axis=1)
    row3 = np.concatenate((M13, M14, np.zeros((K, M)), np.zeros((K, M))), axis=1)
    row4 = np.concatenate((M15, M16, np.zeros((K, M)), np.zeros((K, M))), axis=1)

    A = np.concatenate((row1, row2, row3, row4))

    I = np.zeros((N, 1), complex)
    I[0] = 1e-2
    rhs = np.concatenate(((M2 - M1) @ I, (M4 + M3)@I, -M13@I, -M15@I))

    x = np.linalg.solve(A, rhs)
    Rn = x[0:N]
    Tn = x[N:2*N]
    Un = x[2*N:2*N+M]
    Dn = x[2*N+M: 2*N+2*M]

    return np.sum(Tn * np.conj(Tn)), np.sum(Rn * np.conj(Rn))


v = np.arange(0.1, 0.20, 0.001)
t = []
r = []
results = Parallel(n_jobs=-1)(delayed(calculate_transmission)(i) for i in v)
for result in results:
    transmission, reflection = result
    t.append(transmission)
    r.append(reflection)

plt.plot(v, t)
plt.show()