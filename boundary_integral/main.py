import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

eV = 1.6e-19
hbar = 1.05e-34
mass = 0.067 * 9.1e-31
W = 5e-9
D = 10e-9
L = 10e-9
N = 10


def calculate_transmission(e):
    energy = e * eV
    beta = np.array([np.sqrt(energy * 2 * mass / hbar ** 2 - ((n + 1) * np.pi / 2 / W) ** 2 + 0j) for n in range(N)])
    gamma = np.array([np.sqrt(energy * 2 * mass / hbar ** 2 - ((n + 1) * np.pi / 2 / L) ** 2 + 0j) for n in range(N)])

    def psi_up(m, y, z):
        return np.sinh(1j * gamma[m] * (y - D)) * np.sin(m * np.pi / 2 / L * (z + L))

    def psi_down(m, y, z):
        return np.sinh(1j * gamma[m] * (y + D)) * np.sin(m * np.pi / 2 / L * (z + L))

    def psi_left_in(n, y, z):
        return np.exp(1j * beta[n] * (z + L)) * np.sin(n * np.pi / 2 / W * (y + W))

    def psi_left_out(n, y, z):
        return np.exp(-1j * beta[n] * (z + L)) * np.sin(n * np.pi / 2 / W * (y + W))

    def psi_right(n, y, z):
        return np.exp(1j * beta[n] * (z - L)) * np.sin(n * np.pi / 2 / W * (y + W))

