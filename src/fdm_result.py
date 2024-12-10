import numpy as np
from setting import *


# Function to simulate the 2D heat equation using finite difference
def simulate_heat_eq_2d(initial_u, left_bc, right_bc, top_bc, bottom_bc, Nt, Nx, Ny, alpha, dx, dy, dt):
    u = np.zeros((Nt, Nx, Ny))
    u[0, :, :] = initial_u
    for t in range(1, Nt):
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                u[t, i, j] = u[t - 1, i, j] + alpha * dt * (
                        (u[t - 1, i + 1, j] - 2 * u[t - 1, i, j] + u[t - 1, i - 1, j]) / dx ** 2 +
                        (u[t - 1, i, j + 1] - 2 * u[t - 1, i, j] + u[t - 1, i, j - 1]) / dy ** 2
                )
        # Apply boundary conditions
        u[t, 0, :] = left_bc[t]  # Left boundary
        u[t, -1, :] = right_bc[t]  # Right boundary
        u[t, :, 0] = bottom_bc[t]  # Bottom boundary
        u[t, :, -1] = top_bc[t]  # Top boundary
    return u
