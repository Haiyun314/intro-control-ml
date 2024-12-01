import os
import numpy as np

# Parameters for the 2D heat equation
Nx, Ny = 50, 50   # Number of grid points in the x and y directions
Lx, Ly = 1.0, 1.0 # Physical size of the plate in the x and y directions
alpha = 0.01      # Thermal diffusivity
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
Nt = 1000          # Number of time steps
dt = 1/Nt       # Time step size

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
src_path = os.path.join(root_path, 'src')
test_path = os.path.join(root_path, 'test')
results_path = os.path.join(root_path, 'results')
# Generate initial condition and boundary conditions
initial_u = np.zeros((Nx, Ny))  # Initially no heat
left_bc = np.ones(Nt)  # Heat source on left boundary
right_bc = np.zeros(Nt)         # No heat on the right boundary
top_bc = np.zeros(Nt)           # No heat on the top boundary
bottom_bc = np.ones(Nt) # Heat source on bottom boundary

