import os

# Parameters for the 2D heat equation
Nx, Ny = 50, 50   # Number of grid points in the x and y directions
Lx, Ly = 1.0, 1.0 # Physical size of the plate in the x and y directions
alpha = 0.01      # Thermal diffusivity
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = 0.001        # Time step size
Nt = 200          # Number of time steps
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
src_path = os.path.join(root_path, 'src')
test_path = os.path.join(root_path, 'test')
results_path = os.path.join(root_path, 'results')

