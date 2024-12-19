from setting import *
import numpy as np
from fdm_result import simulate_heat_eq_2d
from show_images import show_image, create_animation
import torch
from pinns import HeatEquationNN, train_heat_equation_model
import tensorflow as tf
from pinns_tf import nn_model, train_step, train_with_early_stopping, plot_loss_history
from datetime import datetime
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def compute_l1_error(pinn_result, fdm_result):
    # Compute L1 norm (mean absolute error) between PINN and FDM results
    return np.mean(np.abs(pinn_result - fdm_result))

def compute_l2_error(pinn_result, fdm_result):
    # Compute L2 norm (mean squared error) between PINN and FDM results
    return np.sqrt(np.mean(np.square(pinn_result - fdm_result)))

def simu_fdm():
    # Simulate the 2D heat equation (uncontrolled case)
    u_uncontrolled = simulate_heat_eq_2d(initial_u, left_bc, right_bc, top_bc, bottom_bc, Nt, Nx, Ny, alpha, dx, dy, dt)
    return u_uncontrolled  # Return the full sequence for animation

def simu_pinns_tf(train: bool = True,
                  alpha: float = 0.01,
                  epochs: int = 5000,
                  optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=0.001),
                  activation_function='tanh',
                  patience: int = 100,  # Early stopping patience
                  use_early_stopping: bool = True,
                  dropout_rate: float = 0.1,  # rate of neurons to dropout
                  dt: datetime = datetime.now(),
                  verbose: bool = True):
    x = tf.linspace(-1, 1, 50)
    y = tf.linspace(-1, 1, 50)
    x, y = tf.meshgrid(x, y)

    init_output = np.zeros_like(x, dtype=np.float32)
    init_output[0, :] = 1
    init_output = tf.reshape(init_output, (-1, 1))

    x = tf.reshape(x, (-1, 1))
    y = tf.reshape(y, (-1, 1))
    t = tf.linspace(0, 10, 100)  # 100 time steps for animation
    t_init = tf.zeros_like(x, dtype=tf.float32)
    pred_data = np.concatenate([x, y, t.numpy().reshape(-1, 1)], axis=-1)  # Time evolution input data

    if train:
        # Model, optimizer, and training loop
        model = nn_model(input_shape=(3,), layers=[16, 32, 32, 16],
                         activation_function=activation_function,
                         dropout_rate=dropout_rate)

        if use_early_stopping:
            model = train_with_early_stopping(
                model=model,
                init_points=np.concatenate([x, y, t_init], axis=-1),
                input_bound=None,
                input_interior=tf.convert_to_tensor(pred_data, dtype=tf.float32),
                output_init=init_output,
                output_bound=None,
                alpha=alpha,
                optimizer=optimizer,
                epochs=epochs,
                patience=patience,
                img_name='PINN_loss_history.png',
                verbose=True
            )
        model.save(os.path.join(models_path, 'tf_pinns.h5'))
    else:
        model = tf.keras.models.load_model(os.path.join(models_path, 'tf_pinns.h5'))

    # Generate predictions for animation
    results = []
    for t_i in t:
        t_grid = tf.fill(x.shape, t_i.numpy())  # Fill time grid for current time step
        input_data = tf.concat([x, y, t_grid], axis=1)
        u_pred = model(input_data)  # Predict heat distribution
        results.append(u_pred.numpy().reshape((50, 50)))

    return results

def create_animation(results, filename="D:\intro-control-ml-main (2)\intro-control-ml-main\src\pinns_animation.mp4", save=True):
    # Create an animation of the heat distribution over time
    fig, ax = plt.subplots()
    cax = ax.imshow(results[0], extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    fig.colorbar(cax)
    ax.set_title('Time: 0.00')

    def update(frame):
        cax.set_array(results[frame])
        ax.set_title(f'Time: {frame / 10:.2f}')
        return cax,

    anim = FuncAnimation(fig, update, frames=len(results), interval=100, blit=True)

    if save:
        anim.save(filename, writer='ffmpeg', dpi=300)
    plt.show()

def main():
    # Run FDM simulation for comparison
    fdm_result = simu_fdm()
    show_image(fdm_result[-1], image_name=['fdm_result'], save=True)  # Show final state
    create_animation(fdm_result, filename="D:\intro-control-ml-main (2)\intro-control-ml-main\src\fdm_animation.mp4", save=True)  # Animate FDM results

    # Run PINN simulation and generate animation
    execution_parameters = {
        'train': True,
        'alpha': 0.01,
        'epochs': 5000,
        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
        'activation_function': 'tanh',
        'dropout_rate': 0.1,
        'patience': 100,
        'use_early_stopping': True,
        'verbose': True
    }

    pinn_results = simu_pinns_tf(**execution_parameters)
    create_animation(pinn_results, filename="D:\intro-control-ml-main (2)\intro-control-ml-main\src\pinns_animation.mp4", save=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from pinns_tf import nn_model, train_with_early_stopping
from setting import *
from fdm_result import simulate_heat_eq_2d

def create_animation(results, filename="heat_animation.mp4", save=True):
    fig, ax = plt.subplots()
    cax = ax.imshow(results[0], extent=[0, Lx, 0, Ly], origin='lower', cmap='hot')
    fig.colorbar(cax)
    ax.set_title('Time: 0.00')

    def update(frame):
        cax.set_array(results[frame])
        ax.set_title(f'Time: {frame * dt:.2f}')
        return cax,

    anim = FuncAnimation(fig, update, frames=len(results), interval=100, blit=True)

    if save:
        anim.save(filename, writer='ffmpeg', dpi=300)
    plt.show()

# Function to simulate the 2D heat equation
def simulate_heat_eq_2d(initial_u, left_bc, right_bc, top_bc, bottom_bc, Nt, Nx, Ny, alpha, dx, dy, dt):
    # Initialize the temperature array
    u = initial_u.copy()
    results = [u.copy()]

    for n in range(Nt):
        # Update the temperature using the finite difference method
        u_new = u.copy()
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                u_new[i, j] = (u[i, j] + 
                               alpha * dt / dx**2 * (u[i+1, j] - 2*u[i, j] + u[i-1, j]) + 
                               alpha * dt / dy**2 * (u[i, j+1] - 2*u[i, j] + u[i, j-1]))
        
        # Apply boundary conditions
        u_new[0, :] = left_bc[n]  # Left boundary
        u_new[-1, :] = right_bc[n]  # Right boundary
        u_new[:, 0] = top_bc[n]  # Top boundary
        u_new[:, -1] = bottom_bc[n]  # Bottom boundary
        
        u = u_new
        results.append(u.copy())

    return results

def main():
    # Generate initial condition and boundary conditions
    initial_u = np.zeros((Nx, Ny))  # Initially no heat
    left_bc = np.zeros(Nt)  # Heat source on left boundary
    right_bc = np.zeros(Nt)  # No heat on the right boundary
    top_bc = np.zeros(Nt)  # No heat on the top boundary
    bottom_bc = np.ones(Nt)  # Heat source on bottom boundary

    # Run the simulation
    results = simulate_heat_eq_2d(initial_u, left_bc, right_bc, top_bc, bottom_bc, Nt, Nx, Ny, alpha, dx, dy, dt)

    # Create the animation
    output_file = os.path.join(os.path.dirname(__file__), "heat_animation.mp4")  # Save in the current directory
    create_animation(results, filename=output_file, save=True)

if __name__ == "__main__":
    main()
