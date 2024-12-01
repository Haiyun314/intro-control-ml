from setting import *
import numpy as np
from fdm_result import simulate_heat_eq_2d
from show_images import show_image, create_animation
import torch
from pinns import HeatEquationNN, train_heat_equation_model

def simu_fdm():
    # Simulate the 2D heat equation (uncontrolled case)
    u_uncontrolled = simulate_heat_eq_2d(initial_u, left_bc, right_bc, top_bc, bottom_bc, Nt, Nx, Ny, alpha, dx, dy, dt) # this is a sequence of updating
    return u_uncontrolled[-1] # the final result


def simu_pinns():
    # Initialize and train the model
    model = HeatEquationNN()

    # here try different optimization methods
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # alpha = 0.01  # Thermal diffusivity
    alpha = 0.1  # Thermal diffusivity

    train_heat_equation_model(model, optimizer, alpha)
    
    # Create and display the animation
    ani = create_animation(model, ['pinns_result'], save= 1)


def main():
    pinns_result = simu_pinns()

    # fdm_result = simu_fdm()
    # image_names = ['result']
    # show_image(fdm_result, image_name= image_names)


if __name__ == "__main__":
    main()