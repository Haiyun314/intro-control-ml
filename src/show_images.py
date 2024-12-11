import matplotlib.pyplot as plt
from matplotlib import animation
from setting import *
import numpy as np
import torch

def show_image(*images, image_name: list, save: bool | int = False,
               vmin: float = 0.0, vmax: float = 1.0):
    """
    Args:
        images: numpy matrix
        image_name (list): such as [image1, image2, image3, ...], based on the given images
        save (int): save = 1 to save the imaegs, 
    """
    number_of_images = len(images)
    if number_of_images == 1:
        plt.figure()
        plt.imshow(images[0], cmap='hot', vmin=vmin, vmax=vmax)
        plt.title(image_name[0])
        plt.colorbar(label='Temperature')
        plt.xlabel('x')
        plt.ylabel('y')
        if save:
            image_path = os.path.join(results_path, f'{image_name[0]}.png')
            plt.savefig(image_path)
        plt.show()
        
    else:
        _, ax = plt.subplots(nrows=1, ncols= number_of_images, figsize=(5 * number_of_images, 5))
        for i in range(number_of_images):
            ax[i].imshow(images[i], vmin=vmin, vmax=vmax)
            ax[i].set_title(image_name[i])
        if save:
            image_path = os.path.join(results_path, f'{image_name[-1]}.png')
            plt.savefig(image_path)
        plt.show()


def show_3d(image, image_name: list, save: int = 0):
    # 3D plot of the temperature distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
    ax.plot_surface(X, Y, image, cmap='hot')
    plt.title(f'{image_name[0]}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')
    if save:
        image_path = os.path.join(results_path, f'{image_name[0]}.png')
        plt.imsave(image_path, image_name[0])
    plt.show()


# Visualization of the heat distribution at different time steps
def plot_heat_distribution(model, t_value, ax, heatmap):
    # Create a grid of points in the spatial domain
    x = torch.linspace(-1, 1, 50).unsqueeze(1).repeat(1, 50)
    y = torch.linspace(-1, 1, 50).unsqueeze(0).repeat(50, 1)
    t = torch.ones_like(x) * t_value  # Set all t points to the same time value

    # Flatten the grid to pass through the model
    x_flat = x.flatten().unsqueeze(1)
    y_flat = y.flatten().unsqueeze(1)
    t_flat = t.flatten().unsqueeze(1)

    # Predict the temperature (u)
    with torch.no_grad():
        u = model(x_flat, y_flat, t_flat).reshape(x.shape).detach().numpy()

    # Update the heatmap data
    heatmap.set_data(u)

# Function to create animation for the heat distribution
def create_animation(model,image_name: list, save: int = 0, num_frames=50):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a static colorbar for the heatmap
    x = torch.linspace(-1, 1, 50).unsqueeze(1).repeat(1, 50)
    y = torch.linspace(-1, 1, 50).unsqueeze(0).repeat(50, 1)
    t = torch.ones_like(x) * 0  # Initial time
    x_flat = x.flatten().unsqueeze(1)
    y_flat = y.flatten().unsqueeze(1)
    t_flat = t.flatten().unsqueeze(1)

    u = model(x_flat, y_flat, t_flat).reshape(x.shape).detach().numpy()

    # Create initial heatmap and colorbar
    heatmap = ax.imshow(u, extent=(-1, 1, -1, 1), origin='lower', cmap='hot')
    cbar = plt.colorbar(heatmap, ax=ax, label="Temperature")

    def update(frame):
        t_value = frame / num_frames  # Vary time from 0 to 1
        plot_heat_distribution(model, t_value, ax, heatmap)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    plt.close()
    if save:
        image_path = os.path.join(results_path, f'{image_name[0]}.png')
        os.makedirs(results_path, exist_ok=True)
        ani.save(image_path, writer='imagemagick', fps=10)

    return ani


def show_gif(data, image_name: list, save: int = 0):
    assert len(data.shape) == 3, f'to show a gif, we need a sequence of images with shape (n, m, number_of_frames), instead {data.shape}'
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clf()
        ax.imshow(data[frame])
        ax.set_title(image_name)
    
    ani = animation.FuncAnimation(fig, update)
    plt.show