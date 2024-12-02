from setting import *
import numpy as np
from fdm_result import simulate_heat_eq_2d
from show_images import show_image, create_animation
import torch
from pinns import HeatEquationNN, train_heat_equation_model
import tensorflow as tf
from pinns_tf import nn_model, train_step

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

def simu_pinns_tf(train: int= 1):
    x = tf.linspace(-1, 1, 50)
    y = tf.linspace(-1, 1, 50)
    x, y = tf.meshgrid(x, y)
    x = tf.reshape(x, (-1, 1))
    y = tf.reshape(y, (-1, 1))
    t = tf.reshape(tf.ones_like(y), (-1, 1))
    pred_data = np.concatenate([x, y, t], axis= -1)

    # bound mask
    bound_l = pred_data[:, 0] == -1
    bound_r = pred_data[:, 0] == 1
    bound_b = pred_data[:, 1] == -1
    bound_t = pred_data[:, 1] == 1
    bound = bound_l + bound_r + bound_b + bound_t

    interior_input = tf.convert_to_tensor(pred_data[~bound], dtype= tf.float32)
    bound_input = tf.convert_to_tensor(pred_data[bound], dtype= tf.float32)
    bound_output = tf.cast(tf.reshape(bound_l[bound], (-1, 1)), dtype= tf.float32)


    alpha = 0.01

    epochs = 100
    if train:
        # Model, optimizer, and training loop
        model = nn_model(input_shape=(3,), layers=[32, 64, 32])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        for epoch in range(epochs):
            loss_value = train_step(model, bound_input, interior_input, bound_output, alpha, optimizer)
            print(f"Epoch {epoch+1}, Loss: {loss_value.numpy()}")
        
        model.save(os.path.join(models_path, 'tf_pinns.h5'))
    else: 
        model = tf.keras.models.load_model(os.path.join(models_path, 'tf_pinns.h5'))

    u = model(pred_data)
    return u


def main():
    # pinns_result = simu_pinns()

    fdm_result = simu_fdm()
    image_names = ['fdm_result']
    show_image(fdm_result, image_name= image_names, save= 1)

    # u = simu_pinns_tf(0)
    # result = tf.reshape(u, (50, 50))
    # show_image(result, image_name=['pinns_tf'], save= 1)

if __name__ == "__main__":
    main()