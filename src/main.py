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

    init_output = np.zeros_like(x, dtype= np.float32)
    init_output[0, :] = 1
    init_output = tf.reshape(init_output, (-1, 1))

    x = tf.reshape(x, (-1, 1))
    y = tf.reshape(y, (-1, 1))
    t = tf.reshape(tf.repeat(tf.range(0, 10, dtype= tf.float32)/10, repeats= 250), (-1, 1)) # time from 0 to 10
    t = tf.random.shuffle(t, seed= 1)
    t_init = tf.zeros_like(t, dtype= tf.float32)
    t_test = tf.ones_like(t, dtype= tf.float32) * 2.5

    pred_data = np.concatenate([x, y, t], axis= -1) # train
    input_init = np.concatenate([x, y, t_init], axis= -1) # train init state
    test_data = np.concatenate([x, y, t_test], axis= -1) # test the reult at time = 1

    # bound mask
    bound_l = pred_data[:, 0] == -1
    bound_r = pred_data[:, 0] == 1
    bound_b = pred_data[:, 1] == -1
    bound_t = pred_data[:, 1] == 1
    bound = bound_l + bound_r + bound_b + bound_t

    interior_input = tf.convert_to_tensor(pred_data[~bound], dtype= tf.float32)
    bound_input = tf.convert_to_tensor(pred_data[bound], dtype= tf.float32)
    bound_output = tf.cast(tf.reshape(bound_l[bound], (-1, 1)), dtype= tf.float32)
    output_init = tf.zeros_like(x, dtype= tf.float32)

    alpha = 0.01

    epochs = 5000
    if train:
        # Model, optimizer, and training loop
        model = nn_model(input_shape=(3,), layers=[16, 32, 32, 16])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        for epoch in range(epochs):
            loss_value = train_step(model,input_init, bound_input, interior_input,output_init ,bound_output, alpha, optimizer)
            print(f"Epoch {epoch+1}, Loss: {loss_value.numpy()}")
        
        model.save(os.path.join(models_path, 'tf_pinns.h5'))
    else: 
        model = tf.keras.models.load_model(os.path.join(models_path, 'tf_pinns.h5'))

    u = model(test_data)
    return u


def main():
    # pinns_result = simu_pinns()

    # fdm_result = simu_fdm()
    # image_names = ['fdm_result']
    # show_image(fdm_result, image_name= image_names, save= 1)

    u = simu_pinns_tf(0)
    result = tf.reshape(u, (50, 50))
    show_image(result, image_name=['pinns_tf'], save= 1)

if __name__ == "__main__":
    main()