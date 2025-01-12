from setting import *
import numpy as np
from fdm_result import simulate_heat_eq_2d
from show_images import show_image, create_animation
import torch
from pinns import HeatEquationNN, train_heat_equation_model
import tensorflow as tf
from pinns_tf import nn_model, train_step, train_with_early_stopping, plot_loss_history
from datetime import datetime
import pandas as pd
from fourier_result import get_heat_eq_solution

def compute_l1_error(pinn_result, fdm_result):
    # Compute L1 norm (mean absolute error) between PINN and FDM results
    return np.mean(np.abs(pinn_result - fdm_result))

def compute_l2_error(pinn_result, fdm_result):
    # Compute L2 norm (mean squared error) between PINN and FDM results
    return np.sqrt(np.mean(np.square(pinn_result - fdm_result)))

def simu_fdm():
    # Simulate the 2D heat equation (uncontrolled case)
    u_uncontrolled = simulate_heat_eq_2d(initial_u, left_bc, right_bc, top_bc, bottom_bc, Nt, Nx, Ny, alpha, dx, dy,
                                         dt)  # this is a sequence of updating
    return u_uncontrolled[-1]  # the final result

def simu_fourier():
    # Simulate the 2D heat equation using Fourier transform
    u_fourier = get_heat_eq_solution()
    return u_fourier

def simu_pinns():
    # Initialize and train the model
    model = HeatEquationNN()

    # here try different optimization methods
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # alpha = 0.01  # Thermal diffusivity
    alpha = 0.1  # Thermal diffusivity

    train_heat_equation_model(model, optimizer, alpha)

    # Create and display the animation
    ani = create_animation(model, ['pinns_result'], save=1)


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

    init_output = np.zeros_like(x, dtype= np.float32)
    init_output[0, :] = 1
    init_output = tf.reshape(init_output, (-1, 1))

    x = tf.reshape(x, (-1, 1))
    y = tf.reshape(y, (-1, 1))
    t = tf.reshape(tf.repeat(tf.range(0, 10, dtype= tf.float32)/10, repeats= 250), (-1, 1)) # time from 0 to 10
    t = tf.random.shuffle(t, seed= 1)
    t_init = tf.zeros_like(t, dtype= tf.float32)
    t_target = tf.ones_like(t, dtype= tf.float32)
    t_test = tf.ones_like(t, dtype= tf.float32)


    pred_data = np.concatenate([x, y, t], axis= -1) # train
    input_init = np.concatenate([x, y, t_init], axis= -1) # train init state
    input_target = np.concatenate([x, y, t_target], axis= -1)
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
    output_target = tf.cast(simu_fdm(), dtype= tf.float32)
    output_target = tf.reshape(output_target, (-1, 1))
    if train:
        # Model, optimizer, and training loop
        model = nn_model(input_shape=(3,), layers=[16, 32, 32, 16],
                         activation_function = activation_function,
                         dropout_rate = dropout_rate)
        optimizer = optimizer
        
        if use_early_stopping:
            model = train_with_early_stopping(
                model=model,
                init_points=input_init,
                input_bound=bound_input,
                input_interior=interior_input,
                input_target= input_target,
                output_init=output_init,
                output_bound=bound_output,
                output_target= output_target,
                alpha=alpha,
                optimizer=optimizer,
                epochs=epochs,
                patience=patience,
                img_name='PINN_loss_history.png',
                verbose=True
            )
        else:
            history = []
            for epoch in range(epochs):
                loss_value = train_step(model,
                                        input_init, 
                                        bound_input, 
                                        interior_input, 
                                        input_target, 
                                        output_init ,
                                        bound_output,
                                        output_target, 
                                        alpha, 
                                        optimizer)
                history.append(loss_value)
                if verbose:
                    print(f"Epoch {epoch+1}, Loss: {loss_value.numpy()}")
            plot_loss_history(history, img_name='PINN_loss_history.png') 
        
        model.save(os.path.join(models_path, 'tf_pinns.h5'))
    else:
        model = tf.keras.models.load_model(os.path.join(models_path, 'tf_pinns.h5'))

    u = model(test_data)
    return u


def main(
    execution_parameters = {
        'train': False,
        'alpha': 0.01,
        'epochs': 10000,
        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
        'activation_function':'tanh',
        'dropout_rate': 0.,  # rate of neurons to dropout
        'patience': 100,  # Early stopping patience
        'use_early_stopping': False, # stop training if there's no improvement
        'verbose': True,
        'other_models': False
    }
):
    # fetch current timestamp
    dt = datetime.now().strftime('%y%m%d_%H%M%S')
    execution_parameters['dt'] = dt
    
    if execution_parameters['other_models']:
        # run Fourier simulation
        fourier_result = simu_fourier()
        show_image(fourier_result, image_name=['fourier_result'], save=True)

        # pinns_result = simu_pinns()

        # run FinDiff simulation
        fdm_result = simu_fdm()
        show_image(fdm_result, image_name=['fdm_result'], save=True)

    # run PINN simulation
    u = simu_pinns_tf(**execution_parameters)
    pinn_result = tf.reshape(u, (50, 50))
    show_image(pinn_result, image_name=['pinns_tf'], save=True)
    
    # Reshape both results for comparison (assuming 50x50 grid)
    fdm_result_reshaped = fdm_result.reshape((50, 50))
    pinn_result_reshaped = pinn_result.numpy().reshape((50, 50))
    
    # Compute L1 and L2 errors
    l1_error = compute_l1_error(pinn_result_reshaped, fdm_result_reshaped)
    l2_error = compute_l2_error(pinn_result_reshaped, fdm_result_reshaped)
    
    # saving optimizer informations
    optimizer = execution_parameters.get('optimizer')
    optimizer_name = optimizer.__class__.__name__
    learning_rate = optimizer.get_config().get('learning_rate', 'N/A')  # default to 'N/A' if not found
    execution_parameters['optimizer_name'] = optimizer_name
    execution_parameters['optimizer_learning_rate'] = learning_rate
    
    
    # build error matrix, with execution parameters
    errors_df = pd.DataFrame({'l1_error': l1_error,
                              'l2_error': l2_error} | execution_parameters,
                             index=[0])
    print(errors_df)
    errors_df.to_csv(os.path.join('results', f'results_{dt}.csv'))

if __name__ == "__main__":
    optimizers = ['Adam', 'Adamax', 'AdamW', 'SGD']
    learning_rates = [0.001, 0.005, 0.01]

    for i in optimizers:
        for j in learning_rates:
            if i == 'Adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=j)
            elif i == 'Adamax':
                optimizer = tf.keras.optimizers.Adamax(learning_rate=j)
            elif i == 'AdamW':
                optimizer = tf.keras.optimizers.AdamW(learning_rate=j)
            elif i == 'SGD':
                optimizer = tf.keras.optimizers.SGD(learning_rate=j)
            else:
                raise ValueError(f"Unsupported optimizer: {i}")
            
            execution_parameters = {
                'train': False,
                'alpha': 0.01,
                'epochs': 10000,
                'optimizer': optimizer,
                'activation_function': 'tanh',
                'dropout_rate': 0., 
                'patience': 100,  
                'use_early_stopping': False, 
                'verbose': True,
                'other_models': False
            }
            main(execution_parameters)


