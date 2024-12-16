import tensorflow as tf
import matplotlib.pyplot as plt
import os

def plot_loss_history(history, img_name='PINN_loss_history.png'):
    plt.plot(history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss History')
    plt.savefig(os.path.join('results', img_name))
    plt.show()
    
def nn_model(input_shape, layers, 
             activation_function: str = 'tanh',
             dropout_rate: float = 0.0,  # rate of neurons to dropout
             ):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    for layer in layers:
        x = tf.keras.layers.Dense(layer, activation=activation_function)(x)
        if dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1)(x)  # Output layer with 1 neuron
    return tf.keras.models.Model(inputs, outputs)


def compute_loss(model, init_points, input_bound, input_interior, input_target,output_init, output_bound, output_target, alpha):
    x_i, y_i, t_i = tf.split(input_interior, num_or_size_splits=3, axis=1)

    u_fdm = model(input_target)
    u_bound = model(input_bound)
    u_init = model(init_points)
    # Gradient computation using GradientTape
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x_i, y_i])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_i, y_i, t_i])
            u = model(tf.concat([x_i, y_i, t_i], axis=1))  # Temperature predictions

        # First-order gradients
        u_t = tape2.gradient(u, t_i)
        u_x = tape2.gradient(u, x_i)
        u_y = tape2.gradient(u, y_i)
        # del tape2
    u_xx = tape1.gradient(u_x, x_i)
    u_yy = tape1.gradient(u_y, y_i)

    if u_xx is None or u_yy is None:
        raise ValueError("Second-order gradients are None.")

    loss_fdm = tf.reduce_mean(tf.square(u_fdm - output_target))*(1/len(input_interior))
    loss_interior = tf.reduce_mean(tf.square(u_t - alpha * (u_xx + u_yy)))*(1/len(input_interior))
    loss_boundary = tf.reduce_mean(tf.square(u_bound - output_bound))*(1/len(input_bound))
    loss_init = tf.reduce_mean(tf.square(u_init - output_init))*(1/len(init_points))

    return loss_interior + loss_boundary + loss_init + loss_fdm


# Training step
@tf.function
def train_step(model, init_points, input_bound, input_interior,input_target, output_init, output_bound, output_target, alpha, optimizer):
    with tf.GradientTape() as tape:
        loss_value = compute_loss(model, init_points, input_bound, input_interior,input_target, output_init, output_bound,output_target, alpha)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value

def train_with_early_stopping(model, init_points, input_bound, input_interior, input_target,
                              output_init, output_bound,output_target, alpha, optimizer, 
                              epochs=5000, patience=100, 
                              img_name='PINN_loss_history.png',
                              verbose=True):
    history = []  # Store loss history
    best_loss = float('inf')
    wait = 0  # Counter for early stopping

    for epoch in range(epochs):
        loss_value = train_step(model, init_points, input_bound, input_interior, input_target, 
                                output_init, output_bound,output_target, alpha, optimizer)
        history.append(loss_value.numpy())

        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {loss_value.numpy()}")

        # Early stopping logic
        if loss_value.numpy() < best_loss:
            best_loss = loss_value.numpy()
            wait = 0  # Reset wait counter
            model.save_weights(os.path.join('results', 'best_model_weights.weights.h5'))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                model.load_weights(os.path.join('results', 'best_model_weights.weights.h5'))  # Restore best model weights
                break
    plot_loss_history(history, img_name)     

    return model