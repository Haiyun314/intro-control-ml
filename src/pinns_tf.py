import tensorflow as tf

def nn_model(input_shape, layers):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    for layer in layers:
        x = tf.keras.layers.Dense(layer, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(1)(x)  # Output layer with 1 neuron
    return tf.keras.models.Model(inputs, outputs)

def compute_loss(model, input_bound, input_interior, output_bound, alpha):
    x_i, y_i, t_i = tf.split(input_interior, num_or_size_splits=3, axis=1)

    u_bound = model(input_bound)

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

    loss_interior = tf.reduce_mean(tf.square(u_t - alpha * (u_xx + u_yy)))*(1/len(input_interior))
    loss_boundary = tf.reduce_mean(tf.square(u_bound - output_bound))*(1/len(input_bound))

    return loss_interior + loss_boundary

# Training step
@tf.function
def train_step(model, input_bound, input_interior, output_bound, alpha, optimizer):
    with tf.GradientTape() as tape:
        loss_value = compute_loss(model, input_bound, input_interior, output_bound, alpha)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value

