import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

def create_model(input_dim: int) -> Model:
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=42)
    bias_initializer = tf.keras.initializers.Zeros()

    model = Sequential([
        layers.Dense(16, activation="relu", input_shape=(input_dim,), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    ])

    return model
