# src/backend/bot/custom_layers.py
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class ExtractWeight(tf.keras.layers.Layer):
    """
    ExtractWeight is a custom layer that takes branch weights (shape: (batch_size, 3))
    and returns a tensor of shape (batch_size, 1) corresponding to the specified index.
    """
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return tf.reshape(inputs[:, self.index], (-1, 1))

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config
