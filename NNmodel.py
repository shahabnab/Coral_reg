import tensorflow as tf

import numpy as np
import tensorflow as tf

class CoralLambdaRampUp(tf.keras.callbacks.Callback):
    """
    Updates a tf.Variable lambda_var from 0 -> max_lambda over rampup_epochs.
    mode: "linear" or "sigmoid" (sigmoid is common in DA papers)
    """
    def __init__(self, lambda_var: tf.Variable, max_lambda: float, rampup_epochs: int,
                 mode: str = "sigmoid", verbose: int = 0):
        super().__init__()
        self.lambda_var = lambda_var
        self.max_lambda = float(max_lambda)
        self.rampup_epochs = max(1, int(rampup_epochs))
        self.mode = mode
        self.verbose = verbose

    def _schedule(self, epoch: int) -> float:
        # progress in [0, 1]
        p = min(1.0, max(0.0, (epoch + 1) / self.rampup_epochs))

        if self.mode == "linear":
            return self.max_lambda * p

        # "sigmoid" ramp (very common): smoothly rises from near 0 to max
        # scaled so p=0 -> ~0, p=1 -> max
        k = 10.0
        s = 1.0 / (1.0 + np.exp(-k * (p - 0.5)))
        s0 = 1.0 / (1.0 + np.exp(-k * (0.0 - 0.5)))
        s1 = 1.0 / (1.0 + np.exp(-k * (1.0 - 0.5)))
        s = (s - s0) / (s1 - s0 + 1e-12)
        return self.max_lambda * float(s)

    def on_epoch_begin(self, epoch, logs=None):
        new_val = self._schedule(epoch)
        self.lambda_var.assign(new_val)
        if self.verbose:
            print(f"[CoralLambdaRampUp] epoch={epoch} coral_lambda={float(self.lambda_var.numpy()):.6f}")

def build_coral_cir_reg_model(input_shape, latent_dim=64, base_filters=32, dropout=0.10):
    x_in = tf.keras.Input(shape=input_shape, name="x")

    x = tf.keras.layers.Conv1D(base_filters, 9, padding="same", activation="relu")(x_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)

    x = tf.keras.layers.Conv1D(base_filters * 2, 7, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)

    x = tf.keras.layers.Conv1D(base_filters * 4, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dropout(dropout)(x)
    z = tf.keras.layers.Dense(latent_dim, activation=None, name="latent")(x)

    h = tf.keras.layers.LayerNormalization()(z)
    h = tf.keras.layers.Dense(max(latent_dim // 2, 8), activation="relu")(h)
    h = tf.keras.layers.Dropout(dropout)(h)

    pred = tf.keras.layers.Dense(1, activation=None, name="pred")(h)

    return tf.keras.Model(x_in, {"pred": pred, "latent": z}, name="coral_cir_reg_model")
