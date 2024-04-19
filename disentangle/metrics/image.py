import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def peak_signal_to_noise_ratio(x_pred, x_true):
    assert x_pred.dtype == x_true.dtype == jnp.uint8    # [0, 1, ..., 255]
    assert x_pred.ndim == x_true.ndim == 3              # not batched!
    mse = jnp.mean(jnp.square(x_pred - x_true))
    max_value = 255.
    return 20 * jnp.log10(max_value) - 10 * jnp.log10(mse)
