import ipdb
import jax
import jax.numpy as jnp
import equinox as eqx

def round_with_straight_through_estimator(z):
    z_rounded = jnp.round(z, decimals=0)
    return z + jax.lax.stop_gradient(z_rounded - z)


class Quantizer(eqx.Module):
    num_latents: int
    num_values_per_latent: int

    def __init__(self, num_latents, num_values_per_latent, *, key=None):
        self.num_latents = num_latents
        self.num_values_per_latent = num_values_per_latent

    def __call__(self, z, *, key=None):
        z_c = jnp.tanh(z)   # [-1, 1]
        z_q = ((z_c + 1) / 2) * (self.num_values_per_latent - 1)  # [0, v - 1]
        z_q = round_with_straight_through_estimator(z_q)  # {0, 1, ..., v - 1}
        z_q = 2 * z_q / (self.num_values_per_latent - 1) - 1  # [-1, 1]
        return {
            'z_c': z_c,
            'z_q': z_q,
        }


def test_quantizer():
    import matplotlib
    import matplotlib.pyplot as plt
    quantizer = Quantizer(num_latents=1, num_values_per_latent=10)
    x = jnp.linspace(-3, 3, 1000)[None]
    y = quantizer(x)
    plt.plot(x[0], y[0])
    plt.show()