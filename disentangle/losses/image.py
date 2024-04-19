import jax
import jax.numpy as jnp
import optax
import dm_pix as pix
import einops

def binary_cross_entropy(x_pred_logits, x_true_probs):
    assert x_pred_logits.ndim == x_true_probs.ndim == 3
    return jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            logits=x_pred_logits,
            labels=x_true_probs,
        )
    )


def mean_squared_error(x_pred, x_true):
    assert x_pred.ndim == x_true.ndim == 3
    return jnp.mean(jnp.square(x_pred - x_true))


def structural_dissimilarity_index_measure(x_pred, x_true):
    assert x_pred.ndim == x_true.ndim == 3
    assert x_pred.shape[0] == x_true.shape[0] == 3
    x_pred = einops.rearrange(x_pred, 'c h w -> h w c')
    x_true = einops.rearrange(x_true, 'c h w -> h w c')
    return (1 - pix.ssim(
        x_pred,
        x_true,
        max_val=1,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        return_map=False,
    )) / 2