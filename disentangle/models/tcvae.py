import ipdb
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import einops
import numpy as np
from typing import *

import disentangle.nn
import disentangle.losses
import disentangle.utils


class TCVAE(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    lambdas: Dict[str, float]
    reconstruction_loss_fn: str
    regularized_attributes: List[str]
    dataset_size: int
    quantize_latents: bool = False

    def __init__(
        self,
        encoder,
        decoder,
        latent_size,
        lambdas,
        reconstruction_loss_fn,
        regularized_attributes,
        dataset_size,
        *,
        key
    ):
        keys = iter(jax.random.split(key, 100))
        self.encoder = encoder(latent_size=2 * latent_size, key=next(keys))
        self.decoder = decoder(latent_size=latent_size, key=next(keys))
        self.lambdas = lambdas
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.regularized_attributes = regularized_attributes
        self.dataset_size = dataset_size

    @eqx.filter_jit
    def batched_loss(self, model, batch, *, key):
        outs = {}

        mu_q, rho_q = jnp.split(jax.vmap(model.encoder)(batch['x']), 2, axis=-1)
        sigma_q = jax.nn.softplus(rho_q)
        z_sample = mu_q + sigma_q * jax.random.normal(key, mu_q.shape)

        outs['z_mu'] = mu_q

        log_qk_zIk_given_zJk = jax.vmap(  # input: (b, nz) (b, nz) (b, nz), output: (b, b, nz)
            in_axes=(0, None, None),
            fun=jax.vmap(  # input: (nz,) (b, nz) (b, nz), output: (b, nz)
                in_axes=(None, 0, 0),
                fun=jax.vmap(  # input: (nz,) (nz,) (nz,), output: (nz,)
                    in_axes=(0, 0, 0),
                    fun=jax.scipy.stats.norm.logpdf  # input: () () (), output: ()
                )
            )
        )(z_sample, mu_q, sigma_q)

        log_q_zI_given_zJ = einops.reduce(log_qk_zIk_given_zJk, 'I J k -> I J', 'sum')
        log_q_zI = jax.scipy.special.logsumexp(log_q_zI_given_zJ, axis=1) - jnp.log(z_sample.shape[0]) - jnp.log(
            model.dataset_size)

        log_qk_zIk = jax.scipy.special.logsumexp(log_qk_zIk_given_zJk, axis=1) - jnp.log(z_sample.shape[0]) - jnp.log(model.dataset_size)
        log_prodk_qk_zIk = einops.reduce(log_qk_zIk, 'I k -> I', 'sum')

        log_q_zIj_given_xI = jax.vmap(  # input: (b, nz) (b, nz) (b, nz), output: (b, nz)
            in_axes=(0, 0, 0),
            fun=jax.vmap(  # input: (nz,) (nz,) (nz,), output: (nz,)
                in_axes=(0, 0, 0),
                fun=jax.scipy.stats.norm.logpdf  # input: () () (), output: ()
            )
        )(z_sample, mu_q, sigma_q)
        log_q_zI_given_xI = einops.reduce(log_q_zIj_given_xI, 'I j -> I', 'sum')

        log_p_zIj = jax.scipy.stats.norm.logpdf(z_sample, 0, 1)
        log_p_zI = einops.reduce(log_p_zIj, 'I j -> I', 'sum')

        x_dim = 64 * 64 * 3

        losses = {}
        losses['total_correlation'] = (log_q_zI - log_prodk_qk_zIk) / x_dim
        losses['mutual_information'] = (log_q_zI_given_xI - log_q_zI) / x_dim
        losses['dimensionwise_kl'] = (log_prodk_qk_zIk - log_p_zI) / x_dim

        outs['decoder'] = jax.vmap(model.decoder)(z_sample)

        if model.reconstruction_loss_fn == 'binary_cross_entropy':
            losses['reconstruct'] = jax.vmap(disentangle.losses.binary_cross_entropy)(
                x_pred_logits=outs['decoder']['x_logits'],
                x_true_probs=(batch['x'] + 1) / 2
            )
        elif model.reconstruction_loss_fn == 'structural_dissimilarity_index_measure':
            losses['reconstruct'] = jax.vmap(disentangle.losses.structural_dissimilarity_index_measure)(
                x_pred=jax.nn.sigmoid(outs['decoder']['x_logits']),
                x_true=(batch['x'] + 1) / 2
            )
        else:
            raise ValueError(f'Unknown reconstruction loss function: {model.reconstruction_loss_fn}')

        losses['total'] = sum(model.lambdas[k] * losses[k] for k in model.lambdas.keys())
        outs['log'] = {f'loss/{k}': v for k, v in losses.items()}

        return jnp.mean(losses['total']), outs

    def construct_optimizer(self, config):
        weight_decay = config.optim.weight_decay
        optimizer = optax.multi_transform(
            {
                'regularized': optax.chain(
                    optax.clip(config.optim.clip),
                    optax.adamw(
                        learning_rate=config.optim.learning_rate,
                        weight_decay=weight_decay
                    )
                ),
                'unregularized': optax.chain(
                    optax.clip(config.optim.clip),
                    optax.adamw(
                        learning_rate=config.optim.learning_rate,
                        weight_decay=0.0
                    )
                )
            },
            param_labels=disentangle.utils.optax_wrap(self.param_labels())
        )
        optimizer_state = optimizer.init(disentangle.utils.optax_wrap(self.filter()))
        return optimizer, optimizer_state

    def param_labels(self):
        param_labels = jax.tree_map(lambda _: 'unregularized', self.filter())
        for attr in self.regularized_attributes:
            param_labels = disentangle.utils.relabel_attr(param_labels, attr, 'regularized')
        print(f'param_labels: {param_labels}')
        return param_labels

    def filter(self, x=None):
        if x is None:
            x = self
        x = eqx.filter(x, eqx.is_array)
        return x

    @staticmethod
    def latent_multiinformation(z, H, h):
        """

        :param z: (b, nz) points
        :param H: (nz, nz) kernel density covariance matrix
        :param h: (nz,) kernel density bandwidths per dimension (scale)
        :return:
        """

        # uppercase: superscript; lowercase: subscript
        log_qk_zIk_given_zJk = jax.vmap(  # input: (b, nz) (b, nz) (nz,), output: (b, b, nz)
            in_axes=(0, None, None),
            fun=jax.vmap(  # input: (nz,) (b, nz) (nz,), output: (b, nz)
                in_axes=(None, 0, None),
                fun=jax.vmap(  # input: (nz,) (nz,) (nz,), output: (nz,)
                    in_axes=(0, 0, 0),
                    fun=jax.scipy.stats.norm.logpdf  # input: () () (), output: ()
                )
            )
        )(z, z, h)

        log_qk_zIk = jax.scipy.special.logsumexp(log_qk_zIk_given_zJk, axis=1) - jnp.log(z.shape[0])
        log_prodk_qk_zIk = einops.reduce(log_qk_zIk, 'I k -> I', 'sum')

        log_q_zIk_given_zJk = jax.vmap(  # input: (b, nz) (b, nz) (nz,), output: (b, b, nz)
            in_axes=(0, None, None),
            fun=jax.vmap(  # input: (nz,) (b, nz) (nz,), output: (b, nz)
                in_axes=(None, 0, None),
                fun=jax.vmap(  # input: (nz,) (nz,) (nz,), output: (nz,)
                    in_axes=(0, 0, 0),
                    fun=jax.scipy.stats.norm.logpdf  # input: () () (), output: ()
                )
            )
        )(z, z, jnp.sqrt(jnp.diagonal(H)))

        log_q_zI_given_zJ = einops.reduce(log_q_zIk_given_zJk, 'I J k -> I J', 'sum')
        log_q_zI = jax.scipy.special.logsumexp(log_q_zI_given_zJ, axis=1) - jnp.log(z.shape[0])
        return log_q_zI - log_prodk_qk_zIk



