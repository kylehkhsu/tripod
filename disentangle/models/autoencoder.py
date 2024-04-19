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


class Autoencoder(eqx.Module):
    encoder: eqx.Module
    latent: eqx.Module
    decoder: eqx.Module
    lambdas: Dict[str, float]
    reconstruction_loss_fn: str
    regularized_attributes: List[str]
    hessian_regularization_on: List[str]
    num_perturbations: int
    epsilon: float
    quantize_latents: bool

    def __init__(
        self,
        encoder,
        decoder,
        latent_size,
        lambdas,
        reconstruction_loss_fn,
        regularized_attributes,
        hessian_regularization_on,
        num_perturbations,
        epsilon,
        quantize_latents,
        num_quantized_values,
        *,
        key
    ):
        keys = iter(jax.random.split(key, 100))
        self.encoder = encoder(latent_size=latent_size, key=next(keys))
        if quantize_latents:
            latent = disentangle.nn.Quantizer(num_latents=latent_size, num_values_per_latent=num_quantized_values,
                                              key=next(keys))
        else:
            latent = eqx.nn.Lambda(lambda z: {'z_c': z})
        self.latent = latent
        self.decoder = decoder(latent_size=latent_size, key=next(keys))
        self.lambdas = lambdas
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.regularized_attributes = regularized_attributes
        self.hessian_regularization_on = hessian_regularization_on
        self.num_perturbations = num_perturbations
        self.epsilon = epsilon
        self.quantize_latents = quantize_latents

    @eqx.filter_jit
    def batched_loss(self, model, batch, *, key):
        outs = {}
        outs.update(jax.vmap(model.latent)(jax.vmap(model.encoder)(batch['x'])))

        if model.quantize_latents:
            z = outs['z_q']
        else:
            z = outs['z_c']

        outs['decoder'] = jax.vmap(model.decoder)(z)
        losses = {}

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

        latent_multiinformation_key, hessian_regularization_key = jax.random.split(key, 2)

        z_std = jnp.std(outs['z_c'], axis=0) + 1e-6

        nz = z.shape[1]
        b = z.shape[0]

        silvermans = ((4 / (nz + 2)) ** (1 / (nz + 4)) * b ** (-1 / (nz + 4))) ** 2
        H = jnp.diag(z_std ** 2 * silvermans)

        # silvermans_marginal = (4 / (1 + 2)) ** (1 / (1 + 4)) * b ** (-1 / (1 + 4))
        # h = z_std * silvermans_marginal

        losses['latent_multiinformation'] = model.latent_multiinformation(
            z, H, z_std
        )

        # returns (batch_size,) per regularizer per layer
        decoder_hessian_regularization = jax.vmap(
            model.hessian_regularization,
            in_axes=(None, 0, None, 0, None, None, 0)
        )(
            model.decoder,
            z,
            z_std,
            outs['decoder'],
            model.num_perturbations,
            model.epsilon,
            jax.random.split(hessian_regularization_key, z.shape[0])
        )
        for regularizer in decoder_hessian_regularization.keys():
            losses[regularizer] = sum([
                decoder_hessian_regularization[regularizer][k] for k in model.hessian_regularization_on
            ])   # excludes some layers

        losses['total'] = sum(model.lambdas[k] * losses[k] for k in model.lambdas.keys())
        outs['log'] = {f'loss/{k}': v for k, v in losses.items()}

        for regularizer in decoder_hessian_regularization.keys():
            outs['log'].update({f'{regularizer}/{k}': v for k, v in decoder_hessian_regularization[regularizer].items()})
        return jnp.mean(losses['total']), outs

    @staticmethod
    def hessian_regularization(f, z, z_std, f_z, k, epsilon, key):
        rademacher_key, normal_key = jax.random.split(key, 2)

        vs = epsilon * jax.random.rademacher(rademacher_key, shape=(k, *z.shape)) * z_std
        finite_diffs = jax.vmap(
            Autoencoder.second_order_central_finite_difference,
            in_axes=(None, None, None, 0, None)
        )(f, z, f_z, vs, epsilon)
        twice_sum_off_diag_squares = {k: jnp.var(v, axis=0) for k, v in finite_diffs.items()}

        vs = epsilon * jax.random.normal(normal_key, shape=(k, *z.shape)) * z_std
        finite_diffs = jax.vmap(
            Autoencoder.second_order_central_finite_difference,
            in_axes=(None, None, None, 0, None)
        )(f, z, f_z, vs, epsilon)
        twice_sum_squares = {k: jnp.var(v, axis=0) for k, v in finite_diffs.items()}

        ret = {}
        ret['hessian_penalty'] = {k: jnp.mean(twice_sum_off_diag_squares[k]) for k in twice_sum_off_diag_squares.keys()}
        ret['hessian_penalty_normalized'] = {
            k: jnp.mean(twice_sum_off_diag_squares[k]) / (jnp.mean(twice_sum_squares[k]) + 1e-10)
            for k in twice_sum_squares.keys()
        }
        return ret
    @staticmethod
    def second_order_central_finite_difference(f, x, f_x, h, epsilon):
        # (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
        return jax.tree_map(lambda x, y, z: (x - 2 * y + z) / (epsilon ** 2), f(x + h), f_x, f(x - h))

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



