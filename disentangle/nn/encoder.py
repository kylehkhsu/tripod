import jax
import jax.numpy as jnp
import equinox as eqx
import einops
from typing import *

from .layers import LinearLayerNormAct, ResConvBlock, Downsample


class ResidualEncoder(eqx.Module):
    down_blocks: Tuple[Tuple[ResConvBlock, ResConvBlock, Downsample], ...]
    dense: eqx.nn.Sequential

    def __init__(self, *, latent_size, key):
        keys = iter(jax.random.split(key, 100))
        self.down_blocks = (
            (
                ResConvBlock(3, out_channels=32, key=next(keys)),
                ResConvBlock(32, key=next(keys)),
                Downsample(32, out_channels=64, key=next(keys)),
            ),
            (
                ResConvBlock(64, key=next(keys)),
                ResConvBlock(64, key=next(keys)),
                Downsample(64, out_channels=128, key=next(keys)),
            ),
            (
                ResConvBlock(128, key=next(keys)),
                ResConvBlock(128, key=next(keys)),
                Downsample(128, out_channels=256, key=next(keys)),
            ),
            (
                ResConvBlock(256, key=next(keys)),
                ResConvBlock(256, key=next(keys)),
                Downsample(256, out_channels=512, key=next(keys)),
            )
        )
        self.dense = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(lambda x: jnp.mean(x, axis=(1, 2))),  # 512
                LinearLayerNormAct(512, 512, key=next(keys)),
                eqx.nn.Linear(512, latent_size, key=next(keys)),
            ]
        )

    def __call__(self, x, *, key=None):
        for block1, block2, downsample in self.down_blocks:
            x = block1(x)
            x = block2(x)
            x = downsample(x)
        z = self.dense(x)
        return z


class MlpEncoder(eqx.Module):
    mlp: eqx.nn.Sequential

    def __init__(self, *, latent_size, key):
        keys = iter(jax.random.split(key, 100))
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(lambda x: einops.rearrange(x, 'c h w -> (c h w)')),  # 3 * 64 * 64
                LinearLayerNormAct(64 * 64 * 3, 512, key=next(keys)),
                eqx.nn.Linear(512, latent_size, key=next(keys)),
            ]
        )

    def __call__(self, x, *, key=None):
        z = self.mlp(x)
        return z