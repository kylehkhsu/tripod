import jax
import jax.numpy as jnp
import equinox as eqx
from typing import *

from .layers import LinearLayerNormAct, ResCondConvBlock, Upsample

class ResidualDecoder(eqx.Module):
    input: jnp.array
    cond_mlp: eqx.nn.Sequential
    up_blocks: Tuple[Tuple[eqx.Module, ...], ...]

    def __init__(self, *, latent_size, normalize_outputs, cond_size=256, key):
        keys = iter(jax.random.split(key, 100))
        self.input = jnp.ones((512, 4, 4), dtype=jnp.float32)
        self.cond_mlp = eqx.nn.Sequential(
            [
                LinearLayerNormAct(latent_size, cond_size, key=next(keys)),
                LinearLayerNormAct(cond_size, key=next(keys)),
            ]
        )
        self.up_blocks = (
            (
                ResCondConvBlock(cond_size, 512, key=next(keys)),
                ResCondConvBlock(cond_size, 512, key=next(keys)),
                Upsample(512, out_channels=256, key=next(keys)),
                eqx.nn.GroupNorm(256, channelwise_affine=False) if normalize_outputs else eqx.nn.Identity()
            ),
            (
                ResCondConvBlock(cond_size, 256, key=next(keys)),
                ResCondConvBlock(cond_size, 256, key=next(keys)),
                Upsample(256, out_channels=128, key=next(keys)),
                eqx.nn.GroupNorm(128, channelwise_affine=False) if normalize_outputs else eqx.nn.Identity()
            ),
            (
                ResCondConvBlock(cond_size, 128, key=next(keys)),
                ResCondConvBlock(cond_size, 128, key=next(keys)),
                Upsample(128, out_channels=64, key=next(keys)),
                eqx.nn.GroupNorm(64, channelwise_affine=False) if normalize_outputs else eqx.nn.Identity()
            ),
            (
                ResCondConvBlock(cond_size, 64, key=next(keys)),
                ResCondConvBlock(cond_size, 64, key=next(keys)),
                Upsample(64, out_channels=3, key=next(keys)),
                eqx.nn.Identity()
            ),
        )

    @eqx.filter_jit
    def __call__(self, z, *, key=None):
        outs = {}
        cond = self.cond_mlp(z)
        outs['cond'] = cond
        x = self.input
        for i, (block1, block2, upsample, norm) in enumerate(self.up_blocks):
            x = block1(x, cond)
            x = block2(x, cond)
            x = upsample(x)
            x = norm(x)
            if i < len(self.up_blocks) - 1:
                outs[f'block{i}'] = x
        outs['x_logits'] = x
        return outs