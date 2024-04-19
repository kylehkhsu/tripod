import jax
import jax.numpy as jnp
import equinox as eqx
import abc
import einops
from typing import *

class ConditionedModule(eqx.Module):

    @abc.abstractmethod
    def __call__(self, x, cond, *, key=None):
        pass

class ConvINAct(eqx.nn.Sequential):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, *, key):
        if out_channels is None:
            out_channels = in_channels

        layers = [
            eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, key=key),
            eqx.nn.GroupNorm(out_channels, out_channels, channelwise_affine=True),
            eqx.nn.Lambda(jax.nn.gelu)
        ]
        super().__init__(layers)

class ConvAdaINAct(ConditionedModule):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    cond_proj: eqx.nn.Linear
    act: eqx.nn.Lambda

    def __init__(self, cond_size, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, *, key):
        if out_channels is None:
            out_channels = in_channels

        conv_key, cond_proj_key = jax.random.split(key, 2)
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, key=conv_key)
        self.norm = eqx.nn.GroupNorm(groups=out_channels, channelwise_affine=False)
        self.cond_proj = eqx.nn.Linear(cond_size, 2 * out_channels, key=cond_proj_key)
        self.act = eqx.nn.Lambda(jax.nn.gelu)

    def __call__(self, x, cond, *, key=None):
        x = self.conv(x)
        x = self.norm(x)
        scale, shift = jnp.split(self.cond_proj(cond), 2, axis=0)
        scale = einops.rearrange(scale, 'c -> c 1 1')
        shift = einops.rearrange(shift, 'c -> c 1 1')
        x = x * (1 + scale) + shift
        x = self.act(x)
        return x


class LinearLayerNormAct(eqx.nn.Sequential):
    def __init__(self, in_size, out_size=None, *, key):
        if out_size is None:
            out_size = in_size
        layers = [
            eqx.nn.Linear(in_size, out_size, key=key),
            eqx.nn.LayerNorm(out_size),
            eqx.nn.Lambda(jax.nn.gelu)
        ]
        super().__init__(layers)


class Upsample(eqx.nn.Sequential):
    def __init__(self, in_channels, out_channels=None, factor=2, *, key):
        if out_channels is None:
            out_channels = in_channels
        layers = [
            eqx.nn.Lambda(lambda x: jax.image.resize(x, (in_channels, x.shape[1] * factor, x.shape[2] * factor), method='nearest')),
            eqx.nn.Conv2d(in_channels, out_channels, 3, 1, 1, key=key)
        ]
        super().__init__(layers)


class Downsample(eqx.nn.Sequential):
    def __init__(self, in_channels, out_channels=None, factor=2, *, key):
        if out_channels is None:
            out_channels = in_channels
        layers = [
            eqx.nn.Lambda(lambda x: einops.rearrange(x, 'c (h f1) (w f2) -> (c f1 f2) h w', f1=factor, f2=factor)),
            eqx.nn.Conv2d(in_channels * factor * factor, out_channels, 1, 1, 0, key=key),
        ]
        super().__init__(layers)


class ResConvBlock(eqx.Module):
    block1: eqx.Module
    block2: eqx.Module
    identity: eqx.Module

    def __init__(self, in_channels, out_channels=None, *, key):
        if out_channels is None:
            out_channels = in_channels
        sub_keys = jax.random.split(key, 3)
        self.block1 = ConvINAct(in_channels, out_channels, 3, 1, 1, key=sub_keys[0])
        self.block2 = ConvINAct(out_channels, out_channels, 3, 1, 1, key=sub_keys[1])
        if in_channels == out_channels:
            self.identity = eqx.nn.Identity()
        else:
            self.identity = eqx.nn.Conv2d(in_channels, out_channels, 1, 1, 0, key=sub_keys[2])

    def __call__(self, x, *, key=None):
        r = self.block1(x)
        r = self.block2(r)
        return self.identity(x) + r


class ResCondConvBlock(ConditionedModule):
    block1: ConditionedModule
    block2: ConditionedModule
    identity: eqx.Module

    def __init__(self, cond_size, in_channels, out_channels=None, *, key):
        if out_channels is None:
            out_channels = in_channels
        sub_keys = jax.random.split(key, 3)
        self.block1 = ConvAdaINAct(cond_size, in_channels, out_channels=out_channels, key=sub_keys[0])
        self.block2 = ConvAdaINAct(cond_size, out_channels, out_channels=out_channels, key=sub_keys[1])
        if in_channels == out_channels:
            self.identity = eqx.nn.Identity()
        else:
            self.identity = eqx.nn.Conv2d(in_channels, out_channels, 1, 1, 0, key=sub_keys[2])

    def __call__(self, x, cond, *, key=None):
        r = self.block1(x, cond)
        r = self.block2(r, cond)
        return self.identity(x) + r





