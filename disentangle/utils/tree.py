import jax
import jax.numpy as jnp
import equinox as eqx


# list trick: https://github.com/patrick-kidger/equinox/issues/79
def optax_wrap(model):
    return [model]


def optax_unwrap(model):
    return model[0]


def relabel_attr(pytree, attr, label):
    has_attr = lambda x: hasattr(x, attr)
    where_attr = lambda m: tuple(
        getattr(x, attr) for x in jax.tree_util.tree_leaves(m, is_leaf=has_attr) if has_attr(x)
    )
    pytree = eqx.tree_at(where_attr, pytree, replace_fn=lambda _: label, is_leaf=lambda x: x is None)
    return pytree


@eqx.filter_jit
def weight_norm(x):
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_leaves(eqx.filter(x, eqx.is_array))))


def optax_step(optimizer, model, grads, optimizer_state):
    grads = optax_wrap(grads)
    model = optax_wrap(model)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
    model = eqx.apply_updates(model, updates)
    model = optax_unwrap(model)
    return model, optimizer_state