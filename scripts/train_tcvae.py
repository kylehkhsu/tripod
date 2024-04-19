import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import einops
import optax

import pathlib
import collections
import ipdb
import tqdm
import omegaconf
import contextlib
import wandb
import hydra
import os
import pprint
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import disentangle

sns.set_theme()
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'stix'  # 'stix' font includes math symbols

@eqx.filter_jit
def train_step(model, optimizer_state, optimizer, batch, *, key):
    (_, outs), grads = eqx.filter_value_and_grad(model.batched_loss, has_aux=True)(model, batch, key=key)
    model, optimizer_state = disentangle.utils.optax_step(optimizer, model, grads, optimizer_state)
    return model, optimizer_state, outs['log']


def batched_x_to_image(x):
    x = (x + 1) / 2
    x = einops.rearrange(x, '... c h w -> ... h w c')
    x = jnp.clip(x, 0, 1)
    image = jnp.asarray(255 * x, dtype=jnp.uint8)
    return image


def rows_to_grid_image(rows, num_image_rows, num_image_cols):
    rows = jnp.stack([v for v in rows.values()], axis=0)  # (r, b, c, h, w)
    image = batched_x_to_image(rows)  # (r, b, h, w, c)
    image = einops.rearrange(image, 'r b h w c -> (r h) (b w) c')
    image = einops.rearrange(
        image, 'h (rows cols w) c -> (rows h) (cols w) c', rows=num_image_rows, cols=num_image_cols
    )
    return image


@eqx.filter_jit
def hessian_of_decoder(model, z):
    # high-dim ints, low-dim z -> use jacfwd, doubly so for hessian
    return jax.jacfwd(jax.jacfwd(model.decoder))(z)

def mask_diagonal(A):
    assert A.ndim == 2
    return A * (jnp.ones_like(A) - jnp.eye(A.shape[0]))


def compute_stats(A):
    stats = {}
    stats['off_diagonal_max'], stats['off_diagonal_mean'] = off_diagonal_max_and_mean(A)
    stats['diagonal_max'], stats['diagonal_mean'] = diagonal_max_and_mean(A)
    stats['diagonal_mean:off_diagonal_mean'] = stats['diagonal_mean'] / (stats['off_diagonal_mean'] + 1e-10)
    stats['diagonal_mean:(diagonal_mean+off_diagonal_mean)'] = stats['diagonal_mean'] / (stats['diagonal_mean'] +
                                                                                         stats['off_diagonal_mean'] +
                                                                                         1e-10)
    stats['frobenius'] = jnp.linalg.norm(A, ord='fro')
    return stats


def off_diagonal_max_and_mean(A):
    assert A.ndim == 2
    masked_A = mask_diagonal(A)
    n = A.shape[0]
    return jnp.max(masked_A), jnp.sum(masked_A) / (n * (n - 1))


def diagonal_max_and_mean(A):
    assert A.ndim == 2
    diagonal = einops.einsum(A, 'i i -> i')
    return jnp.max(diagonal), jnp.mean(diagonal)


@eqx.filter_jit
def process_layer_hessian(hessians, rescaler):
    # normalization
    if hessians.ndim == 5:
        hessians = einops.rearrange(hessians, 'c h w z1 z2 -> (c h w) z1 z2')
    n = hessians.shape[-1]

    matrices = {}
    matrices['abs_hessian'] = jnp.abs(hessians)
    matrices['rescaled_abs_hessian'] = jnp.abs(hessians * rescaler)

    ret = {}
    for k_mat, v_mat in matrices.items():
        ret[k_mat] = einops.reduce(v_mat, 'b z1 z2 -> z1 z2', 'mean')
        ret.update({f'{k_mat}_{k}': v.mean() for k, v in jax.vmap(compute_stats)(v_mat).items()})
    return ret

def evaluate(model, val_set, config, dataset_info, *, key):
    log = {}
    log.update({
        'weight_norm/decoder': disentangle.utils.weight_norm(model.decoder).item(),
        'weight_norm/encoder': disentangle.utils.weight_norm(model.encoder).item(),
    })

    outs_list = []
    for i, batch in tqdm.tqdm(enumerate(val_set), total=config.data.num_val // config.data.batch_size):
        key, sub_key = jax.random.split(key)
        _, outs = model.batched_loss(model, batch, key=sub_key)
        outs['x_pred'] = jax.nn.sigmoid(outs['decoder']['x_logits']) * 2 - 1
        outs['log']['metrics/psnr'] = jax.vmap(disentangle.metrics.peak_signal_to_noise_ratio)(
            batched_x_to_image(outs['x_pred']),
            batched_x_to_image(batch['x'])
        )
        outs['x'] = batch['x']
        outs['s'] = batch['s']
        if i > 20:
            for k in [k for k in outs.keys() if k.startswith('x')]:
                outs[k] = jnp.zeros((0, *outs[k].shape[1:]), dtype=outs[k].dtype)
            outs['decoder'] = jax.tree_map(lambda x: jnp.zeros((0, *x.shape[1:]), dtype=x.dtype), outs['decoder'])
        outs_list.append(outs)

    outs = jax.tree_map(lambda *leaves: jnp.concatenate(leaves) if leaves[0].ndim > 0 else jnp.stack(leaves), *outs_list)
    log.update({f'{k}/val': v.mean().item() for k, v in outs['log'].items()})

    z = outs['z_mu']

    # infomec
    infomec = disentangle.metrics.compute_infomec(
        sources=outs['s'], latents=z, discrete_latents=model.quantize_latents
    )
    log.update({f'metrics/{k}': v for k, v in infomec.items() if k in ['infom', 'infoe', 'infoc']})

    # DCI
    dci = disentangle.metrics.compute_dci(sources=outs['s'], latents=z)
    log.update({f'metrics/{k}': v for k, v in dci.items()})

    # NMI heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        infomec['nmi'], ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1, cbar=False,
        annot_kws={'fontsize': 8},
        xticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(infomec['nmi'].shape[1])],
        yticklabels=dataset_info['source_names'],
        rasterized=True
    )

    for i, label in enumerate(ax.get_xticklabels()):
        if infomec['active_latents'][i] == 0:
            label.set_color('red')

    fig.tight_layout()
    log.update({f'nmi_heatmap': wandb.Image(fig)})
    plt.close()

    # val generations
    num_samples = config.eval.num_vis_rows.val * config.eval.num_vis_cols
    rows = {k: outs[k][:num_samples] for k in ['x', 'x_pred']}
    rows['x_diff'] = jnp.abs(rows['x'] - rows['x_pred']) - 1
    log.update({'vis/val': wandb.Image(np.array(
        rows_to_grid_image(rows, config.eval.num_vis_rows.val, config.eval.num_vis_cols)
    ))})

    # latent densities
    data = pd.DataFrame(z, columns=[f'z{i}' for i in range(z.shape[1])])
    data['id'] = data.index
    data = data.melt(id_vars='id', var_name='z', value_name='value')
    fig, ax = plt.subplots(figsize=(z.shape[1] ** 0.8, 3))
    sns.violinplot(data=data, ax=ax, x='z', y='value', density_norm='width', cut=0)
    fig.tight_layout()
    log.update({f'latent_densities': wandb.Image(fig)})
    plt.close()

    # decoded latent interventions
    z_max = z.max(axis=0)
    z_min = z.min(axis=0)
    z_sample = z[:config.eval.num_intervene_cols]    # (b, z)
    for i_latent in range(z.shape[1]):
        values = jnp.linspace(z_min[i_latent], z_max[i_latent], config.eval.num_intervene_values)
        z_intervene = einops.repeat(jnp.copy(z_sample), 'b z -> b v z', v=config.eval.num_intervene_values)
        z_intervene = z_intervene.at[:, :, i_latent].set(values)
        x_intervene_logits = jax.vmap(jax.vmap(model.decoder))(z_intervene)['x_logits']
        x_intervene = jax.nn.sigmoid(x_intervene_logits) * 2 - 1  # (b, v, c, h, w)
        image = batched_x_to_image(x_intervene)
        image = einops.rearrange(image, 'b v h w c -> (b h) (v w) c')
        log.update({f'decode_latent_interventions/{i_latent}': wandb.Image(np.array(image))})

    # analysis
    z_std = jnp.std(z, axis=0) + 1e-6
    rescaler = einops.einsum(z_std, z_std, 'i, j -> i j')

    accum = collections.defaultdict(lambda: jnp.array(0.))
    for i in tqdm.tqdm(range(config.eval.num_derivatives)):
        decoder_hessian = hessian_of_decoder(model, z[i])
        for k_layer, hessian in decoder_hessian.items():
            ret = process_layer_hessian(hessian, rescaler)
            for k_stat, v_stat in ret.items():
                # if jnp.isnan(v_stat).any():
                #     ipdb.set_trace()
                accum[f'{k_stat}/{k_layer}'] += v_stat / config.eval.num_derivatives

        decoder_outs = model.decoder(z[i])
        for k_layer, v_layer in decoder_outs.items():
            stats = {}
            stats['norm'] = jnp.linalg.norm(v_layer)
            for k_stat, v_stat in stats.items():
                accum[f'{k_stat}/{k_layer}'] += v_stat / config.eval.num_derivatives

    log.update({k: v.item() for k, v in accum.items() if v.ndim == 0})

    num_latents = z.shape[1]
    for k, v in {k: v for k, v in accum.items() if v.ndim == 2}.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            v,
            ax=ax, annot=True, fmt='.2f', square=True,
            cbar=True,
            annot_kws={'fontsize': 8},
            xticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(num_latents)],
            yticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(num_latents)],
            rasterized=True
        )
        fig.tight_layout()
        log.update({k: wandb.Image(fig)})
        plt.close()

    nz = z.shape[1]
    fig, axes = plt.subplots(nz, nz, figsize=(4 * nz, 4 * nz))
    for i in range(nz):
        for j in range(i + 1, nz):
            ax = axes[i][j]
            if model.quantize_latents:
                sns.histplot(
                    ax=ax,
                    x=z[:, i],
                    y=z[:, j],
                    rasterized=True,
                    binrange=((-1, 1), (-1, 1)),
                    bins=config.model.num_quantized_values
                )
            else:
                sns.histplot(
                    ax=ax,
                    x=z[:, i],
                    y=z[:, j],
                    rasterized=True,
                )
            ax.set_xlabel(rf'$z_{{{i}}}$')
            ax.set_ylabel(rf'$z_{{{j}}}$')
            ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    log.update({'pairwise_latents': wandb.Image(fig)})
    plt.close()
    return log


def save(path, model, optimizer_state):
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / 'model.eqx', model)
    eqx.tree_serialise_leaves(path / 'optimizer_state.eqx', optimizer_state)
    print(f'saved model and optimizer state to {path}')


def main(config):
    api = wandb.Api()
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    pprint.pprint(config_dict)
    run = wandb.init(
        project=config.wandb.project,
        config=config_dict,
        save_code=True,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        name=config.wandb.name,
        mode='disabled' if config.debug else 'online'
    )
    wandb.run.log_code(hydra.utils.get_original_cwd())
    wandb.config.update({'wandb_run_dir': wandb.run.dir})
    wandb.config.update({'hydra_run_dir': os.getcwd()})
    checkpoints_path = pathlib.Path(run.dir) / 'checkpoints'

    dataset_info, train_set, val_set = disentangle.datasets.load(config)
    config.model.latent_size = 2 * dataset_info['num_sources']
    wandb.config.update({'model.latent_size': config.model.latent_size})

    keys = iter(jax.random.split(jax.random.PRNGKey(config.experiment.seed), 100))

    model = hydra.utils.instantiate(config.model)(dataset_size=dataset_info['num_train'], key=next(keys))
    optimizer, optimizer_state = model.construct_optimizer(config)

    pbar = tqdm.tqdm(train_set, total=int(config.optim.num_steps))
    context = jax.disable_jit if config.debug else contextlib.nullcontext
    train_key = next(keys)
    eval_key = next(keys)
    with context():
        for step, batch in enumerate(pbar):
            if step >= config.optim.num_steps:
                break

            if (step + 1) % config.checkpoint.period == 0:
                path = checkpoints_path / f'step={step}'
                save(path, model, optimizer_state)
                wandb.save(str(path / '*'), base_path=run.dir)

            if (step == 0 and not config.debug) or \
                (step + 1) % config.eval.period == 0 or \
                ((step + 1 < config.eval.period) and (step + 1) % (config.eval.period // 5) == 0):
                model = eqx.nn.inference_mode(model, True)
                log = evaluate(model, val_set, config, dataset_info, key=eval_key)
                wandb.log(log, step=step)
                model = eqx.nn.inference_mode(model, False)

            train_key, sub_key = jax.random.split(train_key)
            model, optimizer_state, log = train_step(model, optimizer_state, optimizer, batch, key=sub_key)
            wandb.log({f'{k}/train': v.mean().item() for k, v in log.items()}, step=step)
    wandb.finish()
