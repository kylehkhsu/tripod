# Introduction
This codebase for disentangled representation learning accompanies the paper [Tripod: Three Complementary Inductive Biases for Disentangled Representation Learning
](https://arxiv.org/abs/2404.10282) by Kyle Hsu<sup>&ast;</sup>, Jubayer Ibn Hamid<sup>&ast;</sup>, Kaylee Burns, Chelsea Finn, and Jiajun Wu

It uses: 
- [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox) for automatic differentiation
- [Hydra](https://hydra.cc/) for configuration management
- [Weights & Biases](https://wandb.ai/) for experiment logging
- [TensorFlow Datasets](https://www.tensorflow.org/datasets) for dataset management and data loading

# Installation

```
mamba create -n tripod python=3.10 -y && mamba activate tripod
git clone --recurse-submodules https://github.com/kylehkhsu/tripod.git
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

## Add environment variables to `mamba activate`
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mamba deactivate && mamba activate tripod
```

[Datasets](./disentangle/datasets) will be installed via the TensorFlow Datasets API on first use.

To use Weights & Biases logging, you may have to create a free account at [wandb.ai](https://wandb.ai/). Then, run `wandb login` and enter the API key from your account.

# Usage
Main entry points are in [scripts](./scripts). Each configurable script has a corresponding [config](./configs) file and [launcher](./launchers) file.

`train_autoencoder.py` trains autoencoder variants, including Tripod and ablations.

`train_tcvae.py` trains variational autoencoder variants, including TCVAE and VAE.

Both of these automatically log model and optimizer checkpoints. 

### Example
To train a Tripod autoencoder, do `python launchers/train_autoencoder.py`. This will use the configuration defaults in `configs/train_autoencoder.yaml`. To override these defaults, do `python launchers/train_autoencoder.py key=value`. For example, `python launchers/train_autoencoder.py dataset=isaac3d` will train a Tripod autoencoder on the Isaac3D dataset.

To run a sweep, add the `--multirun` flag. The sweep will run over all combinations of configurations specified in `hydra.sweeper.params` in the config file. 

By default, using `--multirun` will invoke the SubmitIt launcher, which submits jobs to a Slurm cluster. Configure this [here](./configs/hydra/launcher/slurm.yaml). To instead run locally, add `hydra/launcher=submitit_local` to the command.

Good Tripod hyperparameters for each dataset are included in the config file.

# Citation
If you find this code useful for your work, please cite:
```
@article{hsu2024tripod,
  title={Tripod: Three Complementary Inductive Biases for Disentangled Representation Learning},
  author={Kyle Hsu and Jubayer Ibn Hamid and Kaylee Burns and Chelsea Finn and Jiajun Wu},
  journal={arXiv preprint arXiv:2404.10282},
  year={2024},
}
```
