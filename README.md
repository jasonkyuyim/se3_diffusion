# Protein diffusion


## Data location

The repository is set-up to work on both MIT (`rosetta`) and UW (`digs`) servers.
The location can be switched with the command line flags

* `experiment.data_location=rosetta`
* `experiment.data_location=digs`

The only difference is the data_loader that gets used -- either `rosetta_data_loader.py` or `digs_data_loader.py`.

TODO: verify both data_loaders do the same preprocessing.

## Installation

A quick and hacky option is to activate Jason's conda. 

```bash
conda activate /data/rsg/chemistry/jyim/miniconda3/envs/dev
```

Install the protein_diffusion package: Run `pip install -e .` with `protein_diffusion` as the current directory.

Install hydra joblib 
https://hydra.cc/docs/plugins/joblib_launcher/

(from Ivan A.) To install NVIDIA's SE3-transformer

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd ./DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer
python ./setup.py install
```

* An up to date version of dgl (deep graph library) is also needed.

Install https://github.com/e3nn/e3nn 


## Training

Add any good model or valuable experiments to this [preedsheet of models](https://docs.google.com/spreadsheets/d/1I3AsxLPpaMEPgekprvyJnZ4oO_Ii_qhNmCPjKW5lAbw/edit?usp=sharing).


### Launching experiments

`train_se3_diffusion.py` is the training script. It utilizes [Hydra](https://hydra.cc). Configs for training are in the `config` directory.
Hydra does a nice thing where it will save the output, config, and overrides of each run to the `outputs/` directory organized by date and time. 

Training can be done with three different accelerator strategies

```bash
python experiments/train_se3_diffusion.py experiment.dist_mode=single  # Single node single GPU
python experiments/train_se3_diffusion.py experiment.dist_mode=multi  # Single node multi-GPU
python experiments/train_se3_diffusion.py experiment.dist_mode=slurm  # Slurm multi-GPU
```

For now we only use Single process/GPU.

By default, the experiment uses `config/base.yaml` but this can be overridden
to use a different config with the `-cn` or `--config-name` flag. See [hydra docs](https://hydra.cc/docs/advanced/hydra-command-line-flags/) for more options of modifying the config file in the command line.
The config must exist in the `config/` directory.

```bash
python experiments/train_se3_diffusion.py -cn lin_exp_schedule
```

Specific config fields can be modified on the command line. For instance, in base.yaml we have

```yaml
experiment:
    use_wandb: True
```

However, you probably don't want to use wandb when debugging locally. Add `experiment.use_wandb=False` flag to turn off Wandb. This patern applies to any field in the config.

```bash
python experiments/train_se3_diffusion.py experiment.use_wandb=False
```

Use `tmux` (or `longtmux` on the rosettas) to start experiments and have them continue running.
Another option is to send the experiments into the background.

```bash
python experiments/train_se3_diffusion.py &
```

Hydra saves all logging into outputs/<date>/<time>/<train_se3_diffusion.log
as well as the configs used to run the experiment in the same directory.

Multi-run can be achieved with the `-m` flag. The config must specify the sweep.
For an example, in `config/base.yaml` we have the following:
```yaml
defaults:
  - override hydra/launcher: joblib

hydra:
  sweeper:
    params:
      data.rosetta.filtering.subset: 1000,null
```

This instructs hydra to use [joblib](https://joblib.readthedocs.io/en/latest/)
as a pipeline for launching a sweep over `data.rosetta.filtering.subset` for two
different values. You can specify a swep with the `-m` flag. The training script
will automatically decide which GPUs to use for each sweep. You have to make sure
enough GPUs are available on the server.

```bash
python experiments/train_se3_diffusion.py -cn base -m
```

### Evaluation

Training also performs continuous evaluation everytime a checkpoint is saved.
Wandb will log evaluation metrics as well as brief summaries of each sample.
To my surprise, Wandb can [log visualizations of proteins](https://docs.wandb.ai/guides/track/log/media#molecules). However, it doesn't allow for customization so the visualizations are poor and misleading.
Instead, we save the samples in `results/`. Your experiment will print out where
checkpoints and evaluation samples are saved:

```bash
[2022-09-13 15:10:25,097][__main__][INFO] - Checkpoints saved to: ./pkl_jar/ckpt/old_schedule_no_aatype_0/13D_09M_2022Y_15h_10m_25s
[2022-09-13 15:10:25,097][__main__][INFO] - Evaluation saved to: ./results/old_schedule_no_aatype_0/13D_09M_2022Y_15h_10m_25s
```

This can also be found in the config in Wandb. You may visualize the PDBs in the
results directory either using pymol or VS code's [protein viewer](https://marketplace.visualstudio.com/items?itemName=ArianJamasb.protein-viewer).


## Inference

TODO
