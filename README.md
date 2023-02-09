# SE(3) diffusion model with application to protein backbone generation

Implementation for "SE(3) diffusion model with application to protein backbone generation" [arxiv link](https://arxiv.org/abs/2302.02277).
While our work is tailored towards protein backbone generation, it is in principle applicable to other domains where SE(3) is utilized.

> For those interested in non-protein applications, we have prepared a minimal notebook with SO(3) diffusion
> https://colab.research.google.com/github/blt2114/SO3_diffusion_example/blob/main/SO3_diffusion_example.ipynb 

We welcome pull requests (especially bugs) and contributions.
We will try out best to improve readability and answer questions!

![framediff-landing-page](https://github.com/jasonkyuyim/se3_diffusion/blob/master/media/denoising_framediff.gif)

## Installation

We recommend [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies.
```bash
conda env create -f se3.yml
```

Next, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

**We are working on sharing the preprocessed training dataset.**

In the meantime, it will be necessary to download PDB then preprocess it with our provided scripts.
PDB can be downloaded from RCSB: https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb.
Our scripts assume you download in **mmCIF format**.
Navigate down to "Download Protocols" and follow the instructions depending on your location.

> WARNING: Downloading PDB can take up to 1TB of space.

After downloading, you should have a directory formatted like this:
https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/ 
```
00/
01/
02/
..
zz/
```
Then run the following with <path_pdb_dir> replaced with the location of PDB.
```python
python processed_pdb_dataset.py --mmcif_dir <pdb_dir> 
```
See the script for more options. Each mmCIF will be written as a pickle file that
we read and process in the data loading pipeline. A `metadata.csv` will be saved
that contains the pickle path of each example as well as additional information
about each example for faster filtering.

### Third party source code

Our repo keeps a fork of [OpenFold](https://github.com/aqlaboratory/openfold) since we made a few changes to the source code.
Likewise, we keep a fork of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN). 
Each of these codebases are actively under development and you may want to refork.
Go give each of these repos a star if you use this codebase!

## Training

`train_se3_diffusion.py` is the training script. It utilizes [Hydra](https://hydra.cc).
Hydra does a nice thing where it will save the output, config, and overrides of each run to the `outputs/` directory organized by date and time. 

Training can be done with the following.
```python
python experiments/train_se3_diffusion.py
```
The config for training is in `config/base.yaml`.
See the config for different training options.
Training will write losses and additional information in the terminal.

We support wandb which can be turned on by setting the following option in the config.

```yaml
experiment:
    use_wandb: True
```

Multi-run can be achieved with the `-m` flag. The config must specify the sweep.
For an example, in `config/base.yaml` we can have the following:
```yaml
defaults:
  - override hydra/launcher: joblib

hydra:
  sweeper:
    params:
      model.node_embed_size: 128,256
```
This instructs hydra to use [joblib](https://joblib.readthedocs.io/en/latest/)
as a pipeline for launching a sweep over `data.rosetta.filtering.subset` for two
different values. You can specify a swep with the `-m` flag. The training script
will automatically decide which GPUs to use for each sweep. You have to make sure
enough GPUs are available on your server.

```python
python experiments/train_se3_diffusion.py -m
```
Each training run outputs the following.
* Model checkpoints are saved to `ckpt`.
* Hydra logging is saved to `outputs/` including configs and overrides.
* Samples during intermittent evaluation (see next section) are saved to `eval_outputs`.
* Wandb outputs are saved to `wandb`.

### Intermittent evaluation

Training also performs evaluation everytime a checkpoint is saved.
We select equally spaced lengths between the minimum and maximum lengths seen during training and sample multiple backbones for each length.
These are then evaluated with different metrics such as secondary structure composition, radius of gyration, chain breaks, and clashes.

> We additionally evaluate TM-score of the sample with a selected example from the training set.
> This was part of a older research effort for something like protein folding.
> We keep this since it'll likely be useful to others but it can be ignored for 
> the task of unconditional generation.

All samples and their metrics can be visualized in wandb (if it was turned on).
The terminal will print the paths to which the checkpoint and samples are saved.
```bash
[2023-02-09 15:10:25,097][__main__][INFO] - Checkpoints saved to: <ckpt_path>
[2022-02-09 15:10:25,097][__main__][INFO] - Evaluation saved to: <sample_path>
```
This can also be found in the config in Wandb by searching `ckpt_dir`.
Once you have a good run, you can copy and save the weights somewhere for inference.

## Inference

`inference_se3_diffusion.py` is the inference script. It utilizes [Hydra](https://hydra.cc).
Training can be done with the following.
```python
python experiments/inference_se3_diffusion.py
```
The config for inference is in `config/inference.yaml`.
See the config for different inference options.
By default, inference will use the published paper weights in `weights/paper_weights.pth`.
Simply change the `weights_path` to use your custom weights.
```yaml
inference:
    weights_path: <path>
```

