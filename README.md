# SE(3) diffusion model with application to protein backbone generation

## Description
Implementation for "SE(3) diffusion model with application to protein backbone generation" [arxiv link](https://arxiv.org/abs/2302.02277).
While our work is tailored towards protein backbone generation, it is in principle applicable to other domains where SE(3) is utilized.

> For those interested in non-protein applications, we have prepared a minimal notebook with SO(3) diffusion
> https://colab.research.google.com/github/blt2114/SO3_diffusion_example/blob/main/SO3_diffusion_example.ipynb 

We have codebase updates we plan to get around to.

* [In the works] Refactor score framework to be more readable and match the paper's math. See the [refactor branch](https://github.com/jasonkyuyim/se3_diffusion/tree/unsupported_refactor).
* Set-up easily downloadable training data.

We welcome pull requests (especially bug fixes) and contributions.
We will try out best to improve readability and answer questions!

If you use our work then please cite
```
@article{yim2023se,
  title={SE (3) diffusion model with application to protein backbone generation},
  author={Yim, Jason and Trippe, Brian L and De Bortoli, Valentin and Mathieu, Emile and Doucet, Arnaud and Barzilay, Regina and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2302.02277},
  year={2023}
}
```


Other protein diffusion codebases:
* Pretrained protein backbone diffusion: [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
* Protein-ligand docking: [DiffDock](https://github.com/gcorso/DiffDock)
* Protein torsion angles: [FoldingDiff](https://github.com/microsoft/foldingdiff/)
* Protein C-alpha backbone generation: [ProtDiff/SMCDiff](https://github.com/blt2114/ProtDiff_SMCDiff)

LICENSE: MIT

![framediff-landing-page](https://github.com/jasonkyuyim/se3_diffusion/blob/master/media/denoising.gif)

# Table of Contents
- [SE(3) diffusion model with application to protein backbone generation](#se3-diffusion-model-with-application-to-protein-backbone-generation)
  - [Description](#description)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
    - [Third party source code](#third-party-source-code)
- [Inference](#inference)
- [Training](#training)
    - [Downloading the PDB for training](#downloading-the-pdb-for-training)
    - [Launching training](#launching-training)
    - [Intermittent evaluation](#intermittent-evaluation)
- [Acknowledgements](#acknowledgements)


# Installation

We recommend [miniconda](https://docs.conda.io/en/main/miniconda.html) (or anaconda).
Run the following to install a conda environment with the necessary dependencies.
```bash
conda env create -f se3.yml
```

Next, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

### Third party source code

Our repo keeps a fork of [OpenFold](https://github.com/aqlaboratory/openfold) since we made a few changes to the source code.
Likewise, we keep a fork of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).
Each of these codebases are actively under development and you may want to refork.
We use copied and adapted several files from the [AlphaFold](https://github.com/deepmind/alphafold) primarily in `/data/`, and have left the DeepMind license at the top of these files.
For a differentiable pytorch implementation of the Logarithmic map on SO(3) we adapted two functions form [geomstats](https://github.com/geomstats/geomstats).
Go give these repos a star if you use this codebase!

# Inference

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
Samples will be saved to `output_dir` in the `inference.yaml`. By default it is
set to `./inference_outputs/`. Sample outputs will be saved as follows,

```shell
inference_outputs
└── 12D_02M_2023Y_20h_46m_13s           # Date time of inference.
    ├── inference_conf.yaml             # Config used during inference.
    └── length_100                      # Sampled length 
        ├── sample_0                    # Sample ID for length
        │   ├── bb_traj_1.pdb           # x_{t-1} diffusion trajectory
        │   ├── sample_1.pdb            # Final sample
        │   ├── self_consistency        # Self consistency results        
        │   │   ├── esmf                # ESMFold predictions using ProteinMPNN sequences
        │   │   │   ├── sample_0.pdb
        │   │   │   ├── sample_1.pdb
        │   │   │   ├── sample_2.pdb
        │   │   │   ├── sample_3.pdb
        │   │   │   ├── sample_4.pdb
        │   │   │   ├── sample_5.pdb
        │   │   │   ├── sample_6.pdb
        │   │   │   ├── sample_7.pdb
        │   │   │   └── sample_8.pdb
        │   │   ├── parsed_pdbs.jsonl   # Parsed chains for ProteinMPNN
        │   │   ├── sample_1.pdb
        │   │   ├── sc_results.csv      # Summary metrics CSV 
        │   │   └── seqs                
        │   │       └── sample_1.fa     # ProteinMPNN sequences
        │   └── x0_traj_1.pdb           # x_0 model prediction trajectory
        └── sample_1                    # Next sample
```

# Training

### Downloading the PDB for training
To get the training dataset, first download PDB then preprocess it with our provided scripts.
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
In this directory, unzip all the files: 
```
gzip -d **/*.gz
```
Then run the following with <path_pdb_dir> replaced with the location of PDB.
```python
python process_pdb_dataset.py --mmcif_dir <pdb_dir> 
```
See the script for more options. Each mmCIF will be written as a pickle file that
we read and process in the data loading pipeline. A `metadata.csv` will be saved
that contains the pickle path of each example as well as additional information
about each example for faster filtering.

For PDB files, we provide some starter code in `process_pdb_files.py`  of how to
modify `process_pdb_dataset.py` to work with PDB files (as we did at an earlier
point in the project). **This has not been tested.** Please make a pull request
if you create a PDB file processing script. 

### Launching training 
`train_se3_diffusion.py` is the training script. It utilizes [Hydra](https://hydra.cc).
Hydra does a nice thing where it will save the output, config, and overrides of each run to the `outputs/` directory organized by date and time. By default we use 2 GPUs to fit proteins up to length 512. The number of GPUs can be changed with the `num_gpus` field in `base.yml`.

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

# Acknowledgements

Thank you to the following for pointing out bugs:
* longlongman

