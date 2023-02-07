#!/bin/bash
#SBATCH -p gpu
#SBATCH -J protein_diffusion_v2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --output=slurm/job-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=36843

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

source activate /home/jyim/.conda/envs/protdiff

# Command line for running training

python experiments/train_se3_diffusion.py \
    -cn base \
    experiment.data_location=digs \
    experiment.dist_mode=slurm
