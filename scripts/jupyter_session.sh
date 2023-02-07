#!/bin/bash
#SBATCH -p gpu
#SBATCH -J jupyter_lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --nodes=1
#SBATCH --gres=gpu:quadro:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/jupyter-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=36843

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

source activate /home/jyim/.conda/envs/protdiff
jupyter lab --ip=$(hostname) --no-browser
