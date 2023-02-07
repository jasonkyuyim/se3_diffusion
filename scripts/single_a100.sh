#!/bin/bash
#SBATCH -p gpu
#SBATCH -J protein_diffusion_v2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/job-%j.out


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=36843

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

source activate /home/jyim/.conda/envs/protdiff

# Command line for running training

# python experiments/train_se3_diffusion.py \
#     -cn sanity_check_no_crop
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000 \
#     experiment.name=subset_1000_cosine \
#     experiment.dist_mode=single \
#     diffuser.trans_schedule=cosine

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000 \
#     experiment.name=subset_1000_t_1000 \
#     experiment.dist_mode=single \
#     diffuser.num_t=1000

# python experiments/train_se3_diffusion.py \
#     -cn subset_100
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn base \
#     experiment.name=base_big \
#     experiment.data_location=digs \
#     experiment.dist_mode=single \
#     model.num_blocks=4 \
#     model.ipa.embed_size=256 \
#     model.ipa.qk_points=8 

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_only_rot \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_only_trans \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn pdb_only_rot \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn pdb_no_trans_loss \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_exp_lin \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_lin_lin \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_lin_log \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_lin_log_2 \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_lin_log_3 \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn subset_1000_lin_log_4 \
#     experiment.dist_mode=single

python experiments/train_se3_diffusion.py \
    -cn base \
    experiment.data_location=digs \
    experiment.dist_mode=single \
    experiment.batch_size=32 \
    model.edge_embed_size=256 \
    model.ipa.embed_size=256

# python experiments/train_se3_diffusion.py \
#     -cn pdb_exp_lin \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn pdb_lin_lin \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn pdb_lin_log_3 \
#     experiment.dist_mode=single

# python experiments/train_se3_diffusion.py \
#     -cn pdb_lin_log_4 \
#     experiment.dist_mode=single
