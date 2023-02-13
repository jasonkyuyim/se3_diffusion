#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 2
#SBATCH --output=example_1.out

# source activate mlfold

folder_with_pdbs="/data/rsg/chemistry/jyim/projects/protein_diffusion/samples/sweep_0/24D_10M_2022Y_16h_25m_04s/rsv_site_5_16-34/sample_4/mpnn/"

output_dir='/data/rsg/chemistry/jyim/projects/protein_diffusion/samples/sweep_0/24D_10M_2022Y_16h_25m_04s/rsv_site_5_16-34/sample_4/mpnn/'
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --num_seq_per_target 10 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1
