"""Runs ESMFold over sequences in fasta file.

python scripts/run_esmfold.py --fasta_path /data/rsg/chemistry/jyim/projects/protein_diffusion/notebooks/unconditional_sequences.fa --output_dir /data/scratch/jyim/esmfold_outputs
"""

import argparse
import torch
import esm
import random
import os
import GPUtil
from biotite.sequence.io import fasta


# Define the parser
parser = argparse.ArgumentParser(
    description='mmCIF processing script.')
parser.add_argument(
    '--fasta_path',
    help='Path to Fasta file.',
    type=str)
parser.add_argument(
    '--output_dir',
    help='Directory to save outputs.',
    type=str)

def main(args):
    esm_dir = '/data/rsg/chemistry/jyim/.cache/torch'
    torch.hub.set_dir(esm_dir)
    folding_model = esm.pretrained.esmfold_v1().eval()
    available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8)])
    device = f'cuda:{available_gpus[0]}'
    print(f'Using GPU: {device}')
    folding_model = folding_model.to(device)
    fasta_seqs = fasta.FastaFile.read(args.fasta_path).items()
    fasta_seqs = list(enumerate(fasta_seqs))
    random.shuffle(fasta_seqs)

    output_dir = args.output_dir
    for i, (header, string) in fasta_seqs:
        if i % 100 == 0:
            print(f'Done with {i}')
        output_path = os.path.join(output_dir, header+'.pdb')
        if os.path.exists(output_path):
            continue
        print(f'Running {header}')
        with torch.no_grad():
            output = folding_model.infer_pdb(string)
        with open(output_path, "w") as f:
            f.write(output)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)