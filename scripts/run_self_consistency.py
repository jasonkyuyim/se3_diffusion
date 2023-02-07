"""Runs self-consistency over PDB samples in a directory

python scripts/run_self_consistency.py --pdb_dir <dir>
"""
import argparse
import torch
import esm
import random
import os
import GPUtil
import numpy as np
import tree

from data import utils as du
import torch
import esm
from analysis import metrics
import subprocess
import shutil
import logging
import pandas as pd
from biotite.sequence.io import fasta

PATH_TO_MPNN = '/data/rsg/chemistry/jyim/projects/protein_diffusion/ProteinMPNN'
ESM_DIR = '/data/rsg/chemistry/jyim/.cache/torch'
torch.hub.set_dir(ESM_DIR)

FOLDING_MODEL = esm.pretrained.esmfold_v1().eval()
SELECTED_GPU = ''.join(
    [str(x) for x in GPUtil.getAvailable(
        order='memory', limit = 8)])[0]
print(f'Using GPU: {SELECTED_GPU}')
FOLDING_MODEL = FOLDING_MODEL.to(f'cuda:{SELECTED_GPU}')


# Define the parser
parser = argparse.ArgumentParser(
    description='Self consistency script.')
parser.add_argument(
    '--pdb_dir',
    help='Directory to save outputs.',
    type=str)


def run_folding(sequence, save_path):
    """Run ESMFold on sequence."""
    with torch.no_grad():
        output = FOLDING_MODEL.infer_pdb(sequence)

    with open(save_path, "w") as f:
        f.write(output)
    return output

def run_self_consistency(decoy_pdb_dir: str, reference_pdb_path: str):

    # Run PorteinMPNN
    output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
    process = subprocess.Popen([
        'python',
        f'{PATH_TO_MPNN}/helper_scripts/parse_multiple_chains.py',
        f'--input_path={decoy_pdb_dir}',
        f'--output_path={output_path}',
    ])
    _ = process.wait()
    process = subprocess.Popen([
        'python',
        f'{PATH_TO_MPNN}/protein_mpnn_run.py',
        '--out_folder',
        decoy_pdb_dir,
        '--jsonl_path',
        output_path,
        '--device',
        SELECTED_GPU,
        '--num_seq_per_target',
        '25',
        '--sampling_temp',
        '0.1',
        '--seed',
        '38',
        '--batch_size',
        '1',
    ])
    _ = process.wait()
    mpnn_fasta_path = os.path.join(
        decoy_pdb_dir,
        'seqs',
        os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
    )

    # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
    mpnn_results = {
        'tm_score': [],
        'sample_path': [],
        'header': [],
        'sequence': [],
        'rmsd': [],
        'pLDDT': [],
    }
    esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
    os.makedirs(esmf_dir, exist_ok=True)
    fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
    sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
    bb_mask = np.copy(sample_feats["bb_mask"]).astype(bool)
    sample_feats = tree.map_structure(lambda x: x[bb_mask], sample_feats)
    for i, (header, string) in enumerate(fasta_seqs.items()):
        try:
            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], esmf_feats['bb_positions'])
        except:
            continue
        mpnn_results['rmsd'].append(rmsd)
        mpnn_results['tm_score'].append(tm_score)
        mpnn_results['sample_path'].append(esmf_sample_path)
        mpnn_results['header'].append(header)
        mpnn_results['sequence'].append(string)
        mpnn_results['pLDDT'].append(np.mean(esmf_feats["b_factors"][:, 1]))

    # Save results to CSV
    csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
    mpnn_results = pd.DataFrame(mpnn_results)
    mpnn_results.to_csv(csv_path)

def main(args):
    logger = logging.getLogger(__name__)
    pdb_dir = args.pdb_dir

    # Create directories of each sample
    all_files = os.listdir(pdb_dir)
    random.shuffle(all_files)
    for sample_file in all_files:
        sample_path = os.path.join(pdb_dir, sample_file)
        if os.path.isdir(sample_path):
            continue
        sample_dir = sample_path.replace('.pdb', '')
        logger.info(f'Running self-consistency on {sample_file}')
        if os.path.exists(sample_dir):
            continue
        reference_path = os.path.join(sample_dir, sample_file)
        os.makedirs(sample_dir)
        shutil.copy(sample_path, reference_path)
        run_self_consistency(sample_dir, reference_path)
        logger.info(f'Done with {sample_file}')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)