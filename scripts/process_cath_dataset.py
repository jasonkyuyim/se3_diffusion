"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Removes all files that are too large.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to either prespecified or default path.
"""

import dataclasses
import functools as fn
import mdtraj as md
import pandas as pd
import os
import multiprocessing as mp
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from data import mmcif_parsing
from data import utils as du
from data import errors
from data import parsers
from analysis import utils as au
import json

CATH_DIR = '/data/rsg/chemistry/jyim/projects/protein_diffusion/data/cath'
CATH_DOMAIN_FILE = os.path.join(CATH_DIR, 'cath-domain-list.txt')
CATH_NR40_FILE = os.path.join(CATH_DIR, 'cath-dataset-nonredundant-S40.list')
CATH_SPLIT_FILE = os.path.join(CATH_DIR, 'chain_set_splits.json')
MMCIF_DIR = '/data/rsg/chemistry/jyim/large_data/pdb/30_08_2021/mmCIF'
WRITE_DIR = '/data/rsg/chemistry/jyim/large_data/cath'
MAX_RESOLUTION = 5.0
MAX_LEN = 500

def _parse_cath_files():
    with open(CATH_NR40_FILE) as f:
        cath_nr40_ids = f.read().split('\n')[:-1]
    cath_nr40_chains = list(set(cath_id[:5] for cath_id in cath_nr40_ids))
    chain_set = sorted([(name[:4], name[4]) for name in  cath_nr40_chains])

    cath_nodes = defaultdict(list)
    with open(CATH_DOMAIN_FILE,'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
        for line in lines:
            entries = line.split()
            cath_id, cath_node = entries[0], '.'.join(entries[1:4])
            chain_name = cath_id[:4] + '.' + cath_id[4]
            cath_nodes[chain_name].append(cath_node)
    cath_nodes = {key:list(set(val)) for key,val in cath_nodes.items()}

    with open(CATH_SPLIT_FILE) as f:
        cath_splits = json.load(f)
    chain_to_split = {}
    for split, split_chains in cath_splits.items():
        if split not in ['train', 'test', 'validation']:
            continue
        for chain in split_chains:
            chain_to_split[chain] = split
    return chain_set, cath_nodes, chain_to_split

CHAIN_SET, CATH_NODES, CHAIN_TO_SPLIT = _parse_cath_files()


def process_mmcif(chain_name: str):
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        chain_name: CATH example name.
        cath_code: CATH cluster eample belongs to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    pdb_name, cath_chain_id = chain_name.split('.')
    pdb_subdir = pdb_name[1:3]
    mmcif_subdir = os.path.join(MMCIF_DIR, pdb_subdir)
    mmcif_path = os.path.join(mmcif_subdir, f'{pdb_name}.cif')

    metadata = {}
    metadata['chain_name'] = chain_name
    # import pdb; pdb.set_trace()
    try:
        metadata['cath_code'] = CATH_NODES[f'{pdb_name.lower()}.{cath_chain_id}']
    except KeyError as e:
        raise errors.DataError(f'{chain_name} not found in CATH.')
    metadata['cath_split'] = CHAIN_TO_SPLIT[chain_name]

    mmcif_subdir = os.path.join(WRITE_DIR, pdb_name[1:3].lower())
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    processed_mmcif_path = os.path.join(mmcif_subdir, f'{chain_name}.pkl')
    processed_mmcif_path = os.path.abspath(processed_mmcif_path)

    metadata['processed_path'] = processed_mmcif_path
    try:
        with open(mmcif_path, 'r') as f:
            parsed_mmcif = mmcif_parsing.parse(
                file_id=pdb_name, mmcif_string=f.read())
    except FileNotFoundError as e:
        raise errors.MmcifParsingError(
            f'{pdb_name} not found.'
        )
    metadata['raw_path'] = mmcif_path
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string
    if '_pdbx_struct_assembly.oligomeric_count' in raw_mmcif:
        raw_olig_count = raw_mmcif['_pdbx_struct_assembly.oligomeric_count']
        oligomeric_count = ','.join(raw_olig_count).lower()
    else:
        oligomeric_count = None
    if '_pdbx_struct_assembly.oligomeric_details' in raw_mmcif:
        raw_olig_detail = raw_mmcif['_pdbx_struct_assembly.oligomeric_details']
        oligomeric_detail = ','.join(raw_olig_detail).lower()
    else:
        oligomeric_detail = None
    metadata['oligomeric_count'] = oligomeric_count
    metadata['oligomeric_detail'] = oligomeric_detail

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header['resolution']
    metadata['resolution'] = mmcif_resolution
    metadata['structure_method'] = mmcif_header['structure_method']
    if mmcif_resolution >= MAX_RESOLUTION:
        raise errors.ResolutionError(
            f'Too high resolution {mmcif_resolution}')
    # if mmcif_resolution == 0.0:
    #     raise errors.ResolutionError(
    #         f'Invalid resolution {mmcif_resolution}')

    # Extract all chains
    struct_chains = {
        (i, chain.id.upper()): chain
        for i, chain in enumerate(parsed_mmcif.structure.get_chains())}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    chain_dict = None
    chain_int = None
    for (i, chain_id), chain_feats in struct_chains.items():
        if cath_chain_id != chain_id:
            continue 
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain_feats, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        chain_int = i
    if chain_dict is None:
        raise errors.DataError(f'Missing chain {cath_chain_id}')

    # Process geometry features
    complex_aatype = chain_dict['aatype']
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['seq_len'] = len(complex_aatype)
    modeled_seq_len = max_modeled_idx - min_modeled_idx + 1
    metadata['modeled_seq_len'] = modeled_seq_len 
    chain_dict['modeled_idx'] = modeled_idx
    if metadata['modeled_seq_len'] > MAX_LEN:
        raise errors.LengthError(
            f'Too long {complex_aatype.shape[0]}')

    # MDtraj
    tmp_file_path = f'./{chain_name}.pdb'
    au.write_prot_to_pdb(
        chain_dict['atom_positions'],
        tmp_file_path,
        chain_dict['aatype'],
        no_indexing=True
    )
    traj = md.load(tmp_file_path)

    # SS percentage
    pdb_ss = md.compute_dssp(traj, simplified=True)
    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / modeled_seq_len
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / modeled_seq_len
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / modeled_seq_len

    # Radius of gyration
    metadata['radius_gyration'] = md.compute_rg(traj)

    os.remove(tmp_file_path)

    # Write features to pickles.
    du.write_pkl(processed_mmcif_path, chain_dict)
    # print(f'Success {chain_name}')
    # Return metadata
    return metadata


def process_serially(
        all_mmcif_paths, max_resolution, max_len, write_dir):
    all_metadata = []
    for i, mmcif_path in enumerate(all_mmcif_paths):
        try:
            start_time = time.time()
            metadata = process_mmcif(
                mmcif_path,
                max_resolution,
                max_len,
                write_dir)
            elapsed_time = time.time() - start_time
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {mmcif_path}: {e}')
    return all_metadata


def process_fn(
        mmcif_path,
        verbose=None,
        max_resolution=None,
        max_len=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_mmcif(
            mmcif_path,
            max_resolution,
            max_len,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {mmcif_path}: {e}')

def process_fn(chain_name):
    try:
        return process_mmcif(chain_name)
    except errors.DataError as e:
        print(f'Failed {chain_name}: {e}')
        return None

def main():

    if not os.path.exists(WRITE_DIR):
        os.makedirs(WRITE_DIR)

    # Parallel
    with mp.Pool(processes=100) as pool:
        all_metadata = pool.map(process_fn, CHAIN_TO_SPLIT.keys())
    all_metadata = [x for x in all_metadata if x is not None]

    # Serial
    # all_metadata = []
    # for chain_name in tqdm(CHAIN_TO_SPLIT):
    #     try:
    #         chain_metadata = process_mmcif(chain_name)
    #         all_metadata.append(chain_metadata)
    #     except errors.DataError as e:
    #         print(f'Failed {chain_name}: {e}')

    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = os.path.join(WRITE_DIR, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)

    print(
        f'Finished processing {succeeded}/{len(CHAIN_TO_SPLIT)} files')


if __name__ == "__main__":
    main()