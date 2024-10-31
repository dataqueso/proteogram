"""
Create a training dataset with protein structures from SCOPe not in test set.
This script finds the SCOPe levels for a protein, creates proteograms, and
places the proteograms in folders according to a certain SCOPe level.
"""
import os
import pandas as pd
import shutil
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

import pyrotein as pr

from Bio.SCOP import Scop
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning
from Bio.PDB.Polypeptide import PPBuilder

from proteogram.constants import (
    HYDROPHOBICITY_LIST_BINARY,
    HYDROPHOBICITY_LIST,
    CHARGE_LIST
)
from proteogram.utils import read_yaml


# Distance cutoff for measuring possible residue interactions in Angstroms
ATOM_DISTANCE_CUTOFF = 15
HYDROPHOBICITY_CUTOFF = 20
SEQUENCE_LEN_LOWER_CUTOFF = 20
SEQUENCE_LEN_UPPER_CUTOFF = 1e9

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

def calc_distogram(sequence, pdb_path, chain):
    nterm = 1
    cterm = len(sequence)
    
    # Define atoms and chain used for distance matrix analysis
    backbone = ["CA"]
    #chain = 'A'

    # Read coordinates from a PDB file
    atoms_pdb = pr.atom.read(pdb_path)
    
    # Create a lookup table for this pdb
    atom_dict = pr.atom.create_lookup_table(atoms_pdb)
    
    # Obtain the chain to process
    chain_dict = atom_dict[chain]
    
    # Obtain coordinates
    xyzs = pr.atom.extract_xyz_by_atom(backbone, chain_dict, nterm, cterm)
    
    # Calculate distance matrix
    dmat = pr.distance.calc_dmat(xyzs, xyzs)
    tri_lower_diag = np.tril(dmat, k=0)
    
    # Assign upper triangle same values in symmetry
    tri_upper_diag = np.rot90(tri_lower_diag, 2)
    tri_diag = tri_lower_diag + tri_upper_diag

    tri_diag = np.nan_to_num(tri_diag)

    return tri_diag

def calc_simple_polarity_map(sequence, disto_map):
    polarity_map = np.zeros((len(sequence), len(sequence)))

    for row in range(len(sequence)):
        for col in range(len(sequence)):
            # If less residues less than cutoff num of Angstroms
            if disto_map[row,col] < ATOM_DISTANCE_CUTOFF:
                row_val = HYDROPHOBICITY_LIST_BINARY[sequence[row]]
                col_val = HYDROPHOBICITY_LIST_BINARY[sequence[col]]
                if row_val == col_val:
                    polarity_map[row,col] = 1
                    polarity_map[col,row] = 1
                # If on the diag, set to 0
                if row == col:
                    polarity_map[row,col] = 0
    return polarity_map

def calc_hydrophobicity_map(sequence, disto_map):
    hydro_map = np.zeros((len(sequence), len(sequence)))

    for row in range(len(sequence)):
        for col in range(len(sequence)):
            # If less residues less than cutoff num of Angstroms
            if disto_map[row,col] < ATOM_DISTANCE_CUTOFF:
                row_val = np.abs(HYDROPHOBICITY_LIST[sequence[row]])
                col_val = np.abs(HYDROPHOBICITY_LIST[sequence[col]])
                delta = np.abs(row_val - col_val)
                hydro_map[row,col] = delta
                hydro_map[col,row] = delta
                # If on the diag, set to 0
                if row == col:
                    hydro_map[row,col] = 0
    return hydro_map

def calc_charge_map(sequence, disto_map):
    charge_map = np.zeros((len(sequence), len(sequence)))

    for row in range(len(sequence)):
        for col in range(len(sequence)):
            # If less residues less than cutoff num of Angstroms
            if disto_map[row,col] < ATOM_DISTANCE_CUTOFF:
                row_val = CHARGE_LIST[sequence[row]]
                col_val = CHARGE_LIST[sequence[col]]
                if row_val == -1 and col_val == 1:
                    charge_map[row,col] =  1
                    charge_map[col,row] =  1
                if row_val == 1 and col_val == -1:
                    charge_map[row,col] =  1
                    charge_map[col,row] =  1
                if row == col:
                    charge_map[row,col] = 0
    return charge_map

def stack_data(distance_map, hydro_map, charge_map):
    # normalize maps
    if np.max(distance_map) != np.min(distance_map):
        distance_map = distance_map / (np.max(distance_map) - np.min(distance_map))
    if np.max(hydro_map) != np.min(hydro_map):
        hydro_map = hydro_map / (np.max(hydro_map) - np.min(hydro_map))
    final_data = np.dstack([distance_map, hydro_map, charge_map])
    return final_data

def get_sequence(pdb_file):
    """Get a protein sequence from a PDB file"""
    p = PDBParser(PERMISSIVE=0)
    structure = p.get_structure('xyz', pdb_file)
    ppb = PPBuilder()
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    return seq

if __name__ == '__main__':

    config = read_yaml('config.yml')
    scope_eval_set = config['scope_eval_set']
    scope_structures_dir = config['scope_structures_dir']
    scope_cla_file = config['scope_cla_file']
    scope_des_file = config['scope_des_file']
    scope_hie_file = config['scope_hie_file']
    training_structures_dir = config['training_structures_dir']
    training_proteograms_dir = config['training_proteograms_dir']
    label_df_out = config['label_df_out']

    # All scope 2.08 structure files
    scope_struct_files =  glob.glob(os.path.join(scope_structures_dir,
                                                '**',
                                                '*.ent'),
                                    recursive=True)

    # Create a Scop object
    scop = Scop(cla_handle=open(scope_cla_file, 'r'),
                des_handle=open(scope_des_file, 'r'),
                hie_handle=open(scope_hie_file, 'r'))

    # scope_prots has the following structure - {file_path: (cls, fold, sfam, fam)}
    scope_prots = []
    for scope_struct_file in tqdm(scope_struct_files):
        sid = os.path.basename(scope_struct_file).replace('.ent', '')
        with open(scope_struct_file, 'r') as fin:
            try:
                # Get a specific domain by its SCOP identifier (sid)
                scop_entry = scop.getDomainBySid(sid)
                # Parse out info for our dataframe
                sccs = scop_entry.sccs
                sccs_spl = sccs.split('.')
                pdb_id = sid[1:5].upper()
                chain = sid[5].upper()
                cls, fold, sfam, fam = sccs_spl[0], '.'.join(sccs_spl[:2]), '.'.join(sccs_spl[:3]), sccs
                scope_prots.append([pdb_id, chain, scope_struct_file, cls, fold, sfam, fam])
            except Exception as e:
                print(e)
    # Place data into a dataframe for easier access
    label_df = pd.DataFrame(scope_prots,
                            columns=['pdb_id', 'chain', 'structure_file', 'class', 'fold', 'superfamily', 'family'])
    
    # Save annotations from SCOPe for the proteins in the 2.08 db
    label_df.to_csv(label_df_out, sep='\t', index=False)

    # What proteins are we already using for testing proteograms
    sampled_proteins = []
    with open(scope_eval_set, 'r') as f:
        for line in f:
            sampled_proteins.append(line.rstrip())

    # Create some output directories (delete and recreate if exists)
    if os.path.exists(training_structures_dir):
        shutil.rmtree(training_structures_dir)
    os.makedirs(training_structures_dir)

    # Create some output directories (delete and recreate if exists)
    if os.path.exists(training_proteograms_dir):
        shutil.rmtree(training_proteograms_dir)
    os.makedirs(training_proteograms_dir)

    file_dict = {str(k): [] for k in label_df['class'].unique()}
    for i in range(label_df.shape[0]):
        fam = str(label_df.loc[i,'class'])
        prot_file = label_df.loc[i, 'structure_file']
        # If already using this protein in the eval experiment, skip
        if os.path.basename(prot_file) in sampled_proteins:
            continue
        if fam in file_dict:
            tmp = file_dict[fam]
            tmp.append(prot_file)
            file_dict[fam] = tmp
        else:
            file_dict[fam] = [prot_file]

    # Put 80% of the files found for the family into train folder and rest into val
    file_dict_train = {k: v[:int(len(v)*0.8)] for k, v in file_dict.items()}
    file_dict_val = {k: v[int(len(v)*0.8):] for k, v in file_dict.items()}
    train_cnt_limit = 100
    val_cnt_limit = 30
    pdb_files_final = []
    for fam in tqdm(file_dict_train.keys()):

        # Create dirs for families
        fam_train_dir = os.path.join(training_proteograms_dir, 'train', fam)
        if os.path.exists(fam_train_dir):
            shutil.rmtree(fam_train_dir)
        os.makedirs(fam_train_dir)
        fam_val_dir = os.path.join(training_proteograms_dir, 'val', fam)
        if os.path.exists(fam_val_dir):
            shutil.rmtree(fam_val_dir)
        os.makedirs(fam_val_dir)

        # Create proteograms and put into correct SCOPe level folder for
        # training model
        cnt_train_fam = 0
        for pdb_file in file_dict_train[fam]:
            sid = os.path.basename(pdb_file).replace('.ent','')
            if len(sid) > 6:
                sid = sid[:6]
            pdb_id = sid[1:5].upper()
            chain_id = sid[5].upper()
            try:
                sequence = get_sequence(pdb_file)
                if len(sequence) >= SEQUENCE_LEN_LOWER_CUTOFF and \
                    len(sequence) < SEQUENCE_LEN_UPPER_CUTOFF:
                    distance_map = calc_distogram(sequence, pdb_file, chain_id)
                    hydro_map = calc_hydrophobicity_map(sequence, distance_map)
                    charge_map = calc_charge_map(sequence, distance_map)
                    final_data = stack_data(distance_map, hydro_map, charge_map)
                    image_file = os.path.join(fam_train_dir,
                                            f'{pdb_id}_{chain_id}.jpg')
                    img = final_data.astype(float)
                    plt.imsave(image_file, img)
                    cnt_train_fam+=1
                    pdb_files_final.append(pdb_file)
            except Exception as e: 
                print(f'problem with creating or saving a proteogram for {pdb_file}: {e}.')
            # Have as many images for this fam as we wish
            if cnt_train_fam == train_cnt_limit:
                break
        
        cnt_val_fam = 0
        for pdb_file in file_dict_val[fam]:
            sid = os.path.basename(pdb_file).replace('.ent','')
            if len(sid) > 6:
                sid = sid[:6]
            pdb_id = sid[1:5].upper()
            chain_id = sid[5].upper()
            try:
                sequence = get_sequence(pdb_file)
                if len(sequence) >= SEQUENCE_LEN_LOWER_CUTOFF and \
                    len(sequence) < SEQUENCE_LEN_UPPER_CUTOFF:
                    distance_map = calc_distogram(sequence, pdb_file, chain_id)
                    hydro_map = calc_hydrophobicity_map(sequence, distance_map)
                    charge_map = calc_charge_map(sequence, distance_map)
                    final_data = stack_data(distance_map, hydro_map, charge_map)
                    image_file = os.path.join(fam_val_dir,
                                            f'{pdb_id}_{chain_id}.jpg')
                    img = final_data.astype(float)
                    plt.imsave(image_file, img)
                    cnt_val_fam+=1
                    pdb_files_final.append(pdb_file)
            except Exception as e:
                print(f'problem with creating or saving a proteogram for {pdb_file}: {e}.')
            if cnt_val_fam == val_cnt_limit:
                break
        
        # If empty folders remove the fam folder
        if len(glob.glob(os.path.join(fam_train_dir, '*.jpg'))) == 0:
            print(f'Empty dir: {fam_train_dir}')
            if os.path.exists(fam_train_dir):
                shutil.rmtree(fam_train_dir)
            if os.path.exists(fam_val_dir):
                    shutil.rmtree(fam_val_dir)
        if len(glob.glob(os.path.join(fam_val_dir, '*.jpg'))) == 0:
            print(f'Empty dir {fam_val_dir}')
            if os.path.exists(fam_train_dir):
                shutil.rmtree(fam_train_dir)
            if os.path.exists(fam_val_dir):
                shutil.rmtree(fam_val_dir)

    # Copy structure files from final list above for gtalign experiments
    for pdb_file in pdb_files_final:
        bname = os.path.basename(pdb_file)
        shutil.copy(pdb_file, os.path.join(training_structures_dir, bname))
