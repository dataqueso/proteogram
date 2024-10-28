import glob
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
import warnings
from tqdm import tqdm

import pyrotein as pr
from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning
from Bio.PDB.Polypeptide import PPBuilder

from proteogram.constants import (
    HYDROPHOBICITY_LIST_BINARY,
    HYDROPHOBICITY_LIST,
    CHARGE_LIST
)


# Distance cutoff for measuring possible residue interactions in Angstroms
ATOM_DISTANCE_CUTOFF = 15
HYDROPHOBICITY_CUTOFF = 20
SEQUENCE_LEN_LOWER_CUTOFF = 20
SEQUENCE_LEN_UPPER_CUTOFF = 1e9

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

def plot_maps(data, sequence, color_bar=True, filename=None):
    """Plot the numpy array"""
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(7, 7)
    # fig.set_dpi(300)
    # img = ax.imshow(data)

    # X-tick labels (sequence)
    ax.set_xticks(range(len(sequence)))
    ax.tick_params()
    x_label_list=list(sequence)
    ax.set_xticklabels(x_label_list)

    # X-tick labels (sequence)
    ax.set_yticks(range(len(sequence)))
    ax.tick_params()
    ax.set_yticklabels(x_label_list, rotation=90)

    # Colorbar
    if color_bar == True:
        cbar = fig.colorbar(img, ax=ax)

    if filename:
        # Visualize a few distance matrix
        fl_dmat = os.path.join(filename)
        #img = data.astype('uint8')
        img = data.astype(float)
        plt.imsave(fl_dmat, img)


def calc_distogram(sequence, pdb_path, chain):
    nterm = 1
    cterm = len(sequence)
    
    # Define atoms and chain used for distance matrix analysis
    backbone = ["CA"]

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
    structures_dir = './data/queries_scope_gtalign_experiment/scope_pdbs_from_downloaded_structures'
    proteograms_output_dir = './data/queries_scope_gtalign_experiment/proteograms_gtalign_list'

    # If not exists, make output dir
    if os.path.exists(proteograms_output_dir):
        shutil.rmtree(proteograms_output_dir)
    os.makedirs(proteograms_output_dir)

    start = time()
    pdb_files = glob.glob(os.path.join(structures_dir, '**', '*.ent'), recursive=True)
    keyerrs = 0
    for pdb_file in tqdm(pdb_files):
        bname =  os.path.basename(pdb_file)
        chain_id =bname[1:5].upper()+':'+ bname[5].upper()
        try:
            sequence = get_sequence(pdb_file)
            if len(sequence) >= SEQUENCE_LEN_LOWER_CUTOFF and \
                len(sequence) < SEQUENCE_LEN_UPPER_CUTOFF:
                distance_map = calc_distogram(sequence, pdb_file, chain_id[-1])
                hydro_map = calc_hydrophobicity_map(sequence, distance_map)
                charge_map = calc_charge_map(sequence, distance_map)
                final_data = stack_data(distance_map, hydro_map, charge_map)
                image_file = os.path.join(proteograms_output_dir,
                                          f'{chain_id.replace(":","_")}.jpg')
                img = final_data.astype(float)
                plt.imsave(image_file, img)
        except KeyError as e:
            # print(f'KeyError for {e}')
            keyerrs+=1
    print(f'Number of key errors = {keyerrs}')

    print(f'Computation took {time()-start} seconds')

