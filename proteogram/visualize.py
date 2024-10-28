import nglview as nv
from Bio.PDB import PDBParser
from Bio import SeqIO
from ase import Atom, Atoms
from .constants import BACKBONE_POSITIONS
from .utils import get_3letter_res_name


def draw_atoms_ngl(pdb_filename):
    """
    PDB protein structure visualization of backbone atoms using
    NGLView.
    
    Arguments
    ---------
    pdb_filename : str
        PDB protein structure file

    Returns
    -------
    view : nglview.widget.NGLWidget
    """
    pdb_parser = PDBParser()
    pdb_structure = pdb_parser.get_structure("pdb_struct", pdb_filename)
    query_seqres = SeqIO.parse(pdb_filename, 'pdb-seqres')
    
    pdb_sequence = []
    for chain in query_seqres:
        pdb_sequence.extend(chain.seq)
    
    pdb_atoms = pdb_structure.get_atoms()
    
    formula = ""
    positions = []
    for i, r in enumerate(pdb_sequence):
        res_name = get_3letter_res_name(r)
        for meta, at in zip(BACKBONE_POSITIONS[res_name], pdb_atoms):
            symbol = meta[0] if meta[0] != "CA" else "C"
            positions.append(at.coord)
            formula += symbol
    view = nv.show_ase(Atoms(formula, positions=positions))
    return view