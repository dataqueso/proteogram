"""
Proteogram (image) search for a single (new) structure
- Use a structure file from PDB (example below)
- Separate out chains/domains
- Save as separate domains in SCOPe naming format
- Convert domains to proteograms
- Search against a large DB of proteograms for similarity
"""
from time import time
import os
import pickle
import torch
import matplotlib.pyplot as plt

from proteogram.image_similarity import Img2Vec
from proteogram.utils import read_yaml, split_by_chain_and_save
from proteogram.proteogram import Proteogram


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # Run embedding vs loading saved embeddings (set to True if any 
    # changes to proteograms, but only if on CUDA-supported machine if
    # originally created with CUDA or CPU if only created on CPU).
    embed = True

    # Modify the following:
    # example downloaded structure path (in PDB)
    new_structure_path = './data/search_experiments/query/AF-A0A3M6TU40-F1-model_v4.pdb'
    # this can be made up, just make 7 characters long
    sid_id = 'daf-a1_'
    # choose your chain of interest
    chain_id = 'A'
    # output for processed structure, proteogram and search result
    struct_dir = './data/search_experiments/query/' 

    # Read from the config.yml
    config = read_yaml('config.yml')
    top_k = config['top_k']
    model_file = config['model_file']
    embed_file = config['embed_file']
    results_file = config['proteogram_sim_results']
    dataset_dir = config['proteograms_dir']
    save_images_dir =config['search_images_dir']

    # If the output dir exists, don't recreate, otherwise make one
    if os.path.exists(save_images_dir):
        print(f'Directory {save_images_dir} exists, will use and may overwrite.')
    else:
        os.makedirs(save_images_dir)

    start = time()
    # Initialize Img2Vec with model from torchvision
    img_sim = Img2Vec(model_file, weights='DEFAULT')
    print(f'Took {time()-start} seconds to initialize Img2Vec object.')

    # Create dataset and create embeddings
    start = time()
    with torch.no_grad():
        if embed:
           img_sim.embed_dataset(str(dataset_dir))
           # Save embeddings
           with open(embed_file, 'wb') as pklout:
               pickle.dump(img_sim.dataset, pklout)
           print(f'Took {time()-start} seconds to create image embeddings.')
        else:
           with open(embed_file, 'rb') as pklin:
                img_sim.dataset = pickle.load(pklin)


    # Get chain "A" and make up sid (for now)
    pdb_id = os.path.basename(new_structure_path)[:-4]

    scope_struct_path = struct_dir + os.sep + sid_id + '.ent'
    split_by_chain_and_save(new_structure_path,
                            chain_id=chain_id,
                            scope_sid=sid_id,
                            scope_pdbs_dir=struct_dir)

    # Create proteogram
    proteogram = Proteogram(new_structure_path,
                            atom_distance_cutoff=15)
    sequence = proteogram.get_sequence()
    distance_map = proteogram.calc_distogram(sequence, chain_id)
    hydro_map = proteogram.calc_hydrophobicity_map(sequence, distance_map)
    charge_map = proteogram.calc_charge_map(sequence, distance_map)
    final_data = proteogram.stack_data(distance_map, hydro_map, charge_map)
    image_file = os.path.join(struct_dir,
                                        f'{pdb_id}_{chain_id}.jpg')
    img = final_data.astype(float)
    plt.imsave(image_file, img)
        
    # Search to find similar images using cosine-similarity amongst embeddings
    # Save search results as images (TOP_K) to save_images_dir
    start = time()
    scores_n_arr = img_sim.similarities_new_image(image_file,
                                                  n=top_k,
                                                  save_result_images_dir=struct_dir)

    print(scores_n_arr)
    
    print(f'Took {time()-start} seconds overall.')

    




