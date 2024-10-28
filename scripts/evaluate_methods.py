"""
Evaluate proteogram approach vs. gtalign.
Pertinent metric calculation info can be found at 
https://weaviate.io/blog/retrieval-evaluation-metrics
"""
import pandas as pd
import numpy as np
import os
import glob
import re
import shutil

from Bio.SCOP import Scop


# Number of top matches returned (k)
TOP_K = 5

def read_gtalign_results(gtalign_results_dir):
    """Read gtalign results into a pandads dataframe"""
    files = glob.glob(os.path.join(gtalign_results_dir, '*.out'))

    results = []
    for file in files:
        pdb_id_query = os.path.basename(file)
        pdb_id_query = pdb_id_query[1:5].upper() + '_' + pdb_id_query[5].upper()
        tmp = [pdb_id_query]
        idx = 0
        with open(file, 'r') as fin:
            i = 0
            for line in fin:
                if line[:7] == '     1 ':
                    idx = i
                    break
                i+=1
        with open(file, 'r') as fin:
            i = 0
            for line in fin:
                if i >= idx and i < (idx + TOP_K):
                    pdb_id = ''
                    try:
                        pdb_id = re.search('/d(.*?)\.ent', line).group(1).upper()
                        pdb_id = pdb_id[:4]+'_'+pdb_id[4]
                    except:
                        pass
                    tmp.append(pdb_id)
                i+=1
                if i >= (idx + TOP_K):
                    break
        results.append(tmp)

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Files and folders (output gets recreated)
    scope_label_file = './data/queries_scope_gtalign_experiment/scope_pdbs_from_downloaded_structures_list.txt'
    proteogram_sim_results = './data/queries_scope_gtalign_experiment/results_fold_ft/proteogram_similarity_results.tsv'
    proteogram_dir = './data/queries_scope_gtalign_experiment/proteograms_gtalign_list'
    gtalign_results_dir = './data/queries_scope_gtalign_experiment/gtalign_out'
    search_images_dir = './data/queries_scope_gtalign_experiment/results_fold_ft_images_search'
    save_bad_searches_dir = './data/queries_scope_gtalign_experiment/bad_proteogram_searches_fold_ft'
    save_good_searches_dir = './data/queries_scope_gtalign_experiment/good_proteogram_searches_fold_ft'

    scope_cla_handle = './data/dir.cla.scope.2.08-stable.txt'
    scope_des_handle = './data/dir.des.scope.2.08-stable.txt'
    scope_hie_handle = './data/dir.hie.scope.2.08-stable.txt'

    # Create some output directories (delete and recreate if exists)
    if os.path.exists(save_bad_searches_dir):
        shutil.rmtree(save_bad_searches_dir)
    os.makedirs(save_bad_searches_dir)
  
    if os.path.exists(save_good_searches_dir):
        shutil.rmtree(save_good_searches_dir)
    os.makedirs(save_good_searches_dir)

    # Create a Scop object
    scop = Scop(cla_handle=open(scope_cla_handle, 'r'),
                des_handle=open(scope_des_handle, 'r'),
                hie_handle=open(scope_hie_handle, 'r'))

    # Use the scope_label_file for a protein list (using the sid in the
    # first column to query local SCOPe info)
    scope_prots = []
    with open(scope_label_file, 'r') as fin:
        for line in fin:
            try:
                spl_line = line.split()
                # Get a specific domain by its SCOP identifier (sid) found in scope_label_file
                scop_entry = scop.getDomainBySid(spl_line[0].replace('.ent',''))
                # Parse out info for our dataframe
                sccs = scop_entry.sccs
                sccs_spl = sccs.split('.')
                pdb_id = spl_line[0][1:5].upper()
                chain = spl_line[0][5].upper()
                pdb_id_chain = pdb_id+'_'+chain
                prot_file = f'{pdb_id}_{chain}.jpg'
                cls, fold, sfam, fam = sccs_spl[0], '.'.join(sccs_spl[:2]), '.'.join(sccs_spl[:3]), sccs
                #print(f'from label file | scope entry: {scop_entry} | family: {fam}')
                scope_prots.append([pdb_id, pdb_id_chain, prot_file, cls, fold, sfam, fam])
            except Exception as e:
                print(e)
    # Place data into a dataframe for easier access
    label_df = pd.DataFrame(scope_prots,
                            columns=['pdb_id', 'pdb_id_chain', 'proteogram_file', 'class', 'fold', 'superfamily', 'family'])
    print(label_df.nunique())

    # Calculate Precision@K's and average for protegram approach
    proteogram_res_df = pd.read_csv(proteogram_sim_results, sep='\t')
    precision_at_ks_fams = []
    precision_at_ks_sfams = []
    precision_at_ks_folds = []
    precision_at_ks_classes = []
    for i in range(proteogram_res_df.shape[0]):
        prot_file = os.path.basename(proteogram_res_df.iloc[i,0])
        query_fam = label_df.loc[label_df['proteogram_file'] == prot_file, 
                                     'family'].iloc[0]
        query_sfam = label_df.loc[label_df['proteogram_file'] == prot_file, 
                                      'superfamily'].iloc[0]
        query_fold = label_df.loc[label_df['proteogram_file'] == prot_file, 
                                      'fold'].iloc[0]
        query_class = label_df.loc[label_df['proteogram_file'] == prot_file,
                                      'class'].iloc[0]
        # Go through search results and find family
        tp_fam = 0
        tp_sfam = 0
        tp_fold = 0
        tp_class = 0
        for target in proteogram_res_df.iloc[i,1:]:
            target_file = os.path.basename(target.split(',')[0])
            try:
                target_fam = label_df.loc[label_df['proteogram_file'] == target_file, 
                                          'family'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_fam}.')
            try:
                target_sfam = label_df.loc[label_df['proteogram_file'] == target_file, 
                                           'superfamily'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_sfam}.')
            try:
                target_fold = label_df.loc[label_df['proteogram_file'] == target_file, 
                                           'fold'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_fold}.')
            try:
                target_class = label_df.loc[label_df['proteogram_file'] == target_file,
                                           'class'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_class}.')
            if query_fam == target_fam:
                tp_fam+=1
            if query_sfam == target_sfam:
                tp_sfam+=1
            if query_fold == target_fold:
                tp_fold+=1
            if query_class == target_class:
                tp_class+=1
        precision_at_ks_fams.append(tp_fam/TOP_K)
        precision_at_ks_sfams.append(tp_sfam/TOP_K)
        precision_at_ks_folds.append(tp_fold/TOP_K)
        precision_at_ks_classes.append(tp_class/TOP_K)
        # Save images of those with no family agreement (tp/TOP_K)
        pdb_id = os.path.basename(prot_file)[0:4]
        chain_id = os.path.basename(prot_file)[5]
        score_for_top_sims = tp_fold/TOP_K
        if score_for_top_sims == 0.2:
            to_copy = f'{pdb_id}_{chain_id}_top_sims.jpg'
            shutil.copy(os.path.join(search_images_dir,
                                     to_copy),
                        os.path.join(save_bad_searches_dir, to_copy.replace('top_sims', f'top_sims_{query_fold}_pk{score_for_top_sims:.4f}')))
        # Save images of those with complete family agreement (tp_fam/TOP_K is 1)
        if score_for_top_sims == 1.0:
            to_copy = f'{pdb_id}_{chain_id}_top_sims.jpg'
            shutil.copy(os.path.join(search_images_dir,
                                     to_copy),
                                     os.path.join(save_good_searches_dir, to_copy.replace('top_sims', f'top_sims_{query_fold}_pk{score_for_top_sims:.4f}')))
            

    print(f'Proteogram average precision@K for families = {np.mean(precision_at_ks_fams)}')
    print(f'Proteogram average precision@K for superfamilies = {np.mean(precision_at_ks_sfams)}')
    print(f'Proteogram average precision@K for folds = {np.mean(precision_at_ks_folds)}')
    print(f'Proteogram average precision@K for classes = {np.mean(precision_at_ks_classes)}')

    # Calculate Precision@K's and average for protegram approach
    gtalign_res_df = read_gtalign_results(gtalign_results_dir)
    # Save it
    gtalign_res_df.to_csv(os.path.join(gtalign_results_dir, 'combined_gtalign_results.tsv'),
            sep='\t',
            index=False)
    precision_at_ks_fams = []
    precision_at_ks_sfams = []
    precision_at_ks_folds = []
    precision_at_ks_classes =[]
    for i in range(gtalign_res_df.shape[0]):
        pdb_id = gtalign_res_df.iloc[i,0].upper()
        try:
            query_fam = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                     'family'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_fam.')
        try:
            query_sfam = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                      'superfamily'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_sfam.')
        try:
            query_fold = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                      'fold'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_fold.')
        try:
            query_class = label_df.loc[label_df['pdb_id_chain'] == pdb_id,
                                      'class'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_class.')
        # Go through search results and find family
        tp_fam = 0
        tp_sfam = 0
        tp_fold = 0
        tp_class = 0
        for target in gtalign_res_df.iloc[i,1:]:
            try:
                target_fam = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'family'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_fam = -1
            try:
                target_sfam = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'superfamily'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                targe_sfam = -1
            try:
                target_fold = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'fold'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_fold = -1
            try:
                target_class = label_df.loc[label_df['pdb_id_chain'] == target,
                                            'class'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_class = -1
            if query_fam == target_fam:
                tp_fam+=1
            if query_sfam == target_sfam:
                tp_sfam+=1
            if query_fold == target_fold:
                tp_fold+=1
            if query_class == target_class:
                tp_class+=1
        precision_at_ks_fams.append(tp_fam/TOP_K)
        precision_at_ks_sfams.append(tp_sfam/TOP_K)
        precision_at_ks_folds.append(tp_fold/TOP_K)
        precision_at_ks_classes.append(tp_class/TOP_K)
    print(f'gtalign average precision@K for families = {np.mean(precision_at_ks_fams)}')
    print(f'gtalign average precision@K for superfamilies = {np.mean(precision_at_ks_sfams)}')
    print(f'gtalign average precision@K for folds = {np.mean(precision_at_ks_folds)}')
    print(f'gtalign average precision@K for classes = {np.mean(precision_at_ks_classes)}')

