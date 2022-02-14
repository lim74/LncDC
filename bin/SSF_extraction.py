import RNA
import os
import tqdm
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from sys import platform
    


# Secondary Structure Features (SSF)

#---------------------------------------------------------------------------------------
# extract the secondary structure of a transcript
def run_rnafold(data):
    ss, mfe = RNA.fold(data['Sequence'])
    data['Secondary_structure'] = ss
    return data

def run_rnafold_parallel(data):
    data = data.apply(run_rnafold, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 1: Secondary structure score of kmer 1
def ss_word_generator(seq, seq_second, step_size, k, frame = 0):
    for i in range(frame, len(seq), step_size):
        word = seq[i:i+k] + seq_second[i:i+k]
        if len(word) == k*2:
                yield word

def ss_kmer_ratio(seq, seq_second, step_size, k, mrna, lncrna):
    if len(seq) < k*2:
        return 0
    sum_log_ratio = 0
    ss_num = 0
    for i in ss_word_generator(seq = seq, seq_second = seq_second,
                            step_size = step_size, k = k, frame = 0):
        if (i not in mrna) or (i not in lncrna):
            continue
        if mrna[i]>0 and lncrna[i]>0:
            ratio = np.log(mrna[i]/lncrna[i])
            sum_log_ratio += ratio
        elif mrna[i]>0 and lncrna[i] == 0:
            sum_log_ratio += 1
        elif mrna[i] == 0 and lncrna[i] > 0:
            sum_log_ratio -= 1
        else:
            continue
        ss_num += 1
    ss_score = sum_log_ratio/ss_num

    return ss_score

def run_ss_k1(data, mrna_mer, lncrna_mer):
    data['SS_score_k1'] = ss_kmer_ratio(data['Sequence'],
                                        data['Secondary_structure'],
                                        1, # step size
                                        1, # kmer
                                        mrna_mer,
                                        lncrna_mer)
    return data

def run_ss_k1_parallel(data, mrna_mer, lncrna_mer):
    data = data.apply(run_ss_k1, args = [mrna_mer, lncrna_mer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 2: Secondary structure score of kmer 2
def run_ss_k2(data, mrna_mer, lncrna_mer):
    data['SS_score_k2'] = ss_kmer_ratio(data['Sequence'],
                                        data['Secondary_structure'],
                                        1, # step size
                                        2, # kmer
                                        mrna_mer,
                                        lncrna_mer)
    return data

def run_ss_k2_parallel(data, mrna_mer, lncrna_mer):
    data = data.apply(run_ss_k2, args = [mrna_mer, lncrna_mer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 3: Secondary structure score of kmer 3
def run_ss_k3(data, mrna_mer, lncrna_mer):
    data['SS_score_k3'] = ss_kmer_ratio(data['Sequence'],
                                        data['Secondary_structure'],
                                        1, # step size
                                        3, # kmer
                                        mrna_mer,
                                        lncrna_mer)
    return data

def run_ss_k3_parallel(data, mrna_mer, lncrna_mer):
    data = data.apply(run_ss_k3, args = [mrna_mer, lncrna_mer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 4: Secondary structure score of kmer 4
def run_ss_k4(data, mrna_mer, lncrna_mer):
    data['SS_score_k4'] = ss_kmer_ratio(data['Sequence'],
                                        data['Secondary_structure'],
                                        1, # step size
                                        4, # kmer
                                        mrna_mer,
                                        lncrna_mer)
    return data

def run_ss_k4_parallel(data, mrna_mer, lncrna_mer):
    data = data.apply(run_ss_k4, args = [mrna_mer, lncrna_mer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 5: Secondary structure score of kmer 5
def run_ss_k5(data, mrna_mer, lncrna_mer):
    data['SS_score_k5'] = ss_kmer_ratio(data['Sequence'],
                                        data['Secondary_structure'],
                                        1, # step size
                                        5, # kmer
                                        mrna_mer,
                                        lncrna_mer)
    return data

def run_ss_k5_parallel(data, mrna_mer, lncrna_mer):
    data = data.apply(run_ss_k5, args = [mrna_mer, lncrna_mer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 6: GC content of a transcript region where the secondary structure are paired
def GC_content_paired_ss(seq, secondary_structure):
    base_dict = {'A':0,
            'T':0,
            'G':0,
            'C':0}
    for i,v in enumerate(secondary_structure):
        if v != '.':
            base = seq[i]
            base_dict[base]+=1
    if sum(base_dict.values()) != 0:
        ratio = (base_dict['G']+base_dict['C'])/sum(base_dict.values())
    else:
        ratio = 'NaN'
    return ratio

def run_GC_paired(data):
    seq = data['Sequence']
    secondary_structure = data['Secondary_structure']
    data['GC_content_paired_ss'] = GC_content_paired_ss(seq,secondary_structure)
    return data

def run_GC_paired_parallel(data):
    data = data.apply(run_GC_paired, axis = 1)
    return data
#---------------------------------------------------------------------------------------
# feature extraction
def feature_extract_ss(dataset, thread, feature):
    run_feature_parallel = feature
    df_chunks = np.array_split(dataset, thread*10)
    
    print("Total number of dataframe chunks divided: " + str(len(df_chunks)))
    print("Chunks of secondary structure calculation finished: ")
    results = []
    with multiprocessing.Pool(thread) as pool:
        for result in tqdm.tqdm(pool.imap(run_feature_parallel, df_chunks), total = len(df_chunks), miniters = 1):
            results.append(result)
    dataset = pd.concat(results, ignore_index = False)
    
    return dataset

def feature_extract(dataset, thread, feature):
    run_feature_parallel = feature
    df_chunks = np.array_split(dataset, thread*10)
    with multiprocessing.Pool(thread) as pool:
        dataset = pd.concat(pool.map(run_feature_parallel, df_chunks), ignore_index = False)
    return dataset

def feature_extract_kmer(dataset, thread, feature, mrna_mer, lncrna_mer):
    df_chunks = np.array_split(dataset, thread*10)
    parallel_function = partial(feature, mrna_mer=mrna_mer, lncrna_mer=lncrna_mer)
    with multiprocessing.Pool(thread) as pool:
        dataset = pd.concat(pool.map(parallel_function, df_chunks), ignore_index = False)
    return dataset

def ssf_extract(dataset, thread, mrna_1mer, lncrna_1mer, mrna_2mer, lncrna_2mer,
                mrna_3mer, lncrna_3mer, mrna_4mer, lncrna_4mer, mrna_5mer, lncrna_5mer):
    # extract the secondary structure of a transcript
    print("Calculating secondary structures of the transcripts by RNAfold ... (This process may take a long time!)")
    dataset = feature_extract_ss(dataset, thread, run_rnafold_parallel)
    
    print("Extracting SSF features ...")
    # feature 1: Secondary structure score of kmer 1
    dataset = feature_extract_kmer(dataset, thread, run_ss_k1_parallel, mrna_1mer, lncrna_1mer)
    # feature 2: Secondary structure score of kmer 2
    dataset = feature_extract_kmer(dataset, thread, run_ss_k2_parallel, mrna_2mer, lncrna_2mer)
    # feature 3: Secondary structure score of kmer 3
    dataset = feature_extract_kmer(dataset, thread, run_ss_k3_parallel, mrna_3mer, lncrna_3mer)
    # feature 4: Secondary structure score of kmer 4
    dataset = feature_extract_kmer(dataset, thread, run_ss_k4_parallel, mrna_4mer, lncrna_4mer)
    # feature 5: Secondary structure score of kmer 5
    dataset = feature_extract_kmer(dataset, thread, run_ss_k5_parallel, mrna_5mer, lncrna_5mer)
    # feature 6: GC content of a transcript region where the secondary structure are paired
    dataset = feature_extract(dataset, thread, run_GC_paired_parallel)
    return dataset
