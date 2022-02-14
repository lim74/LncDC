import RNA
import os
import itertools
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from collections import Counter
from sys import platform
import tqdm
    



# Secondary Structure-Based Features (SBF)

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
# make ss kmer tables
def all_ss_combi(k):
    nucleotides = [''.join(x) for x in itertools.product('ATGC',repeat = k)]
    structure_type = [''.join(x) for x in itertools.product('.()',repeat = k)]
    for i in itertools.product(nucleotides, structure_type):
        yield ''.join(i)

def ss_word_generator(seq, seq_second, step_size, k, frame = 0):
    for i in range(frame, len(seq), step_size):
        word = seq[i:i+k] + seq_second[i:i+k]
        if len(word) == k*2:
                yield word

def ss_frequency_seqs(seq_set,seq_second_set,step_size=1,k=1,frame=0,min_count=0):
    seq_num = 0
    for i in range(len(seq_set)):
        seq_num += 1
        if seq_num == 1:
            count_table = Counter(ss_word_generator(seq_set[i],seq_second_set[i],
                                                    step_size=step_size,k=k,frame=frame))
        else:
            count_table.update(ss_word_generator(seq_set[i],seq_second_set[i],
                                                 step_size=step_size,k=k,frame=frame))

    count_table_new = {}
    for kmer in all_ss_combi(k):
        if kmer not in count_table:
                count_table_new[kmer] = 0
        if count_table[kmer] >= min_count:
                count_table_new[kmer] = count_table[kmer]
    return count_table_new
#---------------------------------------------------------------------------------------
# feature 1: Secondary structure score of kmer 1
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

def ssf_extract(dataset, thread, output_table):
    # extract the secondary structure of a transcript
    print("Calculating secondary structures of the transcripts by RNAfold ... (This process may take a long time!)")
    dataset = feature_extract_ss(dataset, thread, run_rnafold_parallel)
    
    print("Generating sequence and secondary structure k-mer tables ...")
    # make ss kmer tables
    mrna_seq = dataset[dataset['type']=='mrna']['Sequence'].tolist()
    lnc_seq = dataset[dataset['type']=='lncrna']['Sequence'].tolist()
    mrna_second = dataset[dataset['type']=='mrna']['Secondary_structure'].tolist()
    lnc_second = dataset[dataset['type']=='lncrna']['Secondary_structure'].tolist()

    # k = 1
    mrna_ss_k1 = ss_frequency_seqs(mrna_seq, mrna_second, step_size = 1, k = 1, frame = 0)
    lnc_ss_k1 = ss_frequency_seqs(lnc_seq, lnc_second, step_size = 1, k = 1, frame = 0)
    mrna_ss_k1_total = sum(mrna_ss_k1.values())
    lnc_ss_k1_total = sum(lnc_ss_k1.values())
    mrna_1mer = {}
    lncrna_1mer = {}
    
    ss_table_1_name = output_table+'_ss_table_k1.csv'
    
    with open(output_table+'_ss_table_k1.csv', 'w') as ss_file:
        ss_file.write('ss' + ',' + 'mrna' + ',' + 'lncrna' + '\n')
        for k,v in mrna_ss_k1.items():
            mrna_1mer_ratio = v / mrna_ss_k1_total
            lnc_1mer_ratio = lnc_ss_k1[k] / lnc_ss_k1_total
            ss_file.write(k + ',' + str(mrna_1mer_ratio) + ',' + str(lnc_1mer_ratio) + '\n')
            mrna_1mer[k] = mrna_1mer_ratio
            lncrna_1mer[k] = lnc_1mer_ratio

    # k = 2
    mrna_ss_k2 = ss_frequency_seqs(mrna_seq, mrna_second, step_size = 1, k = 2, frame = 0)
    lnc_ss_k2 = ss_frequency_seqs(lnc_seq, lnc_second, step_size = 1, k = 2, frame = 0)
    mrna_ss_k2_total = sum(mrna_ss_k2.values())
    lnc_ss_k2_total = sum(lnc_ss_k2.values())
    mrna_2mer = {}
    lncrna_2mer = {}
    with open(output_table+'_ss_table_k2.csv', 'w') as ss_file:
        ss_file.write('ss' + ',' + 'mrna' + ',' + 'lncrna' + '\n')
        for k,v in mrna_ss_k2.items():
            mrna_2mer_ratio = v / mrna_ss_k2_total
            lnc_2mer_ratio = lnc_ss_k2[k] / lnc_ss_k2_total
            ss_file.write(k + ',' + str(mrna_2mer_ratio) + ',' + str(lnc_2mer_ratio) + '\n')
            mrna_2mer[k] = mrna_2mer_ratio
            lncrna_2mer[k] = lnc_2mer_ratio

    # k = 3
    mrna_ss_k3 = ss_frequency_seqs(mrna_seq, mrna_second, step_size = 1, k = 3, frame = 0)
    lnc_ss_k3 = ss_frequency_seqs(lnc_seq, lnc_second, step_size = 1, k = 3, frame = 0)
    mrna_ss_k3_total = sum(mrna_ss_k3.values())
    lnc_ss_k3_total = sum(lnc_ss_k3.values())
    mrna_3mer = {}
    lncrna_3mer = {}
    with open(output_table+'_ss_table_k3.csv', 'w') as ss_file:
        ss_file.write('ss' + ',' + 'mrna' + ',' + 'lncrna' + '\n')
        for k,v in mrna_ss_k3.items():
            mrna_3mer_ratio = v / mrna_ss_k3_total
            lnc_3mer_ratio = lnc_ss_k3[k] / lnc_ss_k3_total
            ss_file.write(k + ',' + str(mrna_3mer_ratio) + ',' + str(lnc_3mer_ratio) + '\n')
            mrna_3mer[k] = mrna_3mer_ratio
            lncrna_3mer[k] = lnc_3mer_ratio

    # k = 4
    mrna_ss_k4 = ss_frequency_seqs(mrna_seq, mrna_second, step_size = 1, k = 4, frame = 0)
    lnc_ss_k4 = ss_frequency_seqs(lnc_seq, lnc_second, step_size = 1, k = 4, frame = 0)
    mrna_ss_k4_total = sum(mrna_ss_k4.values())
    lnc_ss_k4_total = sum(lnc_ss_k4.values())
    mrna_4mer = {}
    lncrna_4mer = {}
    with open(output_table+'_ss_table_k4.csv', 'w') as ss_file:
        ss_file.write('ss' + ',' + 'mrna' + ',' + 'lncrna' + '\n')
        for k,v in mrna_ss_k4.items():
            mrna_4mer_ratio = v / mrna_ss_k4_total
            lnc_4mer_ratio = lnc_ss_k4[k] / lnc_ss_k4_total
            ss_file.write(k + ',' + str(mrna_4mer_ratio) + ',' + str(lnc_4mer_ratio) + '\n')
            mrna_4mer[k] = mrna_4mer_ratio
            lncrna_4mer[k] = lnc_4mer_ratio

    # k = 5
    mrna_ss_k5 = ss_frequency_seqs(mrna_seq, mrna_second, step_size = 1, k = 5, frame = 0)
    lnc_ss_k5 = ss_frequency_seqs(lnc_seq, lnc_second, step_size = 1, k = 5, frame = 0)
    mrna_ss_k5_total = sum(mrna_ss_k5.values())
    lnc_ss_k5_total = sum(lnc_ss_k5.values())
    mrna_5mer = {}
    lncrna_5mer = {}
    with open(output_table+'_ss_table_k5.csv', 'w') as ss_file:
        ss_file.write('ss' + ',' + 'mrna' + ',' + 'lncrna' + '\n')
        for k,v in mrna_ss_k5.items():
            mrna_5mer_ratio = v / mrna_ss_k5_total
            lnc_5mer_ratio = lnc_ss_k5[k] / lnc_ss_k5_total
            ss_file.write(k + ',' + str(mrna_5mer_ratio) + ',' + str(lnc_5mer_ratio) + '\n')
            mrna_5mer[k] = mrna_5mer_ratio
            lncrna_5mer[k] = lnc_5mer_ratio
    
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
