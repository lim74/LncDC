import warnings
warnings.filterwarnings('ignore')

import os
import re
import pickle
import pandas as pd
import SIF_PF_extraction
import argparse


def load_fasta(filename):
    '''
    load_fasta
    Inputs:
        filename - name of FASTA file to load
    Returns:
        a list of sequences
    '''
    sequences = []
    seq = ''
    with open(filename, 'r') as fd:
        for line in fd:
            if '>' in line:
                if seq != '':
                    sequences.append((seq.replace('a','A').replace('t','T').replace('g','G').replace('c','C'), description))
                seq = ''
                description = line.strip()[1:]
            else:
                seq += line.strip()
    if seq != '':
        sequences.append((seq.replace('a','A').replace('t','T').replace('g','G').replace('c','C'), description))

    return sequences

def main():
    filepath = os.path.dirname(__file__)[:-3]
    default_data_path = os.path.join(filepath,'data/')
    parser = argparse.ArgumentParser(description='Long noncoding RNA decipherer')
    parser.add_argument('-i','--input', help = 'input transcripts in fasta format',
                        type = str, required = True, default = None)
    parser.add_argument('-o','--output', help = 'output file in csv format',
                        type = str, default = 'lncdc.output.csv')
    parser.add_argument('-x','--hexamer', help = 'prebuilt hexamer table',
                        type = str, default = default_data_path+'train_hexamer_table.csv')
    parser.add_argument('-m','--model', help = 'prebuilt trainging model',
                        type = str, default = default_data_path+'XGB_model_SIF_PF.pkl')
    parser.add_argument('-p','--imputer', help = 'prebuilt imputer from training data',
                        type = str, default = default_data_path+'imputer_SIF_PF.pkl')
    parser.add_argument('-s','--scaler', help = 'prebuilt scaler from training data',
                        type = str, default = default_data_path+'scaler_SIF_PF.pkl')
    parser.add_argument('-r','--secondary', help = 'predict with secondary structure based features included',
                        action = "store_true")
    parser.add_argument('-k','--kmer', help = 'prefix of the secondary structure kmer tables',
                        type = str, default = default_data_path+'train_ss_table')
    parser.add_argument('-t','--thread', help = 'number of thread assigned to use, set -1 to use all cpus',
                        type = int, default = -1)

    args = parser.parse_args()
    inputfile = args.input
    outputfile = args.output
    hexamer_file = args.hexamer
    model = args.model
    imputer = args.imputer
    scaler = args.scaler
    ss_feature = args.secondary
    ss_kmer_file = args.kmer

    thread = args.thread
    if thread == -1:
        thread = os.cpu_count()
    else:
        thread = thread

    test_data = load_fasta(inputfile)
    # initialize a dataframe
    dataset = pd.DataFrame(index = range(len(test_data)), columns = ['Sequence','Description'])
    for i in range(dataset.index.size):
        dataset.loc[i,'Sequence'] = test_data[i][0]
        dataset.loc[i,'Description'] = test_data[i][1]

    # remove the sequences that have non [ATGC] inside
    for i in range(dataset.index.size):
        if len(re.findall(r'[^ATGC]',dataset.loc[i,'Sequence'])) > 0:
            dataset.loc[i,'Sequence'] = float('NaN')
    dataset.dropna(how = 'any', inplace = True)
    # reset the index of the dataframe
    dataset.reset_index(drop = True, inplace = True)

    # Calculate the length of the transcripts
    for i in range(dataset.index.size):
        dataset.loc[i,'Transcript_length'] = len(dataset.loc[i, 'Sequence'])

    # load pretrained hexamer table
    coding_hexamer = {}
    noncoding_hexamer = {}
    with open(hexamer_file) as hexamer_table:
        hexamer_table.readline()
        for line in hexamer_table:
            line_split = line.strip('\n').split(',')
            hexamer = line_split[0]
            coding_hexamer[hexamer] = float(line_split[1])
            noncoding_hexamer[hexamer] = float(line_split[2])

    # IF only use SIF and PF features
    if not ss_feature:
        # Filter out sequence length less than 200nt
        dataset = dataset[dataset['Transcript_length'] >= 200]
        dataset = dataset.reset_index(drop=True)

        # extract features
        dataset = SIF_PF_extraction.sif_pf_extract(dataset, thread, coding_hexamer, noncoding_hexamer)
        columns = ['Description','Transcript_length','GC_content', 'Fickett_score', 'ORF_T0_length', 'ORF_T1_length',
                   'ORF_T2_length','ORF_T0_coverage', 'ORF_T1_coverage', 'ORF_T3_coverage', 'Hexamer_score_ORF_T0',
                   'Hexamer_score_ORF_T1', 'Hexamer_score_ORF_T2', 'Hexamer_score_ORF_T3', 'RCB_T0',
                   'RCB_T1', 'ORF_T0_PI', 'ORF_T0_MW', 'ORF_T0_aromaticity', 'ORF_T0_instability',
                   'ORF_T1_MW', 'ORF_T1_instability', 'ORF_T2_MW', 'ORF_T3_MW']
        dataset = dataset[columns]
        x_test = dataset.drop(['Description','Transcript_length'], axis = 1)

        # imputation of missing values
        with open(imputer, 'rb') as file:
            pickle_imputer = pickle.load(file)
        x_test[x_test.columns] = pickle_imputer.transform(x_test[x_test.columns])

        # Standardization
        with open(scaler, 'rb') as file:
            pickle_scaler = pickle.load(file)
        x_test[x_test.columns] = pickle_scaler.transform(x_test[x_test.columns])

        # Load model
        with open(model, 'rb') as file:
            XGB_model = pickle.load(file)

        y_pred = XGB_model.predict(x_test)
        y_prob = XGB_model.predict_proba(x_test)[:,0]

        # Append predictions and output
        dataset['Noncoding_prob'] = y_prob
        dataset['predict'] = y_pred

        # output to a result file
        dataset.to_csv(outputfile, index = False)
    else:
        # Use SIF + PF + SSF
        import SSF_extraction
        # Filter out sequence length less than 200nt
        dataset = dataset[dataset['Transcript_length'] >= 200]
        dataset = dataset.reset_index(drop=True)

        # extract SIF + PF features
        dataset = SIF_PF_extraction.sif_pf_extract(dataset, thread, coding_hexamer, noncoding_hexamer)
        columns = ['Sequence','Description', 'Transcript_length', 'GC_content', 'Fickett_score', 'ORF_T0_length', 'ORF_T1_length',
                   'ORF_T2_length', 'ORF_T0_coverage', 'ORF_T1_coverage', 'ORF_T3_coverage', 'Hexamer_score_ORF_T0',
                   'Hexamer_score_ORF_T1', 'Hexamer_score_ORF_T2', 'Hexamer_score_ORF_T3', 'RCB_T0',
                   'RCB_T1', 'ORF_T0_PI', 'ORF_T0_MW', 'ORF_T0_aromaticity', 'ORF_T0_instability',
                   'ORF_T1_MW', 'ORF_T1_instability', 'ORF_T2_MW', 'ORF_T3_MW']
        dataset = dataset[columns]

        # Secondary Structure Features
        # load pretrained secondary structure kmer table k1
        mrna_1mer = {}
        lncrna_1mer = {}
        with open(ss_kmer_file + '_k1.csv') as ss_table_k1:
            ss_table_k1.readline()
            for line in ss_table_k1:
                line_split = line.strip('\n').split(',')
                ss = line_split[0]
                mrna_1mer[ss] = float(line_split[1])
                lncrna_1mer[ss] = float(line_split[2])

        # load pretrained secondary structure kmer table k2
        mrna_2mer = {}
        lncrna_2mer = {}
        with open(ss_kmer_file + '_k2.csv') as ss_table_k2:
            ss_table_k2.readline()
            for line in ss_table_k2:
                line_split = line.strip('\n').split(',')
                ss = line_split[0]
                mrna_2mer[ss] = float(line_split[1])
                lncrna_2mer[ss] = float(line_split[2])

        # load pretrained secondary structure kmer table k3
        mrna_3mer = {}
        lncrna_3mer = {}
        with open(ss_kmer_file + '_k3.csv') as ss_table_k3:
            ss_table_k3.readline()
            for line in ss_table_k3:
                line_split = line.strip('\n').split(',')
                ss = line_split[0]
                mrna_3mer[ss] = float(line_split[1])
                lncrna_3mer[ss] = float(line_split[2])

        # load pretrained secondary structure kmer table k4
        mrna_4mer = {}
        lncrna_4mer = {}
        with open(ss_kmer_file + '_k4.csv') as ss_table_k4:
            ss_table_k4.readline()
            for line in ss_table_k4:
                line_split = line.strip('\n').split(',')
                ss = line_split[0]
                mrna_4mer[ss] = float(line_split[1])
                lncrna_4mer[ss] = float(line_split[2])

        # load pretrained secondary structure kmer table k5
        mrna_5mer = {}
        lncrna_5mer = {}
        with open(ss_kmer_file + '_k5.csv') as ss_table_k5:
            ss_table_k5.readline()
            for line in ss_table_k5:
                line_split = line.strip('\n').split(',')
                ss = line_split[0]
                mrna_5mer[ss] = float(line_split[1])
                lncrna_5mer[ss] = float(line_split[2])

        # extract SSF features
        dataset = SSF_extraction.ssf_extract(dataset, thread, mrna_1mer, lncrna_1mer,
                                             mrna_2mer, lncrna_2mer, mrna_3mer, lncrna_3mer,
                                             mrna_4mer, lncrna_4mer, mrna_5mer, lncrna_5mer)

        full_columns = ['Description', 'Transcript_length', 'GC_content', 'Fickett_score', 'ORF_T0_length',
                   'ORF_T1_length','ORF_T2_length', 'ORF_T0_coverage', 'ORF_T1_coverage', 'ORF_T3_coverage',
                   'Hexamer_score_ORF_T0','Hexamer_score_ORF_T1', 'Hexamer_score_ORF_T2', 'Hexamer_score_ORF_T3',
                   'RCB_T0','RCB_T1', 'SS_score_k1','SS_score_k2','SS_score_k3','SS_score_k4',
                   'SS_score_k5','GC_content_paired_ss',
                   'ORF_T0_PI', 'ORF_T0_MW', 'ORF_T0_aromaticity', 'ORF_T0_instability',
                   'ORF_T1_MW', 'ORF_T1_instability', 'ORF_T2_MW', 'ORF_T3_MW']
        dataset = dataset[full_columns]
        x_test = dataset.drop(['Description', 'Transcript_length'], axis=1)

        # imputation of missing values
        with open(imputer, 'rb') as file:
            pickle_imputer = pickle.load(file)
        x_test[x_test.columns] = pickle_imputer.transform(x_test[x_test.columns])

        # Standardization
        with open(scaler, 'rb') as file:
            pickle_scaler = pickle.load(file)
        x_test[x_test.columns] = pickle_scaler.transform(x_test[x_test.columns])

        # Load model
        with open(model, 'rb') as file:
            XGB_model = pickle.load(file)

        y_pred = XGB_model.predict(x_test)
        y_prob = XGB_model.predict_proba(x_test)[:, 0]

        # Append predictions and output
        dataset['Noncoding_prob'] = y_prob
        dataset['predict'] = y_pred

        # output to a result file
        dataset.to_csv(outputfile, index=False)

if __name__ == '__main__':
    main()
