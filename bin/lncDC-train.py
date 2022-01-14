import warnings
warnings.filterwarnings('ignore')

import os
import re
import pickle
import numpy as np
import pandas as pd
import argparse
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import train_SIF_PF_extraction

seed = 666


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
                    sequences.append(seq.replace('a','A').replace('t','T').replace('g','G').replace('c','C'))
                seq = ''
            else:
                seq += line.strip()
    if seq != '':
        sequences.append(seq.replace('a','A').replace('t','T').replace('g','G').replace('c','C'))

    return sequences

def under_over_process(x, y, njobs):
    ratio = y.value_counts()[0] / y.value_counts()[1]
    # undersample first
    if(ratio < 2/3 or ratio > 1.5):
        rus = RandomUnderSampler(sampling_strategy=2/3,random_state=seed)
        x_underSampled, y_underSampled = rus.fit_resample(x, y)
    else:
        x_underSampled = x
        y_underSampled = y

    # then oversample
    smote = SMOTE(random_state=seed, n_jobs = njobs)
    x_resampled, y_resampled = smote.fit_resample(x_underSampled, y_underSampled)
    return x_resampled, y_resampled

def main():
    parser = argparse.ArgumentParser(description='Long noncoding RNA decipherer')
    parser.add_argument('-m','--mrna', help = 'mrna transcripts in fasta format',
                        type = str, required = True, default = None)
    parser.add_argument('-c','--cds', help = 'CDS sequences of the mrna transcripts in fasta format, same size, same order',
                        type = str, required = True, default = None)
    parser.add_argument('-l','--lncrna', help = 'lncrna transcripts in fasta format')
    parser.add_argument('-o','--output', help = 'prefix of output files',
                        type = str, default = 'self_train')
    parser.add_argument('-r','--secondary', help = 'predict with secondary structure features included',
                        action = "store_true")
    parser.add_argument('-t','--thread', help = 'number of thread assigned to use, set -1 to use all cpus',
                        type = int, default = -1)

    args = parser.parse_args()
    mrna = args.mrna
    cds = args.cds
    lncrna = args.lncrna
    ss_feature = args.secondary
    output_prefix = args.output

    thread = args.thread
    if thread == -1:
        thread = os.cpu_count()
    else:
        thread = thread

    # load mrna data
    mrna_data = load_fasta(mrna)
    # load cds data
    cds_data = load_fasta(cds)

    # initialize a mrna dataframe
    mrna_dataset = pd.DataFrame(index=range(len(mrna_data)), columns=['Sequence', 'type', 'CDS_seq'])
    for i in range(mrna_dataset.index.size):
        mrna_dataset.loc[i, 'Sequence'] = mrna_data[i]
        mrna_dataset.loc[i,'type'] = 'mrna'
        mrna_dataset.loc[i, 'CDS_seq'] = cds_data[i]

    # load lncrna data
    lncrna_data = load_fasta(lncrna)

    # initialize a lncrna dataframe
    lncrna_dataset = pd.DataFrame(index=range(len(lncrna_data)), columns=['Sequence', 'type', 'CDS_seq'])
    for i in range(lncrna_dataset.index.size):
        lncrna_dataset.loc[i, 'Sequence'] = lncrna_data[i]
        lncrna_dataset.loc[i,'type'] = 'lncrna'
        lncrna_dataset.loc[i, 'CDS_seq'] = 0

    # combine to a dataframe
    dataset = pd.concat([mrna_dataset, lncrna_dataset])
    # reset the index of the dataframe
    dataset.reset_index(drop=True, inplace=True)

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

    # IF only use SIF and PF features
    if not ss_feature:
        # Filter out sequence length less than 200nt or more than 20000nt
        dataset = dataset[(dataset['Transcript_length'] >= 200) & (dataset['Transcript_length'] <= 20000)]
        dataset = dataset.reset_index(drop=True)

        # extract features
        dataset = train_SIF_PF_extraction.sif_pf_extract(dataset, thread, output_prefix)
        columns = ['type','Transcript_length','GC_content', 'Fickett_score', 'ORF_T0_length', 'ORF_T1_length',
                   'ORF_T2_length','ORF_T0_coverage', 'ORF_T1_coverage', 'ORF_T3_coverage', 'Hexamer_score_ORF_T0',
                   'Hexamer_score_ORF_T1', 'Hexamer_score_ORF_T2', 'Hexamer_score_ORF_T3', 'RCB_T0',
                   'RCB_T1', 'ORF_T0_PI', 'ORF_T0_MW', 'ORF_T0_aromaticity', 'ORF_T0_instability',
                   'ORF_T1_MW', 'ORF_T1_instability', 'ORF_T2_MW', 'ORF_T3_MW']
        dataset = dataset[columns]
        x_train = dataset.drop(['type','Transcript_length'], axis = 1)
        y_train = dataset[['type']]

        # imputation of missing values
        imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_mean.fit(x_train[x_train.columns])
        x_train[x_train.columns] = imputer_mean.transform(x_train[x_train.columns])

        # write imputer to a pickle file
        pkl_filename = output_prefix + '_imputer_SIF_PF.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(imputer_mean, file)

        # standardization
        scaler = StandardScaler()
        scaler.fit(x_train[x_train.columns])
        x_train[x_train.columns] = scaler.transform(x_train[x_train.columns])
        pkl_filename = output_prefix + '_scaler_SIF_PF.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(scaler, file)

        # build standardized training dataset
        train_std = pd.concat([x_train, y_train], axis=1)

        # balancing training data
        x_resampled, y_resampled = under_over_process(
            train_std.drop(['type'], axis=1),
            train_std['type'], thread)

        # fit a model
        xgb_model = xgb.XGBClassifier(alpha = 10, colsample_bytree = 0.8, gamma = 0.01, learning_rate = 0.1,
                                     max_depth = 9, min_child_weight = 4, n_estimators = 1000,
                                     objective = 'binary:logistic', subsample = 0.8, verbosity = 0,
                                     random_state = seed, n_jobs = thread)
        xgb_model.fit(x_resampled, y_resampled)
        pkl_filename = output_prefix + '_xgb_model_SIF_PF.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(xgb_model, file)

    else:
        # Use SIF + PF + SSF
        import train_SSF_extraction
        # Filter out sequence length less than 200nt or more than 20000nt
        dataset = dataset[(dataset['Transcript_length'] >= 200) & (dataset['Transcript_length'] <= 20000)]
        dataset = dataset.reset_index(drop=True)

        # extract features
        dataset = train_SIF_PF_extraction.sif_pf_extract(dataset, thread, output_prefix)
        columns = ['type', 'Sequence','Transcript_length', 'GC_content', 'Fickett_score', 'ORF_T0_length', 'ORF_T1_length',
                   'ORF_T2_length', 'ORF_T0_coverage', 'ORF_T1_coverage', 'ORF_T3_coverage', 'Hexamer_score_ORF_T0',
                   'Hexamer_score_ORF_T1', 'Hexamer_score_ORF_T2', 'Hexamer_score_ORF_T3', 'RCB_T0',
                   'RCB_T1', 'ORF_T0_PI', 'ORF_T0_MW', 'ORF_T0_aromaticity', 'ORF_T0_instability',
                   'ORF_T1_MW', 'ORF_T1_instability', 'ORF_T2_MW', 'ORF_T3_MW']
        dataset = dataset[columns]

        # extract Secondary Structure Features
        dataset = train_SSF_extraction.ssf_extract(dataset, thread, output_prefix)
        full_columns = ['type', 'Transcript_length', 'GC_content', 'Fickett_score', 'ORF_T0_length',
                   'ORF_T1_length','ORF_T2_length', 'ORF_T0_coverage', 'ORF_T1_coverage', 'ORF_T3_coverage',
                   'Hexamer_score_ORF_T0','Hexamer_score_ORF_T1', 'Hexamer_score_ORF_T2', 'Hexamer_score_ORF_T3',
                   'RCB_T0','RCB_T1', 'SS_score_k1','SS_score_k2','SS_score_k3','SS_score_k4',
                   'SS_score_k5','GC_content_paired_ss',
                   'ORF_T0_PI', 'ORF_T0_MW', 'ORF_T0_aromaticity', 'ORF_T0_instability',
                   'ORF_T1_MW', 'ORF_T1_instability', 'ORF_T2_MW', 'ORF_T3_MW']
        dataset = dataset[full_columns]
        x_train = dataset.drop(['type', 'Transcript_length'], axis = 1)
        y_train = dataset[['type']]

        # imputation of missing values
        imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_mean.fit(x_train[x_train.columns])
        x_train[x_train.columns] = imputer_mean.transform(x_train[x_train.columns])

        # write imputer to a pickle file
        pkl_filename = output_prefix + '_imputer_SIF_PF_SSF.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(imputer_mean, file)

        # standardization
        scaler = StandardScaler()
        scaler.fit(x_train[x_train.columns])
        x_train[x_train.columns] = scaler.transform(x_train[x_train.columns])
        pkl_filename = output_prefix + '_scaler_SIF_PF_SSF.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(scaler, file)

        # build standardized training dataset
        train_std = pd.concat([x_train, y_train], axis=1)

        # balancing training data
        x_resampled, y_resampled = under_over_process(
            train_std.drop(['type'], axis=1),
            train_std['type'], thread)

        # fit a model
        xgb_model = xgb.XGBClassifier(alpha=10, colsample_bytree=0.8, gamma=0.01, learning_rate=0.1,
                                      max_depth=9, min_child_weight=4, n_estimators=1000,
                                      objective='binary:logistic', subsample=0.8, verbosity = 0,
                                      random_state=seed, n_jobs=thread)
        xgb_model.fit(x_resampled, y_resampled)
        pkl_filename = output_prefix + '_xgb_model_SIF_PF_SSF.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(xgb_model, file)

if __name__ == '__main__':
    main()
