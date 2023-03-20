import warnings
warnings.filterwarnings('ignore')

import os
import sys

system = sys.version_info
if system[0] < 3 or system[1] < 9:
    sys.stderr.write("ERROR: Your python version is: "+ str(system[0]) + "." + str(system[1]) +"\n" + "LncDC requires python 3.9 or newer!\n" )
    sys.exit(1)


import re
import gzip
import pathlib
import pickle
import pandas as pd
import numpy as np
import SIF_PF_extraction
import argparse

def file_check(filename, rule, filetype):
    if not filename.endswith(rule):
        sys.stderr.write("ERROR: Please use secondary structure trained "+filetype+ " file, which ends with: "+ rule +" \n")
        sys.exit(1)

def file_exist(filename,parser):
    if filename == None:
        parser.print_help()
        sys.exit(1)
    else:
        if not os.path.isfile(filename):
            sys.stderr.write("ERROR: No such file: " + filename + "\n")
            sys.exit(1)

def format_check(filename):
    if filename.endswith((".gz", ".Z", ".z")):
        fd = gzip.open(filename, 'rt')
    else:
        fd = open(filename, 'r')
    
    filetype = 1
    for line in fd:
        if line.startswith('#'):
            continue
        elif line.strip() == None:
            continue
        elif line.startswith('>'):
            filetype = 0
            break
        else:
            filetype = 1
            break
    
    fd.close()
    return filetype
    
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
    
    if filename.endswith((".gz", ".Z", ".z")):
        fd = gzip.open(filename, 'rt')
    else:
        fd = open(filename, 'r')
        
    for line in fd:
        if line.startswith('>'):
            if seq != '':
                sequences.append((seq.replace('a','A').replace('t','T').replace('g','G').replace('c','C'), description))
            seq = ''
            description = line.strip()[1:]
        else:
            seq += line.strip()
    if seq != '':
        sequences.append((seq.replace('a','A').replace('t','T').replace('g','G').replace('c','C'), description))
    
    fd.close()
    return sequences

def main():
    filepath = os.path.dirname(__file__)[:-3]
    default_data_path = os.path.join(filepath,'data/')
    parser = argparse.ArgumentParser(description='LncDC: a machine learning based tool for long non-coding RNA detection from RNA-Seq data', )
    parser.add_argument('-v','--version', action = 'version', version = '%(prog)s version:1.3.5')
    parser.add_argument('-i','--input', help = 'The inputfile with RNA transcript sequences in fasta format. The fasta file could be regular text file or gzip compressed file (*.gz)',
                        type = str, required = True, default = None)
    parser.add_argument('-o','--output', help = 'The output file that will contain the prediction results in csv format. Long noncoding RNAs are labeled as lncrna, and message RNAs are labeled as mrna. Default: lncdc.output.csv',
                        type = str, default = 'lncdc.output.csv')
    parser.add_argument('-x','--hexamer', help = '(Optional) Prebuilt hexamer table in csv format. Run lncDC-train.py to obtain the hexamer table of your own training data. Default: train_hexamer_table.csv',
                        type = str, default = default_data_path+'train_hexamer_table.csv')
    parser.add_argument('-m','--model', help = '(Optional) Prebuilt training model. Run lncDC-train.py to obtain the model trained from your own training data. Default: XGB_model_SIF_PF.pkl',
                        type = str, default = default_data_path+'XGB_model_SIF_PF.pkl')
    parser.add_argument('-p','--imputer', help = '(Optional) Prebuilt imputer from training data. Run lncDC-train.py to obtain the imputer from your own training data. Default: imputer_SIF_PF.pkl',
                        type = str, default = default_data_path+'imputer_SIF_PF.pkl')
    parser.add_argument('-s','--scaler', help = '(Optional) Prebuilt scaler from training data. Run lncDC-train.py to obtain the imputer from your own training data. Default: scaler_SIF_PF.pkl',
                        type = str, default = default_data_path+'scaler_SIF_PF.pkl')
    parser.add_argument('-r','--secondary', help = '(Optional) Turn on to predict with secondary structure features. Default: turned off',
                        action = "store_true")
    parser.add_argument('-k','--kmer', help = '(Optional) Prefix of the sequence and secondary structure kmer tables. Need to specify -r first. For example, the prefix of secondary structure kmer table file mouse_ss_table_k1.csv is mouse_ss_table. Run lncDC-train.py to obtain the tables from your own training data. Default: train_ss_table',
                        type = str, default = default_data_path+'train_ss_table')
    parser.add_argument('-t','--thread', help = '(Optional) The number of threads assigned to use. Set -1 to use all cpus. Default value: -1.',
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
    
    if ss_feature:
        try:
            import RNA
        except:
            sys.stderr.write("ViennaRNA is not properly installed! \n")
            sys.stderr.write("ViennaRNA is required for secondary structure feature extraction. \nYou can install it by: \n1) CONDA: conda install -c bioconda viennarna \n2) Or install from the official ViennaRNA website: https://www.tbi.univie.ac.at/RNA/ \n \nTo confirm that ViennaRNA is properly installed, you can test it by: \n$ python \n>>> import RNA \n \nViennaRNA is successfully installed if there are no error messages poped up. \n")
            sys.exit(1)
    
    print("Process Start.")
    print("Checking if the inputfile exists ...")
    # check if the inputfile exists
    file_exist(inputfile, parser)
    
    print("Checking if the model file and other required files exist ...")
    # check if the requried files exist
    file_exist(hexamer_file,parser)
    file_check(hexamer_file,'hexamer_table.csv','hexamer')
    
    if ss_feature:
        # set up the default model to SSF model
        if model.endswith('XGB_model_SIF_PF.pkl'):
            model = os.path.join(default_data_path, 'XGB_model_SIF_PF_SSF.pkl')
        
        # check if the SSF model file exists
        file_exist(model,parser)
        
        # check if the user inputfile is SSF model file
        file_check(model,'model_SIF_PF_SSF.pkl','model')
        
        # set up the default imputer to SSF imputer
        if imputer.endswith('imputer_SIF_PF.pkl'):
            imputer = os.path.join(default_data_path, 'imputer_SIF_PF_SSF.pkl')
        
        # check if the SSF imputer file exists
        file_exist(imputer,parser)
        
        # check if the user inputfile is SSF imputer file
        file_check(imputer,'imputer_SIF_PF_SSF.pkl','imputer')

        # set up the default scaler to SSF scaler
        if scaler.endswith('scaler_SIF_PF.pkl'):
            scaler = os.path.join(default_data_path, 'scaler_SIF_PF_SSF.pkl')
        
        # check if the SSF scaler file exists
        file_exist(scaler,parser)
        
        # check if the user inputfile is SSF scaler file
        file_check(scaler,'scaler_SIF_PF_SSF.pkl','scaler')
        
        # check if the ss kmer files exist
        file_exist(ss_kmer_file + '_k1.csv',parser)
        file_exist(ss_kmer_file + '_k2.csv',parser)
        file_exist(ss_kmer_file + '_k3.csv',parser)
        file_exist(ss_kmer_file + '_k4.csv',parser)
        file_exist(ss_kmer_file + '_k5.csv',parser)
    else:
        file_exist(model,parser)
        file_check(model,'model_SIF_PF.pkl','model')
        file_exist(imputer,parser)
        file_check(imputer,'imputer_SIF_PF.pkl','imputer')
        file_exist(scaler,parser)
        file_check(scaler,'scaler_SIF_PF.pkl','scaler')
        
    print("Checking if the inputfile is in fasta format ...")
    # check if the inputfile is in fasta format
    file_format = format_check(inputfile)
    if not file_format == 0:
        sys.stderr.write("ERROR: Your inputfile is not in fasta format \n")
        sys.exit(1)
    else:
        print("PASS")
    
    # create any parent directories if needed
    current_path = pathlib.Path().resolve()
    outputfile = os.path.join(current_path, outputfile)

    if not os.path.isfile(outputfile):
        os.makedirs(os.path.dirname(outputfile), exist_ok = True)
    
    test_data = load_fasta(inputfile)
    
    print()
    print("Initializing dataframe ...")
    # initialize a dataframe
    dataset = pd.DataFrame(index = range(len(test_data)), columns = ['Sequence','Description'])
    for i in range(dataset.index.size):
        dataset.loc[i,'Sequence'] = test_data[i][0]
        dataset.loc[i,'Description'] = test_data[i][1]
    
    print("Total Number of transcripts loaded: " + str(dataset.index.size))
    
    # remove the sequences that have non [ATGC] inside
    for i in range(dataset.index.size):
        if len(re.findall(r'[^ATGC]',dataset.loc[i,'Sequence'])) > 0:
            dataset.loc[i,'Sequence'] = float('NaN')
    dataset.dropna(how = 'any', inplace = True)
    # reset the index of the dataframe
    dataset.reset_index(drop = True, inplace = True)
    
    print("Calculating transcript lengths ...")
    print()
    
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
        # print()
        dataset = dataset[dataset['Transcript_length'] >= 200]
        dataset = dataset.reset_index(drop=True)
        
        print("Removing Non-valid transcripts (sequence that have non-ATGCatgc letters & sequence length less than 200 nt) ...")
        print("Number of valid transcripts: " + str(dataset.index.size))
        
        if dataset.index.size == 0:
            sys.stderr.write("No valid transcripts detected! \n")
            sys.exit(1)
        
        print()
        print("Extracting SIF and PF features ...")
        
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
        
        print("Predicting ...")
        # Load model
        with open(model, 'rb') as file:
            XGB_model = pickle.load(file)

        y_pred = XGB_model.predict(x_test)
        y_pred = np.vectorize({1:'mrna',0:'lncrna'}.get)(y_pred)
        y_prob = XGB_model.predict_proba(x_test)[:,0]

        # Append predictions and output
        dataset['Noncoding_prob'] = y_prob
        dataset['predict'] = y_pred
        
        print("Writing the output to file: " + str(outputfile))
        # output to a result file
        dataset.to_csv(outputfile, index = False)
        
        print("Done!")
        
    else:
        # Use SIF + PF + SSF
        import SSF_extraction
        # Filter out sequence length less than 200nt or more than 20000nt
        dataset = dataset[(dataset['Transcript_length'] >= 200) & (dataset['Transcript_length'] <= 20000)]
        dataset = dataset.reset_index(drop=True)
        
        print("Removing Non-valid transcripts (sequence that have non-ATGCatgc" + " letters & sequence length less than 200 nt) ...")
        print("Filtering out transcripts with sequence length greater than 20,000" + "nt due to the limited addressable range of the RNAfold program ...")
        print("We recommand you use LncDC without the '-r' option to predict lncRNAs over 20,000 nts.")
        
        print()
        print("Number of valid transcripts: " + str(dataset.index.size))
        
        if dataset.index.size == 0:
            sys.stderr.write("No valid transcripts detected! \n")
            sys.exit(1)
        
        print()
        print("Extracting SIF and PF features ...")
        
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
        
        print("Predicting ...")
        # Load model
        with open(model, 'rb') as file:
            XGB_model = pickle.load(file)

        y_pred = XGB_model.predict(x_test)
        y_pred = np.vectorize({1:'mrna',0:'lncrna'}.get)(y_pred)
        y_prob = XGB_model.predict_proba(x_test)[:, 0]

        # Append predictions and output
        dataset['Noncoding_prob'] = y_prob
        dataset['predict'] = y_pred

        print("Writing the output to file: " + str(outputfile))
        # output to a result file
        dataset.to_csv(outputfile, index=False)
        
        print("Done!")
        
if __name__ == '__main__':
    main()
