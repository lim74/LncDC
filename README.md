# LncDC
 Long non-coding rna Decipherer (LncDC) is a machine learning based tool for identifying long non-coding transcripts from RNA-Seq data.

## Pre-requisite
python 3 >= 3.7; pandas >= 1.2.3; numpy >= 1.20.2; sklearn >= 0.24.1; xgboost >= 1.3.3; imblearn >= 0.8.0; biopython >= 1.78

ViennaRNA (Optional, required when using secondary structure features for prediction. Can be installed by conda: conda install -c bioconda viennarna)

## Conda environment (Optional & Recommanded)
1. Download and install Anaconda.
https://docs.anaconda.com/anaconda/install/

2. Create an environment 
```
conda create -n lncdc     
```
Here, 'lncdc' is the name of the conda environment, which can be replaced by any names.

2. Install ViennaRNA for secondary structure features extraction (Optional)
```
conda install -c bioconda viennarna
```

3. Activate the environment (before we run LncDC)
```
conda activate lncdc
```

4. Deactivate the environment
```
conda deactivate
```

## Installation
1. Download the source code from https://github.com/lim74/LncDC

2. Unzip the package
```
unzip LncDC-main.zip
```

3. Go to LncDC-main directory
```
cd LncDC-main
```

4. Install LncDC
```
python setup.py install
```
If you want to install the required side-packages in a certain conda environment, make sure the environment is activated. 

## Usage and Examples
### How to perform a prediction
```
python lncDC.py -i input.fa -o output -x hexamer_table.csv -m model.pkl -p imputer.pkl -s scaler.pkl -r -k ss_table -t number_of_threads
```
    -i The sequences of RNA transcripts in fasta format
    -o The output file that contains the prediction results in csv format. Long noncoding RNAs are labeled as 'lncrna', and message RNAs are labeled as 'mrna'. Default: lncdc.output.csv
    -x (Optional) Prebuilt hexamer table in csv format. Run lncDC-train.py to obtain the hexamer table of your own training data. Default: train_hexamer_table.csv
    -m (Optional) Prebuilt training model. Run lncDC-train.py to obtain the model trained from your own training data. Default: XGB_model_SIF_PF.pkl
    -p (Optional) Prebuilt imputer from training data. Run lncDC-train.py to obtain the imputer from your own training data. Default: imputer_SIF_PF.pkl
    -s (Optional) Prebuilt scaler from training data. Run lncDC-train.py to obtain the imputer from your own training data. Default: scaler_SIF_PF.pkl
    -r (Optional) Turn on to predict with secondary structure based features. Default: turned off.
    -k (Optional) -r required. Prefix of the secondary structure kmer tables. Run lncDC-train.py to obtain the tables from your own training data. Default: train_ss_table
    -t (Optional) The number of threads assigned to use. Set -1 to use all cpus. Default value: -1.
### Examples for prediction
1. Predict with the default model (human).
```
cd LncDC-main/
python bin/lncDC.py -i test/human_test.fasta -o lncdc_human.csv -t 8
```
Here, the human_test.fasta file includes human RNA transcripts in fasta format. The prediction results will be stored in the lncdc_human.csv file. 8 threads are used for this prediction. 

2. Predict with the self-trained model (without secondary structure based features).
```
cd LncDC-main/
python bin/lncDC.py -i test/mouse_test.fasta -o lncdc_mouse.csv -x data/train_hexamer_table.csv -m data/XGB_model_SIF_PF.pkl -p data/imputer_SIF_PF.pkl -s data/scaler_SIF_PF.pkl -t 8
```
The input file in this prediction is mouse_test.fasta, which contains the mouse RNA transcripts in fasta format. The output file is lncdc_mouse.csv. The parameter '-x' is applied, and the hexamer table train_hexamer_table.csv is provided. '-m' is used, follewed by the self-trained model XGB_model_SIF_PF.pkl. The self-trained model only used sequence intrinsic and protein features in this case. The imputer file imputer_SIF_PF.pkl and the sclar file scaler_SIF_PF.pkl are provided with parameter '-p' and '-s', respectively.  8 threads are used for the prediction.

3. Predict with the self-trained model (with secondary structure features).
```
cd LncDC-main/
python bin/lncDC.py -i test/mouse_test.fasta -o lncdc_ss_mouse.csv -x data/train_hexamer_table.csv -m data/XGB_model_SIF_PF_SBF.pkl -p data/imputer_SIF_PF_SBF.pkl -s data/scaler_SIF_PF_SBF.pkl -r -k -s data/train_ss_table -t 8
```
The input file is mouse_test.fasta and the output file is lncdc_ss_mouse.csv. In addition to the parameters requried for the prediction without using secondary structure based features, '-r' and '-k' are requried for prediction with secondary structure based features. The '-r' parameter will turn on the prediction with secondary structure based features, while the '-k' parameter will provide the requried train_ss_table(s). The '-t' parameter indicates that the program will use 8 threads for prediction. 

### How to train a model with my own data
```
python lncDC-train.py -m mrna.fa -c cds.fa -l lncrna.fa -o output -t number_of_threads -r
```
    -m The sequences of mRNA transcripts in fasta format
    -c The CDS sequences of the mRNA transcripts in fasta format. The order of the CDS sequences should be the same as the mRNA sequences.
    -l The sequences of lncRNA transcripts in fasta format.
    -o The prefix of the output files, including a hexamer table, a prediction model, an imputer and a scaler. If the '-r' parameter turned on, the output files will also include secondary structure kmer tables.
    -t (Optional) The number of threads assigned to use. Set -1 to use all cpus. Default value: -1.
    -r (Optional) Turn on to train a model with secondary structure features. Will generate secondary structure kmer tables. Default: turned off.

### Examples for training models
1. Train a model (No secondary structure features) using mouse data
```
cd LncDC-main/
python bin/lncDC-train.py -m mrna_mouse.fasta -c cds_mouse.fasta -l lncrna_mouse.fasta -o self_mouse -t 8
```
To train a model with mouse data, we need to provide the mRNA and lncRNA sequence, respectively. The mRNA sequences are stored in the mrna_mouse.fasta file, and their corresponding CDS sequences are stored in the cds_mouse.fasta file. The lncrna_mouse.fasta file includes the long noncoding RNA sequences. 'self_mouse' is the prefix of the output files, which include four files: self_mouse_hexamer_table.csv, self_mouse_xgb_model_SIF_PF.pkl, self_mouse_imputer_SIF_PF.pkl and self_mouse_scaler_SIF_PF.pkl. We set '-t' to 8 so that there are 8 threads will be used for the model training. 

2. Train a model (with secondary structure based features) using mouse data
```
cd LncDC-main/
python bin/lncDC-train.py -m mrna_mouse.fasta -c cds_mouse.fasta -l lncrna_mouse.fasta -o SS_mouse -r -t 8
```
To train a model with secondary structure features, we only need to add the '-r' parameter. In addition to the outputs with a prefix 'SS_mouse' that are similar to the first example, five ss tables will also be generated, including SS_mouse_ss_table_k1.csv, SS_mouse_ss_table_k2.csv, SS_mouse_ss_table_k3.csv, SS_mouse_ss_table_k4.csv and SS_mouse_ss_table_k5.csv.

## Authors
Minghua Li

## Contact
lim74@miamioh.edu

## License
This project is licensed under the MIT License. 
Copyright (c) 2020 lim74
