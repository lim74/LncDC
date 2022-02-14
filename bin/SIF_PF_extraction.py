import re
import os
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from Bio.SeqUtils import ProtParam

# Sequence Intrinsic features (SIF)

#---------------------------------------------------------------------------------------
# Feature 1: GC content
def gcContent(sequence):
    '''

    :param sequence: primary sequence of a transcript
    :return: the GC content of the sequence
    '''
    Gcontent = 0
    Ccontent = 0
    for base in sequence:
        if base == 'G':
            Gcontent += 1
        if base == 'C':
            Ccontent += 1
    GCcontent = (Gcontent+Ccontent)/len(sequence)
    return GCcontent

def run_gcContent(data):
    '''

    :param data: input dataframe
    :return: processed dataframe
    '''
    data['GC_content'] = gcContent(data['Sequence'])
    return data

def run_gcContent_parallel(data):
    '''

    :param data: input dataframe
    :return: processed dataframe
    '''
    data = data.apply(run_gcContent, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# Feature 2: Fickett score

# Fickett TESTCODE data
# NAR 10(17) 5303-531
position_parameter = [1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,0.0]
position_prob_coding = {
'A':[0.94,0.68,0.84,0.93,0.58,0.68,0.45,0.34,0.20,0.22],
'C':[0.80,0.70,0.70,0.81,0.66,0.48,0.51,0.33,0.30,0.23],
'G':[0.90,0.88,0.74,0.64,0.53,0.48,0.27,0.16,0.08,0.08],
'T':[0.97,0.97,0.91,0.68,0.69,0.44,0.54,0.20,0.09,0.09]
}
position_weight = {'A':0.26,'C':0.18,'G':0.31,'T':0.33}

content_parameter = [0.33,0.31,0.29,0.27,0.25,0.23,0.21,0.17,0]
content_prob_coding = {
'A':[0.28,0.49,0.44,0.55,0.62,0.49,0.67,0.65,0.81,0.21],
'C':[0.82,0.64,0.51,0.64,0.59,0.59,0.43,0.44,0.39,0.31],
'G':[0.40,0.54,0.47,0.64,0.64,0.73,0.41,0.41,0.33,0.29],
'T':[0.28,0.24,0.39,0.40,0.55,0.75,0.56,0.69,0.51,0.58]
}
content_weight = {'A':0.11,'C':0.12,'G':0.15,'T':0.14}

def position_map(pos_value, base):
    if pos_value < 0:
        return None
    for i,v in enumerate(position_parameter):
        if pos_value >= v:
            posbyweight = position_prob_coding[base][i]*position_weight[base]
            return posbyweight

def content_map(con_value, base):
    if con_value < 0:
        return None
    for i,v in enumerate(content_parameter):
        if con_value >= v:
            conbyweight = content_prob_coding[base][i]*content_weight[base]
            return conbyweight

def fickett(sequence):
    A123pos = [0, 0, 0]  # A1, A2, A3
    T123pos = [0, 0, 0]  # T1, T2, T3
    G123pos = [0, 0, 0]  # G1, G2, G3
    C123pos = [0, 0, 0]  # C1, C2, C3
    fickett_score = 0
    for i in range(3):
        for j in range(i, len(sequence), 3):
            if sequence[j] == 'A':
                A123pos[i] += 1
            elif sequence[j] == 'T':
                T123pos[i] += 1
            elif sequence[j] == 'G':
                G123pos[i] += 1
            elif sequence[j] == 'C':
                C123pos[i] += 1
    # A, T, G, C positions
    Apos = max(A123pos[0], A123pos[1], A123pos[2]) / (min(A123pos[0], A123pos[1], A123pos[2]) + 1)
    Tpos = max(T123pos[0], T123pos[1], T123pos[2]) / (min(T123pos[0], T123pos[1], T123pos[2]) + 1)
    Gpos = max(G123pos[0], G123pos[1], G123pos[2]) / (min(G123pos[0], G123pos[1], G123pos[2]) + 1)
    Cpos = max(C123pos[0], C123pos[1], C123pos[2]) / (min(C123pos[0], C123pos[1], C123pos[2]) + 1)
    # A, T, G, C content
    Acon = sequence.count('A') / len(sequence)
    Tcon = sequence.count('T') / len(sequence)
    Gcon = sequence.count('G') / len(sequence)
    Ccon = sequence.count('C') / len(sequence)

    # fickett score
    fickett_score += position_map(Apos, 'A')
    fickett_score += position_map(Tpos, 'T')
    fickett_score += position_map(Gpos, 'G')
    fickett_score += position_map(Cpos, 'C')

    fickett_score += content_map(Acon, 'A')
    fickett_score += content_map(Tcon, 'T')
    fickett_score += content_map(Gcon, 'G')
    fickett_score += content_map(Ccon, 'C')
    return fickett_score

def run_fickett(data):
    data['Fickett_score'] = fickett(data['Sequence'])
    return data

def run_fickett_parallel(data):
    data = data.apply(run_fickett, axis = 1)
    return data


#---------------------------------------------------------------------------------------
# feature 3: type 0 ORF length (type 0 ORF Sequence also extracted)
def translation(sequence):
    '''

    :param sequence: input primary sequence of a transcript
    :return: protein sequence
    '''
    trans_dic = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L',
                 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'ATT': 'I', 'ATC': 'I',
                 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V',
                 'GTG': 'V', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACT': 'T',
                 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A',
                 'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*',
                 'TAG': '*', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
                 'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D',
                 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'TGT': 'C', 'TGC': 'C',
                 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R',
                 'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
                 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}
    protein = ''
    for i in range(0, len(sequence)-2, 3):
        protein += trans_dic[sequence[i:i+3]]
    return protein

def ORFfinder_T0(sequence):
    # 3 frame translation
    trans1 = translation(sequence)
    trans2 = translation(sequence[1:])
    trans3 = translation(sequence[2:])

    orf1 = re.finditer(r'M.*?\*', trans1)
    orf2 = re.finditer(r'M.*?\*', trans2)
    orf3 = re.finditer(r'M.*?\*', trans3)
    orf1_seqs = [(m.start(), m.end(), m.group(), 'frame1') for m in orf1]
    orf2_seqs = [(m.start(), m.end(), m.group(), 'frame2') for m in orf2]
    orf3_seqs = [(m.start(), m.end(), m.group(), 'frame3') for m in orf3]

    orfs = orf1_seqs + orf2_seqs + orf3_seqs
    if len(orfs) == 0:
        return ('NaN', 'NaN'), 'NaN', 0
    else:
        orf_sorted = sorted(orfs, key=lambda t: len(t[2]), reverse=True)
        longest_orf_protein_seq = orf_sorted[0][2]
        longest_orf_frame = orf_sorted[0][3]
        longest_orf_pro_position = (int(orf_sorted[0][0]), int(orf_sorted[0][1]))
        if longest_orf_frame == 'frame1':
            longest_orf_rna_seq_position = (longest_orf_pro_position[0] * 3, longest_orf_pro_position[1] * 3)
            longest_orf_rna_seq = sequence[longest_orf_rna_seq_position[0]:longest_orf_rna_seq_position[1]]
            longest_orf_rna_length = longest_orf_rna_seq_position[1] - longest_orf_rna_seq_position[0]
            return longest_orf_rna_seq_position, longest_orf_rna_seq, longest_orf_rna_length
        elif longest_orf_frame == 'frame2':
            longest_orf_rna_seq_position = (longest_orf_pro_position[0] * 3, longest_orf_pro_position[1] * 3)
            longest_orf_rna_seq = sequence[1:][longest_orf_rna_seq_position[0]:longest_orf_rna_seq_position[1]]
            longest_orf_rna_seq_original_position = (
            longest_orf_pro_position[0] * 3 + 1, longest_orf_pro_position[1] * 3 + 1)
            longest_orf_rna_length = longest_orf_rna_seq_original_position[1] - longest_orf_rna_seq_original_position[0]
            return longest_orf_rna_seq_original_position, longest_orf_rna_seq, longest_orf_rna_length
        elif longest_orf_frame == 'frame3':
            longest_orf_rna_seq_position = (longest_orf_pro_position[0] * 3, longest_orf_pro_position[1] * 3)
            longest_orf_rna_seq = sequence[2:][longest_orf_rna_seq_position[0]:longest_orf_rna_seq_position[1]]
            longest_orf_rna_seq_original_position = (
            longest_orf_pro_position[0] * 3 + 2, longest_orf_pro_position[1] * 3 + 2)
            longest_orf_rna_length = longest_orf_rna_seq_original_position[1] - longest_orf_rna_seq_original_position[0]
            return longest_orf_rna_seq_original_position, longest_orf_rna_seq, longest_orf_rna_length

def run_orf_T0(data):
    orf = ORFfinder_T0(data['Sequence'])
    data['ORF_T0_Sequence'] = orf[1]
    data['ORF_T0_length'] = orf[2]
    return data

def run_orf_T0_parallel(data):
    data = data.apply(run_orf_T0, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 4: type 1 ORF length (type 1 ORF Sequence also extracted)
def ORFfinder_T1(sequence):
    end = len(sequence)
    start = end
    current_orf_len = 0
    current_orf_seq = ''
    current_position = (0,0)
    for frame in range(3):
        for i in range(frame,len(sequence),3):
            codon = sequence[i:i+3]
            if codon == 'ATG':
                start = i
                break
            else:
                continue
        current_len = end - start
        if current_len >= current_orf_len:
            current_orf_len = current_len
            current_orf_seq = sequence[start:end]
            current_position = (start, end)
    if int(current_orf_len) == 0:
        return ('NaN','NaN'),'NaN', 0
    else:
        return current_position, current_orf_seq, current_orf_len

def run_orf_T1(data):
    orf = ORFfinder_T1(data['Sequence'])
    data['ORF_T1_Sequence'] = orf[1]
    data['ORF_T1_length'] = orf[2]
    return data

def run_orf_T1_parallel(data):
    data = data.apply(run_orf_T1, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 5: type 2 ORF length (type 2 ORF Sequence also extracted)
def ORFfinder_T2(sequence):
    start = 0
    end = 0
    current_orf_len = 0
    current_orf_seq = ''
    current_position = (0,0)
    for frame in range(3):
        for i in range(frame,len(sequence),3):
            codon = sequence[i:i+3]
            if codon in ['TAG','TGA','TAA']:
                end = i+3
                current_len = end - start
                if current_len >= current_orf_len:
                    current_orf_len = current_len
                    current_orf_seq = sequence[start:end]
                    current_position = (start, end)
                start = end
            else:
                continue
    if int(current_orf_len) == 0:
        return ('NaN','NaN'),'NaN', 0
    else:
        return current_position, current_orf_seq, current_orf_len

def run_orf_T2(data):
    orf = ORFfinder_T2(data['Sequence'])
    data['ORF_T2_Sequence'] = orf[1]
    data['ORF_T2_length'] = orf[2]
    return data

def run_orf_T2_parallel(data):
    data = data.apply(run_orf_T2, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# type 3 ORF length (type 3 ORF Sequence also extracted)
# Not feature, used for calculate type 3 ORF coverage
def ORFfinder_T3(sequence):
    end = len(sequence)
    start = end
    current_orf_len = 0
    current_orf_seq = ''
    current_position = (0, 0)
    for frame in range(3):
        for i in range(frame, len(sequence), 3):
            codon = sequence[i:i + 3]
            if codon == 'ATG':
                start = i
                break
            else:
                continue
        current_len = end - start
        if current_len >= current_orf_len:
            current_orf_len = current_len
            current_orf_seq = sequence[start:end]
            current_position = (start, end)

    start = 0
    end = 0
    for frame in range(3):
        for i in range(frame, len(sequence), 3):
            codon = sequence[i:i + 3]
            if codon in ['TAG', 'TGA', 'TAA']:
                end = i + 3
                current_len = end - start
                if current_len >= current_orf_len:
                    current_orf_len = current_len
                    current_orf_seq = sequence[start:end]
                    current_position = (start, end)
                start = end
            else:
                continue

    if int(current_orf_len) == 0:
        return ('NaN', 'NaN'), 'NaN', 0
    else:
        return current_position, current_orf_seq, current_orf_len

def run_orf_T3(data):
    orf = ORFfinder_T3(data['Sequence'])
    data['ORF_T3_Sequence'] = orf[1]
    data['ORF_T3_length'] = orf[2]
    return data

def run_orf_T3_parallel(data):
    data = data.apply(run_orf_T3, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 6: type 0 ORF coverage
def run_orf_T0_coverage(data):
    data['ORF_T0_coverage'] = data['ORF_T0_length']/data['Transcript_length']
    return data

def run_orf_T0_coverage_parallel(data):
    data = data.apply(run_orf_T0_coverage, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 7: type 1 ORF coverage
def run_orf_T1_coverage(data):
    data['ORF_T1_coverage'] = data['ORF_T1_length']/data['Transcript_length']
    return data

def run_orf_T1_coverage_parallel(data):
    data = data.apply(run_orf_T1_coverage, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 8: type 3 ORF coverage
def run_orf_T3_coverage(data):
    data['ORF_T3_coverage'] = data['ORF_T3_length']/data['Transcript_length']
    return data

def run_orf_T3_coverage_parallel(data):
    data = data.apply(run_orf_T3_coverage, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 9: type 0 ORF hexamer score

def word_generator(seq, word_size, step_size, frame = 0):
    # generate DNA word from sequence using word_size and step_size, Frame is 0, 1 or 2
    for i in range(frame, len(seq), step_size):
        word = seq[i:i+word_size]
        if len(word) == word_size:
            yield word

def hexamer_ratio_orf(seq, word_size, step_size, coding, noncoding):
    '''

    :param seq: primary sequence of a transcript
    :param word_size: length of an word that include adjacent nucleotide acids, like 'ATG'
    :param step_size: the step size that the sliding window move every time
    :param coding: coding hexamer table
    :param noncoding: noncoding hexamer table
    :return:
    '''
    if len(seq) < word_size:
        return 0
    sum_log_ratio = 0
    hexamer_num = 0
    for i in word_generator(seq = seq, word_size = word_size,
                                step_size = step_size, frame = 0):
        if (i not in coding) or (i not in noncoding):
            continue
        if coding[i]>0 and noncoding[i]>0:
            ratio = np.log(coding[i]/noncoding[i])
            sum_log_ratio += ratio
        elif coding[i]>0 and noncoding[i] == 0:
            sum_log_ratio += 1
        elif coding[i] == 0 and noncoding[i] > 0:
            sum_log_ratio -= 1
        else:
            continue
        hexamer_num += 1
    hexamer_score_inframe = sum_log_ratio/hexamer_num

    return hexamer_score_inframe

def run_orf_T0_hexamer(data, coding_hexamer, noncoding_hexamer):
    if data['ORF_T0_length'] != 0:
        data['Hexamer_score_ORF_T0'] = hexamer_ratio_orf(data['ORF_T0_Sequence'], 6, 3, coding_hexamer, noncoding_hexamer)
    else:
        data['Hexamer_score_ORF_T0'] = 'NaN'
    return data

def run_orf_T0_hexamer_parallel(data, coding_hexamer, noncoding_hexamer):
    data = data.apply(run_orf_T0_hexamer, args = [coding_hexamer, noncoding_hexamer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 10: type 1 ORF hexamer score
def run_orf_T1_hexamer(data, coding_hexamer, noncoding_hexamer):
    if data['ORF_T1_length'] != 0:
        data['Hexamer_score_ORF_T1'] = hexamer_ratio_orf(data['ORF_T1_Sequence'], 6, 3, coding_hexamer, noncoding_hexamer)
    else:
        data['Hexamer_score_ORF_T1'] = 'NaN'
    return data

def run_orf_T1_hexamer_parallel(data, coding_hexamer, noncoding_hexamer):
    data = data.apply(run_orf_T1_hexamer, args = [coding_hexamer, noncoding_hexamer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 11: type 2 ORF hexamer score
def run_orf_T2_hexamer(data, coding_hexamer, noncoding_hexamer):
    if data['ORF_T2_length'] != 0:
        data['Hexamer_score_ORF_T2'] = hexamer_ratio_orf(data['ORF_T2_Sequence'], 6, 3, coding_hexamer, noncoding_hexamer)
    else:
        data['Hexamer_score_ORF_T2'] = 'NaN'
    return data

def run_orf_T2_hexamer_parallel(data, coding_hexamer, noncoding_hexamer):
    data = data.apply(run_orf_T2_hexamer, args = [coding_hexamer, noncoding_hexamer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 12: type 3 ORF hexamer score
def run_orf_T3_hexamer(data, coding_hexamer, noncoding_hexamer):
    if data['ORF_T3_length'] != 0:
        data['Hexamer_score_ORF_T3'] = hexamer_ratio_orf(data['ORF_T3_Sequence'], 6, 3, coding_hexamer, noncoding_hexamer)
    else:
        data['Hexamer_score_ORF_T3'] = 'NaN'
    return data

def run_orf_T3_hexamer_parallel(data, coding_hexamer, noncoding_hexamer):
    data = data.apply(run_orf_T3_hexamer, args = [coding_hexamer, noncoding_hexamer], axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 13: type 0 ORF Relative Codon Bias(RCB)
def RCB_score(seq):
    condon_dict = {}         # a dictionary contain the counts of each condon
    condon_list = []         # a list contain the condons in a sequence
    Frequency_1 = {'A': 0,
                'T': 0,
                'G' : 0,
                'C' : 0}
    Frequency_2 = {'A': 0,
                'T': 0,
                'G' : 0,
                'C' : 0}
    Frequency_3 = {'A': 0,
                'T': 0,
                'G' : 0,
                'C' : 0}

    # First position
    for i in range(0,len(seq),3):
        if seq[i] == 'A':
            Frequency_1['A'] += 1
        elif seq[i] == 'T':
            Frequency_1['T'] += 1
        elif seq[i] == 'G':
            Frequency_1['G'] += 1
        elif seq[i] == 'C':
            Frequency_1['C'] += 1

        if len(seq[i:i+3]) != 3:
            continue
        elif seq[i:i+3] not in condon_dict:
            condon_dict[seq[i:i+3]] = 1
        else:
            condon_dict[seq[i:i+3]] += 1

        if len(seq[i:i+3]) == 3:
            condon_list.append(seq[i:i+3])

    # Second position
    for i in range(1,len(seq),3):
        if seq[i] == 'A':
            Frequency_2['A'] += 1
        elif seq[i] == 'T':
            Frequency_2['T'] += 1
        elif seq[i] == 'G':
            Frequency_2['G'] += 1
        elif seq[i] == 'C':
            Frequency_2['C'] += 1

    # Third position
    for i in range(2,len(seq),3):
        if seq[i] == 'A':
            Frequency_3['A'] += 1
        elif seq[i] == 'T':
            Frequency_3['T'] += 1
        elif seq[i] == 'G':
            Frequency_3['G'] += 1
        elif seq[i] == 'C':
            Frequency_3['C'] += 1

    total_condons = len(condon_list)
    condon_sum = 0
    for k in condon_list:
        d = np.log(np.absolute((condon_dict[k]/total_condons) - (Frequency_1[k[0]]/total_condons) *
             (Frequency_2[k[1]]/total_condons) * (Frequency_3[k[2]]/total_condons)) / ((Frequency_1[k[0]]/total_condons) *
             (Frequency_2[k[1]]/total_condons) * (Frequency_3[k[2]]/total_condons)) + 1)
        condon_sum = condon_sum + d

    RCB = np.exp(condon_sum/len(condon_list))-1
    return RCB

def run_orf_T0_RCB(data):
    if data['ORF_T0_length'] != 0:
        data['RCB_T0'] = RCB_score(data['ORF_T0_Sequence'])
    else:
        data['RCB_T0'] = 'NaN'
    return data

def run_orf_T0_RCB_parallel(data):
    data = data.apply(run_orf_T0_RCB, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 14: type 1 ORF Relative Codon Bias(RCB)
def run_orf_T1_RCB(data):
    if data['ORF_T1_length'] != 0:
        data['RCB_T1'] = RCB_score(data['ORF_T1_Sequence'])
    else:
        data['RCB_T1'] = 'NaN'
    return data

def run_orf_T1_RCB_parallel(data):
    data = data.apply(run_orf_T1_RCB, axis = 1)
    return data

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# Protein Features (PF)
def cal_protein_features(seq):
    '''

    :param seq: input primary sequence of a transcript
    :return: pi: (protein isoelectric),
             mw: (molecular weight)
             # gravy: (grand average of hydropathy)
             aromaticity: (relative frequency of Phe+Trp+Tyr)
             instability: (instability index --
                          Any value above 40 means the protein is unstable (=has a short half life)).
    '''
    protein_seq = translation(seq).strip('*')
    protein_object = ProtParam.ProteinAnalysis(protein_seq)
    pi = protein_object.isoelectric_point()
    mw = protein_object.molecular_weight()
    # gravy = protein_object.gravy()
    aromaticity = protein_object.aromaticity()
    instability = protein_object.instability_index()

    return pi, mw, aromaticity, instability

#---------------------------------------------------------------------------------------
# feature 15-18: type 0 ORF protein features, including:
# pi, mw, aromaticity and instability
def run_pf_T0(data):
    if data['ORF_T0_length'] != 0:
        pi,mw,aromaticity,instability = cal_protein_features(data['ORF_T0_Sequence'])
        data['ORF_T0_PI'] = pi
        data['ORF_T0_MW'] = mw
        data['ORF_T0_aromaticity'] = aromaticity
        data['ORF_T0_instability'] = instability
    else:
        data['ORF_T0_PI'] = 'NaN'
        data['ORF_T0_MW'] = 'NaN'
        data['ORF_T0_aromaticity'] = 'NaN'
        data['ORF_T0_instability'] = 'NaN'
    return data

def run_pf_T0_parallel(data):
    data = data.apply(run_pf_T0, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 19-20: type 1 ORF protein features, including:
# mw and instability
def run_pf_T1(data):
    if data['ORF_T1_length'] != 0:
        orf_t1 = data['ORF_T1_Sequence']
        sequence = orf_t1
        for i in range(0,len(orf_t1),3):
            codon = orf_t1[i:i+3]
            if codon in ['TAG','TGA','TAA']:
                sequence = orf_t1[0:i]
                break
            else:
                continue

        pi,mw,aromaticity,instability = cal_protein_features(sequence)
        data['ORF_T1_MW'] = mw
        data['ORF_T1_instability'] = instability
    else:
        data['ORF_T1_MW'] = 'NaN'
        data['ORF_T1_instability'] = 'NaN'
    return data

def run_pf_T1_parallel(data):
    data = data.apply(run_pf_T1, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 21: type 2 ORF protein features, including: mw
def run_pf_T2(data):
    if data['ORF_T2_length'] != 0:
        protein_seq = translation(data['ORF_T2_Sequence']).strip('*')
        protein_object = ProtParam.ProteinAnalysis(protein_seq)
        mw = protein_object.molecular_weight()
        data['ORF_T2_MW'] = mw
    else:
        data['ORF_T2_MW'] = 'NaN'
    return data

def run_pf_T2_parallel(data):
    data = data.apply(run_pf_T2, axis = 1)
    return data

#---------------------------------------------------------------------------------------
# feature 22: type 3 ORF protein features, including: mw
def run_pf_T3(data):
    if data['ORF_T3_length'] != 0:
        orf_t3 = data['ORF_T3_Sequence']
        sequence = orf_t3
        if orf_t3.startswith('ATG'):
            for i in range(0,len(orf_t3),3):
                codon = orf_t3[i:i+3]
                if codon in ['TAG','TGA','TAA']:
                    sequence = orf_t3[0:i]
                    break
                else:
                    continue
        protein_seq = translation(sequence).strip('*')
        protein_object = ProtParam.ProteinAnalysis(protein_seq)
        mw = protein_object.molecular_weight()
        data['ORF_T3_MW'] = mw
    else:
        data['ORF_T3_MW'] = 'NaN'
    return data

def run_pf_T3_parallel(data):
    data = data.apply(run_pf_T3, axis = 1)
    return data


#---------------------------------------------------------------------------------------
# feature extraction
def feature_extract(dataset, thread, feature):
    run_feature_parallel = feature
    df_chunks = np.array_split(dataset, thread*10)
    with multiprocessing.Pool(thread) as pool:
        dataset = pd.concat(pool.map(run_feature_parallel, df_chunks), ignore_index = False)
    return dataset

def feature_extract_hexamer(dataset, thread, feature, coding_hexamer, noncoding_hexamer):
    df_chunks = np.array_split(dataset, thread*10)
    parallel_function = partial(feature, coding_hexamer=coding_hexamer, noncoding_hexamer=noncoding_hexamer)
    with multiprocessing.Pool(thread) as pool:
        dataset = pd.concat(pool.map(parallel_function, df_chunks), ignore_index = False)
    return dataset

def sif_pf_extract(dataset, thread, coding_hexamer, noncoding_hexamer):
    # Feature 1: GC content
    dataset = feature_extract(dataset, thread, run_gcContent_parallel)
    # Feature 2: Fickett score
    dataset = feature_extract(dataset, thread, run_fickett_parallel)
    # feature 3: type 0 ORF length (type 0 ORF Sequence also extracted)
    dataset = feature_extract(dataset, thread, run_orf_T0_parallel)
    # feature 4: type 1 ORF length (type 1 ORF Sequence also extracted)
    dataset = feature_extract(dataset, thread, run_orf_T1_parallel)
    # feature 5: type 2 ORF length (type 2 ORF Sequence also extracted)
    dataset = feature_extract(dataset, thread, run_orf_T2_parallel)
    # type 3 ORF length (type 3 ORF Sequence also extracted)
    # Not feature, used for calculate type 3 ORF coverage
    dataset = feature_extract(dataset, thread, run_orf_T3_parallel)
    # feature 6: type 0 ORF coverage
    dataset = feature_extract(dataset, thread, run_orf_T0_coverage_parallel)
    # feature 7: type 1 ORF coverage
    dataset = feature_extract(dataset, thread, run_orf_T1_coverage_parallel)
    # feature 8: type 3 ORF coverage
    dataset = feature_extract(dataset, thread, run_orf_T3_coverage_parallel)
    # feature 9: type 0 ORF hexamer score
    dataset = feature_extract_hexamer(dataset, thread, run_orf_T0_hexamer_parallel,
                                      coding_hexamer, noncoding_hexamer)
    # feature 10: type 1 ORF hexamer score
    dataset = feature_extract_hexamer(dataset, thread, run_orf_T1_hexamer_parallel,
                                      coding_hexamer, noncoding_hexamer)
    # feature 11: type 2 ORF hexamer score
    dataset = feature_extract_hexamer(dataset, thread, run_orf_T2_hexamer_parallel,
                                      coding_hexamer, noncoding_hexamer)
    # feature 12: type 3 ORF hexamer score
    dataset = feature_extract_hexamer(dataset, thread, run_orf_T3_hexamer_parallel,
                                      coding_hexamer, noncoding_hexamer)
    # feature 13: type 0 ORF Relative Codon Bias(RCB)
    dataset = feature_extract(dataset, thread, run_orf_T0_RCB_parallel)
    # feature 14: type 1 ORF Relative Codon Bias(RCB)
    dataset = feature_extract(dataset, thread, run_orf_T1_RCB_parallel)
    # feature 15-18: type 0 ORF protein features, including:
    # pi, mw, aromaticity and instability
    dataset = feature_extract(dataset, thread, run_pf_T0_parallel)
    # feature 19-20: type 1 ORF protein features, including:
    # mw and instability
    dataset = feature_extract(dataset, thread, run_pf_T1_parallel)
    # feature 21: type 2 ORF protein features, including: mw
    dataset = feature_extract(dataset, thread, run_pf_T2_parallel)
    # feature 22: type 3 ORF protein features, including: mw
    dataset = feature_extract(dataset, thread, run_pf_T3_parallel)
    return dataset

