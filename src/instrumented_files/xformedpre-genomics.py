import sys
from instrument_lib import *
import numpy as np
import os
print(1, 3)
path_0 = os.getcwd()


def read_fasta(fasta_file):
    print('enter scope 1')
    print(1, 6)
    print(3, 7)
    fasta_file_1 = fasta_file
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 12)
    fasta_dict_1 = {}
    with open(fasta_file_1, 'r') as f_1:
        for line_1 in f_1:
            print(5, 15)
            line_1 = line_1.strip()
            if not line_1:
                continue
            if line_1.startswith('>'):
                print(9, 19)
                active_sequence_name_1 = line_1[1:]
                if active_sequence_name_1 not in fasta_dict_1:
                    print(11, 21)
                    fasta_dict_1[active_sequence_name_1] = []
                continue
            print(10, 23)
            sequence_1 = line_1
            fasta_dict_1[active_sequence_name_1].append(sequence_1)
    print('exit scope 1')
    return fasta_dict_1
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 2')
    print(1, 28)
    print(16, 29)
    fastq_file_2 = fastq_file
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 34)
    fastq_dict_2 = {}
    with open(fastq_file_2, 'r') as f_2:
        for line_2 in f_2:
            print(18, 37)
            line_2 = line_2.strip()
            if not line_2:
                continue
            if line_2.startswith('@'):
                print(22, 41)
                active_sequence_name_2 = line_2[1:]
                if active_sequence_name_2 not in fastq_dict_2:
                    print(24, 43)
                    fastq_dict_2[active_sequence_name_2] = []
                continue
            print(23, 45)
            sequence_2 = line_2
            fastq_dict_2[active_sequence_name_2].append(sequence_2)
    print('exit scope 2')
    return fastq_dict_2
    print('exit scope 2')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 3')
    print(1, 50)
    print(29, 51)
    fasta_file_3 = fasta_file
    print(29, 52)
    quality_file_3 = quality_file
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 57)
    fasta_dict_3 = {}
    with open(fasta_file_3, 'r') as f_3:
        for line_3 in f_3:
            print(31, 60)
            line_3 = line_3.strip()
            if not line_3:
                continue
            if line_3.startswith('>'):
                print(35, 64)
                active_sequence_name_3 = line_3[1:]
                if active_sequence_name_3 not in fasta_dict_3:
                    print(37, 66)
                    fasta_dict_3[active_sequence_name_3] = []
                continue
            print(36, 68)
            sequence_3 = line_3
            fasta_dict_3[active_sequence_name_3].append(sequence_3)
    print(32, 70)
    quality_dict_3 = {}
    with open(quality_file_3, 'r') as f_3:
        for line_3 in f_3:
            print(40, 73)
            line_3 = line_3.strip()
            if not line_3:
                continue
            if line_3.startswith('>'):
                print(44, 77)
                active_sequence_name_3 = line_3[1:]
                if active_sequence_name_3 not in quality_dict_3:
                    print(46, 79)
                    quality_dict_3[active_sequence_name_3] = []
                continue
            print(45, 81)
            quality_3 = line_3
            quality_dict_3[active_sequence_name_3].append(quality_3)
    print('exit scope 3')
    return fasta_dict_3, quality_dict_3
    print('exit scope 3')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 4')
    print(1, 86)
    print(51, 89)
    fasta_file_4 = fasta_file
    print(51, 90)
    fastq_file_4 = fastq_file
    print(51, 91)
    output_file_4 = output_file
    print(51, 92)
    min_length_4 = min_length
    print(51, 93)
    max_length_4 = max_length
    print(51, 94)
    min_quality_4 = min_quality
    print(51, 95)
    max_quality_4 = max_quality
    print(51, 96)
    min_length_fraction_4 = min_length_fraction
    print(51, 97)
    max_length_fraction_4 = max_length_fraction
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 101)
    fasta_dict_4 = read_fasta(fasta_file_4)
    print(51, 102)
    fastq_dict_4 = read_fastq(fastq_file_4)
    print(51, 103)
    filtered_fasta_dict_4 = filter_fasta(fasta_dict_4, min_length_4,
        max_length_4, min_length_fraction_4, max_length_fraction_4)
    print(51, 105)
    filtered_fastq_dict_4 = filter_fastq(fastq_dict_4, min_quality_4,
        max_quality_4)
    write_output(filtered_fasta_dict_4, filtered_fastq_dict_4, output_file_4)
    print('exit scope 4')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 5')
    print(1, 110)
    print(54, 112)
    fasta_dict_5 = fasta_dict
    print(54, 113)
    min_length_5 = min_length
    print(54, 114)
    max_length_5 = max_length
    print(54, 115)
    min_length_fraction_5 = min_length_fraction
    print(54, 116)
    max_length_fraction_5 = max_length_fraction
    """
    Filters a fasta dictionary by length
    """
    print(54, 120)
    filtered_fasta_dict_5 = {}
    for key_5 in fasta_dict_5:
        print(56, 122)
        sequence_5 = fasta_dict_5[key_5][0]
        print(56, 123)
        length_5 = len(sequence_5)
        if min_length_fraction_5 > 0:
            if length_5 < min_length_fraction_5 * len(fasta_dict_5[key_5][0]):
                continue
        if max_length_fraction_5 > 0:
            if length_5 > max_length_fraction_5 * len(fasta_dict_5[key_5][0]):
                continue
        if min_length_5 > 0:
            if length_5 < min_length_5:
                continue
        if max_length_5 > 0:
            if length_5 > max_length_5:
                continue
        print(71, 136)
        filtered_fasta_dict_5[key_5] = fasta_dict_5[key_5]
    print('exit scope 5')
    return filtered_fasta_dict_5
    print('exit scope 5')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 6')
    print(1, 140)
    print(77, 141)
    fastq_dict_6 = fastq_dict
    print(77, 142)
    min_quality_6 = min_quality
    print(77, 143)
    max_quality_6 = max_quality
    """
    Filters a fastq dictionary by quality
    """
    print(77, 147)
    filtered_fastq_dict_6 = {}
    for key_6 in fastq_dict_6:
        print(79, 149)
        quality_6 = fastq_dict_6[key_6][0]
        if min_quality_6 > 0:
            if quality_6 < min_quality_6:
                continue
        if max_quality_6 > 0:
            if quality_6 > max_quality_6:
                continue
        print(86, 156)
        filtered_fastq_dict_6[key_6] = fastq_dict_6[key_6]
    print('exit scope 6')
    return filtered_fastq_dict_6
    print('exit scope 6')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 7')
    print(1, 160)
    print(92, 161)
    filtered_fasta_dict_7 = filtered_fasta_dict
    print(92, 162)
    filtered_fastq_dict_7 = filtered_fastq_dict
    print(92, 163)
    output_file_7 = output_file
    """
    Writes output
    """
    with open(output_file_7, 'w') as f_7:
        for key_7 in filtered_fasta_dict_7:
            f_7.write('>' + key_7 + '\n')
            f_7.write(filtered_fasta_dict_7[key_7][0] + '\n')
        for key_7 in filtered_fastq_dict_7:
            f_7.write('@' + key_7 + '\n')
            f_7.write(filtered_fastq_dict_7[key_7][0] + '\n')
            f_7.write('+\n')
            f_7.write(filtered_fastq_dict_7[key_7][1] + '\n')
    print('exit scope 7')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print('enter scope 8')
    print(1, 178)
    print(100, 179)
    seq1_8 = seq1
    print(100, 180)
    seq2_8 = seq2
    print(100, 181)
    match_score_8 = match_score
    print(100, 182)
    mismatch_score_8 = mismatch_score
    print(100, 183)
    gap_score_8 = gap_score
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 187)
    scoring_matrix_8 = [[(0) for j_8 in range(len(seq2_8) + 1)] for i_8 in
        range(len(seq1_8) + 1)]
    print(100, 189)
    traceback_matrix_8 = [[None for j_8 in range(len(seq2_8) + 1)] for i_8 in
        range(len(seq1_8) + 1)]
    print(100, 191)
    max_score_8 = 0
    print(100, 192)
    max_pos_8 = None
    for i_8 in range(1, len(seq1_8) + 1):
        for j_8 in range(1, len(seq2_8) + 1):
            print(104, 195)
            letter1_8 = seq1_8[i_8 - 1]
            print(104, 196)
            letter2_8 = seq2_8[j_8 - 1]
            if letter1_8 == letter2_8:
                print(106, 198)
                diagonal_score_8 = scoring_matrix_8[i_8 - 1][j_8 - 1
                    ] + match_score_8
            else:
                print(108, 201)
                diagonal_score_8 = scoring_matrix_8[i_8 - 1][j_8 - 1
                    ] + mismatch_score_8
            print(107, 203)
            up_score_8 = scoring_matrix_8[i_8 - 1][j_8] + gap_score_8
            print(107, 204)
            left_score_8 = scoring_matrix_8[i_8][j_8 - 1] + gap_score_8
            if diagonal_score_8 >= up_score_8:
                if diagonal_score_8 >= left_score_8:
                    print(115, 207)
                    scoring_matrix_8[i_8][j_8] = diagonal_score_8
                    print(115, 208)
                    traceback_matrix_8[i_8][j_8] = 'D'
                else:
                    print(117, 210)
                    scoring_matrix_8[i_8][j_8] = left_score_8
                    print(117, 211)
                    traceback_matrix_8[i_8][j_8] = 'L'
            elif up_score_8 >= left_score_8:
                print(112, 213)
                scoring_matrix_8[i_8][j_8] = up_score_8
                print(112, 214)
                traceback_matrix_8[i_8][j_8] = 'U'
            else:
                print(114, 216)
                scoring_matrix_8[i_8][j_8] = left_score_8
                print(114, 217)
                traceback_matrix_8[i_8][j_8] = 'L'
            if scoring_matrix_8[i_8][j_8] >= max_score_8:
                print(118, 219)
                max_score_8 = scoring_matrix_8[i_8][j_8]
                print(118, 220)
                max_pos_8 = i_8, j_8
    print(103, 221)
    i_8 = max_pos_8
    print(103, 222)
    j_8 = max_pos_8
    print(103, 223)
    aln1_8 = ''
    print(103, 224)
    aln2_8 = ''
    while traceback_matrix_8[i_8][j_8] != None:
        if traceback_matrix_8[i_8][j_8] == 'D':
            print(123, 227)
            aln1_8 += seq1_8[i_8 - 1]
            print(123, 228)
            aln2_8 += seq2_8[j_8 - 1]
            print(123, 229)
            i_8 -= 1
            print(123, 230)
            j_8 -= 1
        elif traceback_matrix_8[i_8][j_8] == 'L':
            print(126, 232)
            aln1_8 += '-'
            print(126, 233)
            aln2_8 += seq2_8[j_8 - 1]
            print(126, 234)
            j_8 -= 1
        elif traceback_matrix_8[i_8][j_8] == 'U':
            print(129, 236)
            aln1_8 += seq1_8[i_8 - 1]
            print(129, 237)
            aln2_8 += '-'
            print(129, 238)
            i_8 -= 1
    print('exit scope 8')
    return aln1_8[::-1], aln2_8[::-1]
    print('exit scope 8')


def main():
    print('enter scope 9')
    print(1, 242)
    print(134, 243)
    fasta_file_9 = (path_0 +
        '/benchmarks/supplemental_files/fasta_example.fasta')
    print(134, 245)
    fastq_file_9 = path_0 + '/benchmarks/supplemental_files/fastq_large.fastq'
    print(134, 246)
    output_file_9 = path_0 + '/benchmarks/supplemental_files/output_file.txt'
    darwin_wga_workflow(fasta_file_9, fastq_file_9, output_file_9)
    print('exit scope 9')


if __name__ == '__main__':
    main()
