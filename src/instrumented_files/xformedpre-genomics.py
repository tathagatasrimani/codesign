import sys
from instrument_lib import *
print(1, 3)
path_0 = '/nfs/pool0/pmcewen/codesign/codesign/src'


def read_fasta(fasta_file):
    print('enter scope 1')
    print(1, 5)
    fasta_file_1 = fasta_file
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 10)
    fasta_dict_1 = {}
    with open(fasta_file_1, 'r') as f_1:
        for line_1 in f_1:
            print(5, 13)
            line_1 = line_1.strip()
            if not line_1:
                continue
            if line_1.startswith('>'):
                print(9, 17)
                active_sequence_name_1 = line_1[1:]
                if active_sequence_name_1 not in fasta_dict_1:
                    print(11, 19)
                    fasta_dict_1[active_sequence_name_1] = []
                continue
            print(10, 21)
            sequence_1 = line_1
            fasta_dict_1[active_sequence_name_1].append(sequence_1)
    print('exit scope 1')
    return fasta_dict_1
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 2')
    print(1, 26)
    fastq_file_2 = fastq_file
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 31)
    fastq_dict_2 = {}
    with open(fastq_file_2, 'r') as f_2:
        for line_2 in f_2:
            print(18, 34)
            line_2 = line_2.strip()
            if not line_2:
                continue
            if line_2.startswith('@'):
                print(22, 38)
                active_sequence_name_2 = line_2[1:]
                if active_sequence_name_2 not in fastq_dict_2:
                    print(24, 40)
                    fastq_dict_2[active_sequence_name_2] = []
                continue
            print(23, 42)
            sequence_2 = line_2
            fastq_dict_2[active_sequence_name_2].append(sequence_2)
    print('exit scope 2')
    return fastq_dict_2
    print('exit scope 2')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 3')
    print(1, 47)
    fasta_file_3 = fasta_file
    quality_file_3 = quality_file
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 52)
    fasta_dict_3 = {}
    with open(fasta_file_3, 'r') as f_3:
        for line_3 in f_3:
            print(31, 55)
            line_3 = line_3.strip()
            if not line_3:
                continue
            if line_3.startswith('>'):
                print(35, 59)
                active_sequence_name_3 = line_3[1:]
                if active_sequence_name_3 not in fasta_dict_3:
                    print(37, 61)
                    fasta_dict_3[active_sequence_name_3] = []
                continue
            print(36, 63)
            sequence_3 = line_3
            fasta_dict_3[active_sequence_name_3].append(sequence_3)
    print(32, 65)
    quality_dict_3 = {}
    with open(quality_file_3, 'r') as f_3:
        for line_3 in f_3:
            print(40, 68)
            line_3 = line_3.strip()
            if not line_3:
                continue
            if line_3.startswith('>'):
                print(44, 72)
                active_sequence_name_3 = line_3[1:]
                if active_sequence_name_3 not in quality_dict_3:
                    print(46, 74)
                    quality_dict_3[active_sequence_name_3] = []
                continue
            print(45, 76)
            quality_3 = line_3
            quality_dict_3[active_sequence_name_3].append(quality_3)
    print('exit scope 3')
    return fasta_dict_3, quality_dict_3
    print('exit scope 3')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 4')
    print(1, 81)
    fasta_file_4 = fasta_file
    fastq_file_4 = fastq_file
    output_file_4 = output_file
    min_length_4 = min_length
    max_length_4 = max_length
    min_quality_4 = min_quality
    max_quality_4 = max_quality
    min_length_fraction_4 = min_length_fraction
    max_length_fraction_4 = max_length_fraction
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 89)
    fasta_dict_4 = read_fasta(fasta_file_4)
    print(51, 91)
    fastq_dict_4 = read_fastq(fastq_file_4)
    print(51, 93)
    filtered_fasta_dict_4 = filter_fasta(fasta_dict_4, min_length_4,
        max_length_4, min_length_fraction_4, max_length_fraction_4)
    print(51, 96)
    filtered_fastq_dict_4 = filter_fastq(fastq_dict_4, min_quality_4,
        max_quality_4)
    write_output(filtered_fasta_dict_4, filtered_fastq_dict_4, output_file_4)
    print('exit scope 4')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 5')
    print(1, 102)
    fasta_dict_5 = fasta_dict
    min_length_5 = min_length
    max_length_5 = max_length
    min_length_fraction_5 = min_length_fraction
    max_length_fraction_5 = max_length_fraction
    """
    Filters a fasta dictionary by length
    """
    print(54, 106)
    filtered_fasta_dict_5 = {}
    for key_5 in fasta_dict_5:
        print(56, 108)
        sequence_5 = fasta_dict_5[key_5][0]
        print(56, 109)
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
        print(71, 122)
        filtered_fasta_dict_5[key_5] = fasta_dict_5[key_5]
    print('exit scope 5')
    return filtered_fasta_dict_5
    print('exit scope 5')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 6')
    print(1, 126)
    fastq_dict_6 = fastq_dict
    min_quality_6 = min_quality
    max_quality_6 = max_quality
    """
    Filters a fastq dictionary by quality
    """
    print(77, 130)
    filtered_fastq_dict_6 = {}
    for key_6 in fastq_dict_6:
        print(79, 132)
        quality_6 = fastq_dict_6[key_6][0]
        if min_quality_6 > 0:
            if quality_6 < min_quality_6:
                continue
        if max_quality_6 > 0:
            if quality_6 > max_quality_6:
                continue
        print(86, 139)
        filtered_fastq_dict_6[key_6] = fastq_dict_6[key_6]
    print('exit scope 6')
    return filtered_fastq_dict_6
    print('exit scope 6')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 7')
    print(1, 143)
    filtered_fasta_dict_7 = filtered_fasta_dict
    filtered_fastq_dict_7 = filtered_fastq_dict
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
    print(1, 161)
    seq1_8 = seq1
    seq2_8 = seq2
    match_score_8 = match_score
    mismatch_score_8 = mismatch_score
    gap_score_8 = gap_score
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 166)
    scoring_matrix_8 = [[(0) for j_8 in range(len(seq2_8) + 1)] for i_8 in
        range(len(seq1_8) + 1)]
    print(100, 168)
    traceback_matrix_8 = [[None for j_8 in range(len(seq2_8) + 1)] for i_8 in
        range(len(seq1_8) + 1)]
    print(100, 170)
    max_score_8 = 0
    print(100, 171)
    max_pos_8 = None
    for i_8 in range(1, len(seq1_8) + 1):
        for j_8 in range(1, len(seq2_8) + 1):
            print(104, 175)
            letter1_8 = seq1_8[i_8 - 1]
            print(104, 176)
            letter2_8 = seq2_8[j_8 - 1]
            if letter1_8 == letter2_8:
                print(106, 178)
                diagonal_score_8 = scoring_matrix_8[i_8 - 1][j_8 - 1
                    ] + match_score_8
            else:
                print(108, 180)
                diagonal_score_8 = scoring_matrix_8[i_8 - 1][j_8 - 1
                    ] + mismatch_score_8
            print(107, 182)
            up_score_8 = scoring_matrix_8[i_8 - 1][j_8] + gap_score_8
            print(107, 183)
            left_score_8 = scoring_matrix_8[i_8][j_8 - 1] + gap_score_8
            if diagonal_score_8 >= up_score_8:
                if diagonal_score_8 >= left_score_8:
                    print(115, 187)
                    scoring_matrix_8[i_8][j_8] = diagonal_score_8
                    print(115, 188)
                    traceback_matrix_8[i_8][j_8] = 'D'
                else:
                    print(117, 190)
                    scoring_matrix_8[i_8][j_8] = left_score_8
                    print(117, 191)
                    traceback_matrix_8[i_8][j_8] = 'L'
            elif up_score_8 >= left_score_8:
                print(112, 194)
                scoring_matrix_8[i_8][j_8] = up_score_8
                print(112, 195)
                traceback_matrix_8[i_8][j_8] = 'U'
            else:
                print(114, 197)
                scoring_matrix_8[i_8][j_8] = left_score_8
                print(114, 198)
                traceback_matrix_8[i_8][j_8] = 'L'
            if scoring_matrix_8[i_8][j_8] >= max_score_8:
                print(118, 201)
                max_score_8 = scoring_matrix_8[i_8][j_8]
                print(118, 202)
                max_pos_8 = i_8, j_8
    print(103, 204)
    i_8 = max_pos_8
    print(103, 205)
    j_8 = max_pos_8
    print(103, 206)
    aln1_8 = ''
    print(103, 207)
    aln2_8 = ''
    while traceback_matrix_8[i_8][j_8] != None:
        if traceback_matrix_8[i_8][j_8] == 'D':
            print(123, 210)
            aln1_8 += seq1_8[i_8 - 1]
            print(123, 211)
            aln2_8 += seq2_8[j_8 - 1]
            print(123, 212)
            i_8 -= 1
            print(123, 213)
            j_8 -= 1
        elif traceback_matrix_8[i_8][j_8] == 'L':
            print(126, 215)
            aln1_8 += '-'
            print(126, 216)
            aln2_8 += seq2_8[j_8 - 1]
            print(126, 217)
            j_8 -= 1
        elif traceback_matrix_8[i_8][j_8] == 'U':
            print(129, 219)
            aln1_8 += seq1_8[i_8 - 1]
            print(129, 220)
            aln2_8 += '-'
            print(129, 221)
            i_8 -= 1
    print('exit scope 8')
    return aln1_8[::-1], aln2_8[::-1]
    print('exit scope 8')


def main():
    print('enter scope 9')
    print(1, 224)
    print(134, 226)
    fasta_file_9 = (path_0 +
        '/benchmarks/supplemental_files/fasta_example.fasta')
    print(134, 227)
    fastq_file_9 = (path_0 +
        '/benchmarks/supplemental_files/fastq_example.fastq')
    print(134, 228)
    output_file_9 = path_0 + '/benchmarks/supplemental_files/output_file.txt'
    darwin_wga_workflow(fasta_file_9, fastq_file_9, output_file_9)
    print('exit scope 9')


if __name__ == '__main__':
    main()
