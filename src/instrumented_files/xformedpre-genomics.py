import sys
from instrument_lib import *
def read_fasta(fasta_file):
    print('enter scope 1')
    print(1, 3)
    fasta_file__1 = fasta_file
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 8)
    fasta_dict__1 = {}
    with open(fasta_file__1, 'r') as f__1:
        for line__1 in f__1:
            print(5, 11)
            line__1 = line__1.strip()
            if not line__1:
                continue
            if line__1.startswith('>'):
                print(9, 15)
                active_sequence_name__1 = line__1[1:]
                if active_sequence_name__1 not in fasta_dict__1:
                    print(11, 17)
                    fasta_dict__1[active_sequence_name__1] = []
                continue
            print(10, 19)
            sequence__1 = line__1
            fasta_dict__1[active_sequence_name__1].append(sequence__1)
    print('exit scope 1')
    return fasta_dict__1
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 2')
    print(1, 24)
    fastq_file__2 = fastq_file
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 29)
    fastq_dict__2 = {}
    with open(fastq_file__2, 'r') as f__2:
        for line__2 in f__2:
            print(18, 32)
            line__2 = line__2.strip()
            if not line__2:
                continue
            if line__2.startswith('@'):
                print(22, 36)
                active_sequence_name__2 = line__2[1:]
                if active_sequence_name__2 not in fastq_dict__2:
                    print(24, 38)
                    fastq_dict__2[active_sequence_name__2] = []
                continue
            print(23, 40)
            sequence__2 = line__2
            fastq_dict__2[active_sequence_name__2].append(sequence__2)
    print('exit scope 2')
    return fastq_dict__2
    print('exit scope 2')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 3')
    print(1, 45)
    fasta_file__3 = fasta_file
    quality_file__3 = quality_file
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 50)
    fasta_dict__3 = {}
    with open(fasta_file__3, 'r') as f__3:
        for line__3 in f__3:
            print(31, 53)
            line__3 = line__3.strip()
            if not line__3:
                continue
            if line__3.startswith('>'):
                print(35, 57)
                active_sequence_name__3 = line__3[1:]
                if active_sequence_name__3 not in fasta_dict__3:
                    print(37, 59)
                    fasta_dict__3[active_sequence_name__3] = []
                continue
            print(36, 61)
            sequence__3 = line__3
            fasta_dict__3[active_sequence_name__3].append(sequence__3)
    print(32, 63)
    quality_dict__3 = {}
    with open(quality_file__3, 'r') as f__3:
        for line__3 in f__3:
            print(40, 66)
            line__3 = line__3.strip()
            if not line__3:
                continue
            if line__3.startswith('>'):
                print(44, 70)
                active_sequence_name__3 = line__3[1:]
                if active_sequence_name__3 not in quality_dict__3:
                    print(46, 72)
                    quality_dict__3[active_sequence_name__3] = []
                continue
            print(45, 74)
            quality__3 = line__3
            quality_dict__3[active_sequence_name__3].append(quality__3)
    print('exit scope 3')
    return fasta_dict__3, quality_dict__3
    print('exit scope 3')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 4')
    print(1, 79)
    fasta_file__4 = fasta_file
    fastq_file__4 = fastq_file
    output_file__4 = output_file
    min_length__4 = min_length
    max_length__4 = max_length
    min_quality__4 = min_quality
    max_quality__4 = max_quality
    min_length_fraction__4 = min_length_fraction
    max_length_fraction__4 = max_length_fraction
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 87)
    fasta_dict__4 = read_fasta(fasta_file__4)
    print(51, 89)
    fastq_dict__4 = read_fastq(fastq_file__4)
    print(51, 91)
    filtered_fasta_dict__4 = filter_fasta(fasta_dict__4, min_length__4,
        max_length__4, min_length_fraction__4, max_length_fraction__4)
    print(51, 94)
    filtered_fastq_dict__4 = filter_fastq(fastq_dict__4, min_quality__4,
        max_quality__4)
    write_output(filtered_fasta_dict__4, filtered_fastq_dict__4, output_file__4
        )
    print('exit scope 4')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 5')
    print(1, 100)
    fasta_dict__5 = fasta_dict
    min_length__5 = min_length
    max_length__5 = max_length
    min_length_fraction__5 = min_length_fraction
    max_length_fraction__5 = max_length_fraction
    """
    Filters a fasta dictionary by length
    """
    print(54, 104)
    filtered_fasta_dict__5 = {}
    for key__5 in fasta_dict__5:
        print(56, 106)
        sequence__5 = fasta_dict__5[key__5][0]
        print(56, 107)
        length__5 = len(sequence__5)
        if min_length_fraction__5 > 0:
            if length__5 < min_length_fraction__5 * len(fasta_dict__5[
                key__5][0]):
                continue
        if max_length_fraction__5 > 0:
            if length__5 > max_length_fraction__5 * len(fasta_dict__5[
                key__5][0]):
                continue
        if min_length__5 > 0:
            if length__5 < min_length__5:
                continue
        if max_length__5 > 0:
            if length__5 > max_length__5:
                continue
        print(71, 120)
        filtered_fasta_dict__5[key__5] = fasta_dict__5[key__5]
    print('exit scope 5')
    return filtered_fasta_dict__5
    print('exit scope 5')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 6')
    print(1, 124)
    fastq_dict__6 = fastq_dict
    min_quality__6 = min_quality
    max_quality__6 = max_quality
    """
    Filters a fastq dictionary by quality
    """
    print(77, 128)
    filtered_fastq_dict__6 = {}
    for key__6 in fastq_dict__6:
        print(79, 130)
        quality__6 = fastq_dict__6[key__6][0]
        if min_quality__6 > 0:
            if quality__6 < min_quality__6:
                continue
        if max_quality__6 > 0:
            if quality__6 > max_quality__6:
                continue
        print(86, 137)
        filtered_fastq_dict__6[key__6] = fastq_dict__6[key__6]
    print('exit scope 6')
    return filtered_fastq_dict__6
    print('exit scope 6')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 7')
    print(1, 141)
    filtered_fasta_dict__7 = filtered_fasta_dict
    filtered_fastq_dict__7 = filtered_fastq_dict
    output_file__7 = output_file
    """
    Writes output
    """
    with open(output_file__7, 'w') as f__7:
        for key__7 in filtered_fasta_dict__7:
            f__7.write('>' + key__7 + '\n')
            f__7.write(filtered_fasta_dict__7[key__7][0] + '\n')
        for key__7 in filtered_fastq_dict__7:
            f__7.write('@' + key__7 + '\n')
            f__7.write(filtered_fastq_dict__7[key__7][0] + '\n')
            f__7.write('+\n')
            f__7.write(filtered_fastq_dict__7[key__7][1] + '\n')
    print('exit scope 7')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print('enter scope 8')
    print(1, 159)
    seq1__8 = seq1
    seq2__8 = seq2
    match_score__8 = match_score
    mismatch_score__8 = mismatch_score
    gap_score__8 = gap_score
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 164)
    scoring_matrix__8 = [[(0) for j__8 in range(len(seq2__8) + 1)] for i__8 in
        range(len(seq1__8) + 1)]
    print(100, 166)
    traceback_matrix__8 = [[None for j__8 in range(len(seq2__8) + 1)] for
        i__8 in range(len(seq1__8) + 1)]
    print(100, 168)
    max_score__8 = 0
    print(100, 169)
    max_pos__8 = None
    for i__8 in range(1, len(seq1__8) + 1):
        for j__8 in range(1, len(seq2__8) + 1):
            print(104, 173)
            letter1__8 = seq1__8[i__8 - 1]
            print(104, 174)
            letter2__8 = seq2__8[j__8 - 1]
            if letter1__8 == letter2__8:
                print(106, 176)
                diagonal_score__8 = scoring_matrix__8[i__8 - 1][j__8 - 1
                    ] + match_score__8
            else:
                print(108, 178)
                diagonal_score__8 = scoring_matrix__8[i__8 - 1][j__8 - 1
                    ] + mismatch_score__8
            print(107, 180)
            up_score__8 = scoring_matrix__8[i__8 - 1][j__8] + gap_score__8
            print(107, 181)
            left_score__8 = scoring_matrix__8[i__8][j__8 - 1] + gap_score__8
            if diagonal_score__8 >= up_score__8:
                if diagonal_score__8 >= left_score__8:
                    print(115, 185)
                    scoring_matrix__8[i__8][j__8] = diagonal_score__8
                    print(115, 186)
                    traceback_matrix__8[i__8][j__8] = 'D'
                else:
                    print(117, 188)
                    scoring_matrix__8[i__8][j__8] = left_score__8
                    print(117, 189)
                    traceback_matrix__8[i__8][j__8] = 'L'
            elif up_score__8 >= left_score__8:
                print(112, 192)
                scoring_matrix__8[i__8][j__8] = up_score__8
                print(112, 193)
                traceback_matrix__8[i__8][j__8] = 'U'
            else:
                print(114, 195)
                scoring_matrix__8[i__8][j__8] = left_score__8
                print(114, 196)
                traceback_matrix__8[i__8][j__8] = 'L'
            if scoring_matrix__8[i__8][j__8] >= max_score__8:
                print(118, 199)
                max_score__8 = scoring_matrix__8[i__8][j__8]
                print(118, 200)
                max_pos__8 = i__8, j__8
    print(103, 202)
    i__8 = max_pos__8
    print(103, 203)
    j__8 = max_pos__8
    print(103, 204)
    aln1__8 = ''
    print(103, 205)
    aln2__8 = ''
    while traceback_matrix__8[i__8][j__8] != None:
        if traceback_matrix__8[i__8][j__8] == 'D':
            print(123, 208)
            aln1__8 += seq1__8[i__8 - 1]
            print(123, 209)
            aln2__8 += seq2__8[j__8 - 1]
            print(123, 210)
            i__8 -= 1
            print(123, 211)
            j__8 -= 1
        elif traceback_matrix__8[i__8][j__8] == 'L':
            print(126, 213)
            aln1__8 += '-'
            print(126, 214)
            aln2__8 += seq2__8[j__8 - 1]
            print(126, 215)
            j__8 -= 1
        elif traceback_matrix__8[i__8][j__8] == 'U':
            print(129, 217)
            aln1__8 += seq1__8[i__8 - 1]
            print(129, 218)
            aln2__8 += '-'
            print(129, 219)
            i__8 -= 1
    print('exit scope 8')
    return aln1__8[::-1], aln2__8[::-1]
    print('exit scope 8')


def main():
    print('enter scope 9')
    print(1, 222)
    print(134, 223)
    fasta_file__9 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fasta_example.fasta'
        )
    print(134, 224)
    fastq_file__9 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fastq_example.fastq'
        )
    print(134, 225)
    output_file__9 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/output_file.txt'
        )
    darwin_wga_workflow(fasta_file__9, fastq_file__9, output_file__9)
    print('exit scope 9')


if __name__ == '__main__':
    main()
