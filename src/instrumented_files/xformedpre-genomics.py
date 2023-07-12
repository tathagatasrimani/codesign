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
            print('enter scope 2')
            print(5, 11)
            line__1 = line__1.strip()
            if not line__1:
                continue
            if line__1.startswith('>'):
                print(9, 15)
                active_sequence_name__2 = line__1[1:]
                if active_sequence_name__2 not in fasta_dict__1:
                    print(11, 17)
                    fasta_dict__1[active_sequence_name__2] = []
                continue
            print(10, 19)
            sequence__2 = line__1
            fasta_dict__1[active_sequence_name__2].append(sequence__2)
            print('exit scope 2')
    print('exit scope 1')
    return fasta_dict__1
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 3')
    print(1, 24)
    fastq_file__3 = fastq_file
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 29)
    fastq_dict__3 = {}
    with open(fastq_file__3, 'r') as f__3:
        for line__3 in f__3:
            print('enter scope 4')
            print(18, 32)
            line__3 = line__3.strip()
            if not line__3:
                continue
            if line__3.startswith('@'):
                print(22, 36)
                active_sequence_name__4 = line__3[1:]
                if active_sequence_name__4 not in fastq_dict__3:
                    print(24, 38)
                    fastq_dict__3[active_sequence_name__4] = []
                continue
            print(23, 40)
            sequence__4 = line__3
            fastq_dict__3[active_sequence_name__4].append(sequence__4)
            print('exit scope 4')
    print('exit scope 3')
    return fastq_dict__3
    print('exit scope 3')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 5')
    print(1, 45)
    fasta_file__5 = fasta_file
    quality_file__5 = quality_file
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 50)
    fasta_dict__5 = {}
    with open(fasta_file__5, 'r') as f__5:
        for line__5 in f__5:
            print('enter scope 6')
            print(31, 53)
            line__5 = line__5.strip()
            if not line__5:
                continue
            if line__5.startswith('>'):
                print(35, 57)
                active_sequence_name__6 = line__5[1:]
                if active_sequence_name__6 not in fasta_dict__5:
                    print(37, 59)
                    fasta_dict__5[active_sequence_name__6] = []
                continue
            print(36, 61)
            sequence__6 = line__5
            fasta_dict__5[active_sequence_name__6].append(sequence__6)
            print('exit scope 6')
    print(32, 63)
    quality_dict__5 = {}
    with open(quality_file__5, 'r') as f__5:
        for line__5 in f__5:
            print('enter scope 7')
            print(40, 66)
            line__5 = line__5.strip()
            if not line__5:
                continue
            if line__5.startswith('>'):
                print(44, 70)
                active_sequence_name__7 = line__5[1:]
                if active_sequence_name__7 not in quality_dict__5:
                    print(46, 72)
                    quality_dict__5[active_sequence_name__7] = []
                continue
            print(45, 74)
            quality__7 = line__5
            quality_dict__5[active_sequence_name__7].append(quality__7)
            print('exit scope 7')
    print('exit scope 5')
    return fasta_dict__5, quality_dict__5
    print('exit scope 5')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 8')
    print(1, 79)
    fasta_file__8 = fasta_file
    fastq_file__8 = fastq_file
    output_file__8 = output_file
    min_length__8 = min_length
    max_length__8 = max_length
    min_quality__8 = min_quality
    max_quality__8 = max_quality
    min_length_fraction__8 = min_length_fraction
    max_length_fraction__8 = max_length_fraction
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 87)
    fasta_dict__8 = read_fasta(fasta_file__8)
    print(51, 89)
    fastq_dict__8 = read_fastq(fastq_file__8)
    print(51, 91)
    filtered_fasta_dict__8 = filter_fasta(fasta_dict__8, min_length__8,
        max_length__8, min_length_fraction__8, max_length_fraction__8)
    print(51, 94)
    filtered_fastq_dict__8 = filter_fastq(fastq_dict__8, min_quality__8,
        max_quality__8)
    write_output(filtered_fasta_dict__8, filtered_fastq_dict__8, output_file__8
        )
    print('exit scope 8')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 9')
    print(1, 100)
    fasta_dict__9 = fasta_dict
    min_length__9 = min_length
    max_length__9 = max_length
    min_length_fraction__9 = min_length_fraction
    max_length_fraction__9 = max_length_fraction
    """
    Filters a fasta dictionary by length
    """
    print(54, 104)
    filtered_fasta_dict__9 = {}
    for key__9 in fasta_dict__9:
        print('enter scope 10')
        print(56, 106)
        sequence__10 = fasta_dict__9[key__9][0]
        print(56, 107)
        length__10 = len(sequence__10)
        if min_length_fraction__9 > 0:
            if length__10 < min_length_fraction__9 * len(fasta_dict__9[
                key__9][0]):
                continue
        if max_length_fraction__9 > 0:
            if length__10 > max_length_fraction__9 * len(fasta_dict__9[
                key__9][0]):
                continue
        if min_length__9 > 0:
            if length__10 < min_length__9:
                continue
        if max_length__9 > 0:
            if length__10 > max_length__9:
                continue
        print(71, 120)
        filtered_fasta_dict__9[key__9] = fasta_dict__9[key__9]
        print('exit scope 10')
    print('exit scope 9')
    return filtered_fasta_dict__9
    print('exit scope 9')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 11')
    print(1, 124)
    fastq_dict__11 = fastq_dict
    min_quality__11 = min_quality
    max_quality__11 = max_quality
    """
    Filters a fastq dictionary by quality
    """
    print(77, 128)
    filtered_fastq_dict__11 = {}
    for key__11 in fastq_dict__11:
        print('enter scope 12')
        print(79, 130)
        quality__12 = fastq_dict__11[key__11][0]
        if min_quality__11 > 0:
            if quality__12 < min_quality__11:
                continue
        if max_quality__11 > 0:
            if quality__12 > max_quality__11:
                continue
        print(86, 137)
        filtered_fastq_dict__11[key__11] = fastq_dict__11[key__11]
        print('exit scope 12')
    print('exit scope 11')
    return filtered_fastq_dict__11
    print('exit scope 11')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 13')
    print(1, 141)
    filtered_fasta_dict__13 = filtered_fasta_dict
    filtered_fastq_dict__13 = filtered_fastq_dict
    output_file__13 = output_file
    """
    Writes output
    """
    with open(output_file__13, 'w') as f__13:
        for key__13 in filtered_fasta_dict__13:
            print('enter scope 14')
            f__13.write('>' + key__13 + '\n')
            f__13.write(filtered_fasta_dict__13[key__13][0] + '\n')
            print('exit scope 14')
        for key__13 in filtered_fastq_dict__13:
            print('enter scope 15')
            f__13.write('@' + key__13 + '\n')
            f__13.write(filtered_fastq_dict__13[key__13][0] + '\n')
            f__13.write('+\n')
            f__13.write(filtered_fastq_dict__13[key__13][1] + '\n')
            print('exit scope 15')
    print('exit scope 13')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print('enter scope 16')
    print(1, 159)
    seq1__16 = seq1
    seq2__16 = seq2
    match_score__16 = match_score
    mismatch_score__16 = mismatch_score
    gap_score__16 = gap_score
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 164)
    scoring_matrix__16 = [[(0) for j__16 in range(len(seq2__16) + 1)] for
        i__16 in range(len(seq1__16) + 1)]
    print(100, 166)
    traceback_matrix__16 = [[None for j__16 in range(len(seq2__16) + 1)] for
        i__16 in range(len(seq1__16) + 1)]
    print(100, 168)
    max_score__16 = 0
    print(100, 169)
    max_pos__16 = None
    for i__16 in range(1, len(seq1__16) + 1):
        print('enter scope 17')
        for j__16 in range(1, len(seq2__16) + 1):
            print('enter scope 18')
            print(104, 173)
            letter1__18 = seq1__16[i__16 - 1]
            print(104, 174)
            letter2__18 = seq2__16[j__16 - 1]
            if letter1__18 == letter2__18:
                print(106, 176)
                diagonal_score__18 = scoring_matrix__16[i__16 - 1][j__16 - 1
                    ] + match_score__16
            else:
                print(108, 178)
                diagonal_score__18 = scoring_matrix__16[i__16 - 1][j__16 - 1
                    ] + mismatch_score__16
            print(107, 180)
            up_score__18 = scoring_matrix__16[i__16 - 1][j__16] + gap_score__16
            print(107, 181)
            left_score__18 = scoring_matrix__16[i__16][j__16 - 1
                ] + gap_score__16
            if diagonal_score__18 >= up_score__18:
                if diagonal_score__18 >= left_score__18:
                    print(115, 185)
                    scoring_matrix__16[i__16][j__16] = diagonal_score__18
                    print(115, 186)
                    traceback_matrix__16[i__16][j__16] = 'D'
                else:
                    print(117, 188)
                    scoring_matrix__16[i__16][j__16] = left_score__18
                    print(117, 189)
                    traceback_matrix__16[i__16][j__16] = 'L'
            elif up_score__18 >= left_score__18:
                print(112, 192)
                scoring_matrix__16[i__16][j__16] = up_score__18
                print(112, 193)
                traceback_matrix__16[i__16][j__16] = 'U'
            else:
                print(114, 195)
                scoring_matrix__16[i__16][j__16] = left_score__18
                print(114, 196)
                traceback_matrix__16[i__16][j__16] = 'L'
            if scoring_matrix__16[i__16][j__16] >= max_score__16:
                print(118, 199)
                max_score__16 = scoring_matrix__16[i__16][j__16]
                print(118, 200)
                max_pos__16 = i__16, j__16
            print('exit scope 18')
        print('exit scope 17')
    print(103, 202)
    i__16 = max_pos__16
    print(103, 203)
    j__16 = max_pos__16
    print(103, 204)
    aln1__16 = ''
    print(103, 205)
    aln2__16 = ''
    while traceback_matrix__16[i__16][j__16] != None:
        if traceback_matrix__16[i__16][j__16] == 'D':
            print(123, 208)
            aln1__16 += seq1__16[i__16 - 1]
            print(123, 209)
            aln2__16 += seq2__16[j__16 - 1]
            print(123, 210)
            i__16 -= 1
            print(123, 211)
            j__16 -= 1
        elif traceback_matrix__16[i__16][j__16] == 'L':
            print(126, 213)
            aln1__16 += '-'
            print(126, 214)
            aln2__16 += seq2__16[j__16 - 1]
            print(126, 215)
            j__16 -= 1
        elif traceback_matrix__16[i__16][j__16] == 'U':
            print(129, 217)
            aln1__16 += seq1__16[i__16 - 1]
            print(129, 218)
            aln2__16 += '-'
            print(129, 219)
            i__16 -= 1
    print('exit scope 16')
    return aln1__16[::-1], aln2__16[::-1]
    print('exit scope 16')


def main():
    print('enter scope 19')
    print(1, 222)
    print(134, 223)
    fasta_file__19 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fasta_example.fasta'
        )
    print(134, 224)
    fastq_file__19 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fastq_example.fastq'
        )
    print(134, 225)
    output_file__19 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/output_file.txt'
        )
    darwin_wga_workflow(fasta_file__19, fastq_file__19, output_file__19)
    print('exit scope 19')


if __name__ == '__main__':
    main()
