import sys
def read_fasta(fasta_file):
    print(1, 3)
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 8)
    fasta_dict = {}
    with open(fasta_file, 'r') as f:
        print(4, 10)
        for line in f:
            print(4, 10)
            print(5, 11)
            line = line.strip()
            if not line:
                print(5, 12)
                continue
            else:
                print(5, 12)
            if line.startswith('>'):
                print(8, 14)
                print(9, 15)
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta_dict:
                    print(9, 16)
                    print(11, 17)
                    fasta_dict[active_sequence_name] = []
                else:
                    print(9, 16)
                continue
            else:
                print(8, 14)
            print(10, 19)
            sequence = line
            fasta_dict[active_sequence_name].append(sequence)
    return fasta_dict


def read_fastq(fastq_file):
    print(1, 24)
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 29)
    fastq_dict = {}
    with open(fastq_file, 'r') as f:
        print(17, 31)
        for line in f:
            print(17, 31)
            print(18, 32)
            line = line.strip()
            if not line:
                print(18, 33)
                continue
            else:
                print(18, 33)
            if line.startswith('@'):
                print(21, 35)
                print(22, 36)
                active_sequence_name = line[1:]
                if active_sequence_name not in fastq_dict:
                    print(22, 37)
                    print(24, 38)
                    fastq_dict[active_sequence_name] = []
                else:
                    print(22, 37)
                continue
            else:
                print(21, 35)
            print(23, 40)
            sequence = line
            fastq_dict[active_sequence_name].append(sequence)
    return fastq_dict


def read_fasta_with_quality(fasta_file, quality_file):
    print(1, 45)
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 50)
    fasta_dict = {}
    with open(fasta_file, 'r') as f:
        print(30, 52)
        for line in f:
            print(30, 52)
            print(31, 53)
            line = line.strip()
            if not line:
                print(31, 54)
                continue
            else:
                print(31, 54)
            if line.startswith('>'):
                print(34, 56)
                print(35, 57)
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta_dict:
                    print(35, 58)
                    print(37, 59)
                    fasta_dict[active_sequence_name] = []
                else:
                    print(35, 58)
                continue
            else:
                print(34, 56)
            print(36, 61)
            sequence = line
            fasta_dict[active_sequence_name].append(sequence)
    print(32, 63)
    quality_dict = {}
    with open(quality_file, 'r') as f:
        print(39, 65)
        for line in f:
            print(39, 65)
            print(40, 66)
            line = line.strip()
            if not line:
                print(40, 67)
                continue
            else:
                print(40, 67)
            if line.startswith('>'):
                print(43, 69)
                print(44, 70)
                active_sequence_name = line[1:]
                if active_sequence_name not in quality_dict:
                    print(44, 71)
                    print(46, 72)
                    quality_dict[active_sequence_name] = []
                else:
                    print(44, 71)
                continue
            else:
                print(43, 69)
            print(45, 74)
            quality = line
            quality_dict[active_sequence_name].append(quality)
    return fasta_dict, quality_dict


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print(1, 79)
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 87)
    fasta_dict = read_fasta(fasta_file)
    print(51, 89)
    fastq_dict = read_fastq(fastq_file)
    print(51, 91)
    filtered_fasta_dict = filter_fasta(fasta_dict, min_length, max_length,
        min_length_fraction, max_length_fraction)
    print(51, 94)
    filtered_fastq_dict = filter_fastq(fastq_dict, min_quality, max_quality)
    write_output(filtered_fasta_dict, filtered_fastq_dict, output_file)


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print(1, 100)
    """
    Filters a fasta dictionary by length
    """
    print(54, 104)
    filtered_fasta_dict = {}
    print(55, 105)
    for key in fasta_dict:
        print(55, 105)
        print(56, 106)
        sequence = fasta_dict[key][0]
        print(56, 107)
        length = len(sequence)
        if min_length_fraction > 0:
            print(56, 108)
            if length < min_length_fraction * len(fasta_dict[key][0]):
                print(58, 109)
                continue
            else:
                print(58, 109)
        else:
            print(56, 108)
        if max_length_fraction > 0:
            print(59, 111)
            if length > max_length_fraction * len(fasta_dict[key][0]):
                print(62, 112)
                continue
            else:
                print(62, 112)
        else:
            print(59, 111)
        if min_length > 0:
            print(63, 114)
            if length < min_length:
                print(66, 115)
                continue
            else:
                print(66, 115)
        else:
            print(63, 114)
        if max_length > 0:
            print(67, 117)
            if length > max_length:
                print(70, 118)
                continue
            else:
                print(70, 118)
        else:
            print(67, 117)
        print(71, 120)
        filtered_fasta_dict[key] = fasta_dict[key]
    return filtered_fasta_dict


def filter_fastq(fastq_dict, min_quality, max_quality):
    print(1, 124)
    """
    Filters a fastq dictionary by quality
    """
    print(77, 128)
    filtered_fastq_dict = {}
    print(78, 129)
    for key in fastq_dict:
        print(78, 129)
        print(79, 130)
        quality = fastq_dict[key][0]
        if min_quality > 0:
            print(79, 131)
            if quality < min_quality:
                print(81, 132)
                continue
            else:
                print(81, 132)
        else:
            print(79, 131)
        if max_quality > 0:
            print(82, 134)
            if quality > max_quality:
                print(85, 135)
                continue
            else:
                print(85, 135)
        else:
            print(82, 134)
        print(86, 137)
        filtered_fastq_dict[key] = fastq_dict[key]
    return filtered_fastq_dict


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print(1, 141)
    """
    Writes output
    """
    with open(output_file, 'w') as f:
        print(93, 146)
        for key in filtered_fasta_dict:
            print(93, 146)
            f.write('>' + key + '\n')
            f.write(filtered_fasta_dict[key][0] + '\n')
        print(95, 149)
        for key in filtered_fastq_dict:
            print(95, 149)
            f.write('@' + key + '\n')
            f.write(filtered_fastq_dict[key][0] + '\n')
            f.write('+\n')
            f.write(filtered_fastq_dict[key][1] + '\n')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print(1, 159)
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 164)
    scoring_matrix = [[(0) for j in range(len(seq2) + 1)] for i in range(
        len(seq1) + 1)]
    print(100, 166)
    traceback_matrix = [[None for j in range(len(seq2) + 1)] for i in range
        (len(seq1) + 1)]
    print(100, 168)
    max_score = 0
    print(100, 169)
    max_pos = None
    print(101, 170)
    for i in range(1, len(seq1) + 1):
        print(101, 170)
        print(102, 171)
        for j in range(1, len(seq2) + 1):
            print(102, 171)
            print(104, 173)
            letter1 = seq1[i - 1]
            print(104, 174)
            letter2 = seq2[j - 1]
            if letter1 == letter2:
                print(104, 175)
                print(106, 176)
                diagonal_score = scoring_matrix[i - 1][j - 1] + match_score
            else:
                print(104, 175)
                print(108, 178)
                diagonal_score = scoring_matrix[i - 1][j - 1] + mismatch_score
            print(107, 180)
            up_score = scoring_matrix[i - 1][j] + gap_score
            print(107, 181)
            left_score = scoring_matrix[i][j - 1] + gap_score
            if diagonal_score >= up_score:
                print(107, 183)
                if diagonal_score >= left_score:
                    print(109, 184)
                    print(115, 185)
                    scoring_matrix[i][j] = diagonal_score
                    print(115, 186)
                    traceback_matrix[i][j] = 'D'
                else:
                    print(109, 184)
                    print(117, 188)
                    scoring_matrix[i][j] = left_score
                    print(117, 189)
                    traceback_matrix[i][j] = 'L'
            else:
                print(107, 183)
                if up_score >= left_score:
                    print(111, 191)
                    print(112, 192)
                    scoring_matrix[i][j] = up_score
                    print(112, 193)
                    traceback_matrix[i][j] = 'U'
                else:
                    print(111, 191)
                    print(114, 195)
                    scoring_matrix[i][j] = left_score
                    print(114, 196)
                    traceback_matrix[i][j] = 'L'
            if scoring_matrix[i][j] >= max_score:
                print(110, 198)
                print(118, 199)
                max_score = scoring_matrix[i][j]
                print(118, 200)
                max_pos = i, j
            else:
                print(110, 198)
    print(103, 202)
    i = max_pos
    print(103, 203)
    j = max_pos
    print(103, 204)
    aln1 = ''
    print(103, 205)
    aln2 = ''
    while traceback_matrix[i][j] != None:
        if traceback_matrix[i][j] == 'D':
            print(121, 207)
            print(123, 208)
            aln1 += seq1[i - 1]
            print(123, 209)
            aln2 += seq2[j - 1]
            print(123, 210)
            i -= 1
            print(123, 211)
            j -= 1
        else:
            print(121, 207)
            if traceback_matrix[i][j] == 'L':
                print(125, 212)
                print(126, 213)
                aln1 += '-'
                print(126, 214)
                aln2 += seq2[j - 1]
                print(126, 215)
                j -= 1
            else:
                print(125, 212)
                if traceback_matrix[i][j] == 'U':
                    print(128, 216)
                    print(129, 217)
                    aln1 += seq1[i - 1]
                    print(129, 218)
                    aln2 += '-'
                    print(129, 219)
                    i -= 1
                else:
                    print(128, 216)
    return aln1[::-1], aln2[::-1]


def main():
    print(1, 222)
    print(134, 223)
    fasta_file = (
        '/home/ubuntu/codesign/src/cfg/benchmarks/supplemental_files/fasta_example.fasta'
        )
    print(134, 224)
    fastq_file = (
        '/home/ubuntu/codesign/src/cfg/benchmarks/supplemental_files/fastq_example.fastq'
        )
    print(134, 225)
    output_file = (
        '/home/ubuntu/codesign/src/cfg/benchmarks/supplemental_files/output_file.txt'
        )
    darwin_wga_workflow(fasta_file, fastq_file, output_file)


if __name__ == '__main__':
    print(1, 228)
    main()
else:
    print(1, 228)
