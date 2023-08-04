# Python Implementation Denovo Single Gene Nanopore Sequencing by DARWIN-Whole Genome Alignment Workload

path = '/nfs/pool0/pmcewen/codesign/codesign/src'

def read_fasta(fasta_file):
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    fasta_dict = {}
    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta_dict:
                    fasta_dict[active_sequence_name] = []
                continue
            sequence = line
            fasta_dict[active_sequence_name].append(sequence)
    return fasta_dict


def read_fastq(fastq_file):
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    fastq_dict = {}
    with open(fastq_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fastq_dict:
                    fastq_dict[active_sequence_name] = []
                continue
            sequence = line
            fastq_dict[active_sequence_name].append(sequence)
    return fastq_dict


def read_fasta_with_quality(fasta_file, quality_file):
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    fasta_dict = {}
    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta_dict:
                    fasta_dict[active_sequence_name] = []
                continue
            sequence = line
            fasta_dict[active_sequence_name].append(sequence)
    quality_dict = {}
    with open(quality_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in quality_dict:
                    quality_dict[active_sequence_name] = []
                continue
            quality = line
            quality_dict[active_sequence_name].append(quality)
    return fasta_dict, quality_dict


def darwin_wga_workflow(fasta_file, fastq_file, output_file,
                        min_length=0, max_length=0,
                        min_quality=0, max_quality=0,
                        min_length_fraction=0, max_length_fraction=0,):
    """
    DARWIN-Whole Genome Alignment workflow
    """
    # Read fasta file
    fasta_dict = read_fasta(fasta_file)
    # Read fastq file
    fastq_dict = read_fastq(fastq_file)
    # Filter fasta file
    filtered_fasta_dict = filter_fasta(fasta_dict, min_length, max_length,
                                       min_length_fraction, max_length_fraction)
    # Filter fastq file
    filtered_fastq_dict = filter_fastq(fastq_dict, min_quality, max_quality)
    # Write output
    write_output(filtered_fasta_dict, filtered_fastq_dict, output_file)



def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction, max_length_fraction):
    """
    Filters a fasta dictionary by length
    """
    filtered_fasta_dict = {}
    for key in fasta_dict:
        sequence = fasta_dict[key][0]
        length = len(sequence)
        if min_length_fraction > 0:
            if length < min_length_fraction * len(fasta_dict[key][0]):
                continue
        if max_length_fraction > 0:
            if length > max_length_fraction * len(fasta_dict[key][0]):
                continue
        if min_length > 0:
            if length < min_length:
                continue
        if max_length > 0:
            if length > max_length:
                continue
        filtered_fasta_dict[key] = fasta_dict[key]
    return filtered_fasta_dict


def filter_fastq(fastq_dict, min_quality, max_quality):
    """
    Filters a fastq dictionary by quality
    """
    filtered_fastq_dict = {}
    for key in fastq_dict:
        quality = fastq_dict[key][0]
        if min_quality > 0:
            if quality < min_quality:
                continue
        if max_quality > 0:
            if quality > max_quality:
                continue
        filtered_fastq_dict[key] = fastq_dict[key]
    return filtered_fastq_dict


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    """
    Writes output
    """
    with open(output_file, "w") as f:
        for key in filtered_fasta_dict:
            f.write(">" + key + "\n")
            f.write(filtered_fasta_dict[key][0] + "\n")
        for key in filtered_fastq_dict:
            f.write("@" + key + "\n")
            f.write(filtered_fastq_dict[key][0] + "\n")
            f.write("+\n")
            f.write(filtered_fastq_dict[key][1] + "\n")



# Fast implementation of Smith Watterman Matrix for Genome Sequence Aligment using minion

def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    # Initialize scoring matrix
    scoring_matrix = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
    # Initialize traceback matrix
    traceback_matrix = [[None for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
    # Fill scoring matrix
    max_score = 0
    max_pos   = None
    for i in range(1, len(seq1)+1):
        for j in range(1, len(seq2)+1):
            # Calculate match score
            letter1 = seq1[i-1]
            letter2 = seq2[j-1]
            if letter1 == letter2:
                diagonal_score = scoring_matrix[i-1][j-1] + match_score
            else:
                diagonal_score = scoring_matrix[i-1][j-1] + mismatch_score
            # Calculate gap scores
            up_score   = scoring_matrix[i-1][j] + gap_score
            left_score = scoring_matrix[i][j-1] + gap_score
            # Calculate maximum score and determine the correct trace
            if diagonal_score >= up_score:
                if diagonal_score >= left_score:
                    scoring_matrix[i][j] = diagonal_score
                    traceback_matrix[i][j] = "D"
                else:
                    scoring_matrix[i][j] = left_score
                    traceback_matrix[i][j] = "L"
            else:
                if up_score >= left_score:
                    scoring_matrix[i][j] = up_score
                    traceback_matrix[i][j] = "U"
                else:
                    scoring_matrix[i][j] = left_score
                    traceback_matrix[i][j] = "L"
            # Update maximum score
            if scoring_matrix[i][j] >= max_score:
                max_score = scoring_matrix[i][j]
                max_pos   = (i, j)
    # Traceback
    i = max_pos
    j = max_pos
    aln1 = ""
    aln2 = ""
    while traceback_matrix[i][j] != None:
        if traceback_matrix[i][j] == "D":
            aln1 += seq1[i-1]
            aln2 += seq2[j-1]
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == "L":
            aln1 += "-"
            aln2 += seq2[j-1]
            j -= 1
        elif traceback_matrix[i][j] == "U":
            aln1 += seq1[i-1]
            aln2 += "-"
            i -= 1
    return aln1[::-1], aln2[::-1]

def main():
    
    fasta_file = path + "/benchmarks/supplemental_files/fasta_example.fasta"
    fastq_file = path + "/benchmarks/supplemental_files/fastq_large.fastq"
    output_file = path + "/benchmarks/supplemental_files/output_file.txt"
    darwin_wga_workflow(fasta_file, fastq_file, output_file)

if __name__ == "__main__":
    main()