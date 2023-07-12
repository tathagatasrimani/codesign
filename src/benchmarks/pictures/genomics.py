digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="def read_fasta(fasta_file):...
def read_fastq(fastq_file):...
def read_fasta_with_quality(fasta_file, quality_file):...
def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,...
def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,...
def filter_fastq(fastq_dict, min_quality, max_quality):...
def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):...
def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):...
def main():...
if __name__ == '__main__':
"]
	136 [label="main()
"]
	"136_calls" [label=main shape=box]
	136 -> "136_calls" [label=calls style=dashed]
	1 -> 136 [label="__name__ == '__main__'"]
	subgraph clusterread_fasta {
		graph [label=read_fasta]
		3 [label="\"\"\"
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    \"\"\"
fasta_dict = {}
"]
		"3_calls" [label=open shape=box]
		3 -> "3_calls" [label=calls style=dashed]
		4 [label="for line in f:
"]
		5 [label="line = line.strip()
if not line:
"]
		"5_calls" [label="line.strip" shape=box]
		5 -> "5_calls" [label=calls style=dashed]
		8 [label="if line.startswith('>'):
"]
		9 [label="active_sequence_name = line[1:]
if active_sequence_name not in fasta_dict:
"]
		11 [label="fasta_dict[active_sequence_name] = []
"]
		11 -> 4 [label=""]
		9 -> 11 [label="active_sequence_name not in fasta_dict"]
		9 -> 4 [label="(active_sequence_name in fasta_dict)"]
		8 -> 9 [label="line.startswith('>')"]
		10 [label="sequence = line
fasta_dict[active_sequence_name].append(sequence)
"]
		"10_calls" [label="fasta_dict.append" shape=box]
		10 -> "10_calls" [label=calls style=dashed]
		10 -> 4 [label=""]
		8 -> 10 [label="(not line.startswith('>'))"]
		5 -> 8 [label="(not not line)"]
		5 -> 4 [label="not line"]
		4 -> 5 [label=f]
		6 [label="return fasta_dict
"]
		4 -> 6 [label=""]
		3 -> 4 [label=""]
	}
	subgraph clusterread_fastq {
		graph [label=read_fastq]
		16 [label="\"\"\"
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    \"\"\"
fastq_dict = {}
"]
		"16_calls" [label=open shape=box]
		16 -> "16_calls" [label=calls style=dashed]
		17 [label="for line in f:
"]
		18 [label="line = line.strip()
if not line:
"]
		"18_calls" [label="line.strip" shape=box]
		18 -> "18_calls" [label=calls style=dashed]
		21 [label="if line.startswith('@'):
"]
		22 [label="active_sequence_name = line[1:]
if active_sequence_name not in fastq_dict:
"]
		24 [label="fastq_dict[active_sequence_name] = []
"]
		24 -> 17 [label=""]
		22 -> 24 [label="active_sequence_name not in fastq_dict"]
		22 -> 17 [label="(active_sequence_name in fastq_dict)"]
		21 -> 22 [label="line.startswith('@')"]
		23 [label="sequence = line
fastq_dict[active_sequence_name].append(sequence)
"]
		"23_calls" [label="fastq_dict.append" shape=box]
		23 -> "23_calls" [label=calls style=dashed]
		23 -> 17 [label=""]
		21 -> 23 [label="(not line.startswith('@'))"]
		18 -> 21 [label="(not not line)"]
		18 -> 17 [label="not line"]
		17 -> 18 [label=f]
		19 [label="return fastq_dict
"]
		17 -> 19 [label=""]
		16 -> 17 [label=""]
	}
	subgraph clusterread_fasta_with_quality {
		graph [label=read_fasta_with_quality]
		29 [label="\"\"\"
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    \"\"\"
fasta_dict = {}
"]
		"29_calls" [label=open shape=box]
		29 -> "29_calls" [label=calls style=dashed]
		30 [label="for line in f:
"]
		31 [label="line = line.strip()
if not line:
"]
		"31_calls" [label="line.strip" shape=box]
		31 -> "31_calls" [label=calls style=dashed]
		34 [label="if line.startswith('>'):
"]
		35 [label="active_sequence_name = line[1:]
if active_sequence_name not in fasta_dict:
"]
		37 [label="fasta_dict[active_sequence_name] = []
"]
		37 -> 30 [label=""]
		35 -> 37 [label="active_sequence_name not in fasta_dict"]
		35 -> 30 [label="(active_sequence_name in fasta_dict)"]
		34 -> 35 [label="line.startswith('>')"]
		36 [label="sequence = line
fasta_dict[active_sequence_name].append(sequence)
"]
		"36_calls" [label="fasta_dict.append" shape=box]
		36 -> "36_calls" [label=calls style=dashed]
		36 -> 30 [label=""]
		34 -> 36 [label="(not line.startswith('>'))"]
		31 -> 34 [label="(not not line)"]
		31 -> 30 [label="not line"]
		30 -> 31 [label=f]
		32 [label="quality_dict = {}
"]
		"32_calls" [label=open shape=box]
		32 -> "32_calls" [label=calls style=dashed]
		39 [label="for line in f:
"]
		40 [label="line = line.strip()
if not line:
"]
		"40_calls" [label="line.strip" shape=box]
		40 -> "40_calls" [label=calls style=dashed]
		43 [label="if line.startswith('>'):
"]
		44 [label="active_sequence_name = line[1:]
if active_sequence_name not in quality_dict:
"]
		46 [label="quality_dict[active_sequence_name] = []
"]
		46 -> 39 [label=""]
		44 -> 46 [label="active_sequence_name not in quality_dict"]
		44 -> 39 [label="(active_sequence_name in quality_dict)"]
		43 -> 44 [label="line.startswith('>')"]
		45 [label="quality = line
quality_dict[active_sequence_name].append(quality)
"]
		"45_calls" [label="quality_dict.append" shape=box]
		45 -> "45_calls" [label=calls style=dashed]
		45 -> 39 [label=""]
		43 -> 45 [label="(not line.startswith('>'))"]
		40 -> 43 [label="(not not line)"]
		40 -> 39 [label="not line"]
		39 -> 40 [label=f]
		41 [label="return fasta_dict, quality_dict
"]
		39 -> 41 [label=""]
		32 -> 39 [label=""]
		30 -> 32 [label=""]
		29 -> 30 [label=""]
	}
	subgraph clusterdarwin_wga_workflow {
		graph [label=darwin_wga_workflow]
		51 [label="\"\"\"
    DARWIN-Whole Genome Alignment workflow
    \"\"\"
fasta_dict = read_fasta(fasta_file)
fastq_dict = read_fastq(fastq_file)
filtered_fasta_dict = filter_fasta(fasta_dict, min_length, max_length,
    min_length_fraction, max_length_fraction)
filtered_fastq_dict = filter_fastq(fastq_dict, min_quality, max_quality)
write_output(filtered_fasta_dict, filtered_fastq_dict, output_file)
"]
		"51_calls" [label="read_fasta
read_fastq
filter_fasta
filter_fastq
write_output" shape=box]
		51 -> "51_calls" [label=calls style=dashed]
	}
	subgraph clusterfilter_fasta {
		graph [label=filter_fasta]
		54 [label="\"\"\"
    Filters a fasta dictionary by length
    \"\"\"
filtered_fasta_dict = {}
"]
		55 [label="for key in fasta_dict:
"]
		56 [label="sequence = fasta_dict[key][0]
length = len(sequence)
if min_length_fraction > 0:
"]
		"56_calls" [label=len shape=box]
		56 -> "56_calls" [label=calls style=dashed]
		58 [label="if length < min_length_fraction * len(fasta_dict[key][0]):
"]
		58 -> 55 [label="length < min_length_fraction * len(fasta_dict[key][0])"]
		59 [label="if max_length_fraction > 0:
"]
		62 [label="if length > max_length_fraction * len(fasta_dict[key][0]):
"]
		62 -> 55 [label="length > max_length_fraction * len(fasta_dict[key][0])"]
		63 [label="if min_length > 0:
"]
		66 [label="if length < min_length:
"]
		66 -> 55 [label="length < min_length"]
		67 [label="if max_length > 0:
"]
		70 [label="if length > max_length:
"]
		70 -> 55 [label="length > max_length"]
		71 [label="filtered_fasta_dict[key] = fasta_dict[key]
"]
		71 -> 55 [label=""]
		70 -> 71 [label="(length <= max_length)"]
		67 -> 70 [label="max_length > 0"]
		67 -> 71 [label="(max_length <= 0)"]
		66 -> 67 [label="(length >= min_length)"]
		63 -> 66 [label="min_length > 0"]
		63 -> 67 [label="(min_length <= 0)"]
		62 -> 63 [label="(length <= max_length_fraction * len(fasta_dict[key][0]))"]
		59 -> 62 [label="max_length_fraction > 0"]
		59 -> 63 [label="(max_length_fraction <= 0)"]
		58 -> 59 [label="(length >= min_length_fraction * len(fasta_dict[key][0]))"]
		56 -> 58 [label="min_length_fraction > 0"]
		56 -> 59 [label="(min_length_fraction <= 0)"]
		55 -> 56 [label=fasta_dict]
		57 [label="return filtered_fasta_dict
"]
		55 -> 57 [label=""]
		54 -> 55 [label=""]
	}
	subgraph clusterfilter_fastq {
		graph [label=filter_fastq]
		77 [label="\"\"\"
    Filters a fastq dictionary by quality
    \"\"\"
filtered_fastq_dict = {}
"]
		78 [label="for key in fastq_dict:
"]
		79 [label="quality = fastq_dict[key][0]
if min_quality > 0:
"]
		81 [label="if quality < min_quality:
"]
		81 -> 78 [label="quality < min_quality"]
		82 [label="if max_quality > 0:
"]
		85 [label="if quality > max_quality:
"]
		85 -> 78 [label="quality > max_quality"]
		86 [label="filtered_fastq_dict[key] = fastq_dict[key]
"]
		86 -> 78 [label=""]
		85 -> 86 [label="(quality <= max_quality)"]
		82 -> 85 [label="max_quality > 0"]
		82 -> 86 [label="(max_quality <= 0)"]
		81 -> 82 [label="(quality >= min_quality)"]
		79 -> 81 [label="min_quality > 0"]
		79 -> 82 [label="(min_quality <= 0)"]
		78 -> 79 [label=fastq_dict]
		80 [label="return filtered_fastq_dict
"]
		78 -> 80 [label=""]
		77 -> 78 [label=""]
	}
	subgraph clusterwrite_output {
		graph [label=write_output]
		92 [label="\"\"\"
    Writes output
    \"\"\"
"]
		"92_calls" [label=open shape=box]
		92 -> "92_calls" [label=calls style=dashed]
		93 [label="for key in filtered_fasta_dict:
"]
		94 [label="f.write('>' + key + '\n')
f.write(filtered_fasta_dict[key][0] + '\n')
"]
		"94_calls" [label="f.write
f.write" shape=box]
		94 -> "94_calls" [label=calls style=dashed]
		94 -> 93 [label=""]
		93 -> 94 [label=filtered_fasta_dict]
		95 [label="for key in filtered_fastq_dict:
"]
		96 [label="f.write('@' + key + '\n')
f.write(filtered_fastq_dict[key][0] + '\n')
f.write('+\n')
f.write(filtered_fastq_dict[key][1] + '\n')
"]
		"96_calls" [label="f.write
f.write
f.write
f.write" shape=box]
		96 -> "96_calls" [label=calls style=dashed]
		96 -> 95 [label=""]
		95 -> 96 [label=filtered_fastq_dict]
		93 -> 95 [label=""]
		92 -> 93 [label=""]
	}
	subgraph clustersmith_waterman {
		graph [label=smith_waterman]
		100 [label="\"\"\"
    Computes the local alignment between two sequences using a scoring matrix.
    \"\"\"
scoring_matrix = [[(0) for j in range(len(seq2) + 1)] for i in range(len(
    seq1) + 1)]
traceback_matrix = [[None for j in range(len(seq2) + 1)] for i in range(len
    (seq1) + 1)]
max_score = 0
max_pos = None
"]
		"100_calls" [label="range
range
range
range" shape=box]
		100 -> "100_calls" [label=calls style=dashed]
		101 [label="for i in range(1, len(seq1) + 1):
"]
		102 [label="for j in range(1, len(seq2) + 1):
"]
		104 [label="letter1 = seq1[i - 1]
letter2 = seq2[j - 1]
if letter1 == letter2:
"]
		106 [label="diagonal_score = scoring_matrix[i - 1][j - 1] + match_score
"]
		107 [label="up_score = scoring_matrix[i - 1][j] + gap_score
left_score = scoring_matrix[i][j - 1] + gap_score
if diagonal_score >= up_score:
"]
		109 [label="if diagonal_score >= left_score:
"]
		115 [label="scoring_matrix[i][j] = diagonal_score
traceback_matrix[i][j] = 'D'
"]
		110 [label="if scoring_matrix[i][j] >= max_score:
"]
		118 [label="max_score = scoring_matrix[i][j]
max_pos = i, j
"]
		118 -> 102 [label=""]
		110 -> 118 [label="scoring_matrix[i][j] >= max_score"]
		110 -> 102 [label="(scoring_matrix[i][j] < max_score)"]
		115 -> 110 [label=""]
		109 -> 115 [label="diagonal_score >= left_score"]
		117 [label="scoring_matrix[i][j] = left_score
traceback_matrix[i][j] = 'L'
"]
		117 -> 110 [label=""]
		109 -> 117 [label="(diagonal_score < left_score)"]
		107 -> 109 [label="diagonal_score >= up_score"]
		111 [label="if up_score >= left_score:
"]
		112 [label="scoring_matrix[i][j] = up_score
traceback_matrix[i][j] = 'U'
"]
		112 -> 110 [label=""]
		111 -> 112 [label="up_score >= left_score"]
		114 [label="scoring_matrix[i][j] = left_score
traceback_matrix[i][j] = 'L'
"]
		114 -> 110 [label=""]
		111 -> 114 [label="(up_score < left_score)"]
		107 -> 111 [label="(diagonal_score < up_score)"]
		106 -> 107 [label=""]
		104 -> 106 [label="letter1 == letter2"]
		108 [label="diagonal_score = scoring_matrix[i - 1][j - 1] + mismatch_score
"]
		108 -> 107 [label=""]
		104 -> 108 [label="(letter1 != letter2)"]
		102 -> 104 [label="range(1, len(seq2) + 1)"]
		102 -> 101 [label=""]
		101 -> 102 [label="range(1, len(seq1) + 1)"]
		103 [label="i = max_pos
j = max_pos
aln1 = ''
aln2 = ''
"]
		120 [label="while traceback_matrix[i][j] != None:
"]
		121 [label="if traceback_matrix[i][j] == 'D':
"]
		123 [label="aln1 += seq1[i - 1]
aln2 += seq2[j - 1]
i -= 1
j -= 1
"]
		123 -> 120 [label=""]
		121 -> 123 [label="traceback_matrix[i][j] == 'D'"]
		125 [label="if traceback_matrix[i][j] == 'L':
"]
		126 [label="aln1 += '-'
aln2 += seq2[j - 1]
j -= 1
"]
		126 -> 120 [label=""]
		125 -> 126 [label="traceback_matrix[i][j] == 'L'"]
		128 [label="if traceback_matrix[i][j] == 'U':
"]
		129 [label="aln1 += seq1[i - 1]
aln2 += '-'
i -= 1
"]
		129 -> 120 [label=""]
		128 -> 129 [label="traceback_matrix[i][j] == 'U'"]
		128 -> 120 [label="(traceback_matrix[i][j] != 'U')"]
		125 -> 128 [label="(traceback_matrix[i][j] != 'L')"]
		121 -> 125 [label="(traceback_matrix[i][j] != 'D')"]
		120 -> 121 [label="traceback_matrix[i][j] != None"]
		122 [label="return aln1[::-1], aln2[::-1]
"]
		120 -> 122 [label="(traceback_matrix[i][j] == None)"]
		103 -> 120 [label=""]
		101 -> 103 [label=""]
		100 -> 101 [label=""]
	}
	subgraph clustermain {
		graph [label=main]
		134 [label="fasta_file = (
    '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fasta_example.fasta'
    )
fastq_file = (
    '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fastq_example.fastq'
    )
output_file = (
    '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/output_file.txt'
    )
darwin_wga_workflow(fasta_file, fastq_file, output_file)
"]
		"134_calls" [label=darwin_wga_workflow shape=box]
		134 -> "134_calls" [label=calls style=dashed]
	}
}
