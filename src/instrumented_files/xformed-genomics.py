import sys
from instrument_lib import *
import sys
from instrument_lib import *
print(1, 3)
path_0 = '/nfs/pool0/pmcewen/codesign/codesign/src'
write_instrument_read(path_0, 'path_0')
print('malloc', sys.getsizeof(path_0), 'path_0')


def read_fasta(fasta_file):
    print('enter scope 1')
    print(1, 5)
    fasta_file_1 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file_1, 'fasta_file_1')
    print('malloc', sys.getsizeof(fasta_file_1), 'fasta_file_1')
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 10)
    fasta_dict_1 = {}
    write_instrument_read(fasta_dict_1, 'fasta_dict_1')
    print('malloc', sys.getsizeof(fasta_dict_1), 'fasta_dict_1')
    with open(instrument_read(fasta_file_1, 'fasta_file_1'), 'r') as f_1:
        for line_1 in instrument_read(f_1, 'f_1'):
            print(5, 13)
            line_1 = instrument_read(line_1, 'line_1').strip()
            write_instrument_read(line_1, 'line_1')
            print('malloc', sys.getsizeof(line_1), 'line_1')
            if not instrument_read(line_1, 'line_1'):
                continue
            if instrument_read(line_1, 'line_1').startswith('>'):
                print(9, 17)
                active_sequence_name_1 = instrument_read_sub(instrument_read
                    (line_1, 'line_1'), 'line_1', None, 1, None, True)
                write_instrument_read(active_sequence_name_1,
                    'active_sequence_name_1')
                print('malloc', sys.getsizeof(active_sequence_name_1),
                    'active_sequence_name_1')
                if instrument_read(active_sequence_name_1,
                    'active_sequence_name_1') not in instrument_read(
                    fasta_dict_1, 'fasta_dict_1'):
                    print(11, 19)
                    fasta_dict_1[instrument_read(instrument_read(
                        active_sequence_name_1, 'active_sequence_name_1'),
                        'active_sequence_name_1')] = []
                    write_instrument_read_sub(fasta_dict_1, 'fasta_dict_1',
                        instrument_read(instrument_read(
                        active_sequence_name_1, 'active_sequence_name_1'),
                        'active_sequence_name_1'), None, None, False)
                continue
            print(10, 21)
            sequence_1 = instrument_read(line_1, 'line_1')
            write_instrument_read(sequence_1, 'sequence_1')
            print('malloc', sys.getsizeof(sequence_1), 'sequence_1')
            instrument_read_sub(instrument_read(fasta_dict_1,
                'fasta_dict_1'), 'fasta_dict_1', instrument_read(
                active_sequence_name_1, 'active_sequence_name_1'), None,
                None, False).append(instrument_read(sequence_1, 'sequence_1'))
    print('exit scope 1')
    return instrument_read(fasta_dict_1, 'fasta_dict_1')
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 2')
    print(1, 26)
    fastq_file_2 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file_2, 'fastq_file_2')
    print('malloc', sys.getsizeof(fastq_file_2), 'fastq_file_2')
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 31)
    fastq_dict_2 = {}
    write_instrument_read(fastq_dict_2, 'fastq_dict_2')
    print('malloc', sys.getsizeof(fastq_dict_2), 'fastq_dict_2')
    with open(instrument_read(fastq_file_2, 'fastq_file_2'), 'r') as f_2:
        for line_2 in instrument_read(f_2, 'f_2'):
            print(18, 34)
            line_2 = instrument_read(line_2, 'line_2').strip()
            write_instrument_read(line_2, 'line_2')
            print('malloc', sys.getsizeof(line_2), 'line_2')
            if not instrument_read(line_2, 'line_2'):
                continue
            if instrument_read(line_2, 'line_2').startswith('@'):
                print(22, 38)
                active_sequence_name_2 = instrument_read_sub(instrument_read
                    (line_2, 'line_2'), 'line_2', None, 1, None, True)
                write_instrument_read(active_sequence_name_2,
                    'active_sequence_name_2')
                print('malloc', sys.getsizeof(active_sequence_name_2),
                    'active_sequence_name_2')
                if instrument_read(active_sequence_name_2,
                    'active_sequence_name_2') not in instrument_read(
                    fastq_dict_2, 'fastq_dict_2'):
                    print(24, 40)
                    fastq_dict_2[instrument_read(instrument_read(
                        active_sequence_name_2, 'active_sequence_name_2'),
                        'active_sequence_name_2')] = []
                    write_instrument_read_sub(fastq_dict_2, 'fastq_dict_2',
                        instrument_read(instrument_read(
                        active_sequence_name_2, 'active_sequence_name_2'),
                        'active_sequence_name_2'), None, None, False)
                continue
            print(23, 42)
            sequence_2 = instrument_read(line_2, 'line_2')
            write_instrument_read(sequence_2, 'sequence_2')
            print('malloc', sys.getsizeof(sequence_2), 'sequence_2')
            instrument_read_sub(instrument_read(fastq_dict_2,
                'fastq_dict_2'), 'fastq_dict_2', instrument_read(
                active_sequence_name_2, 'active_sequence_name_2'), None,
                None, False).append(instrument_read(sequence_2, 'sequence_2'))
    print('exit scope 2')
    return instrument_read(fastq_dict_2, 'fastq_dict_2')
    print('exit scope 2')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 3')
    print(1, 47)
    fasta_file_3 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file_3, 'fasta_file_3')
    print('malloc', sys.getsizeof(fasta_file_3), 'fasta_file_3')
    quality_file_3 = instrument_read(quality_file, 'quality_file')
    write_instrument_read(quality_file_3, 'quality_file_3')
    print('malloc', sys.getsizeof(quality_file_3), 'quality_file_3')
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 52)
    fasta_dict_3 = {}
    write_instrument_read(fasta_dict_3, 'fasta_dict_3')
    print('malloc', sys.getsizeof(fasta_dict_3), 'fasta_dict_3')
    with open(instrument_read(fasta_file_3, 'fasta_file_3'), 'r') as f_3:
        for line_3 in instrument_read(f_3, 'f_3'):
            print(31, 55)
            line_3 = instrument_read(line_3, 'line_3').strip()
            write_instrument_read(line_3, 'line_3')
            print('malloc', sys.getsizeof(line_3), 'line_3')
            if not instrument_read(line_3, 'line_3'):
                continue
            if instrument_read(line_3, 'line_3').startswith('>'):
                print(35, 59)
                active_sequence_name_3 = instrument_read_sub(instrument_read
                    (line_3, 'line_3'), 'line_3', None, 1, None, True)
                write_instrument_read(active_sequence_name_3,
                    'active_sequence_name_3')
                print('malloc', sys.getsizeof(active_sequence_name_3),
                    'active_sequence_name_3')
                if instrument_read(active_sequence_name_3,
                    'active_sequence_name_3') not in instrument_read(
                    fasta_dict_3, 'fasta_dict_3'):
                    print(37, 61)
                    fasta_dict_3[instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3')] = []
                    write_instrument_read_sub(fasta_dict_3, 'fasta_dict_3',
                        instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3'), None, None, False)
                continue
            print(36, 63)
            sequence_3 = instrument_read(line_3, 'line_3')
            write_instrument_read(sequence_3, 'sequence_3')
            print('malloc', sys.getsizeof(sequence_3), 'sequence_3')
            instrument_read_sub(instrument_read(fasta_dict_3,
                'fasta_dict_3'), 'fasta_dict_3', instrument_read(
                active_sequence_name_3, 'active_sequence_name_3'), None,
                None, False).append(instrument_read(sequence_3, 'sequence_3'))
    print(32, 65)
    quality_dict_3 = {}
    write_instrument_read(quality_dict_3, 'quality_dict_3')
    print('malloc', sys.getsizeof(quality_dict_3), 'quality_dict_3')
    with open(instrument_read(quality_file_3, 'quality_file_3'), 'r') as f_3:
        for line_3 in instrument_read(f_3, 'f_3'):
            print(40, 68)
            line_3 = instrument_read(line_3, 'line_3').strip()
            write_instrument_read(line_3, 'line_3')
            print('malloc', sys.getsizeof(line_3), 'line_3')
            if not instrument_read(line_3, 'line_3'):
                continue
            if instrument_read(line_3, 'line_3').startswith('>'):
                print(44, 72)
                active_sequence_name_3 = instrument_read_sub(instrument_read
                    (line_3, 'line_3'), 'line_3', None, 1, None, True)
                write_instrument_read(active_sequence_name_3,
                    'active_sequence_name_3')
                print('malloc', sys.getsizeof(active_sequence_name_3),
                    'active_sequence_name_3')
                if instrument_read(active_sequence_name_3,
                    'active_sequence_name_3') not in instrument_read(
                    quality_dict_3, 'quality_dict_3'):
                    print(46, 74)
                    quality_dict_3[instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3')] = []
                    write_instrument_read_sub(quality_dict_3,
                        'quality_dict_3', instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3'), None, None, False)
                continue
            print(45, 76)
            quality_3 = instrument_read(line_3, 'line_3')
            write_instrument_read(quality_3, 'quality_3')
            print('malloc', sys.getsizeof(quality_3), 'quality_3')
            instrument_read_sub(instrument_read(quality_dict_3,
                'quality_dict_3'), 'quality_dict_3', instrument_read(
                active_sequence_name_3, 'active_sequence_name_3'), None,
                None, False).append(instrument_read(quality_3, 'quality_3'))
    print('exit scope 3')
    return instrument_read(fasta_dict_3, 'fasta_dict_3'), instrument_read(
        quality_dict_3, 'quality_dict_3')
    print('exit scope 3')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 4')
    print(1, 81)
    fasta_file_4 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file_4, 'fasta_file_4')
    print('malloc', sys.getsizeof(fasta_file_4), 'fasta_file_4')
    fastq_file_4 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file_4, 'fastq_file_4')
    print('malloc', sys.getsizeof(fastq_file_4), 'fastq_file_4')
    output_file_4 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file_4, 'output_file_4')
    print('malloc', sys.getsizeof(output_file_4), 'output_file_4')
    min_length_4 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length_4, 'min_length_4')
    print('malloc', sys.getsizeof(min_length_4), 'min_length_4')
    max_length_4 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length_4, 'max_length_4')
    print('malloc', sys.getsizeof(max_length_4), 'max_length_4')
    min_quality_4 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality_4, 'min_quality_4')
    print('malloc', sys.getsizeof(min_quality_4), 'min_quality_4')
    max_quality_4 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality_4, 'max_quality_4')
    print('malloc', sys.getsizeof(max_quality_4), 'max_quality_4')
    min_length_fraction_4 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction_4, 'min_length_fraction_4')
    print('malloc', sys.getsizeof(min_length_fraction_4),
        'min_length_fraction_4')
    max_length_fraction_4 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction_4, 'max_length_fraction_4')
    print('malloc', sys.getsizeof(max_length_fraction_4),
        'max_length_fraction_4')
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 89)
    fasta_dict_4 = read_fasta(instrument_read(fasta_file_4, 'fasta_file_4'))
    write_instrument_read(fasta_dict_4, 'fasta_dict_4')
    print('malloc', sys.getsizeof(fasta_dict_4), 'fasta_dict_4')
    print(51, 91)
    fastq_dict_4 = read_fastq(instrument_read(fastq_file_4, 'fastq_file_4'))
    write_instrument_read(fastq_dict_4, 'fastq_dict_4')
    print('malloc', sys.getsizeof(fastq_dict_4), 'fastq_dict_4')
    print(51, 93)
    filtered_fasta_dict_4 = filter_fasta(instrument_read(fasta_dict_4,
        'fasta_dict_4'), instrument_read(min_length_4, 'min_length_4'),
        instrument_read(max_length_4, 'max_length_4'), instrument_read(
        min_length_fraction_4, 'min_length_fraction_4'), instrument_read(
        max_length_fraction_4, 'max_length_fraction_4'))
    write_instrument_read(filtered_fasta_dict_4, 'filtered_fasta_dict_4')
    print('malloc', sys.getsizeof(filtered_fasta_dict_4),
        'filtered_fasta_dict_4')
    print(51, 96)
    filtered_fastq_dict_4 = filter_fastq(instrument_read(fastq_dict_4,
        'fastq_dict_4'), instrument_read(min_quality_4, 'min_quality_4'),
        instrument_read(max_quality_4, 'max_quality_4'))
    write_instrument_read(filtered_fastq_dict_4, 'filtered_fastq_dict_4')
    print('malloc', sys.getsizeof(filtered_fastq_dict_4),
        'filtered_fastq_dict_4')
    write_output(instrument_read(filtered_fasta_dict_4,
        'filtered_fasta_dict_4'), instrument_read(filtered_fastq_dict_4,
        'filtered_fastq_dict_4'), instrument_read(output_file_4,
        'output_file_4'))
    print('exit scope 4')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 5')
    print(1, 102)
    fasta_dict_5 = instrument_read(fasta_dict, 'fasta_dict')
    write_instrument_read(fasta_dict_5, 'fasta_dict_5')
    print('malloc', sys.getsizeof(fasta_dict_5), 'fasta_dict_5')
    min_length_5 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length_5, 'min_length_5')
    print('malloc', sys.getsizeof(min_length_5), 'min_length_5')
    max_length_5 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length_5, 'max_length_5')
    print('malloc', sys.getsizeof(max_length_5), 'max_length_5')
    min_length_fraction_5 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction_5, 'min_length_fraction_5')
    print('malloc', sys.getsizeof(min_length_fraction_5),
        'min_length_fraction_5')
    max_length_fraction_5 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction_5, 'max_length_fraction_5')
    print('malloc', sys.getsizeof(max_length_fraction_5),
        'max_length_fraction_5')
    """
    Filters a fasta dictionary by length
    """
    print(54, 106)
    filtered_fasta_dict_5 = {}
    write_instrument_read(filtered_fasta_dict_5, 'filtered_fasta_dict_5')
    print('malloc', sys.getsizeof(filtered_fasta_dict_5),
        'filtered_fasta_dict_5')
    for key_5 in instrument_read(fasta_dict_5, 'fasta_dict_5'):
        print(56, 108)
        sequence_5 = instrument_read_sub(instrument_read_sub(
            instrument_read(fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5',
            instrument_read(key_5, 'key_5'), None, None, False),
            'fasta_dict_5[key_5]', 0, None, None, False)
        write_instrument_read(sequence_5, 'sequence_5')
        print('malloc', sys.getsizeof(sequence_5), 'sequence_5')
        print(56, 109)
        length_5 = len(instrument_read(sequence_5, 'sequence_5'))
        write_instrument_read(length_5, 'length_5')
        print('malloc', sys.getsizeof(length_5), 'length_5')
        if instrument_read(min_length_fraction_5, 'min_length_fraction_5') > 0:
            if instrument_read(length_5, 'length_5') < instrument_read(
                min_length_fraction_5, 'min_length_fraction_5') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5',
                instrument_read(key_5, 'key_5'), None, None, False),
                'fasta_dict_5[key_5]', 0, None, None, False)):
                continue
        if instrument_read(max_length_fraction_5, 'max_length_fraction_5') > 0:
            if instrument_read(length_5, 'length_5') > instrument_read(
                max_length_fraction_5, 'max_length_fraction_5') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5',
                instrument_read(key_5, 'key_5'), None, None, False),
                'fasta_dict_5[key_5]', 0, None, None, False)):
                continue
        if instrument_read(min_length_5, 'min_length_5') > 0:
            if instrument_read(length_5, 'length_5') < instrument_read(
                min_length_5, 'min_length_5'):
                continue
        if instrument_read(max_length_5, 'max_length_5') > 0:
            if instrument_read(length_5, 'length_5') > instrument_read(
                max_length_5, 'max_length_5'):
                continue
        print(71, 122)
        filtered_fasta_dict_5[instrument_read(instrument_read(key_5,
            'key_5'), 'key_5')] = instrument_read_sub(instrument_read(
            fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5', instrument_read(
            key_5, 'key_5'), None, None, False)
        write_instrument_read_sub(filtered_fasta_dict_5,
            'filtered_fasta_dict_5', instrument_read(instrument_read(key_5,
            'key_5'), 'key_5'), None, None, False)
    print('exit scope 5')
    return instrument_read(filtered_fasta_dict_5, 'filtered_fasta_dict_5')
    print('exit scope 5')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 6')
    print(1, 126)
    fastq_dict_6 = instrument_read(fastq_dict, 'fastq_dict')
    write_instrument_read(fastq_dict_6, 'fastq_dict_6')
    print('malloc', sys.getsizeof(fastq_dict_6), 'fastq_dict_6')
    min_quality_6 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality_6, 'min_quality_6')
    print('malloc', sys.getsizeof(min_quality_6), 'min_quality_6')
    max_quality_6 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality_6, 'max_quality_6')
    print('malloc', sys.getsizeof(max_quality_6), 'max_quality_6')
    """
    Filters a fastq dictionary by quality
    """
    print(77, 130)
    filtered_fastq_dict_6 = {}
    write_instrument_read(filtered_fastq_dict_6, 'filtered_fastq_dict_6')
    print('malloc', sys.getsizeof(filtered_fastq_dict_6),
        'filtered_fastq_dict_6')
    for key_6 in instrument_read(fastq_dict_6, 'fastq_dict_6'):
        print(79, 132)
        quality_6 = instrument_read_sub(instrument_read_sub(instrument_read
            (fastq_dict_6, 'fastq_dict_6'), 'fastq_dict_6', instrument_read
            (key_6, 'key_6'), None, None, False), 'fastq_dict_6[key_6]', 0,
            None, None, False)
        write_instrument_read(quality_6, 'quality_6')
        print('malloc', sys.getsizeof(quality_6), 'quality_6')
        if instrument_read(min_quality_6, 'min_quality_6') > 0:
            if instrument_read(quality_6, 'quality_6') < instrument_read(
                min_quality_6, 'min_quality_6'):
                continue
        if instrument_read(max_quality_6, 'max_quality_6') > 0:
            if instrument_read(quality_6, 'quality_6') > instrument_read(
                max_quality_6, 'max_quality_6'):
                continue
        print(86, 139)
        filtered_fastq_dict_6[instrument_read(instrument_read(key_6,
            'key_6'), 'key_6')] = instrument_read_sub(instrument_read(
            fastq_dict_6, 'fastq_dict_6'), 'fastq_dict_6', instrument_read(
            key_6, 'key_6'), None, None, False)
        write_instrument_read_sub(filtered_fastq_dict_6,
            'filtered_fastq_dict_6', instrument_read(instrument_read(key_6,
            'key_6'), 'key_6'), None, None, False)
    print('exit scope 6')
    return instrument_read(filtered_fastq_dict_6, 'filtered_fastq_dict_6')
    print('exit scope 6')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 7')
    print(1, 143)
    filtered_fasta_dict_7 = instrument_read(filtered_fasta_dict,
        'filtered_fasta_dict')
    write_instrument_read(filtered_fasta_dict_7, 'filtered_fasta_dict_7')
    print('malloc', sys.getsizeof(filtered_fasta_dict_7),
        'filtered_fasta_dict_7')
    filtered_fastq_dict_7 = instrument_read(filtered_fastq_dict,
        'filtered_fastq_dict')
    write_instrument_read(filtered_fastq_dict_7, 'filtered_fastq_dict_7')
    print('malloc', sys.getsizeof(filtered_fastq_dict_7),
        'filtered_fastq_dict_7')
    output_file_7 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file_7, 'output_file_7')
    print('malloc', sys.getsizeof(output_file_7), 'output_file_7')
    """
    Writes output
    """
    with open(instrument_read(output_file_7, 'output_file_7'), 'w') as f_7:
        for key_7 in instrument_read(filtered_fasta_dict_7,
            'filtered_fasta_dict_7'):
            instrument_read(f_7, 'f_7').write('>' + instrument_read(key_7,
                'key_7') + '\n')
            instrument_read(f_7, 'f_7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fasta_dict_7,
                'filtered_fasta_dict_7'), 'filtered_fasta_dict_7',
                instrument_read(key_7, 'key_7'), None, None, False),
                'filtered_fasta_dict_7[key_7]', 0, None, None, False) + '\n')
        for key_7 in instrument_read(filtered_fastq_dict_7,
            'filtered_fastq_dict_7'):
            instrument_read(f_7, 'f_7').write('@' + instrument_read(key_7,
                'key_7') + '\n')
            instrument_read(f_7, 'f_7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict_7,
                'filtered_fastq_dict_7'), 'filtered_fastq_dict_7',
                instrument_read(key_7, 'key_7'), None, None, False),
                'filtered_fastq_dict_7[key_7]', 0, None, None, False) + '\n')
            instrument_read(f_7, 'f_7').write('+\n')
            instrument_read(f_7, 'f_7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict_7,
                'filtered_fastq_dict_7'), 'filtered_fastq_dict_7',
                instrument_read(key_7, 'key_7'), None, None, False),
                'filtered_fastq_dict_7[key_7]', 1, None, None, False) + '\n')
    print('exit scope 7')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print('enter scope 8')
    print(1, 161)
    seq1_8 = instrument_read(seq1, 'seq1')
    write_instrument_read(seq1_8, 'seq1_8')
    print('malloc', sys.getsizeof(seq1_8), 'seq1_8')
    seq2_8 = instrument_read(seq2, 'seq2')
    write_instrument_read(seq2_8, 'seq2_8')
    print('malloc', sys.getsizeof(seq2_8), 'seq2_8')
    match_score_8 = instrument_read(match_score, 'match_score')
    write_instrument_read(match_score_8, 'match_score_8')
    print('malloc', sys.getsizeof(match_score_8), 'match_score_8')
    mismatch_score_8 = instrument_read(mismatch_score, 'mismatch_score')
    write_instrument_read(mismatch_score_8, 'mismatch_score_8')
    print('malloc', sys.getsizeof(mismatch_score_8), 'mismatch_score_8')
    gap_score_8 = instrument_read(gap_score, 'gap_score')
    write_instrument_read(gap_score_8, 'gap_score_8')
    print('malloc', sys.getsizeof(gap_score_8), 'gap_score_8')
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 166)
    scoring_matrix_8 = [[(0) for j_8 in range(len(instrument_read(seq2_8,
        'seq2_8')) + 1)] for i_8 in range(len(instrument_read(seq1_8,
        'seq1_8')) + 1)]
    write_instrument_read(scoring_matrix_8, 'scoring_matrix_8')
    print('malloc', sys.getsizeof(scoring_matrix_8), 'scoring_matrix_8')
    print(100, 168)
    traceback_matrix_8 = [[None for j_8 in range(len(instrument_read(seq2_8,
        'seq2_8')) + 1)] for i_8 in range(len(instrument_read(seq1_8,
        'seq1_8')) + 1)]
    write_instrument_read(traceback_matrix_8, 'traceback_matrix_8')
    print('malloc', sys.getsizeof(traceback_matrix_8), 'traceback_matrix_8')
    print(100, 170)
    max_score_8 = 0
    write_instrument_read(max_score_8, 'max_score_8')
    print('malloc', sys.getsizeof(max_score_8), 'max_score_8')
    print(100, 171)
    max_pos_8 = None
    write_instrument_read(max_pos_8, 'max_pos_8')
    print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8')
    for i_8 in range(1, len(instrument_read(seq1_8, 'seq1_8')) + 1):
        for j_8 in range(1, len(instrument_read(seq2_8, 'seq2_8')) + 1):
            print(104, 175)
            letter1_8 = instrument_read_sub(instrument_read(seq1_8,
                'seq1_8'), 'seq1_8', instrument_read(i_8, 'i_8') - 1, None,
                None, False)
            write_instrument_read(letter1_8, 'letter1_8')
            print('malloc', sys.getsizeof(letter1_8), 'letter1_8')
            print(104, 176)
            letter2_8 = instrument_read_sub(instrument_read(seq2_8,
                'seq2_8'), 'seq2_8', instrument_read(j_8, 'j_8') - 1, None,
                None, False)
            write_instrument_read(letter2_8, 'letter2_8')
            print('malloc', sys.getsizeof(letter2_8), 'letter2_8')
            if instrument_read(letter1_8, 'letter1_8') == instrument_read(
                letter2_8, 'letter2_8'):
                print(106, 178)
                diagonal_score_8 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                    'scoring_matrix_8', instrument_read(i_8, 'i_8') - 1,
                    None, None, False), 'scoring_matrix_8[i_8 - 1]', 
                    instrument_read(j_8, 'j_8') - 1, None, None, False
                    ) + instrument_read(match_score_8, 'match_score_8')
                write_instrument_read(diagonal_score_8, 'diagonal_score_8')
                print('malloc', sys.getsizeof(diagonal_score_8),
                    'diagonal_score_8')
            else:
                print(108, 180)
                diagonal_score_8 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                    'scoring_matrix_8', instrument_read(i_8, 'i_8') - 1,
                    None, None, False), 'scoring_matrix_8[i_8 - 1]', 
                    instrument_read(j_8, 'j_8') - 1, None, None, False
                    ) + instrument_read(mismatch_score_8, 'mismatch_score_8')
                write_instrument_read(diagonal_score_8, 'diagonal_score_8')
                print('malloc', sys.getsizeof(diagonal_score_8),
                    'diagonal_score_8')
            print(107, 182)
            up_score_8 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                'scoring_matrix_8', instrument_read(i_8, 'i_8') - 1, None,
                None, False), 'scoring_matrix_8[i_8 - 1]', instrument_read(
                j_8, 'j_8'), None, None, False) + instrument_read(gap_score_8,
                'gap_score_8')
            write_instrument_read(up_score_8, 'up_score_8')
            print('malloc', sys.getsizeof(up_score_8), 'up_score_8')
            print(107, 183)
            left_score_8 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                'scoring_matrix_8', instrument_read(i_8, 'i_8'), None, None,
                False), 'scoring_matrix_8[i_8]', instrument_read(j_8, 'j_8'
                ) - 1, None, None, False) + instrument_read(gap_score_8,
                'gap_score_8')
            write_instrument_read(left_score_8, 'left_score_8')
            print('malloc', sys.getsizeof(left_score_8), 'left_score_8')
            if instrument_read(diagonal_score_8, 'diagonal_score_8'
                ) >= instrument_read(up_score_8, 'up_score_8'):
                if instrument_read(diagonal_score_8, 'diagonal_score_8'
                    ) >= instrument_read(left_score_8, 'left_score_8'):
                    print(115, 187)
                    scoring_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = instrument_read(diagonal_score_8,
                        'diagonal_score_8')
                    write_instrument_read_sub(scoring_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ], "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
                    print(115, 188)
                    traceback_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = 'D'
                    write_instrument_read_sub(traceback_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ],
                        "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
                else:
                    print(117, 190)
                    scoring_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = instrument_read(left_score_8,
                        'left_score_8')
                    write_instrument_read_sub(scoring_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ], "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
                    print(117, 191)
                    traceback_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = 'L'
                    write_instrument_read_sub(traceback_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ],
                        "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
            elif instrument_read(up_score_8, 'up_score_8') >= instrument_read(
                left_score_8, 'left_score_8'):
                print(112, 194)
                scoring_matrix_8[instrument_read(instrument_read(i_8, 'i_8'
                    ), 'i_8')][instrument_read(instrument_read(j_8, 'j_8'),
                    'j_8')] = instrument_read(up_score_8, 'up_score_8')
                write_instrument_read_sub(scoring_matrix_8[instrument_read(
                    instrument_read(i_8, 'i_8'), 'i_8')],
                    "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
                print(112, 195)
                traceback_matrix_8[instrument_read(instrument_read(i_8,
                    'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                    'j_8'), 'j_8')] = 'U'
                write_instrument_read_sub(traceback_matrix_8[
                    instrument_read(instrument_read(i_8, 'i_8'), 'i_8')],
                    "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
            else:
                print(114, 197)
                scoring_matrix_8[instrument_read(instrument_read(i_8, 'i_8'
                    ), 'i_8')][instrument_read(instrument_read(j_8, 'j_8'),
                    'j_8')] = instrument_read(left_score_8, 'left_score_8')
                write_instrument_read_sub(scoring_matrix_8[instrument_read(
                    instrument_read(i_8, 'i_8'), 'i_8')],
                    "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
                print(114, 198)
                traceback_matrix_8[instrument_read(instrument_read(i_8,
                    'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                    'j_8'), 'j_8')] = 'L'
                write_instrument_read_sub(traceback_matrix_8[
                    instrument_read(instrument_read(i_8, 'i_8'), 'i_8')],
                    "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
            if instrument_read_sub(instrument_read_sub(instrument_read(
                scoring_matrix_8, 'scoring_matrix_8'), 'scoring_matrix_8',
                instrument_read(i_8, 'i_8'), None, None, False),
                'scoring_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
                None, False) >= instrument_read(max_score_8, 'max_score_8'):
                print(118, 201)
                max_score_8 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                    'scoring_matrix_8', instrument_read(i_8, 'i_8'), None,
                    None, False), 'scoring_matrix_8[i_8]', instrument_read(
                    j_8, 'j_8'), None, None, False)
                write_instrument_read(max_score_8, 'max_score_8')
                print('malloc', sys.getsizeof(max_score_8), 'max_score_8')
                print(118, 202)
                max_pos_8 = instrument_read(i_8, 'i_8'), instrument_read(j_8,
                    'j_8')
                write_instrument_read(max_pos_8, 'max_pos_8')
                print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8')
    print(103, 204)
    i_8 = instrument_read(max_pos_8, 'max_pos_8')
    write_instrument_read(i_8, 'i_8')
    print('malloc', sys.getsizeof(i_8), 'i_8')
    print(103, 205)
    j_8 = instrument_read(max_pos_8, 'max_pos_8')
    write_instrument_read(j_8, 'j_8')
    print('malloc', sys.getsizeof(j_8), 'j_8')
    print(103, 206)
    aln1_8 = ''
    write_instrument_read(aln1_8, 'aln1_8')
    print('malloc', sys.getsizeof(aln1_8), 'aln1_8')
    print(103, 207)
    aln2_8 = ''
    write_instrument_read(aln2_8, 'aln2_8')
    print('malloc', sys.getsizeof(aln2_8), 'aln2_8')
    while instrument_read_sub(instrument_read_sub(instrument_read(
        traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
        instrument_read(i_8, 'i_8'), None, None, False),
        'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None, None,
        False) != None:
        if instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
            instrument_read(i_8, 'i_8'), None, None, False),
            'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
            None, False) == 'D':
            print(123, 210)
            aln1_8 += instrument_read_sub(instrument_read(seq1_8, 'seq1_8'),
                'seq1_8', instrument_read(i_8, 'i_8') - 1, None, None, False)
            write_instrument_read(aln1_8, 'aln1_8')
            print(123, 211)
            aln2_8 += instrument_read_sub(instrument_read(seq2_8, 'seq2_8'),
                'seq2_8', instrument_read(j_8, 'j_8') - 1, None, None, False)
            write_instrument_read(aln2_8, 'aln2_8')
            print(123, 212)
            i_8 -= 1
            write_instrument_read(i_8, 'i_8')
            print(123, 213)
            j_8 -= 1
            write_instrument_read(j_8, 'j_8')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
            instrument_read(i_8, 'i_8'), None, None, False),
            'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
            None, False) == 'L':
            print(126, 215)
            aln1_8 += '-'
            write_instrument_read(aln1_8, 'aln1_8')
            print(126, 216)
            aln2_8 += instrument_read_sub(instrument_read(seq2_8, 'seq2_8'),
                'seq2_8', instrument_read(j_8, 'j_8') - 1, None, None, False)
            write_instrument_read(aln2_8, 'aln2_8')
            print(126, 217)
            j_8 -= 1
            write_instrument_read(j_8, 'j_8')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
            instrument_read(i_8, 'i_8'), None, None, False),
            'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
            None, False) == 'U':
            print(129, 219)
            aln1_8 += instrument_read_sub(instrument_read(seq1_8, 'seq1_8'),
                'seq1_8', instrument_read(i_8, 'i_8') - 1, None, None, False)
            write_instrument_read(aln1_8, 'aln1_8')
            print(129, 220)
            aln2_8 += '-'
            write_instrument_read(aln2_8, 'aln2_8')
            print(129, 221)
            i_8 -= 1
            write_instrument_read(i_8, 'i_8')
    print('exit scope 8')
    return instrument_read_sub(instrument_read(aln1_8, 'aln1_8'), 'aln1_8',
        None, None, None, True), instrument_read_sub(instrument_read(aln2_8,
        'aln2_8'), 'aln2_8', None, None, None, True)
    print('exit scope 8')


def main():
<<<<<<< HEAD
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
=======
    print('enter scope 9')
    print(1, 224)
    print(134, 226)
    fasta_file_9 = instrument_read(path_0, 'path_0'
        ) + '/benchmarks/supplemental_files/fasta_example.fasta'
    write_instrument_read(fasta_file_9, 'fasta_file_9')
    print('malloc', sys.getsizeof(fasta_file_9), 'fasta_file_9')
    print(134, 227)
    fastq_file_9 = instrument_read(path_0, 'path_0'
        ) + '/benchmarks/supplemental_files/fastq_large.fastq'
    write_instrument_read(fastq_file_9, 'fastq_file_9')
    print('malloc', sys.getsizeof(fastq_file_9), 'fastq_file_9')
    print(134, 228)
    output_file_9 = instrument_read(path_0, 'path_0'
        ) + '/benchmarks/supplemental_files/output_file.txt'
    write_instrument_read(output_file_9, 'output_file_9')
    print('malloc', sys.getsizeof(output_file_9), 'output_file_9')
    darwin_wga_workflow(instrument_read(fasta_file_9, 'fasta_file_9'),
        instrument_read(fastq_file_9, 'fastq_file_9'), instrument_read(
        output_file_9, 'output_file_9'))
    print('exit scope 9')
>>>>>>> master


if instrument_read(__name__, '__name__') == '__main__':
    main()
