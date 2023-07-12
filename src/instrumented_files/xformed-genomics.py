import sys
from instrument_lib import *
import sys
from instrument_lib import *


def read_fasta(fasta_file):
    print('enter scope 1')
    print(1, 3)
    fasta_file__1 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file__1, 'fasta_file__1')
    print('malloc', sys.getsizeof(fasta_file__1), 'fasta_file__1')
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 8)
    fasta_dict__1 = {}
    write_instrument_read(fasta_dict__1, 'fasta_dict__1')
    print('malloc', sys.getsizeof(fasta_dict__1), 'fasta_dict__1')
    with open(instrument_read(fasta_file__1, 'fasta_file__1'), 'r') as f__1:
        for line__1 in instrument_read(f__1, 'f__1'):
            print('enter scope 2')
            print(5, 11)
            line__1 = instrument_read(line__1, 'line__1').strip()
            write_instrument_read(line__1, 'line__1')
            print('malloc', sys.getsizeof(line__1), 'line__1')
            if not instrument_read(line__1, 'line__1'):
                continue
            if instrument_read(line__1, 'line__1').startswith('>'):
                print(9, 15)
                active_sequence_name__2 = instrument_read_sub(instrument_read
                    (line__1, 'line__1'), 'line__1', None, 1, None, True)
                write_instrument_read(active_sequence_name__2,
                    'active_sequence_name__2')
                print('malloc', sys.getsizeof(active_sequence_name__2),
                    'active_sequence_name__2')
                if instrument_read(active_sequence_name__2,
                    'active_sequence_name__2') not in instrument_read(
                    fasta_dict__1, 'fasta_dict__1'):
                    print(11, 17)
                    fasta_dict__1[instrument_read(instrument_read(
                        active_sequence_name__2, 'active_sequence_name__2'),
                        'active_sequence_name__2')] = []
                    write_instrument_read_sub(fasta_dict__1,
                        'fasta_dict__1', instrument_read(instrument_read(
                        active_sequence_name__2, 'active_sequence_name__2'),
                        'active_sequence_name__2'), None, None, False)
                continue
            print(10, 19)
            sequence__2 = instrument_read(line__1, 'line__1')
            write_instrument_read(sequence__2, 'sequence__2')
            print('malloc', sys.getsizeof(sequence__2), 'sequence__2')
            instrument_read_sub(instrument_read(fasta_dict__1,
                'fasta_dict__1'), 'fasta_dict__1', instrument_read(
                active_sequence_name__2, 'active_sequence_name__2'), None,
                None, False).append(instrument_read(sequence__2, 'sequence__2')
                )
            print('exit scope 2')
    print('exit scope 1')
    return instrument_read(fasta_dict__1, 'fasta_dict__1')
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 3')
    print(1, 24)
    fastq_file__3 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file__3, 'fastq_file__3')
    print('malloc', sys.getsizeof(fastq_file__3), 'fastq_file__3')
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 29)
    fastq_dict__3 = {}
    write_instrument_read(fastq_dict__3, 'fastq_dict__3')
    print('malloc', sys.getsizeof(fastq_dict__3), 'fastq_dict__3')
    with open(instrument_read(fastq_file__3, 'fastq_file__3'), 'r') as f__3:
        for line__3 in instrument_read(f__3, 'f__3'):
            print('enter scope 4')
            print(18, 32)
            line__3 = instrument_read(line__3, 'line__3').strip()
            write_instrument_read(line__3, 'line__3')
            print('malloc', sys.getsizeof(line__3), 'line__3')
            if not instrument_read(line__3, 'line__3'):
                continue
            if instrument_read(line__3, 'line__3').startswith('@'):
                print(22, 36)
                active_sequence_name__4 = instrument_read_sub(instrument_read
                    (line__3, 'line__3'), 'line__3', None, 1, None, True)
                write_instrument_read(active_sequence_name__4,
                    'active_sequence_name__4')
                print('malloc', sys.getsizeof(active_sequence_name__4),
                    'active_sequence_name__4')
                if instrument_read(active_sequence_name__4,
                    'active_sequence_name__4') not in instrument_read(
                    fastq_dict__3, 'fastq_dict__3'):
                    print(24, 38)
                    fastq_dict__3[instrument_read(instrument_read(
                        active_sequence_name__4, 'active_sequence_name__4'),
                        'active_sequence_name__4')] = []
                    write_instrument_read_sub(fastq_dict__3,
                        'fastq_dict__3', instrument_read(instrument_read(
                        active_sequence_name__4, 'active_sequence_name__4'),
                        'active_sequence_name__4'), None, None, False)
                continue
            print(23, 40)
            sequence__4 = instrument_read(line__3, 'line__3')
            write_instrument_read(sequence__4, 'sequence__4')
            print('malloc', sys.getsizeof(sequence__4), 'sequence__4')
            instrument_read_sub(instrument_read(fastq_dict__3,
                'fastq_dict__3'), 'fastq_dict__3', instrument_read(
                active_sequence_name__4, 'active_sequence_name__4'), None,
                None, False).append(instrument_read(sequence__4, 'sequence__4')
                )
            print('exit scope 4')
    print('exit scope 3')
    return instrument_read(fastq_dict__3, 'fastq_dict__3')
    print('exit scope 3')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 5')
    print(1, 45)
    fasta_file__5 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file__5, 'fasta_file__5')
    print('malloc', sys.getsizeof(fasta_file__5), 'fasta_file__5')
    quality_file__5 = instrument_read(quality_file, 'quality_file')
    write_instrument_read(quality_file__5, 'quality_file__5')
    print('malloc', sys.getsizeof(quality_file__5), 'quality_file__5')
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 50)
    fasta_dict__5 = {}
    write_instrument_read(fasta_dict__5, 'fasta_dict__5')
    print('malloc', sys.getsizeof(fasta_dict__5), 'fasta_dict__5')
    with open(instrument_read(fasta_file__5, 'fasta_file__5'), 'r') as f__5:
        for line__5 in instrument_read(f__5, 'f__5'):
            print('enter scope 6')
            print(31, 53)
            line__5 = instrument_read(line__5, 'line__5').strip()
            write_instrument_read(line__5, 'line__5')
            print('malloc', sys.getsizeof(line__5), 'line__5')
            if not instrument_read(line__5, 'line__5'):
                continue
            if instrument_read(line__5, 'line__5').startswith('>'):
                print(35, 57)
                active_sequence_name__6 = instrument_read_sub(instrument_read
                    (line__5, 'line__5'), 'line__5', None, 1, None, True)
                write_instrument_read(active_sequence_name__6,
                    'active_sequence_name__6')
                print('malloc', sys.getsizeof(active_sequence_name__6),
                    'active_sequence_name__6')
                if instrument_read(active_sequence_name__6,
                    'active_sequence_name__6') not in instrument_read(
                    fasta_dict__5, 'fasta_dict__5'):
                    print(37, 59)
                    fasta_dict__5[instrument_read(instrument_read(
                        active_sequence_name__6, 'active_sequence_name__6'),
                        'active_sequence_name__6')] = []
                    write_instrument_read_sub(fasta_dict__5,
                        'fasta_dict__5', instrument_read(instrument_read(
                        active_sequence_name__6, 'active_sequence_name__6'),
                        'active_sequence_name__6'), None, None, False)
                continue
            print(36, 61)
            sequence__6 = instrument_read(line__5, 'line__5')
            write_instrument_read(sequence__6, 'sequence__6')
            print('malloc', sys.getsizeof(sequence__6), 'sequence__6')
            instrument_read_sub(instrument_read(fasta_dict__5,
                'fasta_dict__5'), 'fasta_dict__5', instrument_read(
                active_sequence_name__6, 'active_sequence_name__6'), None,
                None, False).append(instrument_read(sequence__6, 'sequence__6')
                )
            print('exit scope 6')
    print(32, 63)
    quality_dict__5 = {}
    write_instrument_read(quality_dict__5, 'quality_dict__5')
    print('malloc', sys.getsizeof(quality_dict__5), 'quality_dict__5')
    with open(instrument_read(quality_file__5, 'quality_file__5'), 'r'
        ) as f__5:
        for line__5 in instrument_read(f__5, 'f__5'):
            print('enter scope 7')
            print(40, 66)
            line__5 = instrument_read(line__5, 'line__5').strip()
            write_instrument_read(line__5, 'line__5')
            print('malloc', sys.getsizeof(line__5), 'line__5')
            if not instrument_read(line__5, 'line__5'):
                continue
            if instrument_read(line__5, 'line__5').startswith('>'):
                print(44, 70)
                active_sequence_name__7 = instrument_read_sub(instrument_read
                    (line__5, 'line__5'), 'line__5', None, 1, None, True)
                write_instrument_read(active_sequence_name__7,
                    'active_sequence_name__7')
                print('malloc', sys.getsizeof(active_sequence_name__7),
                    'active_sequence_name__7')
                if instrument_read(active_sequence_name__7,
                    'active_sequence_name__7') not in instrument_read(
                    quality_dict__5, 'quality_dict__5'):
                    print(46, 72)
                    quality_dict__5[instrument_read(instrument_read(
                        active_sequence_name__7, 'active_sequence_name__7'),
                        'active_sequence_name__7')] = []
                    write_instrument_read_sub(quality_dict__5,
                        'quality_dict__5', instrument_read(instrument_read(
                        active_sequence_name__7, 'active_sequence_name__7'),
                        'active_sequence_name__7'), None, None, False)
                continue
            print(45, 74)
            quality__7 = instrument_read(line__5, 'line__5')
            write_instrument_read(quality__7, 'quality__7')
            print('malloc', sys.getsizeof(quality__7), 'quality__7')
            instrument_read_sub(instrument_read(quality_dict__5,
                'quality_dict__5'), 'quality_dict__5', instrument_read(
                active_sequence_name__7, 'active_sequence_name__7'), None,
                None, False).append(instrument_read(quality__7, 'quality__7'))
            print('exit scope 7')
    print('exit scope 5')
    return instrument_read(fasta_dict__5, 'fasta_dict__5'), instrument_read(
        quality_dict__5, 'quality_dict__5')
    print('exit scope 5')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 8')
    print(1, 79)
    fasta_file__8 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file__8, 'fasta_file__8')
    print('malloc', sys.getsizeof(fasta_file__8), 'fasta_file__8')
    fastq_file__8 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file__8, 'fastq_file__8')
    print('malloc', sys.getsizeof(fastq_file__8), 'fastq_file__8')
    output_file__8 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file__8, 'output_file__8')
    print('malloc', sys.getsizeof(output_file__8), 'output_file__8')
    min_length__8 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length__8, 'min_length__8')
    print('malloc', sys.getsizeof(min_length__8), 'min_length__8')
    max_length__8 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length__8, 'max_length__8')
    print('malloc', sys.getsizeof(max_length__8), 'max_length__8')
    min_quality__8 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality__8, 'min_quality__8')
    print('malloc', sys.getsizeof(min_quality__8), 'min_quality__8')
    max_quality__8 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality__8, 'max_quality__8')
    print('malloc', sys.getsizeof(max_quality__8), 'max_quality__8')
    min_length_fraction__8 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction__8, 'min_length_fraction__8')
    print('malloc', sys.getsizeof(min_length_fraction__8),
        'min_length_fraction__8')
    max_length_fraction__8 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction__8, 'max_length_fraction__8')
    print('malloc', sys.getsizeof(max_length_fraction__8),
        'max_length_fraction__8')
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 87)
    fasta_dict__8 = read_fasta(instrument_read(fasta_file__8, 'fasta_file__8'))
    write_instrument_read(fasta_dict__8, 'fasta_dict__8')
    print('malloc', sys.getsizeof(fasta_dict__8), 'fasta_dict__8')
    print(51, 89)
    fastq_dict__8 = read_fastq(instrument_read(fastq_file__8, 'fastq_file__8'))
    write_instrument_read(fastq_dict__8, 'fastq_dict__8')
    print('malloc', sys.getsizeof(fastq_dict__8), 'fastq_dict__8')
    print(51, 91)
    filtered_fasta_dict__8 = filter_fasta(instrument_read(fasta_dict__8,
        'fasta_dict__8'), instrument_read(min_length__8, 'min_length__8'),
        instrument_read(max_length__8, 'max_length__8'), instrument_read(
        min_length_fraction__8, 'min_length_fraction__8'), instrument_read(
        max_length_fraction__8, 'max_length_fraction__8'))
    write_instrument_read(filtered_fasta_dict__8, 'filtered_fasta_dict__8')
    print('malloc', sys.getsizeof(filtered_fasta_dict__8),
        'filtered_fasta_dict__8')
    print(51, 94)
    filtered_fastq_dict__8 = filter_fastq(instrument_read(fastq_dict__8,
        'fastq_dict__8'), instrument_read(min_quality__8, 'min_quality__8'),
        instrument_read(max_quality__8, 'max_quality__8'))
    write_instrument_read(filtered_fastq_dict__8, 'filtered_fastq_dict__8')
    print('malloc', sys.getsizeof(filtered_fastq_dict__8),
        'filtered_fastq_dict__8')
    write_output(instrument_read(filtered_fasta_dict__8,
        'filtered_fasta_dict__8'), instrument_read(filtered_fastq_dict__8,
        'filtered_fastq_dict__8'), instrument_read(output_file__8,
        'output_file__8'))
    print('exit scope 8')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 9')
    print(1, 100)
    fasta_dict__9 = instrument_read(fasta_dict, 'fasta_dict')
    write_instrument_read(fasta_dict__9, 'fasta_dict__9')
    print('malloc', sys.getsizeof(fasta_dict__9), 'fasta_dict__9')
    min_length__9 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length__9, 'min_length__9')
    print('malloc', sys.getsizeof(min_length__9), 'min_length__9')
    max_length__9 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length__9, 'max_length__9')
    print('malloc', sys.getsizeof(max_length__9), 'max_length__9')
    min_length_fraction__9 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction__9, 'min_length_fraction__9')
    print('malloc', sys.getsizeof(min_length_fraction__9),
        'min_length_fraction__9')
    max_length_fraction__9 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction__9, 'max_length_fraction__9')
    print('malloc', sys.getsizeof(max_length_fraction__9),
        'max_length_fraction__9')
    """
    Filters a fasta dictionary by length
    """
    print(54, 104)
    filtered_fasta_dict__9 = {}
    write_instrument_read(filtered_fasta_dict__9, 'filtered_fasta_dict__9')
    print('malloc', sys.getsizeof(filtered_fasta_dict__9),
        'filtered_fasta_dict__9')
    for key__9 in instrument_read(fasta_dict__9, 'fasta_dict__9'):
        print('enter scope 10')
        print(56, 106)
        sequence__10 = instrument_read_sub(instrument_read_sub(
            instrument_read(fasta_dict__9, 'fasta_dict__9'),
            'fasta_dict__9', instrument_read(key__9, 'key__9'), None, None,
            False), 'fasta_dict__9[key__9]', 0, None, None, False)
        write_instrument_read(sequence__10, 'sequence__10')
        print('malloc', sys.getsizeof(sequence__10), 'sequence__10')
        print(56, 107)
        length__10 = len(instrument_read(sequence__10, 'sequence__10'))
        write_instrument_read(length__10, 'length__10')
        print('malloc', sys.getsizeof(length__10), 'length__10')
        if instrument_read(min_length_fraction__9, 'min_length_fraction__9'
            ) > 0:
            if instrument_read(length__10, 'length__10') < instrument_read(
                min_length_fraction__9, 'min_length_fraction__9') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict__9, 'fasta_dict__9'), 'fasta_dict__9',
                instrument_read(key__9, 'key__9'), None, None, False),
                'fasta_dict__9[key__9]', 0, None, None, False)):
                continue
        if instrument_read(max_length_fraction__9, 'max_length_fraction__9'
            ) > 0:
            if instrument_read(length__10, 'length__10') > instrument_read(
                max_length_fraction__9, 'max_length_fraction__9') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict__9, 'fasta_dict__9'), 'fasta_dict__9',
                instrument_read(key__9, 'key__9'), None, None, False),
                'fasta_dict__9[key__9]', 0, None, None, False)):
                continue
        if instrument_read(min_length__9, 'min_length__9') > 0:
            if instrument_read(length__10, 'length__10') < instrument_read(
                min_length__9, 'min_length__9'):
                continue
        if instrument_read(max_length__9, 'max_length__9') > 0:
            if instrument_read(length__10, 'length__10') > instrument_read(
                max_length__9, 'max_length__9'):
                continue
        print(71, 120)
        filtered_fasta_dict__9[instrument_read(instrument_read(key__9,
            'key__9'), 'key__9')] = instrument_read_sub(instrument_read(
            fasta_dict__9, 'fasta_dict__9'), 'fasta_dict__9',
            instrument_read(key__9, 'key__9'), None, None, False)
        write_instrument_read_sub(filtered_fasta_dict__9,
            'filtered_fasta_dict__9', instrument_read(instrument_read(
            key__9, 'key__9'), 'key__9'), None, None, False)
        print('exit scope 10')
    print('exit scope 9')
    return instrument_read(filtered_fasta_dict__9, 'filtered_fasta_dict__9')
    print('exit scope 9')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 11')
    print(1, 124)
    fastq_dict__11 = instrument_read(fastq_dict, 'fastq_dict')
    write_instrument_read(fastq_dict__11, 'fastq_dict__11')
    print('malloc', sys.getsizeof(fastq_dict__11), 'fastq_dict__11')
    min_quality__11 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality__11, 'min_quality__11')
    print('malloc', sys.getsizeof(min_quality__11), 'min_quality__11')
    max_quality__11 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality__11, 'max_quality__11')
    print('malloc', sys.getsizeof(max_quality__11), 'max_quality__11')
    """
    Filters a fastq dictionary by quality
    """
    print(77, 128)
    filtered_fastq_dict__11 = {}
    write_instrument_read(filtered_fastq_dict__11, 'filtered_fastq_dict__11')
    print('malloc', sys.getsizeof(filtered_fastq_dict__11),
        'filtered_fastq_dict__11')
    for key__11 in instrument_read(fastq_dict__11, 'fastq_dict__11'):
        print('enter scope 12')
        print(79, 130)
        quality__12 = instrument_read_sub(instrument_read_sub(
            instrument_read(fastq_dict__11, 'fastq_dict__11'),
            'fastq_dict__11', instrument_read(key__11, 'key__11'), None,
            None, False), 'fastq_dict__11[key__11]', 0, None, None, False)
        write_instrument_read(quality__12, 'quality__12')
        print('malloc', sys.getsizeof(quality__12), 'quality__12')
        if instrument_read(min_quality__11, 'min_quality__11') > 0:
            if instrument_read(quality__12, 'quality__12') < instrument_read(
                min_quality__11, 'min_quality__11'):
                continue
        if instrument_read(max_quality__11, 'max_quality__11') > 0:
            if instrument_read(quality__12, 'quality__12') > instrument_read(
                max_quality__11, 'max_quality__11'):
                continue
        print(86, 137)
        filtered_fastq_dict__11[instrument_read(instrument_read(key__11,
            'key__11'), 'key__11')] = instrument_read_sub(instrument_read(
            fastq_dict__11, 'fastq_dict__11'), 'fastq_dict__11',
            instrument_read(key__11, 'key__11'), None, None, False)
        write_instrument_read_sub(filtered_fastq_dict__11,
            'filtered_fastq_dict__11', instrument_read(instrument_read(
            key__11, 'key__11'), 'key__11'), None, None, False)
        print('exit scope 12')
    print('exit scope 11')
    return instrument_read(filtered_fastq_dict__11, 'filtered_fastq_dict__11')
    print('exit scope 11')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 13')
    print(1, 141)
    filtered_fasta_dict__13 = instrument_read(filtered_fasta_dict,
        'filtered_fasta_dict')
    write_instrument_read(filtered_fasta_dict__13, 'filtered_fasta_dict__13')
    print('malloc', sys.getsizeof(filtered_fasta_dict__13),
        'filtered_fasta_dict__13')
    filtered_fastq_dict__13 = instrument_read(filtered_fastq_dict,
        'filtered_fastq_dict')
    write_instrument_read(filtered_fastq_dict__13, 'filtered_fastq_dict__13')
    print('malloc', sys.getsizeof(filtered_fastq_dict__13),
        'filtered_fastq_dict__13')
    output_file__13 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file__13, 'output_file__13')
    print('malloc', sys.getsizeof(output_file__13), 'output_file__13')
    """
    Writes output
    """
    with open(instrument_read(output_file__13, 'output_file__13'), 'w'
        ) as f__13:
        for key__13 in instrument_read(filtered_fasta_dict__13,
            'filtered_fasta_dict__13'):
            print('enter scope 14')
            instrument_read(f__13, 'f__13').write('>' + instrument_read(
                key__13, 'key__13') + '\n')
            instrument_read(f__13, 'f__13').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fasta_dict__13,
                'filtered_fasta_dict__13'), 'filtered_fasta_dict__13',
                instrument_read(key__13, 'key__13'), None, None, False),
                'filtered_fasta_dict__13[key__13]', 0, None, None, False) +
                '\n')
            print('exit scope 14')
        for key__13 in instrument_read(filtered_fastq_dict__13,
            'filtered_fastq_dict__13'):
            print('enter scope 15')
            instrument_read(f__13, 'f__13').write('@' + instrument_read(
                key__13, 'key__13') + '\n')
            instrument_read(f__13, 'f__13').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict__13,
                'filtered_fastq_dict__13'), 'filtered_fastq_dict__13',
                instrument_read(key__13, 'key__13'), None, None, False),
                'filtered_fastq_dict__13[key__13]', 0, None, None, False) +
                '\n')
            instrument_read(f__13, 'f__13').write('+\n')
            instrument_read(f__13, 'f__13').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict__13,
                'filtered_fastq_dict__13'), 'filtered_fastq_dict__13',
                instrument_read(key__13, 'key__13'), None, None, False),
                'filtered_fastq_dict__13[key__13]', 1, None, None, False) +
                '\n')
            print('exit scope 15')
    print('exit scope 13')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print('enter scope 16')
    print(1, 159)
    seq1__16 = instrument_read(seq1, 'seq1')
    write_instrument_read(seq1__16, 'seq1__16')
    print('malloc', sys.getsizeof(seq1__16), 'seq1__16')
    seq2__16 = instrument_read(seq2, 'seq2')
    write_instrument_read(seq2__16, 'seq2__16')
    print('malloc', sys.getsizeof(seq2__16), 'seq2__16')
    match_score__16 = instrument_read(match_score, 'match_score')
    write_instrument_read(match_score__16, 'match_score__16')
    print('malloc', sys.getsizeof(match_score__16), 'match_score__16')
    mismatch_score__16 = instrument_read(mismatch_score, 'mismatch_score')
    write_instrument_read(mismatch_score__16, 'mismatch_score__16')
    print('malloc', sys.getsizeof(mismatch_score__16), 'mismatch_score__16')
    gap_score__16 = instrument_read(gap_score, 'gap_score')
    write_instrument_read(gap_score__16, 'gap_score__16')
    print('malloc', sys.getsizeof(gap_score__16), 'gap_score__16')
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 164)
    scoring_matrix__16 = [[(0) for j__16 in range(len(instrument_read(
        seq2__16, 'seq2__16')) + 1)] for i__16 in range(len(instrument_read
        (seq1__16, 'seq1__16')) + 1)]
    write_instrument_read(scoring_matrix__16, 'scoring_matrix__16')
    print('malloc', sys.getsizeof(scoring_matrix__16), 'scoring_matrix__16')
    print(100, 166)
    traceback_matrix__16 = [[None for j__16 in range(len(instrument_read(
        seq2__16, 'seq2__16')) + 1)] for i__16 in range(len(instrument_read
        (seq1__16, 'seq1__16')) + 1)]
    write_instrument_read(traceback_matrix__16, 'traceback_matrix__16')
    print('malloc', sys.getsizeof(traceback_matrix__16), 'traceback_matrix__16'
        )
    print(100, 168)
    max_score__16 = 0
    write_instrument_read(max_score__16, 'max_score__16')
    print('malloc', sys.getsizeof(max_score__16), 'max_score__16')
    print(100, 169)
    max_pos__16 = None
    write_instrument_read(max_pos__16, 'max_pos__16')
    print('malloc', sys.getsizeof(max_pos__16), 'max_pos__16')
    for i__16 in range(1, len(instrument_read(seq1__16, 'seq1__16')) + 1):
        print('enter scope 17')
        for j__16 in range(1, len(instrument_read(seq2__16, 'seq2__16')) + 1):
            print('enter scope 18')
            print(104, 173)
            letter1__18 = instrument_read_sub(instrument_read(seq1__16,
                'seq1__16'), 'seq1__16', instrument_read(i__16, 'i__16') - 
                1, None, None, False)
            write_instrument_read(letter1__18, 'letter1__18')
            print('malloc', sys.getsizeof(letter1__18), 'letter1__18')
            print(104, 174)
            letter2__18 = instrument_read_sub(instrument_read(seq2__16,
                'seq2__16'), 'seq2__16', instrument_read(j__16, 'j__16') - 
                1, None, None, False)
            write_instrument_read(letter2__18, 'letter2__18')
            print('malloc', sys.getsizeof(letter2__18), 'letter2__18')
            if instrument_read(letter1__18, 'letter1__18') == instrument_read(
                letter2__18, 'letter2__18'):
                print(106, 176)
                diagonal_score__18 = instrument_read_sub(instrument_read_sub
                    (instrument_read(scoring_matrix__16,
                    'scoring_matrix__16'), 'scoring_matrix__16', 
                    instrument_read(i__16, 'i__16') - 1, None, None, False),
                    'scoring_matrix__16[i__16 - 1]', instrument_read(j__16,
                    'j__16') - 1, None, None, False) + instrument_read(
                    match_score__16, 'match_score__16')
                write_instrument_read(diagonal_score__18, 'diagonal_score__18')
                print('malloc', sys.getsizeof(diagonal_score__18),
                    'diagonal_score__18')
            else:
                print(108, 178)
                diagonal_score__18 = instrument_read_sub(instrument_read_sub
                    (instrument_read(scoring_matrix__16,
                    'scoring_matrix__16'), 'scoring_matrix__16', 
                    instrument_read(i__16, 'i__16') - 1, None, None, False),
                    'scoring_matrix__16[i__16 - 1]', instrument_read(j__16,
                    'j__16') - 1, None, None, False) + instrument_read(
                    mismatch_score__16, 'mismatch_score__16')
                write_instrument_read(diagonal_score__18, 'diagonal_score__18')
                print('malloc', sys.getsizeof(diagonal_score__18),
                    'diagonal_score__18')
            print(107, 180)
            up_score__18 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix__16, 'scoring_matrix__16'),
                'scoring_matrix__16', instrument_read(i__16, 'i__16') - 1,
                None, None, False), 'scoring_matrix__16[i__16 - 1]',
                instrument_read(j__16, 'j__16'), None, None, False
                ) + instrument_read(gap_score__16, 'gap_score__16')
            write_instrument_read(up_score__18, 'up_score__18')
            print('malloc', sys.getsizeof(up_score__18), 'up_score__18')
            print(107, 181)
            left_score__18 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix__16, 'scoring_matrix__16'),
                'scoring_matrix__16', instrument_read(i__16, 'i__16'), None,
                None, False), 'scoring_matrix__16[i__16]', instrument_read(
                j__16, 'j__16') - 1, None, None, False) + instrument_read(
                gap_score__16, 'gap_score__16')
            write_instrument_read(left_score__18, 'left_score__18')
            print('malloc', sys.getsizeof(left_score__18), 'left_score__18')
            if instrument_read(diagonal_score__18, 'diagonal_score__18'
                ) >= instrument_read(up_score__18, 'up_score__18'):
                if instrument_read(diagonal_score__18, 'diagonal_score__18'
                    ) >= instrument_read(left_score__18, 'left_score__18'):
                    print(115, 185)
                    scoring_matrix__16[instrument_read(instrument_read(
                        i__16, 'i__16'), 'i__16')][instrument_read(
                        instrument_read(j__16, 'j__16'), 'j__16')
                        ] = instrument_read(diagonal_score__18,
                        'diagonal_score__18')
                    write_instrument_read_sub(scoring_matrix__16[
                        instrument_read(instrument_read(i__16, 'i__16'),
                        'i__16')],
                        "scoring_matrix__16[instrument_read(i__16, 'i__16')]",
                        instrument_read(instrument_read(j__16, 'j__16'),
                        'j__16'), None, None, False)
                    print(115, 186)
                    traceback_matrix__16[instrument_read(instrument_read(
                        i__16, 'i__16'), 'i__16')][instrument_read(
                        instrument_read(j__16, 'j__16'), 'j__16')] = 'D'
                    write_instrument_read_sub(traceback_matrix__16[
                        instrument_read(instrument_read(i__16, 'i__16'),
                        'i__16')],
                        "traceback_matrix__16[instrument_read(i__16, 'i__16')]"
                        , instrument_read(instrument_read(j__16, 'j__16'),
                        'j__16'), None, None, False)
                else:
                    print(117, 188)
                    scoring_matrix__16[instrument_read(instrument_read(
                        i__16, 'i__16'), 'i__16')][instrument_read(
                        instrument_read(j__16, 'j__16'), 'j__16')
                        ] = instrument_read(left_score__18, 'left_score__18')
                    write_instrument_read_sub(scoring_matrix__16[
                        instrument_read(instrument_read(i__16, 'i__16'),
                        'i__16')],
                        "scoring_matrix__16[instrument_read(i__16, 'i__16')]",
                        instrument_read(instrument_read(j__16, 'j__16'),
                        'j__16'), None, None, False)
                    print(117, 189)
                    traceback_matrix__16[instrument_read(instrument_read(
                        i__16, 'i__16'), 'i__16')][instrument_read(
                        instrument_read(j__16, 'j__16'), 'j__16')] = 'L'
                    write_instrument_read_sub(traceback_matrix__16[
                        instrument_read(instrument_read(i__16, 'i__16'),
                        'i__16')],
                        "traceback_matrix__16[instrument_read(i__16, 'i__16')]"
                        , instrument_read(instrument_read(j__16, 'j__16'),
                        'j__16'), None, None, False)
            elif instrument_read(up_score__18, 'up_score__18'
                ) >= instrument_read(left_score__18, 'left_score__18'):
                print(112, 192)
                scoring_matrix__16[instrument_read(instrument_read(i__16,
                    'i__16'), 'i__16')][instrument_read(instrument_read(
                    j__16, 'j__16'), 'j__16')] = instrument_read(up_score__18,
                    'up_score__18')
                write_instrument_read_sub(scoring_matrix__16[
                    instrument_read(instrument_read(i__16, 'i__16'),
                    'i__16')],
                    "scoring_matrix__16[instrument_read(i__16, 'i__16')]",
                    instrument_read(instrument_read(j__16, 'j__16'),
                    'j__16'), None, None, False)
                print(112, 193)
                traceback_matrix__16[instrument_read(instrument_read(i__16,
                    'i__16'), 'i__16')][instrument_read(instrument_read(
                    j__16, 'j__16'), 'j__16')] = 'U'
                write_instrument_read_sub(traceback_matrix__16[
                    instrument_read(instrument_read(i__16, 'i__16'),
                    'i__16')],
                    "traceback_matrix__16[instrument_read(i__16, 'i__16')]",
                    instrument_read(instrument_read(j__16, 'j__16'),
                    'j__16'), None, None, False)
            else:
                print(114, 195)
                scoring_matrix__16[instrument_read(instrument_read(i__16,
                    'i__16'), 'i__16')][instrument_read(instrument_read(
                    j__16, 'j__16'), 'j__16')] = instrument_read(left_score__18
                    , 'left_score__18')
                write_instrument_read_sub(scoring_matrix__16[
                    instrument_read(instrument_read(i__16, 'i__16'),
                    'i__16')],
                    "scoring_matrix__16[instrument_read(i__16, 'i__16')]",
                    instrument_read(instrument_read(j__16, 'j__16'),
                    'j__16'), None, None, False)
                print(114, 196)
                traceback_matrix__16[instrument_read(instrument_read(i__16,
                    'i__16'), 'i__16')][instrument_read(instrument_read(
                    j__16, 'j__16'), 'j__16')] = 'L'
                write_instrument_read_sub(traceback_matrix__16[
                    instrument_read(instrument_read(i__16, 'i__16'),
                    'i__16')],
                    "traceback_matrix__16[instrument_read(i__16, 'i__16')]",
                    instrument_read(instrument_read(j__16, 'j__16'),
                    'j__16'), None, None, False)
            if instrument_read_sub(instrument_read_sub(instrument_read(
                scoring_matrix__16, 'scoring_matrix__16'),
                'scoring_matrix__16', instrument_read(i__16, 'i__16'), None,
                None, False), 'scoring_matrix__16[i__16]', instrument_read(
                j__16, 'j__16'), None, None, False) >= instrument_read(
                max_score__16, 'max_score__16'):
                print(118, 199)
                max_score__16 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix__16,
                    'scoring_matrix__16'), 'scoring_matrix__16',
                    instrument_read(i__16, 'i__16'), None, None, False),
                    'scoring_matrix__16[i__16]', instrument_read(j__16,
                    'j__16'), None, None, False)
                write_instrument_read(max_score__16, 'max_score__16')
                print('malloc', sys.getsizeof(max_score__16), 'max_score__16')
                print(118, 200)
                max_pos__16 = instrument_read(i__16, 'i__16'), instrument_read(
                    j__16, 'j__16')
                write_instrument_read(max_pos__16, 'max_pos__16')
                print('malloc', sys.getsizeof(max_pos__16), 'max_pos__16')
            print('exit scope 18')
        print('exit scope 17')
    print(103, 202)
    i__16 = instrument_read(max_pos__16, 'max_pos__16')
    write_instrument_read(i__16, 'i__16')
    print('malloc', sys.getsizeof(i__16), 'i__16')
    print(103, 203)
    j__16 = instrument_read(max_pos__16, 'max_pos__16')
    write_instrument_read(j__16, 'j__16')
    print('malloc', sys.getsizeof(j__16), 'j__16')
    print(103, 204)
    aln1__16 = ''
    write_instrument_read(aln1__16, 'aln1__16')
    print('malloc', sys.getsizeof(aln1__16), 'aln1__16')
    print(103, 205)
    aln2__16 = ''
    write_instrument_read(aln2__16, 'aln2__16')
    print('malloc', sys.getsizeof(aln2__16), 'aln2__16')
    while instrument_read_sub(instrument_read_sub(instrument_read(
        traceback_matrix__16, 'traceback_matrix__16'),
        'traceback_matrix__16', instrument_read(i__16, 'i__16'), None, None,
        False), 'traceback_matrix__16[i__16]', instrument_read(j__16,
        'j__16'), None, None, False) != None:
        if instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix__16, 'traceback_matrix__16'),
            'traceback_matrix__16', instrument_read(i__16, 'i__16'), None,
            None, False), 'traceback_matrix__16[i__16]', instrument_read(
            j__16, 'j__16'), None, None, False) == 'D':
            print(123, 208)
            aln1__16 += instrument_read_sub(instrument_read(seq1__16,
                'seq1__16'), 'seq1__16', instrument_read(i__16, 'i__16') - 
                1, None, None, False)
            write_instrument_read(aln1__16, 'aln1__16')
            print(123, 209)
            aln2__16 += instrument_read_sub(instrument_read(seq2__16,
                'seq2__16'), 'seq2__16', instrument_read(j__16, 'j__16') - 
                1, None, None, False)
            write_instrument_read(aln2__16, 'aln2__16')
            print(123, 210)
            i__16 -= 1
            write_instrument_read(i__16, 'i__16')
            print(123, 211)
            j__16 -= 1
            write_instrument_read(j__16, 'j__16')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix__16, 'traceback_matrix__16'),
            'traceback_matrix__16', instrument_read(i__16, 'i__16'), None,
            None, False), 'traceback_matrix__16[i__16]', instrument_read(
            j__16, 'j__16'), None, None, False) == 'L':
            print(126, 213)
            aln1__16 += '-'
            write_instrument_read(aln1__16, 'aln1__16')
            print(126, 214)
            aln2__16 += instrument_read_sub(instrument_read(seq2__16,
                'seq2__16'), 'seq2__16', instrument_read(j__16, 'j__16') - 
                1, None, None, False)
            write_instrument_read(aln2__16, 'aln2__16')
            print(126, 215)
            j__16 -= 1
            write_instrument_read(j__16, 'j__16')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix__16, 'traceback_matrix__16'),
            'traceback_matrix__16', instrument_read(i__16, 'i__16'), None,
            None, False), 'traceback_matrix__16[i__16]', instrument_read(
            j__16, 'j__16'), None, None, False) == 'U':
            print(129, 217)
            aln1__16 += instrument_read_sub(instrument_read(seq1__16,
                'seq1__16'), 'seq1__16', instrument_read(i__16, 'i__16') - 
                1, None, None, False)
            write_instrument_read(aln1__16, 'aln1__16')
            print(129, 218)
            aln2__16 += '-'
            write_instrument_read(aln2__16, 'aln2__16')
            print(129, 219)
            i__16 -= 1
            write_instrument_read(i__16, 'i__16')
    print('exit scope 16')
    return instrument_read_sub(instrument_read(aln1__16, 'aln1__16'),
        'aln1__16', None, None, None, True), instrument_read_sub(
        instrument_read(aln2__16, 'aln2__16'), 'aln2__16', None, None, None,
        True)
    print('exit scope 16')


def main():
    print('enter scope 19')
    print(1, 222)
    print(134, 223)
    fasta_file__19 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fasta_example.fasta'
        )
    write_instrument_read(fasta_file__19, 'fasta_file__19')
    print('malloc', sys.getsizeof(fasta_file__19), 'fasta_file__19')
    print(134, 224)
    fastq_file__19 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fastq_example.fastq'
        )
    write_instrument_read(fastq_file__19, 'fastq_file__19')
    print('malloc', sys.getsizeof(fastq_file__19), 'fastq_file__19')
    print(134, 225)
    output_file__19 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/output_file.txt'
        )
    write_instrument_read(output_file__19, 'output_file__19')
    print('malloc', sys.getsizeof(output_file__19), 'output_file__19')
    darwin_wga_workflow(instrument_read(fasta_file__19, 'fasta_file__19'),
        instrument_read(fastq_file__19, 'fastq_file__19'), instrument_read(
        output_file__19, 'output_file__19'))
    print('exit scope 19')


if instrument_read(__name__, '__name__') == '__main__':
    main()
